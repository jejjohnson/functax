"""Spectral interpolation: band-limited Fourier resampling of a periodic axis.

`Fourier` reads N periodic samples on M equispaced nodes of the same period by
zero-padding (or cropping) the discrete Fourier spectrum, the resampling of
``scipy.signal.resample``. Its read-out `FourierReadout` applies the map with
two FFTs and has a hand-written adjoint of the same form. The Chebyshev
read-out of #43 waits on the Chebyshev transform of #46.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from functax._src.geometry._types import Axis
from functax._src.interp._fit import Identity
from functax._src.interp._plan import Plan
from functax._src.interp._policies import Policies
from functax._src.interp._schemes import Scheme
from functax._src.interp._stencil import Readout
from functax.exceptions import GeometryError, PolicyError, ValidationError


__all__ = [
    "Fourier",
    "FourierReadout",
    "resampling_modes",
]


# a node within 1e-12 P of x₀ + jP/N counts as on the grid (8ε for float32)
_GRID_RTOL = 1e-12


def resampling_modes(
    n_source: int, n_target: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The mode map S of N → M resampling: (output bin, input bin, weight) triples.

    Bins are in full-FFT order (0, 1, …, then the negative wavenumbers). With
    m = min(N, M), the wavenumbers |k| < m/2 copy across with weight 1. For
    even m and N ≠ M the unpaired bin k = m/2 follows ``scipy.signal.resample``:
    upsampling splits input bin N/2 into output bins ±N/2 with weight ½ each,
    and downsampling adds input bins ±M/2 into output bin M/2.

    Args:
        n_source: N, the number of input samples.
        n_target: M, the number of output samples.

    Returns:
        Three 1-D arrays of one length: the output bins, the input bins, and
        the weights (float64).
    """
    m = min(n_source, n_target)
    # the bins k = 0, …, ⌊m/2⌋, then the negative bins k = −1, …, −(m − ⌊m/2⌋ − 1)
    n_nonnegative = m // 2 + 1
    positive = np.arange(n_nonnegative)
    negative = np.arange(1, m - n_nonnegative + 1)
    outputs = [positive, n_target - negative]
    inputs = [positive, n_source - negative]
    weights = [np.ones(positive.size), np.ones(negative.size)]
    if m % 2 == 0 and n_source != n_target:
        half = m // 2
        if n_target > n_source:
            # split input bin N/2 into output bins +N/2 and −N/2
            weights[0][half] = 0.5
            outputs.append(np.array([n_target - half]))
            inputs.append(np.array([half]))
            weights.append(np.array([0.5]))
        else:
            # unite input bins +M/2 and −M/2 into output bin M/2
            outputs.append(np.array([half]))
            inputs.append(np.array([n_source - half]))
            weights.append(np.array([1.0]))
    return (
        np.concatenate(outputs),
        np.concatenate(inputs),
        np.concatenate(weights),
    )


class FourierReadout(Readout):
    """H y = (M/N) Re F⁻¹_M[S F_N y], band-limited resampling by two FFTs.

    F_N is the length-N DFT, F⁻¹_M the inverse length-M DFT, and S the (M, N)
    mode map of `resampling_modes` (zero-padding when M > N, cropping when
    M < N). Writing the DFTs as matrices, H = (1/N) Re(F*_M S F_N), so the
    adjoint has the same shape with the roles swapped:

        Hᵀ r = Re F⁻¹_N[Sᵀ F_M r].

    Args:
        n_source: N, the number of periodic input samples.
        n_target: M, the number of output samples.
        dtype: The floating dtype of the mode weights and of `operator`.

    Raises:
        ValidationError: If N or M is below 1.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.interp import FourierReadout
        >>> read = FourierReadout(4, 8)
        >>> read.apply(jnp.array([1.0, 0.0, -1.0, 0.0])).shape
        (8,)
        >>> # cos(2πx) on 4 nodes, read on 8
        >>> x = jnp.arange(8) / 8
        >>> values = read.apply(jnp.cos(2 * jnp.pi * jnp.arange(4) / 4))
        >>> bool(jnp.allclose(values, jnp.cos(2 * jnp.pi * x)))
        True
    """

    output_modes: Int[Array, " mode"]
    input_modes: Int[Array, " mode"]
    mode_weights: Float[Array, " mode"]
    valid: Bool[Array, " target"]
    n_source: int = eqx.field(static=True)
    n_target: int = eqx.field(static=True)

    def __init__(self, n_source: int, n_target: int, *, dtype: Any = None) -> None:
        for name, size in (("n_source", n_source), ("n_target", n_target)):
            if isinstance(size, bool) or not isinstance(size, int | np.integer):
                msg = f"'{name}' must be an integer, got {size!r}"
                raise ValidationError(msg)
            if size < 1:
                msg = f"'{name}' must be positive, got {size}"
                raise ValidationError(msg)
        outputs, inputs, weights = resampling_modes(int(n_source), int(n_target))
        dtype = jnp.result_type(float) if dtype is None else dtype
        self.output_modes = jnp.asarray(outputs)
        self.input_modes = jnp.asarray(inputs)
        self.mode_weights = jnp.asarray(weights, dtype=dtype)
        self.valid = jnp.ones(int(n_target), dtype=bool)
        self.n_source = int(n_source)
        self.n_target = int(n_target)

    def _map_modes(
        self,
        spectrum: jax.Array,
        gather: jax.Array,
        scatter: jax.Array,
        size: int,
    ) -> jax.Array:
        """Z[scatter] += w · X[gather] over the last axis: S X, or Sᵀ X swapped."""
        weights = self.mode_weights.astype(spectrum.real.dtype)
        contributions = spectrum[..., gather] * weights
        zeros = jnp.zeros((*spectrum.shape[:-1], size), spectrum.dtype)
        return zeros.at[..., scatter].add(contributions)

    def apply(
        self, coefficients: Float[Array, "*batch source"]
    ) -> Float[Array, "*batch target"]:
        """H y = (M/N) Re F⁻¹_M[S F_N y], over any leading batch dims.

        Args:
            coefficients: y, the N periodic samples, last.

        Returns:
            The M resampled values, last.

        Raises:
            ValidationError: If the last axis of ``coefficients`` is not N.
        """
        _check_last_axis(coefficients, self.n_source, "coefficients")
        # F_N y
        spectrum = jnp.fft.fft(coefficients)
        # S F_N y
        padded = self._map_modes(
            spectrum, self.input_modes, self.output_modes, self.n_target
        )
        # (M/N) Re F⁻¹_M[S F_N y]
        resampled = jnp.fft.ifft(padded).real
        return resampled * (self.n_target / self.n_source)

    def apply_adjoint(
        self, residual: Float[Array, "*batch target"]
    ) -> Float[Array, "*batch source"]:
        """Hᵀ r = Re F⁻¹_N[Sᵀ F_M r], over any leading batch dims.

        Args:
            residual: r, the M values, last.

        Returns:
            Hᵀ r on the N samples, last.

        Raises:
            ValidationError: If the last axis of ``residual`` is not M.
        """
        _check_last_axis(residual, self.n_target, "residual")
        # F_M r
        spectrum = jnp.fft.fft(residual)
        # Sᵀ F_M r
        cropped = self._map_modes(
            spectrum, self.output_modes, self.input_modes, self.n_source
        )
        # Re F⁻¹_N[Sᵀ F_M r]
        return jnp.fft.ifft(cropped).real

    def operator(self) -> lx.FunctionLinearOperator:
        """H as an (M, N) lineax operator in the dtype of the mode weights."""
        structure = jax.ShapeDtypeStruct((self.n_source,), self.mode_weights.dtype)
        return lx.FunctionLinearOperator(self.apply, structure)


def _check_last_axis(array: jax.Array, size: int, name: str) -> None:
    if array.ndim < 1 or array.shape[-1] != size:
        msg = (
            f"'{name}' must have a last axis of length {size}, got shape {array.shape}"
        )
        raise ValidationError(msg)


def _grid_tolerance(axis: Axis) -> float:
    """The relative tolerance of a node: 1e-12, or 8ε of a coarser dtype."""
    epsilon = float(np.finfo(np.asarray(axis.values).dtype).eps)
    return max(_GRID_RTOL, 8 * epsilon)


def _periodic_grid(axis: Any, name: str) -> tuple[float, float, int]:
    """(x₀, P, N) of a periodic axis whose nodes are x₀ + jP/N, on the host."""
    if not isinstance(axis, Axis):
        msg = f"'{name}' must be an Axis, got {type(axis).__name__}"
        raise GeometryError(msg)
    if axis.period is None:
        msg = f"'{name}' must be periodic (an Axis with a period)"
        raise GeometryError(msg)
    values = np.asarray(axis.values, dtype=np.float64)
    n_nodes = values.shape[0]
    period = axis.period
    # xⱼ = x₀ + jP/N: ascending, equally spaced, tiling the whole period
    expected = values[0] + np.arange(n_nodes) * (period / n_nodes)
    deviation = np.max(np.abs(values - expected))
    if deviation > _grid_tolerance(axis) * period:
        msg = (
            f"'{name}' must have the nodes x0 + j * period / n (ascending, "
            f"equally spaced over the period), off by up to {deviation!r}"
        )
        raise GeometryError(msg)
    return float(values[0]), period, n_nodes


class Fourier(Scheme):
    """Band-limited Fourier resampling from N to M nodes of one periodic axis.

    The N samples define the trigonometric polynomial with their discrete
    Fourier coefficients ŷ = F_N y, and the plan reads it on M nodes of the same
    period from the same origin by zero-padding the spectrum (M > N) or
    cropping it (M < N):

        f̂ = (M/N) F⁻¹_M[S ŷ].

    For even min(N, M) the unpaired Nyquist bin is split (upsampling) or
    folded (downsampling) as ``scipy.signal.resample`` does, so a field whose
    spectrum lies in |k| < min(N, M)/2 is resampled exactly and upsampling
    then downsampling returns the samples. The axis must be periodic or the
    field band-limited; at a jump the read-out rings (Gibbs).

    Algorithm:
        1. Check both axes: periodic with one period P, ascending, nodes
           x₀ + jP/N and x₀ + jP/M from the same x₀ (host).
        2. Build S, the mode map of `resampling_modes`, into a `FourierReadout`.
        3. Apply: ŷ = F_N y, then f̂ = (M/N) Re F⁻¹_M[S ŷ].

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis
        >>> from functax.interp import Fourier
        >>> coarse = Axis(jnp.arange(16) / 16, period=1.0)
        >>> fine = Axis(jnp.arange(40) / 40, period=1.0)
        >>> plan = Fourier().plan(coarse, fine)
        >>> # sin(2π·3x) is band-limited on 16 nodes, so it resamples exactly
        >>> values = plan(jnp.sin(6 * jnp.pi * coarse.values))
        >>> bool(jnp.allclose(values, jnp.sin(6 * jnp.pi * fine.values)))
        True
        >>> plan.adjoint(jnp.ones(40)).shape
        (16,)
        >>> Fourier().get_config()
        {'class': 'Fourier'}

    References:
        Oppenheim and Schafer, *Discrete-Time Signal Processing*, ch. 4
        (band-limited interpolation); ``scipy.signal.resample``.
    """

    def plan(
        self, source: Any, target: Any, *, policies: Policies | None = None
    ) -> Plan:
        """The plan that reads N periodic samples at the M nodes of ``target``.

        Args:
            source: A periodic, ascending, equally spaced `Axis` whose N nodes
                tile its period.
            target: An `Axis` of M nodes of the same kind, with the same period
                and first node.
            policies: The missing-sample rule (``"propagate"`` or ``"fill"``);
                no target is outside a periodic axis.

        Returns:
            The `Plan`, with the identity fit and a `FourierReadout`.

        Raises:
            GeometryError: If an axis is not periodic, ascending and equally
                spaced over its whole period, or the two differ in period or
                origin.
            PolicyError: Under ``missing="renormalise"``, which needs a local
                stencil.
        """
        policies = Policies() if policies is None else policies
        if policies.missing == "renormalise":
            msg = "Fourier plans have no local stencil for missing='renormalise'"
            raise PolicyError(msg)
        source_origin, period, n_source = _periodic_grid(source, "source")
        target_origin, target_period, n_target = _periodic_grid(target, "target")
        if not math.isclose(period, target_period, rel_tol=_GRID_RTOL):
            msg = (
                f"'target' must have the period of 'source' {period!r}, "
                f"got {target_period!r}"
            )
            raise GeometryError(msg)
        tolerance = max(_grid_tolerance(source), _grid_tolerance(target))
        if abs(target_origin - source_origin) > tolerance * period:
            msg = (
                f"'target' must start at the first node of 'source' "
                f"{source_origin!r}, got {target_origin!r}"
            )
            raise GeometryError(msg)
        read = FourierReadout(n_source, n_target, dtype=source.values.dtype)
        return Plan(Identity(), read, policies)
