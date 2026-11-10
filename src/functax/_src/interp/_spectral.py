"""Spectral interpolation: Fourier resampling and the Chebyshev read-out at points.

`Fourier` reads N periodic samples on M equispaced nodes of the same period by
zero-padding (or cropping) the discrete Fourier spectrum, the resampling of
``scipy.signal.resample``. Its read-out `FourierReadout` applies the map with
two FFTs and has a hand-written adjoint of the same form. `Chebyshev` fits the
Chebyshev coefficients of samples on Chebyshev nodes with the transform of
`ChebyshevTransform1D` (`ChebyshevFit`) and reads the series at arbitrary
points, by `clenshaw` in its interpolant and by the dense matrix
T_pk = T_k(ξ_p) in its plan.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import dataclasses
import math
from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Real

from functax._src._dtypes import promote_integers
from functax._src.basis._legendre import _check_degree
from functax._src.basis._shell import _chebyshev
from functax._src.geometry._chebyshev import CHEBYSHEV_KINDS, _mapped_nodes
from functax._src.geometry._types import AffineGrid, Axis, Points, RectilinearGrid
from functax._src.interp._fit import Fit, Identity
from functax._src.interp._locate import check_outside
from functax._src.interp._plan import Plan
from functax._src.interp._policies import Policies
from functax._src.interp._rbf import KernelReadout
from functax._src.interp._schemes import Scheme
from functax._src.interp._stencil import Readout
from functax._src.interp._tensor import target_coordinates
from functax._src.quad._rules import _interval
from functax._src.transform._chebyshev import ChebyshevTransform1D
from functax.exceptions import GeometryError, PolicyError, ValidationError


__all__ = [
    "Chebyshev",
    "ChebyshevFit",
    "ChebyshevInterpolant",
    "Fourier",
    "FourierReadout",
    "clenshaw",
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


# --- Chebyshev ---------------------------------------------------------------


def _reference_interval(interval: Any) -> tuple[float, float]:
    """(a, b) as finite floats with a < b."""
    lower, upper = _interval(interval)
    if not lower < upper:
        msg = f"'interval' must have a < b, got {interval!r}"
        raise ValidationError(msg)
    return lower, upper


def clenshaw(
    x: Real[ArrayLike, "*batch"],
    coefficients: Real[ArrayLike, " mode"],
    *,
    interval: tuple[float, float] = (-1.0, 1.0),
) -> Float[Array, "*batch"]:
    """The Chebyshev series f(x) = ∑ₖ aₖ Tₖ(ξ(x)) by Clenshaw's recurrence.

    With ξ = (2x − a − b)/(b − a), the affine map of [a, b] onto [−1, 1], and
    K coefficients,

        bₖ = aₖ + 2ξ bₖ₊₁ − bₖ₊₂,  k = K − 1, …, 1,  b_K = b_{K+1} = 0,
        f(x) = a₀ + ξ b₁ − b₂,

    in O(K) per point without forming any Tₖ, stable for any K. It equals
    ``numpy.polynomial.chebyshev.chebval(ξ, a)``. Points outside [a, b] are
    evaluated as the polynomial (extrapolated); `Chebyshev` applies the
    outside policies.

    Algorithm:
        1. Map x onto ξ in [−1, 1].
        2. Scan k = K − 1 down to 1, carrying (bₖ₊₁, bₖ₊₂).
        3. f = a₀ + ξ b₁ − b₂.

    Args:
        x: The points x, any shape. Integers become float32.
        coefficients: The K coefficients a₀, …, a_{K−1}, by degree.
        interval: The interval (a, b), with a < b, that the series lives on.

    Returns:
        f(x), with the shape of ``x``, in the common floating dtype of ``x``
        and ``coefficients``.

    Raises:
        ValidationError: If ``coefficients`` is not 1-D and non-empty, or the
            interval is not two finite numbers with a < b.

    References:
        Clenshaw, C. W. (1955). A note on the summation of Chebyshev series.
        Mathematical Tables and Other Aids to Computation 9(51), 118–120.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.interp import clenshaw
        >>> # T₂(x) = 2x² − 1
        >>> clenshaw(
        ...     jnp.array([0.0, 0.5, 1.0]), jnp.array([0.0, 0.0, 1.0])
        ... ).tolist()
        [-1.0, -0.5, 1.0]
        >>> # on [0, 4], T₁(ξ) = ξ = (x − 2)/2
        >>> clenshaw(
        ...     jnp.array([3.0]), jnp.array([0.0, 1.0]), interval=(0.0, 4.0)
        ... ).tolist()
        [0.5]
        >>> clenshaw(jnp.zeros((2, 3)), jnp.ones(5)).shape
        (2, 3)
    """
    lower, upper = _reference_interval(interval)
    series = promote_integers(jnp.asarray(coefficients))
    if series.ndim != 1 or series.shape[0] == 0:
        msg = f"'coefficients' must be 1-D and non-empty, got shape {series.shape}"
        raise ValidationError(msg)
    points = promote_integers(jnp.asarray(x))
    dtype = jnp.result_type(points, series)
    points = points.astype(dtype)
    series = series.astype(dtype)
    # ξ = (2x − a − b)/(b − a)
    reference = (2.0 * points - (lower + upper)) / (upper - lower)
    # f = a₀ for K = 1
    if series.shape[0] == 1:
        return jnp.full_like(reference, series[0])

    def step(
        carry: tuple[jax.Array, jax.Array], coefficient: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        later, latest = carry
        # bₖ = aₖ + 2ξ bₖ₊₁ − bₖ₊₂
        current = coefficient + 2.0 * reference * later - latest
        return (current, later), None

    zero = jnp.zeros_like(reference)
    # aₖ for k = K − 1, …, 1
    descending = series[:0:-1]
    (first, second), _ = jax.lax.scan(step, (zero, zero), descending)
    # f = a₀ + ξ b₁ − b₂
    return series[0] + reference * first - second


@dataclasses.dataclass(frozen=True)
class _ChebyshevGrid:
    """The degree, interval and order of an `Axis` of Chebyshev nodes (host)."""

    degree: int
    lower: float
    upper: float
    ascending: bool


def _chebyshev_grid(axis: Any, kind: str, name: str) -> _ChebyshevGrid:
    """Read N, [a, b] and the order off an `Axis` of Chebyshev nodes, on the host.

    Lobatto nodes give [a, b] as their ends; Gauss nodes give it from the first
    and last node, x₀ − x_{N−1} = 2h cos(π/(2N)). The nodes must then be those
    of `chebyshev_nodes` on [a, b], in either order.
    """
    if not isinstance(axis, Axis):
        msg = f"'{name}' must be an Axis, got {type(axis).__name__}"
        raise GeometryError(msg)
    if axis.period is not None:
        msg = f"'{name}' must not be periodic: Chebyshev nodes span an interval"
        raise GeometryError(msg)
    # an Axis has at least two nodes, so [a, b] is determined for either kind
    values = np.asarray(axis.values, dtype=np.float64)
    n_nodes = values.shape[0]
    ascending = bool(values[-1] > values[0])
    descending_values = values[::-1] if ascending else values
    first, last = descending_values[0], descending_values[-1]
    if kind == "lobatto":
        degree = n_nodes - 1
        lower, upper = float(last), float(first)
    else:
        degree = n_nodes
        # c = (x₀ + x_{N−1})/2, h = (x₀ − x_{N−1}) / (2 cos(π/(2N)))
        centre = 0.5 * (first + last)
        half_width = (first - last) / (2.0 * math.cos(math.pi / (2 * degree)))
        lower, upper = float(centre - half_width), float(centre + half_width)
    expected = _mapped_nodes(degree, kind, lower, upper)
    deviation = float(np.max(np.abs(descending_values - expected)))
    if deviation > _grid_tolerance(axis) * (upper - lower):
        msg = (
            f"'{name}' must hold the {kind} Chebyshev nodes of an interval (in "
            f"either order), off by up to {deviation!r}"
        )
        raise GeometryError(msg)
    return _ChebyshevGrid(degree, lower, upper, ascending)


class ChebyshevFit(Fit):
    """c = P C y: the first K Chebyshev coefficients of samples on Chebyshev nodes.

    C is the forward transform of `ChebyshevTransform1D` (the values taken in
    its descending node order) and P keeps the coefficients a₀, …, a_{K−1}.
    The adjoint is Cᵀ Pᵀ: Pᵀ pads with zeros, and with cₖ = 2 at the end modes
    and 1 inside,

        Gauss–Lobatto:  C_kj = 2 cos(πjk/N) / (N cₖ cⱼ) is symmetric, so Cᵀ = C,
        Gauss:          Cᵀ g = (2/N) ∑ₖ (gₖ/cₖ) Tₖ(ξⱼ), the synthesis of g/c.

    Args:
        transform: The `ChebyshevTransform1D` of the nodes.
        n_coefficients: K, at most the number of nodes.
        ascending: Whether the data run up the axis (the reverse of the
            transform's order), so they are flipped first.

    Raises:
        ValidationError: If ``n_coefficients`` is not an integer from 1 to the
            number of nodes.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import chebyshev_nodes
        >>> from functax.interp import ChebyshevFit
        >>> from functax.transform import ChebyshevTransform1D
        >>> fit = ChebyshevFit(ChebyshevTransform1D(4), 3)
        >>> x = chebyshev_nodes(4)
        >>> bool(jnp.allclose(fit.apply(1.0 + 2.0 * x), jnp.array([1.0, 2.0, 0.0])))
        True
        >>> fit.apply_adjoint(jnp.ones(3)).shape
        (5,)
    """

    transform: ChebyshevTransform1D
    n_coefficients: int = eqx.field(static=True)
    ascending: bool = eqx.field(static=True, default=False)

    def __check_init__(self) -> None:
        count = _check_degree(self.n_coefficients, "n_coefficients", minimum=1)
        if count > self.transform.n_nodes:
            msg = (
                f"'n_coefficients' must be at most the {self.transform.n_nodes} "
                f"nodes, got {count}"
            )
            raise ValidationError(msg)

    def _oriented(self, values: jax.Array) -> jax.Array:
        """The values in the transform's (descending) node order, or back."""
        return values[..., ::-1] if self.ascending else values

    def apply(self, data: Float[Array, "*batch source"]) -> Float[Array, "*batch mode"]:
        """c = P C y, over any leading batch dims.

        Raises:
            ValidationError: If the last axis of ``data`` is not the node count.
        """
        data = jnp.asarray(data)
        _check_last_axis(data, self.transform.n_nodes, "data")
        coefficients = self.transform.to_spectral(self._oriented(data))
        return coefficients[..., : self.n_coefficients]

    def apply_adjoint(
        self, coefficients: Float[Array, "*batch mode"]
    ) -> Float[Array, "*batch source"]:
        """Cᵀ Pᵀ g, over any leading batch dims.

        Raises:
            ValidationError: If the last axis of ``coefficients`` is not K.
        """
        coefficients = jnp.asarray(coefficients)
        _check_last_axis(coefficients, self.n_coefficients, "coefficients")
        n_nodes = self.transform.n_nodes
        padding = n_nodes - self.n_coefficients
        # Pᵀ g: the dropped coefficients are zero
        padded = coefficients
        if padding:
            zeros = jnp.zeros((*coefficients.shape[:-1], padding), coefficients.dtype)
            padded = einx.id(
                "... kept, ... dropped -> ... (kept + dropped)", coefficients, zeros
            )
        if self.transform.kind == "lobatto":
            # Cᵀ = C on the Lobatto nodes
            back = self.transform.to_spectral(padded)
        else:
            # Cᵀ g = (2/N) synthesis(g/c), with c₀ = 2
            halved = padded.at[..., 0].multiply(0.5)
            back = self.transform.from_spectral(halved) * (2.0 / self.transform.n)
        return self._oriented(back)

    def data_size(self, n_coefficients: int) -> int:
        """The number of nodes, N + 1 (Lobatto) or N (Gauss)."""
        del n_coefficients
        return self.transform.n_nodes


def _reference_points(
    coordinates: Float[Array, " target"],
    grid: _ChebyshevGrid,
    policies: Policies,
) -> tuple[Float[Array, " target"], Bool[Array, " target"]]:
    """The targets as ξ = (2x − a − b)/(b − a) after the outside rule, and validity.

    ``"nan"`` and ``"raise"`` mark targets off [a, b] invalid (``"raise"``
    raises on the host first); ``"clamp"`` moves them onto ±1;
    ``"extrapolate"`` keeps the polynomial. Non-finite targets are invalid
    under every rule, and read ξ = 0 so no NaN reaches a gradient.
    """
    # ξ = (2x − a − b)/(b − a)
    reference = (2.0 * coordinates - (grid.lower + grid.upper)) / (
        grid.upper - grid.lower
    )
    finite = jnp.isfinite(reference)
    # a target within rounding of an end counts as on it: 1e-12, or 8ε of a
    # coarser dtype
    epsilon = float(jnp.finfo(reference.dtype).eps)
    tolerance = max(_GRID_RTOL, 8 * epsilon)
    inside = finite & (jnp.abs(reference) <= 1.0 + tolerance)
    check_outside(inside, policies)
    valid = finite if policies.outside in ("clamp", "extrapolate") else inside
    safe = jnp.where(finite, reference, jnp.zeros((), reference.dtype))
    if policies.outside == "clamp":
        # onto ±1 with a unit derivative up to the ends (jnp.clip halves it
        # there) and zero beyond them
        one = jnp.ones((), safe.dtype)
        safe = jnp.where(safe > 1.0, one, jnp.where(safe < -1.0, -one, safe))
    elif policies.outside != "extrapolate":
        safe = jnp.where(inside, safe, jnp.zeros((), safe.dtype))
    return safe, valid


def _check_chebyshev_policies(policies: Policies) -> None:
    if policies.missing == "renormalise":
        msg = "Chebyshev plans have no local stencil for missing='renormalise'"
        raise PolicyError(msg)


class ChebyshevInterpolant(eqx.Module):
    """A fitted Chebyshev series f̂(x) = ∑ₖ cₖ Tₖ(ξ(x)), read at targets by `clenshaw`.

    Made by ``Chebyshev(n_coefficients=K).fit(axis, y)``. Calling it maps the
    targets onto ξ ∈ [−1, 1], applies the outside rule and sums the series
    by Clenshaw's recurrence inside the trace, so it runs under ``jit``,
    ``vmap``, ``grad`` and ``lax.scan``; it equals
    ``scheme.plan(axis, targets)(y)``.

    Args:
        coefficients: c, the K coefficients last, with any leading batch dims.
        source: The `Axis` of Chebyshev nodes the data lived on.
        scheme: The `Chebyshev` scheme that fitted it.
        policies: The plan policies; ``missing="fill"`` was applied at the fit.
        lower: a, the lower end of the interval.
        upper: b, the upper end.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, chebyshev_nodes
        >>> from functax.interp import Chebyshev
        >>> axis = Axis(chebyshev_nodes(8, interval=(0.0, 2.0)))
        >>> field = Chebyshev(n_coefficients=9).fit(axis, axis.values**3)
        >>> bool(
        ...     jnp.allclose(
        ...         field(jnp.array([[0.5], [1.5]])), jnp.array([0.125, 3.375])
        ...     )
        ... )
        True
        >>> field(jnp.array([[3.0]])).tolist()
        [nan]
    """

    coefficients: Float[Array, "*batch mode"]
    source: Axis
    # Any: the Chebyshev scheme is defined below
    scheme: Any
    policies: Policies
    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)

    def __call__(
        self, targets: Float[Array, "target 1"] | Points
    ) -> Float[Array, "*batch target"]:
        """f̂ at the targets, ``fill_value`` where they are invalid.

        Args:
            targets: The (M, 1) target coordinates, or `Points` with one
                coordinate.

        Returns:
            f̂ with the batch dims of the coefficients and the targets last.

        Raises:
            GeometryError: If the targets do not have one coordinate.
            OutsideDomainError: Under ``outside="raise"`` when a target is
                outside [a, b].
        """
        coordinates = (
            targets.xy if isinstance(targets, Points) else jnp.asarray(targets)
        )
        if coordinates.ndim != 2 or coordinates.shape[1] != 1:
            msg = (
                "'targets' must be (M, 1) coordinates on the Chebyshev axis, "
                f"got shape {coordinates.shape}"
            )
            raise GeometryError(msg)
        grid = _ChebyshevGrid(0, self.lower, self.upper, ascending=False)
        reference, valid = _reference_points(coordinates[:, 0], grid, self.policies)

        def evaluate(series: jax.Array) -> jax.Array:
            # the coefficients are already on [−1, 1]: ξ is the point
            return clenshaw(reference, series)

        for _ in range(self.coefficients.ndim - 1):
            evaluate = jax.vmap(evaluate)
        values = evaluate(self.coefficients)
        fill = jnp.asarray(self.policies.fill_value, dtype=values.dtype)
        return einx.where("target, ... target, -> ... target", valid, values, fill)

    def plan(self, target: Any) -> Plan:
        """The plan that ``scheme.plan(axis, target)`` builds, which reads data."""
        return self.scheme.plan(self.source, target, policies=self.policies)


class Chebyshev(Scheme):
    """The Chebyshev series of samples on Chebyshev nodes, read at arbitrary points.

    Samples y on the nodes of `chebyshev_nodes` on [a, b] (an `Axis`, in either
    order) define the series of their first K Chebyshev coefficients,

        f̂(x) = ∑ₖ cₖ Tₖ(ξ(x)),  c = P C y,  ξ = (2x − a − b)/(b − a),

    with C the forward `ChebyshevTransform1D` and P the truncation to K
    modes (K = N + 1 on Lobatto nodes interpolates the samples). It is linear
    in y: `plan` gives a `Plan` whose fit is `ChebyshevFit` and whose read-out
    is the dense `KernelReadout` T_pk = Tₖ(ξ_p), with the adjoint Cᵀ Pᵀ Tᵀ;
    `fit` gives a `ChebyshevInterpolant` that sums the series by `clenshaw`
    at targets chosen in the trace. A field of degree below K is reproduced
    exactly. Targets off [a, b] follow ``policies.outside``.

    Algorithm:
        1. Read N, [a, b] and the order off the axis (host), checking that its
           nodes are Chebyshev nodes of ``kind``.
        2. Fit: c = P C y, the forward transform truncated to K modes.
        3. Read: f̂ = T c with T_pk = Tₖ(ξ_p) (plan), or Clenshaw's recurrence
           (interpolant).

    Args:
        n_coefficients: K, the number of coefficients kept, from 1 to the
            number of nodes.
        kind: The node kind of the source axis, ``"lobatto"`` (the default) or
            ``"gauss"``.

    Raises:
        ValidationError: If ``n_coefficients`` is not a positive integer, or
            ``kind`` is unknown; when planned or fitted, if K exceeds the
            number of nodes.
        GeometryError: When planned or fitted, if the source is not an `Axis`
            of Chebyshev nodes of ``kind``.
        PolicyError: When planned or fitted under ``missing="renormalise"``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points, chebyshev_nodes
        >>> from functax.interp import Chebyshev
        >>> axis = Axis(chebyshev_nodes(16, interval=(-2.0, 2.0)))
        >>> targets = Points(jnp.array([[-1.5], [0.3], [1.9]]))
        >>> plan = Chebyshev(n_coefficients=17).plan(axis, targets)
        >>> values = plan(jnp.exp(axis.values))
        >>> bool(jnp.allclose(values, jnp.exp(targets.xy[:, 0])))
        True
        >>> plan.adjoint(jnp.ones(3)).shape
        (17,)
        >>> Chebyshev(n_coefficients=4).get_config()
        {'class': 'Chebyshev', 'n_coefficients': 4, 'kind': 'lobatto'}
    """

    n_coefficients: int = eqx.field(static=True)
    kind: str = eqx.field(static=True, default="lobatto")

    def __check_init__(self) -> None:
        _check_degree(self.n_coefficients, "n_coefficients", minimum=1)
        if self.kind not in CHEBYSHEV_KINDS:
            msg = f"'kind' must be one of {list(CHEBYSHEV_KINDS)}, got {self.kind!r}"
            raise ValidationError(msg)

    def _fit(self, source: Any) -> tuple[_ChebyshevGrid, ChebyshevFit]:
        grid = _chebyshev_grid(source, self.kind, "source")
        transform = ChebyshevTransform1D(grid.degree, kind=self.kind)
        fit = ChebyshevFit(transform, self.n_coefficients, grid.ascending)
        return grid, fit

    def plan(
        self, source: Any, target: Any, *, policies: Policies | None = None
    ) -> Plan:
        """The plan that reads samples on the Chebyshev ``source`` at ``target``.

        Args:
            source: An `Axis` of Chebyshev nodes of ``kind``, in either order.
            target: `Points` with one coordinate, an `Axis`, or a 1-D grid.
            policies: The outside and missing rules; ``boundary`` and
                ``extent`` do not apply.

        Returns:
            The `Plan`, with a `ChebyshevFit` and the dense `KernelReadout` T.

        Raises:
            GeometryError: If the source is not an `Axis` of Chebyshev nodes.
            ValidationError: If K exceeds the number of nodes.
            PolicyError: Under ``missing="renormalise"``.
            OutsideDomainError: Under ``outside="raise"`` when a target is
                outside [a, b].
        """
        policies = Policies() if policies is None else policies
        _check_chebyshev_policies(policies)
        grid, fit = self._fit(source)
        coordinates = target_coordinates(source, target)
        if coordinates.shape[1] != 1:
            msg = (
                f"'target' has {coordinates.shape[1]} coordinates per point, but "
                "the Chebyshev source has 1 axis"
            )
            raise GeometryError(msg)
        reference, valid = _reference_points(coordinates[:, 0], grid, policies)
        # T_pk = Tₖ(ξ_p)
        table = _chebyshev(reference, self.n_coefficients)
        is_grid = isinstance(target, RectilinearGrid | AffineGrid)
        target_shape = target.shape if is_grid else None
        return Plan(fit, KernelReadout(table, valid), policies, None, target_shape)

    def fit(
        self,
        source: Axis,
        data: Real[ArrayLike, "*batch source"],
        *,
        policies: Policies | None = None,
    ) -> ChebyshevInterpolant:
        """The series on ``source`` from y, read later at targets chosen in the trace.

        Args:
            source: An `Axis` of Chebyshev nodes of ``kind``, in either order.
            data: y, the samples last, with any leading batch dims.
            policies: The outside and missing rules.

        Returns:
            The `ChebyshevInterpolant`.

        Raises:
            GeometryError: If the source is not an `Axis` of Chebyshev nodes.
            ValidationError: If K exceeds the number of nodes, or the last axis
                of ``data`` is not the node count.
            PolicyError: Under ``missing="renormalise"``.
        """
        policies = Policies() if policies is None else policies
        _check_chebyshev_policies(policies)
        grid, fit = self._fit(source)
        data = promote_integers(jnp.asarray(data))
        if policies.missing == "fill":
            fill = jnp.asarray(policies.fill_value, dtype=data.dtype)
            data = jnp.where(jnp.isfinite(data), data, fill)
        coefficients = fit.apply(data)
        return ChebyshevInterpolant(
            coefficients, source, self, policies, grid.lower, grid.upper
        )
