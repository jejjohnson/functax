"""Chebyshev transforms: nodal values on Chebyshev nodes to Chebyshev coefficients.

`ChebyshevTransform1D` and `ChebyshevTransform2D` are spectraldiffx's
transforms at ``d68f8f6`` (``_src/chebyshev/transforms.py`` and the FFT kernels
of ``ChebyshevGrid1D.transform``), with their method names and normalisation:
u(x) = ∑ₖ aₖ Tₖ(ξ(x)) with the true coefficients aₖ, so a constant c maps to
(c, 0, …, 0) and Tₙ to the unit vector eₙ. The nodes are those of
`chebyshev_nodes`, in spectraldiffx's order from the upper end down; the
derivatives, filters and solvers stay in spectraldiffx.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Real

from functax._src._dtypes import promote_integers
from functax._src.basis._legendre import _check_degree
from functax._src.geometry._chebyshev import CHEBYSHEV_KINDS
from functax.exceptions import ValidationError


__all__ = [
    "ChebyshevTransform1D",
    "ChebyshevTransform2D",
]


def _check_kind(kind: Any) -> None:
    if kind not in CHEBYSHEV_KINDS:
        msg = f"'kind' must be one of {list(CHEBYSHEV_KINDS)}, got {kind!r}"
        raise ValidationError(msg)


def _node_count(degree: int, kind: str) -> int:
    """N + 1 Lobatto nodes, or N Gauss nodes."""
    return degree + 1 if kind == "lobatto" else degree


def _as_real(values: Any, shape: tuple[int, ...], name: str) -> jax.Array:
    """``values`` as a real floating array whose trailing axes are ``shape``."""
    array = promote_integers(jnp.asarray(values))
    if not jnp.issubdtype(array.dtype, jnp.floating):
        msg = f"'{name}' must be real, got dtype {array.dtype}"
        raise ValidationError(msg)
    if array.ndim < len(shape) or array.shape[array.ndim - len(shape) :] != shape:
        msg = f"'{name}' must have trailing axes of shape {shape}, got {array.shape}"
        raise ValidationError(msg)
    return array


def _lobatto(values: jax.Array, degree: int, *, inverse: bool) -> jax.Array:
    """The DCT-I Chebyshev transform on the N + 1 Gauss–Lobatto nodes, last axis.

    Forward: aₖ = (2 / (N cₖ)) ∑ⱼ'' uⱼ cos(πjk/N), with c₀ = c_N = 2 and
    cₖ = 1 otherwise (the double prime halves the j = 0 and j = N terms): a
    DCT-I, evaluated as a length-2N real FFT of the even extension.
    Inverse: uⱼ = ∑ₖ aₖ cos(πjk/N).
    """
    if not inverse:
        # the even extension (u₀, …, u_N, u_{N−1}, …, u₁), of length 2N;
        # for N = 1 the reflected part is empty and u is the extension
        if degree == 1:
            extended = values
        else:
            reflected = values[..., -2:0:-1]
            extended = einx.id(
                "... node, ... interior -> ... (node + interior)", values, reflected
            )
        # rfft/N gives the 2/cₖ-weighted values: halve the two end modes
        coefficients = jnp.fft.rfft(extended).real / degree
        return coefficients.at[..., 0].multiply(0.5).at[..., degree].multiply(0.5)
    # irfft(X, n=2N)[j] = (1/2N)[X₀ + X_N(−1)ʲ + 2 ∑_{0<k<N} Xₖ cos(πjk/N)], so
    # X = N a with the end modes doubled gives ∑ₖ aₖ cos(πjk/N)
    doubled = values.at[..., 0].multiply(2.0).at[..., degree].multiply(2.0)
    spectrum = degree * doubled
    synthesis = jnp.fft.irfft(spectrum + 0j, n=2 * degree)
    return synthesis[..., : degree + 1]


def _gauss(values: jax.Array, degree: int, *, inverse: bool) -> jax.Array:
    """The DCT-II Chebyshev transform on the N Gauss nodes, last axis.

    Forward: a₀ = (1/N) ∑ⱼ uⱼ and aₖ = (2/N) ∑ⱼ uⱼ cos(πk(2j + 1)/(2N)) for
    k > 0, a length-2N real FFT of the zero-padded samples with a half-sample
    phase shift. Inverse: uⱼ = ∑ₖ aₖ Tₖ(xⱼ), the synthesis, as a length-2N
    inverse FFT.
    """
    # the real and complex dtypes of the input (float16 computes in float32),
    # promoted explicitly so float64 stays float64
    real_dtype = jnp.promote_types(values.dtype, jnp.float32)
    complex_dtype = jnp.promote_types(real_dtype, jnp.complex64)
    if not inverse:
        # Zₖ = ∑ⱼ uⱼ exp(−iπ(2j + 1)k/(2N)): the FFT of the zero-padded samples,
        # shifted by half a sample
        spectrum = jnp.fft.rfft(values.astype(real_dtype), n=2 * degree)
        modes = jnp.arange(degree + 1, dtype=real_dtype)
        phase = jnp.exp((-1j * jnp.pi * modes / (2 * degree)).astype(complex_dtype))
        shifted = einx.multiply("... mode, mode -> ... mode", spectrum, phase)
        # aₖ = (2/N) Re Zₖ, with a₀ halved to (1/N) ∑ⱼ uⱼ
        coefficients = shifted[..., :degree].real * (2.0 / degree)
        return coefficients.at[..., 0].multiply(0.5)
    # uₙ = Re ∑ₖ aₖ exp(iπk(2n + 1)/(2N)) = Re[2N IFFT_{2N}(h ⊕ 0)ₙ] with
    # hₖ = aₖ exp(iπk/(2N)); a₀ is not doubled, as the forward halved it
    modes = jnp.arange(degree, dtype=real_dtype)
    phase = jnp.exp((1j * jnp.pi * modes / (2 * degree)).astype(complex_dtype))
    shifted = einx.multiply(
        "... mode, mode -> ... mode", values.astype(complex_dtype), phase
    )
    synthesis = jnp.fft.ifft(shifted, n=2 * degree)
    return (2 * degree * synthesis.real)[..., :degree]


def _transform_last(
    values: jax.Array, degree: int, kind: str, *, inverse: bool
) -> jax.Array:
    """The 1-D transform of either kind along the last axis."""
    if kind == "lobatto":
        return _lobatto(values, degree, inverse=inverse)
    return _gauss(values, degree, inverse=inverse)


class ChebyshevTransform1D(eqx.Module):
    """The forward and inverse Chebyshev transform on N Chebyshev nodes.

    For u on the nodes xⱼ of `chebyshev_nodes` (ξⱼ = (2xⱼ − a − b)/(b − a)),

        u(xⱼ) = ∑ₖ aₖ Tₖ(ξⱼ),    Tₖ(ξ) = cos(k arccos ξ),

    with the true Chebyshev coefficients aₖ:

        Gauss–Lobatto:  aₖ = (2 / (N cₖ)) ∑ⱼ'' uⱼ cos(πjk/N), c₀ = c_N = 2,
        Gauss:          aₖ = (2 / (N cₖ)) ∑ⱼ uⱼ cos(πk(2j + 1)/(2N)), c₀ = 2,

    and cₖ = 1 otherwise (the double prime halves the end terms). The
    coefficients do not depend on the interval [a, b]. Forward then inverse,
    or inverse then forward, is the identity up to rounding. spectraldiffx's
    class (``d68f8f6``) with its method names and normalisation; it wrapped a
    ``ChebyshevGrid1D``, and here it takes the grid's N and node type. The
    values are ordered like the nodes, from the upper end of the interval
    down (spectraldiffx's order); `clenshaw_curtis` ascends, so its rule pairs
    with the values reversed.

    Algorithm:
        1. Gauss–Lobatto: a length-2N real FFT of the even extension of u (a
           DCT-I), divided by N, with the two end modes halved.
        2. Gauss: a length-2N real FFT of u padded with zeros, shifted by half
           a sample (a DCT-II), times 2/N, with a₀ halved.
        3. The inverses synthesise ∑ₖ aₖ Tₖ(ξⱼ) with the matching inverse FFT.

    Args:
        n: N, the polynomial degree (spectraldiffx's ``N``), at least 1: the
            transform acts on N + 1 Lobatto values or N Gauss values.
        kind: ``"lobatto"`` (spectraldiffx's ``"gauss-lobatto"``, the default)
            or ``"gauss"``.

    Raises:
        ValidationError: If ``n`` is not an integer of at least 1, or ``kind``
            is unknown.

    References:
        Trefethen, L. N. (2000). Spectral Methods in MATLAB. SIAM.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import chebyshev_nodes
        >>> from functax.transform import ChebyshevTransform1D
        >>> transform = ChebyshevTransform1D(16)
        >>> u = jnp.sin(jnp.pi * chebyshev_nodes(16))
        >>> bool(jnp.allclose(transform.from_spectral(transform.to_spectral(u)), u))
        True
        >>> # T₃ on the Gauss nodes is the unit vector e₃
        >>> gauss = ChebyshevTransform1D(6, kind="gauss")
        >>> t3 = jnp.cos(3 * jnp.arccos(chebyshev_nodes(6, kind="gauss")))
        >>> bool(jnp.allclose(gauss.to_spectral(t3), jnp.eye(6)[3]))
        True
        >>> transform(jnp.ones((2, 17))).shape
        (2, 17)
    """

    n: int = eqx.field(static=True)
    kind: str = eqx.field(static=True, default="lobatto")

    def __check_init__(self) -> None:
        _check_degree(self.n, "n", minimum=1)
        _check_kind(self.kind)

    @property
    def n_nodes(self) -> int:
        """The number of values: N + 1 on Lobatto nodes, N on Gauss nodes."""
        return _node_count(self.n, self.kind)

    def to_spectral(
        self, u: Real[ArrayLike, "*batch node"]
    ) -> Float[Array, "*batch mode"]:
        """Forward: nodal values u on the last axis to the coefficients aₖ.

        Args:
            u: The values at the nodes, last, with any leading batch dims.
                Integers become float32.

        Returns:
            The coefficients a₀, …, in the dtype of ``u``.

        Raises:
            ValidationError: If ``u`` is complex or its last axis is not the
                node count.
        """
        values = _as_real(u, (self.n_nodes,), "u")
        return _transform_last(values, self.n, self.kind, inverse=False)

    def from_spectral(
        self, a: Real[ArrayLike, "*batch mode"]
    ) -> Float[Array, "*batch node"]:
        """Inverse: the coefficients aₖ on the last axis to the nodal values.

        Args:
            a: The coefficients, last, with any leading batch dims.

        Returns:
            u(xⱼ) = ∑ₖ aₖ Tₖ(ξⱼ) at the nodes, in the dtype of ``a``.

        Raises:
            ValidationError: If ``a`` is complex or its last axis is not the
                node count.
        """
        coefficients = _as_real(a, (self.n_nodes,), "a")
        return _transform_last(coefficients, self.n, self.kind, inverse=True)

    def __call__(
        self, u: Real[ArrayLike, "*batch size"], inverse: bool = False
    ) -> Float[Array, "*batch size"]:
        """`to_spectral`, or `from_spectral` with ``inverse=True``."""
        if inverse:
            return self.from_spectral(u)
        return self.to_spectral(u)


class ChebyshevTransform2D(eqx.Module):
    """The tensor-product Chebyshev transform on an (N_y, N_x) grid of nodes.

    For u on the grid of `chebyshev_nodes` in y (second-to-last axis) and x
    (last axis), spectraldiffx's (Ny_pts, Nx_pts) layout,

        u(yᵢ, xⱼ) = ∑ₗ ∑ₖ a_{lk} Tₗ(ηᵢ) Tₖ(ξⱼ),

    so T₂(ξ) T₃(η) maps to the unit coefficient at [3, 2]. Forward applies the
    1-D transform of `ChebyshevTransform1D` along x and then along y; inverse
    along y and then along x. spectraldiffx's class (``d68f8f6``) with its
    method names and normalisation, taking the grid's Nx, Ny and node type.

    Algorithm:
        1. Forward: the 1-D forward transform on the last axis (x), then on the
           second-to-last (y), moved last and back.
        2. Inverse: the 1-D inverse on y, then on x.

    Args:
        nx: N_x, the polynomial degree in x, at least 1.
        ny: N_y, the polynomial degree in y, at least 1.
        kind: ``"lobatto"`` (the default) or ``"gauss"``, for both axes.

    Raises:
        ValidationError: If ``nx`` or ``ny`` is not an integer of at least 1,
            or ``kind`` is unknown.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import chebyshev_nodes
        >>> from functax.transform import ChebyshevTransform2D
        >>> transform = ChebyshevTransform2D(8, 6)
        >>> x, y = chebyshev_nodes(8), chebyshev_nodes(6)
        >>> # T₂(x) T₃(y), on rows y and columns x
        >>> u = jnp.outer(4 * y**3 - 3 * y, 2 * x**2 - 1)
        >>> unit = jnp.zeros((7, 9)).at[3, 2].set(1.0)
        >>> bool(jnp.allclose(transform.to_spectral(u), unit))
        True
        >>> transform.from_spectral(jnp.ones((3, 7, 9))).shape
        (3, 7, 9)
    """

    nx: int = eqx.field(static=True)
    ny: int = eqx.field(static=True)
    kind: str = eqx.field(static=True, default="lobatto")

    def __check_init__(self) -> None:
        _check_degree(self.nx, "nx", minimum=1)
        _check_degree(self.ny, "ny", minimum=1)
        _check_kind(self.kind)

    @property
    def shape(self) -> tuple[int, int]:
        """The grid shape (y, x): N + 1 per axis on Lobatto nodes, N on Gauss."""
        return (_node_count(self.ny, self.kind), _node_count(self.nx, self.kind))

    def _along_y(self, values: jax.Array, *, inverse: bool) -> jax.Array:
        """The 1-D transform along the second-to-last axis (y)."""
        columns = einx.id("... y x -> ... x y", values)
        transformed = _transform_last(columns, self.ny, self.kind, inverse=inverse)
        return einx.id("... x y -> ... y x", transformed)

    def to_spectral(
        self, u: Real[ArrayLike, "*batch node_y node_x"]
    ) -> Float[Array, "*batch mode_y mode_x"]:
        """Forward: values on the (y, x) grid to the coefficients a_{lk}.

        Args:
            u: The values, (y, x) last, with any leading batch dims.

        Returns:
            The coefficients, indexed [l, k] by the modes in y and in x.

        Raises:
            ValidationError: If ``u`` is complex or its last two axes are not
                the grid shape.
        """
        values = _as_real(u, self.shape, "u")
        along_x = _transform_last(values, self.nx, self.kind, inverse=False)
        return self._along_y(along_x, inverse=False)

    def from_spectral(
        self, a: Real[ArrayLike, "*batch mode_y mode_x"]
    ) -> Float[Array, "*batch node_y node_x"]:
        """Inverse: the coefficients a_{lk} to the values on the (y, x) grid.

        Args:
            a: The coefficients, (l, k) last, with any leading batch dims.

        Returns:
            u(yᵢ, xⱼ) = ∑ₗ ∑ₖ a_{lk} Tₗ(ηᵢ) Tₖ(ξⱼ).

        Raises:
            ValidationError: If ``a`` is complex or its last two axes are not
                the grid shape.
        """
        coefficients = _as_real(a, self.shape, "a")
        along_y = self._along_y(coefficients, inverse=True)
        return _transform_last(along_y, self.nx, self.kind, inverse=True)

    def __call__(
        self, u: Real[ArrayLike, "*batch rows cols"], inverse: bool = False
    ) -> Float[Array, "*batch rows cols"]:
        """`to_spectral`, or `from_spectral` with ``inverse=True``."""
        if inverse:
            return self.from_spectral(u)
        return self.to_spectral(u)
