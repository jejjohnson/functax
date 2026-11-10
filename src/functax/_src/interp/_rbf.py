"""Radial basis function interpolation: a kernel, a polynomial tail, one solve.

The interpolant f̂(x) = ∑ⱼ cⱼ k(x, xⱼ) + ∑ₚ bₚ pₚ(x) is fitted once per field by
`KernelSolve` and read at the targets by a dense `KernelReadout`,
[K* P*], so an `RBF` plan is a `Plan` like any other, with an adjoint and a
lineax operator.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import itertools
import math
from collections.abc import Callable
from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float

from functax._src.geometry._types import Points, RectilinearGrid
from functax._src.interp._fit import KernelSolve
from functax._src.interp._kernels import Kernel, kernel_matrix
from functax._src.interp._plan import Plan
from functax._src.interp._policies import Policies
from functax._src.interp._schemes import Scheme
from functax._src.interp._stencil import Readout
from functax._src.interp._tensor import _target_coordinates
from functax._typing import Shape
from functax.exceptions import GeometryError, PolicyError, ValidationError


__all__ = [
    "RBF",
    "KernelReadout",
    "KernelSystem",
    "PolynomialTail",
    "kernel_system",
]


type Tail = Callable[[jax.Array], jax.Array]
"""P(x), the (n, P) tail or drift matrix at n points."""


class KernelReadout(Readout):
    """The dense read-out of a kernel scheme: f̂(X*) = K* c + P* b.

    Row k holds the cross-kernel k(x*ₖ, xⱼ) to every source and the tail
    terms pₚ(x*ₖ), so H = [K* P*] is (M, N + P); its adjoint is
    Hᵀ r = [K*ᵀ r; P*ᵀ r]. Invalid rows (padding targets) are zero.

    Args:
        matrix: H, the (M, N + P) read-out.
        valid: Whether each row is a real target; all true when ``None``.

    Raises:
        ValidationError: If ``matrix`` is not 2-D and floating, or ``valid``
            is not boolean with one entry per row.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.interp import KernelReadout
        >>> read = KernelReadout(jnp.array([[1.0, 2.0, 1.0], [0.0, 1.0, 1.0]]))
        >>> read.apply(jnp.array([1.0, 1.0, 2.0])).tolist()
        [5.0, 3.0]
        >>> read.apply_adjoint(jnp.array([1.0, 1.0])).tolist()
        [1.0, 3.0, 2.0]
    """

    matrix: Float[Array, "target coefficient"]
    valid: Bool[Array, " target"]

    def __init__(
        self,
        matrix: Float[Array, "target coefficient"],
        valid: Bool[Array, " target"] | None = None,
    ) -> None:
        matrix = jnp.asarray(matrix)
        if matrix.ndim != 2:
            msg = f"'matrix' must be 2-D (target, coefficient), got {matrix.shape}"
            raise ValidationError(msg)
        if not jnp.issubdtype(matrix.dtype, jnp.floating):
            msg = f"'matrix' must have a floating dtype, got {matrix.dtype}"
            raise ValidationError(msg)
        n_target = matrix.shape[0]
        valid = jnp.ones(n_target, dtype=bool) if valid is None else valid
        valid = jnp.asarray(valid)
        if valid.shape != (n_target,) or valid.dtype != jnp.bool_:
            msg = (
                f"'valid' must be boolean with shape ({n_target},), got "
                f"{valid.dtype} {valid.shape}"
            )
            raise ValidationError(msg)
        self.matrix = matrix
        self.valid = valid

    @staticmethod
    def _check_last_axis(array: jax.Array, size: int, name: str) -> None:
        if array.ndim < 1 or array.shape[-1] != size:
            msg = (
                f"'{name}' must have a last axis of length {size}, "
                f"got shape {array.shape}"
            )
            raise ValidationError(msg)

    def apply(
        self, coefficients: Float[Array, "*batch coefficient"]
    ) -> Float[Array, "*batch target"]:
        """f̂ = H c = K* c + P* b, over any leading batch dims.

        Raises:
            ValidationError: If the last axis of ``coefficients`` is not N + P.
        """
        coefficients = jnp.asarray(coefficients)
        self._check_last_axis(coefficients, self.matrix.shape[1], "coefficients")
        return einx.dot(
            "... coefficient, target coefficient -> ... target",
            coefficients,
            self.matrix,
        )

    def apply_adjoint(
        self, residual: Float[Array, "*batch target"]
    ) -> Float[Array, "*batch coefficient"]:
        """g = Hᵀ r = [K*ᵀ r; P*ᵀ r], over any leading batch dims.

        Raises:
            ValidationError: If the last axis of ``residual`` is not M.
        """
        residual = jnp.asarray(residual)
        self._check_last_axis(residual, self.matrix.shape[0], "residual")
        return einx.dot(
            "... target, target coefficient -> ... coefficient",
            residual,
            self.matrix,
        )

    def operator(self) -> lx.MatrixLinearOperator:
        """H as an (M, N + P) lineax operator."""
        return lx.MatrixLinearOperator(self.matrix)


def _monomial_powers(n_dims: int, degree: int) -> tuple[tuple[int, ...], ...]:
    """The exponents of every monomial in D variables of total degree ≤ ``degree``."""
    powers = []
    for total in range(degree + 1):
        for dims in itertools.combinations_with_replacement(range(n_dims), total):
            powers.append(tuple(dims.count(dim) for dim in range(n_dims)))
    return tuple(powers)


class PolynomialTail(eqx.Module):
    """The monomials pₚ(x) = ∏_d x̂_d^{a_pd} of total degree ≤ d in scaled coordinates.

    x̂ = (x − s)/h maps the bounding box of the sources onto [−1, 1]ᴰ (an axis
    of zero extent keeps h = 1), as SciPy's ``RBFInterpolator`` does; the
    span, and so the interpolant, is the same as for the raw monomials, but
    the tail matrix is far better conditioned.

    Args:
        shift: s, the centre of the sources' bounding box.
        scale: h, its half-widths.
        powers: The exponents a_p of each of the P monomials.
    """

    shift: Float[Array, " dim"]
    scale: Float[Array, " dim"]
    powers: tuple[tuple[int, ...], ...] = eqx.field(static=True)

    @classmethod
    def from_sources(
        cls, sources: Float[Array, "source dim"], degree: int
    ) -> PolynomialTail:
        """The tail of degree ``degree`` ≥ 0, scaled to the sources' bounding box."""
        lower = einx.min("[source] dim", sources)
        upper = einx.max("[source] dim", sources)
        shift = (upper + lower) / 2
        half_width = (upper - lower) / 2
        flat = half_width == 0
        scale = jnp.where(flat, jnp.ones((), half_width.dtype), half_width)
        powers = _monomial_powers(sources.shape[1], degree)
        return cls(shift, scale, powers)

    @property
    def n_terms(self) -> int:
        """P, the number of monomials."""
        return len(self.powers)

    def __call__(self, points: Float[Array, "obs dim"]) -> Float[Array, "obs term"]:
        """P with Pₖₚ = pₚ(xₖ), at n points."""
        # x̂ = (x − s)/h
        shifted = einx.subtract("obs dim, dim -> obs dim", points, self.shift)
        scaled = einx.divide("obs dim, dim -> obs dim", shifted, self.scale)
        ones = jnp.ones(points.shape[0], points.dtype)
        columns = []
        for power in self.powers:
            column = ones
            for dim, exponent in enumerate(power):
                if exponent:
                    # integer powers, so the gradient at x̂ = 0 is finite
                    column = column * scaled[:, dim] ** exponent
            columns.append(column)
        return jnp.column_stack(columns)


def _scattered_source(source: Any) -> Float[Array, "source dim"]:
    """The (N, D) coordinates of a `Points` source without padding."""
    if not isinstance(source, Points):
        msg = (
            f"'source' must be Points for a kernel scheme, got {type(source).__name__}"
        )
        raise GeometryError(msg)
    if source.valid is not None:
        msg = "'source' must not carry a 'valid' mask; drop the padding sources first"
        raise GeometryError(msg)
    return source.xy


def _targets(
    target: Any,
) -> tuple[Float[Array, "target dim"], Bool[Array, " target"], Shape | None]:
    """The target coordinates (padding set to 0), their validity, and a grid shape.

    Raises:
        GeometryError: If ``target`` is not `Points`, an `Axis` or a
            `RectilinearGrid`.
    """
    coordinates = _target_coordinates(None, target)
    n_target = coordinates.shape[0]
    valid = jnp.ones(n_target, dtype=bool)
    if isinstance(target, Points) and target.valid is not None:
        valid = target.valid
    zero = jnp.zeros((), coordinates.dtype)
    # padding rows read the origin, so no NaN enters the kernel matrices
    coordinates = einx.where(
        "target, target dim, -> target dim", valid, coordinates, zero
    )
    grid_shape = target.shape if isinstance(target, RectilinearGrid) else None
    return coordinates, valid, grid_shape


class KernelSystem(eqx.Module):
    """The pieces a kernel plan is built from: the fit, the read-out and the targets.

    Args:
        fit: The `KernelSolve`.
        read: The `KernelReadout` [K* P*].
        targets: The target coordinates, padding at the origin.
        target_shape: The shape of a grid target, or ``None``.
    """

    fit: KernelSolve
    read: KernelReadout
    targets: Float[Array, "target dim"]
    target_shape: Shape | None = eqx.field(static=True)


def kernel_system(
    kernel: Kernel,
    source: Any,
    target: Any,
    *,
    tail_for: Callable[[jax.Array], Tail | None],
    diagonal: float | int,
    solver: lx.AbstractLinearSolver | None,
    positive_definite: bool = True,
) -> KernelSystem:
    """Build the saddle-point fit and the dense read-out of a kernel scheme.

    Args:
        kernel: k.
        source: The `Points` sources (no padding).
        target: `Points` (padding allowed), an `Axis` or a `RectilinearGrid`.
        tail_for: Given the source coordinates, the tail P(x) (a
            `PolynomialTail` or a kriging drift), or ``None`` for no tail.
        diagonal: λ ≥ 0, added to the diagonal of K.
        solver: The lineax solver of the inner system, or ``None`` for the
            `KernelSolve` default.
        positive_definite: Whether the inner system is positive definite.

    Returns:
        The `KernelSystem`.

    Raises:
        GeometryError: If the geometries are unsupported.
        ValidationError: If the kernel returns the wrong shape, or the tail
            has no fewer terms than there are sources.
    """
    sources = _scattered_source(source)
    coordinates, valid, target_shape = _targets(target)
    # --- the fit: K̃ = K + λI and P on the sources ---
    gram = kernel_matrix(kernel, sources, sources)
    if diagonal:
        identity = jnp.eye(gram.shape[0], dtype=gram.dtype)
        gram = gram + diagonal * identity
    tail = tail_for(sources)
    source_terms = None if tail is None else tail(sources)
    fit = KernelSolve(
        gram, source_terms, solver=solver, positive_definite=positive_definite
    )
    # --- the read-out: H = [K* P*] at the targets ---
    cross = kernel_matrix(kernel, coordinates, sources)
    if tail is not None:
        target_terms = tail(coordinates)
        cross = einx.id("target a, target b -> target (a + b)", cross, target_terms)
    zero = jnp.zeros((), cross.dtype)
    matrix = einx.where(
        "target, target coefficient, -> target coefficient", valid, cross, zero
    )
    read = KernelReadout(matrix, valid)
    return KernelSystem(fit, read, coordinates, target_shape)


def _check_policies(policies: Policies) -> None:
    """Raise for policies defined only on local read-outs."""
    if policies.missing == "renormalise":
        msg = "a kernel scheme does not support missing='renormalise'"
        raise PolicyError(msg)


class RBF(Scheme):
    """Radial basis function interpolation: f̂(x) = ∑ⱼ cⱼ k(x, xⱼ) + ∑ₚ bₚ pₚ(x).

    With K on the sources, smoothing λ ≥ 0 and the (N, P) matrix P of
    monomials up to ``degree``, the weights solve

        [[K + λI, P], [Pᵀ, 0]] [c; b] = [y; 0],

    and f̂(X*) = K* c + P* b. With λ = 0 the interpolant passes through the
    samples; λ > 0 trades that for smoothness. The plan's fit is a
    `KernelSolve` (factorised once, applied per field) and its read-out a
    dense `KernelReadout`, so the plan has an adjoint 𝒮ᵀ Hᵀ and a lineax
    operator like any other. It equals SciPy's ``RBFInterpolator`` for the
    same kernel, degree and smoothing (`Polyharmonic` holds SciPy's signs).
    Building costs O(N³) for the solve and O(MN) for the read-out; applying
    costs O(N² + MN) per field. Every target is valid (a kernel interpolant
    is defined everywhere), so ``outside`` and ``boundary`` do not apply.

    Args:
        kernel: k, any `Kernel`: `Polyharmonic`, `Wendland`, or a callable
            returning an (M, N) array or lineax operator.
        degree: The polynomial tail's degree, −1 for none; ``None`` (the
            default) takes the kernel's ``min_degree`` (1 for the thin-plate
            spline), or no tail for a kernel without one. A degree below
            ``min_degree`` is allowed, as in SciPy, but the system is then
            not guaranteed to be solvable.
        smoothing: λ ≥ 0, added to the diagonal of K.
        solver: The lineax solver of the inner system on the tail's null
            space; ``None`` (the default) takes ``lineax.Cholesky()``, or
            ``lineax.LU()`` when ``degree`` is below ``min_degree`` and that
            system may be indefinite.

    Raises:
        ValidationError: If ``kernel`` is not callable, ``degree`` is below
            −1, or ``smoothing`` is not a non-negative finite number.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Points
        >>> from functax.interp import RBF, Polyharmonic
        >>> sources = Points(
        ...     jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        ... )
        >>> plan = RBF(Polyharmonic(2)).plan(sources, sources)
        >>> data = jnp.array([1.0, 2.0, 3.0, 5.0])
        >>> bool(jnp.allclose(plan(data), data))
        True
        >>> # a degree-1 tail reproduces an affine field exactly
        >>> midpoint = RBF(Polyharmonic(2)).plan(
        ...     sources, Points(jnp.array([[0.5, 0.25]]))
        ... )
        >>> bool(
        ...     jnp.allclose(
        ...         midpoint(1.0 + sources.xy[:, 0] + 2.0 * sources.xy[:, 1]), 2.0
        ...     )
        ... )
        True
    """

    kernel: Kernel
    # int | None rather than a narrower type, so a bad value reaches the check
    degree: int | None = eqx.field(static=True, default=None)
    smoothing: float | int = eqx.field(static=True, default=0.0)
    solver: lx.AbstractLinearSolver | None = None

    def __check_init__(self) -> None:
        if not callable(self.kernel):
            msg = f"'kernel' must be callable k(targets, sources), got {self.kernel!r}"
            raise ValidationError(msg)
        degree = self.degree
        is_integer = isinstance(degree, int) and not isinstance(degree, bool)
        if degree is not None and (not is_integer or degree < -1):
            msg = f"'degree' must be None or an integer ≥ −1, got {degree!r}"
            raise ValidationError(msg)
        _check_smoothing(self.smoothing, "smoothing")

    @property
    def tail_degree(self) -> int:
        """The degree of the tail the plan uses, −1 for none."""
        if self.degree is not None:
            return self.degree
        return getattr(self.kernel, "min_degree", -1)

    def plan(
        self, source: Any, target: Any, *, policies: Policies | None = None
    ) -> Plan:
        """The plan reading data on the ``source`` points at ``target``.

        Args:
            source: `Points` (no ``valid`` mask), the N sources.
            target: `Points` (padding rows are invalid), an `Axis` or a
                `RectilinearGrid` read at its nodes.
            policies: The missing-sample and fill rules (``missing="fill"``
                fills non-finite samples before the fit).

        Returns:
            The `Plan`, with a `KernelSolve` fit and a `KernelReadout`.

        Raises:
            GeometryError: If the geometries are unsupported.
            PolicyError: Under ``missing="renormalise"``.
            ValidationError: If the tail has as many terms as there are
                sources, or the kernel returns the wrong shape.
        """
        policies = Policies() if policies is None else policies
        _check_policies(policies)
        degree = self.tail_degree

        def tail_for(sources: jax.Array) -> Tail | None:
            if degree < 0:
                return None
            return PolynomialTail.from_sources(sources, degree)

        system = kernel_system(
            self.kernel,
            source,
            target,
            tail_for=tail_for,
            diagonal=self.smoothing,
            solver=self.solver,
            positive_definite=degree >= getattr(self.kernel, "min_degree", -1),
        )
        return Plan(system.fit, system.read, policies, None, system.target_shape)


def _check_smoothing(value: Any, name: str) -> None:
    """Raise unless ``value`` is a non-negative, finite number."""
    is_number = isinstance(value, int | float) and not isinstance(value, bool)
    if not is_number or not math.isfinite(value) or value < 0:
        msg = f"'{name}' must be a non-negative, finite number, got {value!r}"
        raise ValidationError(msg)
