"""Linear fits onto a basis: ridge least squares and Galerkin projection.

Both take the basis already evaluated at the sites, Φ (M, J), so any basis of
`functax.basis` (or a user's feature map) plugs in, and both solve through
lineax.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import math

import einx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, ArrayLike, Float

from functax._src._checks import check_values, require_floating
from functax.exceptions import ValidationError


__all__ = [
    "galerkin",
    "least_squares",
]


def _check_design(design_matrix: jax.Array, data: jax.Array) -> None:
    """Φ is a floating (M, J) matrix and y floating with M along its last axis."""
    if design_matrix.ndim != 2:
        msg = (
            "'design_matrix' must be 2-D (obs, feature), got shape "
            f"{design_matrix.shape}"
        )
        raise ValidationError(msg)
    require_floating(design_matrix, "design_matrix")
    require_floating(data, "y")
    n_obs = design_matrix.shape[0]
    if data.ndim < 1 or data.shape[-1] != n_obs:
        msg = f"'y' must have a last axis of length {n_obs}, got shape {data.shape}"
        raise ValidationError(msg)


def _cholesky_solve(
    normal: Float[Array, "row col"], rhs: Float[Array, "*batch feature"]
) -> Float[Array, "*batch feature"]:
    """c = A⁻¹ b for a symmetric positive-definite A, by Cholesky, over batch dims."""
    operator = lx.MatrixLinearOperator(
        normal, (lx.symmetric_tag, lx.positive_semidefinite_tag)
    )

    def solve_one(vector: jax.Array) -> jax.Array:
        return lx.linear_solve(operator, vector, lx.Cholesky()).value

    return jnp.vectorize(solve_one, signature="(j)->(j)")(rhs)


def least_squares(
    design_matrix: Float[ArrayLike, "obs feature"],
    y: Float[ArrayLike, "*batch obs"],
    *,
    ridge: float = 0.0,
) -> Float[Array, "*batch feature"]:
    """c = argmin ‖Φc − y‖² + α‖c‖², the ridge least-squares coefficients.

    The minimiser solves (ΦᵀΦ + αI) c = Φᵀy. With α > 0 the normal matrix is
    positive definite and is factorised by Cholesky; with α = 0 the problem is
    solved by QR on Φ itself, which avoids squaring its condition number.

    Algorithm:
        1. α = 0: c = R⁻¹ Qᵀ y with Φ = QR (``lineax.QR``).
        2. α > 0: A = ΦᵀΦ + αI and b = Φᵀy, then c = A⁻¹b by Cholesky.

    Args:
        design_matrix: Φ, the J basis functions evaluated at the M sites,
            Φₘⱼ = φⱼ(xₘ); M ≥ J when α = 0.
        y: The data at the sites, M along the last axis, over any leading
            batch dims.
        ridge: α ≥ 0, the ridge weight; a Python number, since it chooses the
            factorisation.

    Returns:
        c, the J coefficients, over the batch dims of ``y``.

    Raises:
        ValidationError: If Φ is not a floating 2-D matrix, ``y`` is not
            floating with M along its last axis, or α is negative or not
            finite.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.project import least_squares
        >>> x = jnp.linspace(0.0, 1.0, 9)
        >>> design = jnp.stack([jnp.ones_like(x), x, x**2], axis=-1)
        >>> c = least_squares(design, 1.0 - 2.0 * x + 3.0 * x**2)
        >>> c.shape, bool(jnp.allclose(c, jnp.array([1.0, -2.0, 3.0])))
        ((3,), True)
        >>> least_squares(design, jnp.zeros((4, 9)), ridge=1e-8).shape
        (4, 3)
    """
    design_matrix = jnp.asarray(design_matrix)
    y = jnp.asarray(y)
    _check_design(design_matrix, y)
    if isinstance(ridge, bool) or not isinstance(ridge, int | float):
        msg = f"'ridge' must be a Python number, got {type(ridge).__name__}"
        raise ValidationError(msg)
    if not (math.isfinite(ridge) and ridge >= 0):
        msg = f"'ridge' must be non-negative and finite, got {ridge!r}"
        raise ValidationError(msg)
    if ridge == 0:
        operator = lx.MatrixLinearOperator(design_matrix)

        def solve_one(vector: jax.Array) -> jax.Array:
            return lx.linear_solve(operator, vector, lx.QR()).value

        return jnp.vectorize(solve_one, signature="(m)->(j)")(y)
    # A = ΦᵀΦ + αI
    gram = einx.dot("obs row, obs col -> row col", design_matrix, design_matrix)
    normal = gram + ridge * jnp.eye(gram.shape[0], dtype=gram.dtype)
    # b = Φᵀy
    rhs = einx.dot("obs feature, ... obs -> ... feature", design_matrix, y)
    return _cholesky_solve(normal, rhs)


def galerkin(
    design_matrix: Float[ArrayLike, "obs feature"],
    y: Float[ArrayLike, "*batch obs"],
    *,
    weights: Float[ArrayLike, " obs"],
) -> Float[Array, "*batch feature"]:
    """c = (ΦᵀWΦ)⁻¹ ΦᵀW y, the Galerkin projection with quadrature weights W.

    With the sites the nodes of a quadrature rule and W = diag(w) its weights,
    ΦᵀWΦ approximates the Gram matrix ∫ φᵢ φⱼ and ΦᵀWy the moments ∫ φᵢ f, so
    c is the L² projection of f onto the span of the basis. For a basis
    orthonormal under the rule, ΦᵀWΦ = I and c = ΦᵀWy.

    Algorithm:
        1. A = ΦᵀWΦ and b = ΦᵀWy.
        2. c = A⁻¹b by Cholesky.

    Args:
        design_matrix: Φ, the J basis functions at the M quadrature nodes.
        y: f at the nodes, M along the last axis, over any leading batch dims.
        weights: w, the M quadrature weights, all positive.

    Returns:
        c, the J coefficients, over the batch dims of ``y``.

    Raises:
        ValidationError: If the shapes disagree, an input is not floating, or
            a weight is not positive (checked when run under ``jit``).

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.project import galerkin
        >>> from functax.quad import gauss_legendre
        >>> nodes, weights = gauss_legendre(8)
        >>> design = jnp.stack([jnp.ones_like(nodes), nodes], axis=-1)
        >>> # the L² projection of x² onto {1, x} on [−1, 1] is 1/3
        >>> c = galerkin(design, nodes**2, weights=weights)
        >>> bool(jnp.allclose(c, jnp.array([1.0 / 3.0, 0.0])))
        True
    """
    design_matrix = jnp.asarray(design_matrix)
    y = jnp.asarray(y)
    weights = jnp.asarray(weights)
    _check_design(design_matrix, y)
    require_floating(weights, "weights")
    n_obs = design_matrix.shape[0]
    if weights.shape != (n_obs,):
        msg = f"'weights' must have shape ({n_obs},), got {weights.shape}"
        raise ValidationError(msg)
    weights = check_values(
        weights, jnp.any(weights <= 0), "'weights' must all be positive"
    )
    # WΦ, the basis scaled by the weight of its node
    weighted = einx.multiply("obs feature, obs -> obs feature", design_matrix, weights)
    # A = ΦᵀWΦ
    normal = einx.dot("obs row, obs col -> row col", design_matrix, weighted)
    # b = ΦᵀWy
    rhs = einx.dot("obs feature, ... obs -> ... feature", weighted, y)
    return _cholesky_solve(normal, rhs)
