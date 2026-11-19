"""Derivatives of Euclidean fields of x ∈ ℝ^D, by automatic differentiation.

A continuous field is a function of its coordinates, so its spatial
derivatives come from automatic differentiation rather than finite
differences, and apply alike to a network, an `Interpolant` or a basis
expansion. Each operator takes a field of one point and returns a callable that
evaluates the derivative at one point; batch with ``jax.vmap`` and compile with
``jax.jit`` at the top. Moved from geonnax ``diffops`` at ``be5537d``, where
`gradient` was ``grad`` and `jacobian_field` was ``jacobian`` (renamed so they no
longer shadow ``jax.grad`` and ``jax.jacobian``).
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

from collections.abc import Callable

import einx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Real

from functax._src._dtypes import promote_integers
from functax.exceptions import ValidationError


__all__ = [
    "curl2d",
    "divergence",
    "gradient",
    "jacobian_field",
    "laplacian",
]


ScalarField = Callable[[Float[Array, " dim"]], Float[Array, ""]]
VectorField = Callable[[Float[Array, " dim"]], Float[Array, " component"]]


def _check_point(x: Real[ArrayLike, "*shape"], name: str = "x") -> Float[Array, " dim"]:
    """One point of shape (D,), with integer coordinates made float32."""
    x = jnp.asarray(x)
    if x.ndim != 1:
        msg = (
            f"'{name}' must be a single point of shape (D,), got shape {x.shape}; "
            "batch with jax.vmap"
        )
        raise ValidationError(msg)
    return promote_integers(x)


def gradient(f: ScalarField) -> Callable[[Float[Array, " dim"]], Float[Array, " dim"]]:
    """The gradient ∇f of a scalar field.

    (∇f)_d = ∂f/∂x_d, by reverse mode (``jax.grad``).

    Args:
        f: A scalar field of one point, (D,) → ().

    Returns:
        A callable (D,) → (D,) that evaluates ∇f.

    Raises:
        ValidationError: When the callable gets more than one point.

    Examples:
        ∇‖x‖² = 2x:

        >>> import einx
        >>> import jax.numpy as jnp
        >>> from functax.diffops import gradient
        >>> grad_f = gradient(lambda x: einx.sum("[dim]", x**2))
        >>> bool(jnp.allclose(grad_f(jnp.array([1.0, 2.0])), jnp.array([2.0, 4.0])))
        True

        Batch over points with ``vmap``:

        >>> import jax
        >>> jax.vmap(gradient(lambda x: x[0] * x[1]))(jnp.ones((3, 2))).shape
        (3, 2)
    """

    def _gradient(x: Real[ArrayLike, "*shape"]) -> Float[Array, " dim"]:
        x = _check_point(x)
        grad_f = jax.grad(f)
        # ∇f(x)
        return grad_f(x)

    return _gradient


def jacobian_field(
    u: VectorField,
) -> Callable[[Float[Array, " dim"]], Float[Array, "component dim"]]:
    """The Jacobian J_kd = ∂u_k/∂x_d of a vector field.

    Computed in forward mode (``jax.jacfwd``), the cheaper mode when the
    number of coordinates D is small, as it is for fields of space and time.

    Args:
        u: A vector field of one point, (D,) → (K,).

    Returns:
        A callable (D,) → (K, D) that evaluates J.

    Raises:
        ValidationError: When the callable gets more than one point.

    Examples:
        u = (x₀x₁, x₁) at (2, 3):

        >>> import jax.numpy as jnp
        >>> from functax.diffops import jacobian_field
        >>> jac = jacobian_field(lambda x: jnp.array([x[0] * x[1], x[1]]))
        >>> expected = jnp.array([[3.0, 2.0], [0.0, 1.0]])
        >>> bool(jnp.allclose(jac(jnp.array([2.0, 3.0])), expected))
        True

        K need not equal D: a field of three components of two coordinates:

        >>> tall = jacobian_field(lambda x: jnp.array([x[0], x[1], x[0] + x[1]]))
        >>> tall(jnp.ones(2)).shape
        (3, 2)
    """

    def _jacobian(x: Real[ArrayLike, "*shape"]) -> Float[Array, "component dim"]:
        x = _check_point(x)
        jacobian_u = jax.jacfwd(u)
        # J_kd = ∂u_k/∂x_d
        return jacobian_u(x)

    return _jacobian


def divergence(u: VectorField) -> Callable[[Float[Array, " dim"]], Float[Array, ""]]:
    """The divergence ∇ · u = ∑_d ∂u_d/∂x_d of a vector field.

    Args:
        u: A vector field of one point, (D,) → (D,).

    Returns:
        A callable (D,) → () that evaluates ∇ · u.

    Raises:
        ValidationError: When the callable gets more than one point, or ``u``
            does not map (D,) to (D,).

    Examples:
        u = (x₀, 2x₁, 3x₂) has ∇ · u = 6 everywhere:

        >>> import jax.numpy as jnp
        >>> from functax.diffops import divergence
        >>> div = divergence(lambda x: x * jnp.array([1.0, 2.0, 3.0]))
        >>> bool(jnp.allclose(div(jnp.zeros(3)), 6.0))
        True

        ∇ · ∇f is the Laplacian:

        >>> from functax.diffops import gradient, laplacian
        >>> def f(x):
        ...     return x[0] ** 2 * x[1]
        >>> x = jnp.array([0.5, -1.0])
        >>> bool(jnp.allclose(divergence(gradient(f))(x), laplacian(f)(x)))
        True
    """

    def _divergence(x: Real[ArrayLike, "*shape"]) -> Float[Array, ""]:
        x = _check_point(x)
        jacobian_u = jax.jacfwd(u)
        # J_kd = ∂u_k/∂x_d
        jac = jacobian_u(x)
        if jac.shape != (x.shape[0], x.shape[0]):
            msg = (
                f"divergence needs u: (D,) → (D,), got a Jacobian of shape {jac.shape}"
            )
            raise ValidationError(msg)
        # ∇ · u = tr J
        return jnp.trace(jac)

    return _divergence


def curl2d(u: VectorField) -> Callable[[Float[Array, " 2"]], Float[Array, ""]]:
    """The scalar curl (vorticity) ∂_x u_y − ∂_y u_x of a planar field.

    Args:
        u: A planar vector field of one point, (2,) → (2,), its inputs
            ordered (x, y).

    Returns:
        A callable (2,) → () that evaluates the curl.

    Raises:
        ValidationError: When the callable gets more than one point, or ``u``
            does not map (2,) to (2,).

    Examples:
        Solid-body rotation u = (−y, x) has vorticity 2:

        >>> import jax.numpy as jnp
        >>> from functax.diffops import curl2d
        >>> curl = curl2d(lambda x: jnp.array([-x[1], x[0]]))
        >>> bool(jnp.allclose(curl(jnp.array([0.3, -0.2])), 2.0))
        True

        A gradient field is irrotational, ∇ × ∇f = 0:

        >>> from functax.diffops import gradient
        >>> potential = gradient(lambda x: x[0] ** 2 * x[1])
        >>> bool(jnp.allclose(curl2d(potential)(jnp.array([0.5, 2.0])), 0.0))
        True
    """

    def _curl(x: Real[ArrayLike, "*shape"]) -> Float[Array, ""]:
        x = _check_point(x)
        jacobian_u = jax.jacfwd(u)
        # J_kd = ∂u_k/∂x_d
        jac = jacobian_u(x)
        if jac.shape != (2, 2):
            msg = f"curl2d needs u: (2,) → (2,), got a Jacobian of shape {jac.shape}"
            raise ValidationError(msg)
        # ∂_x u_y − ∂_y u_x
        return jac[1, 0] - jac[0, 1]

    return _curl


def laplacian(f: ScalarField) -> Callable[[Float[Array, " dim"]], Float[Array, ""]]:
    """The Laplacian Δf = ∑_d ∂²f/∂x_d² of a scalar field.

    Evaluated as ∑_d e_dᵀ ∇²f e_d with forward-over-reverse Hessian-vector
    products (``jax.jvp`` of ``jax.grad``) along the D unit vectors, so the
    dense Hessian is never formed.

    Algorithm:
        1. Build the reverse-mode gradient ∇f.
        2. For each unit vector e_d, push it forward through ∇f (``jax.jvp``)
           to get ∇²f e_d, and take e_dᵀ ∇²f e_d.
        3. Sum the D curvatures.

    Args:
        f: A scalar field of one point, (D,) → ().

    Returns:
        A callable (D,) → () that evaluates Δf.

    Raises:
        ValidationError: When the callable gets more than one point.

    Examples:
        Δ‖x‖² = 2D:

        >>> import einx
        >>> import jax.numpy as jnp
        >>> from functax.diffops import laplacian
        >>> lap = laplacian(lambda x: einx.sum("[dim]", x**2))
        >>> bool(jnp.allclose(lap(jnp.array([0.5, -1.0, 2.0])), 6.0))
        True

        A harmonic function, x₀² − x₁², has Δf = 0:

        >>> harmonic = laplacian(lambda x: x[0] ** 2 - x[1] ** 2)
        >>> bool(jnp.allclose(harmonic(jnp.array([0.3, 1.2])), 0.0))
        True
    """

    def _laplacian(x: Real[ArrayLike, "*shape"]) -> Float[Array, ""]:
        x = _check_point(x)
        grad_f = jax.grad(f)

        def curvature_along(direction: Float[Array, " dim"]) -> Float[Array, ""]:
            # ∇²f e_d, a Hessian-vector product, forward over reverse
            _, hessian_direction = jax.jvp(grad_f, (x,), (direction,))
            # e_dᵀ ∇²f e_d
            return einx.dot("dim, dim ->", hessian_direction, direction)

        # e_d, the D unit vectors as rows
        directions = jnp.eye(x.shape[0], dtype=x.dtype)
        curvature_batch = jax.vmap(curvature_along)
        # ∂²f/∂x_d² for each d
        curvatures = curvature_batch(directions)
        # Δf = ∑_d ∂²f/∂x_d²
        return einx.sum("[dim]", curvatures)

    return _laplacian
