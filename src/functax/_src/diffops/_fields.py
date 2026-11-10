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
from typing import Any

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


def _check_point(x: Any, name: str = "x") -> jax.Array:
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

    Args:
        f: A scalar field of one point, (D,) → ().

    Returns:
        A callable (D,) → (D,) that evaluates ∇f.

    Raises:
        ValidationError: When the callable gets more than one point.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.diffops import gradient
        >>> gradient(lambda x: jnp.sum(x**2))(jnp.array([1.0, 2.0])).tolist()
        [2.0, 4.0]
        >>> # batch with vmap
        >>> import jax
        >>> jax.vmap(gradient(lambda x: x[0] * x[1]))(jnp.ones((3, 2))).shape
        (3, 2)
    """

    def _gradient(x: Real[ArrayLike, "*shape"]) -> Float[Array, " dim"]:
        x = _check_point(x)
        return jax.grad(f)(x)

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
        >>> import jax.numpy as jnp
        >>> from functax.diffops import jacobian_field
        >>> J = jacobian_field(lambda x: jnp.array([x[0] * x[1], x[1]]))
        >>> J(jnp.array([2.0, 3.0])).tolist()
        [[3.0, 2.0], [0.0, 1.0]]
    """

    def _jacobian(x: Real[ArrayLike, "*shape"]) -> Float[Array, "component dim"]:
        x = _check_point(x)
        return jax.jacfwd(u)(x)

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
        >>> import jax.numpy as jnp
        >>> from functax.diffops import divergence
        >>> div = divergence(lambda x: x * jnp.array([1.0, 2.0, 3.0]))
        >>> float(div(jnp.zeros(3)))
        6.0
    """

    def _divergence(x: Real[ArrayLike, "*shape"]) -> Float[Array, ""]:
        x = _check_point(x)
        jac = jax.jacfwd(u)(x)
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
        >>> import jax.numpy as jnp
        >>> from functax.diffops import curl2d
        >>> # solid-body rotation has vorticity 2
        >>> float(
        ...     curl2d(lambda x: jnp.array([-x[1], x[0]]))(jnp.array([0.3, -0.2]))
        ... )
        2.0
    """

    def _curl(x: Real[ArrayLike, "*shape"]) -> Float[Array, ""]:
        x = _check_point(x)
        jac = jax.jacfwd(u)(x)
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

    Args:
        f: A scalar field of one point, (D,) → ().

    Returns:
        A callable (D,) → () that evaluates Δf.

    Raises:
        ValidationError: When the callable gets more than one point.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.diffops import laplacian
        >>> float(laplacian(lambda x: jnp.sum(x**2))(jnp.array([0.5, -1.0, 2.0])))
        6.0
    """

    def _laplacian(x: Real[ArrayLike, "*shape"]) -> Float[Array, ""]:
        x = _check_point(x)
        grad_f = jax.grad(f)

        def second(direction: Float[Array, " dim"]) -> Float[Array, ""]:
            # e_dᵀ ∇²f e_d
            return jnp.vdot(jax.jvp(grad_f, (x,), (direction,))[1], direction)

        directions = jnp.eye(x.shape[0], dtype=x.dtype)
        return jnp.sum(jax.vmap(second)(directions))

    return _laplacian
