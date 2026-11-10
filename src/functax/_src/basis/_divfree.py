"""Divergence-free and curl-free vector bases on the box [−L, L]².

A 2-D incompressible velocity is the skew gradient of a stream function,
u = ∇^⊥ψ = (−∂_y ψ, ∂_x ψ). Expanding ψ in the box Laplacian eigenfunctions φ_j
gives atoms that are divergence-free by construction, the natural reduced
basis for nearly geostrophic ocean currents; their gradients are the curl-free
companions. Each atom carries the eigenvalue λ_j + λ_k of its scalar mode, so
a consumer can impose a spectral energy law. Moved from geonnax ``_divfree``
at ``be5537d``.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import einx
import jax
from jaxtyping import Array, Float

from functax._src.basis._fourier import (
    _basis_value_and_grad_1d,
    _check_boundary,
    _to_tuple,
    fourier_eigenvalues,
)
from functax.exceptions import ValidationError


__all__ = [
    "curlfree_basis",
    "divfree_basis",
]


def _check_plane(xy: jax.Array) -> None:
    if xy.ndim != 2 or xy.shape[-1] != 2:
        msg = f"'xy' must have shape (N, 2), got {xy.shape}"
        raise ValidationError(msg)


def _axis_bases(
    xy: Float[Array, "obs 2"], counts: tuple, widths: tuple, boundaries: tuple
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The 1-D values and derivatives on each axis: (φ_x, φ_x′, φ_y, φ_y′)."""
    out = []
    for d in range(2):
        _check_boundary(boundaries[d])
        out += _basis_value_and_grad_1d(
            xy[:, d], counts[d], float(widths[d]), boundaries[d]
        )
    return out[0], out[1], out[2], out[3]


def divfree_basis(
    xy: Float[Array, "obs 2"],
    num_basis_per_dim: int | tuple[int, int],
    L: float | tuple[float, float],
    *,
    boundary: str | tuple[str, str] = "dirichlet",
) -> tuple[Float[Array, "obs feature 2"], Float[Array, " feature"]]:
    """Divergence-free vector atoms, the skew gradients of box stream functions.

    φ_jk(x, y) = (−φ_j(x) φ_k′(y), φ_j′(x) φ_k(y)), so
    ∂_x φ^(x) + ∂_y φ^(y) = −φ_j′φ_k′ + φ_j′φ_k′ = 0 identically.

    Args:
        xy: Points in [−L, L]².
        num_basis_per_dim: The number of 1-D stream-function modes per axis;
            an int applies to both, and there are M = M_x M_y atoms.
        L: The half-width per axis; a number applies to both.
        boundary: The stream functions' boundary condition, ``"dirichlet"``,
            ``"neumann"`` or ``"periodic"``, one value or one per axis.

    Returns:
        ``(Phi, lam)``: the (u, v) components of each atom at each point, and
        the stream-function eigenvalues λ_j + λ_k, row-major over (j, k).

    Raises:
        ValidationError: If ``xy`` is not (N, 2), a per-axis argument has the
            wrong length, or an axis's 1-D arguments are invalid.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.vector import divfree_basis
        >>> Phi, lam = divfree_basis(jnp.zeros((5, 2)), num_basis_per_dim=3, L=1.0)
        >>> Phi.shape, lam.shape
        ((5, 9, 2), (9,))
    """
    _check_plane(xy)
    counts = _to_tuple(num_basis_per_dim, 2, "num_basis_per_dim")
    widths = _to_tuple(L, 2, "L")
    boundaries = _to_tuple(boundary, 2, "boundary")
    value_x, slope_x, value_y, slope_y = _axis_bases(xy, counts, widths, boundaries)
    # u = −∂_y ψ = −φ_j(x) φ_k′(y) and v = ∂_x ψ = φ_j′(x) φ_k(y), row-major
    # over (j, k) as fourier_eigenvalues lays them out
    u = -einx.multiply("obs left, obs right -> obs (left right)", value_x, slope_y)
    v = einx.multiply("obs left, obs right -> obs (left right)", slope_x, value_y)
    phi = einx.id("obs feature, obs feature -> obs feature (1 + 1)", u, v)
    lam = fourier_eigenvalues(
        num_basis_per_dim, L, 2, dtype=xy.dtype, boundary=boundary
    )
    return phi, lam


def curlfree_basis(
    xy: Float[Array, "obs 2"],
    num_basis_per_dim: int | tuple[int, int],
    L: float | tuple[float, float],
    *,
    boundary: str | tuple[str, str] = "dirichlet",
) -> tuple[Float[Array, "obs feature 2"], Float[Array, " feature"]]:
    """Curl-free vector atoms, the gradients of the box Laplacian eigenfunctions.

    The companion of `divfree_basis`: φ_jk(x, y) = (φ_j′(x) φ_k(y), φ_j(x) φ_k′(y)),
    the gradient of the potential φ_j ⊗ φ_k, so its scalar curl
    ∂_x φ^(y) − ∂_y φ^(x) vanishes identically.

    Args:
        xy: Points in [−L, L]².
        num_basis_per_dim: The number of 1-D modes per axis; an int applies to
            both.
        L: The half-width per axis; a number applies to both.
        boundary: The potential's boundary condition, one value or one per
            axis.

    Returns:
        ``(Phi, lam)``: the (u, v) components of each atom at each point, and
        the potential eigenvalues λ_j + λ_k, row-major over (j, k).

    Raises:
        ValidationError: If ``xy`` is not (N, 2), a per-axis argument has the
            wrong length, or an axis's 1-D arguments are invalid.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.vector import curlfree_basis
        >>> Phi, lam = curlfree_basis(jnp.zeros((5, 2)), num_basis_per_dim=3, L=1.0)
        >>> Phi.shape, lam.shape
        ((5, 9, 2), (9,))
    """
    _check_plane(xy)
    counts = _to_tuple(num_basis_per_dim, 2, "num_basis_per_dim")
    widths = _to_tuple(L, 2, "L")
    boundaries = _to_tuple(boundary, 2, "boundary")
    value_x, slope_x, value_y, slope_y = _axis_bases(xy, counts, widths, boundaries)
    # ∇(φ_j ⊗ φ_k) = (φ_j′(x) φ_k(y), φ_j(x) φ_k′(y))
    u = einx.multiply("obs left, obs right -> obs (left right)", slope_x, value_y)
    v = einx.multiply("obs left, obs right -> obs (left right)", value_x, slope_y)
    phi = einx.id("obs feature, obs feature -> obs feature (1 + 1)", u, v)
    lam = fourier_eigenvalues(
        num_basis_per_dim, L, 2, dtype=xy.dtype, boundary=boundary
    )
    return phi, lam
