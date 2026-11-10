"""Vector spherical harmonics, a basis for tangent vector fields on the sphere.

For each real Y_lm with l ≥ 1, the gradient (poloidal, curl-free) harmonic
Ψ_lm = ∇_S Y_lm and the toroidal (divergence-free) harmonic Φ_lm = r̂ × ∇_S Y_lm
together span the square-integrable tangent fields, a discrete Helmholtz
decomposition: ∇_S · Ψ_lm = −l(l + 1) Y_lm and ∇_S · Φ_lm = 0. Moved from geonnax
``_vector_spherical`` at ``be5537d``.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import einx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from functax._src.basis._legendre import _check_degree
from functax._src.basis._spherical_harmonics import (
    _check_unit_points,
    harmonic_degrees,
    real_spherical_harmonics,
)
from functax._src.geometry._frames import _enu
from functax.exceptions import ValidationError


__all__ = ["vector_spherical_harmonics"]


_OUTPUTS: tuple[str, ...] = ("cartesian", "enu")


def vector_spherical_harmonics(
    unit_xyz: Float[Array, "obs 3"],
    l_max: int,
    *,
    output: str = "cartesian",
    normalized: bool = True,
) -> tuple[
    Float[Array, "obs feature component"],
    Float[Array, "obs feature component"],
    Int[Array, " feature"],
]:
    """The gradient and toroidal vector spherical harmonics for 1 ≤ l ≤ l_max.

    The surface gradient is the tangential part of the ambient gradient of
    any smooth extension: ∇_S Y(u) = (I − u uᵀ) ∇Ỹ(u). `real_spherical_harmonics`
    is a polynomial in (x, y, z), so its Jacobian (``jax.jacfwd``) gives Ψ_lm
    smoothly everywhere, poles included, and Φ_lm = u × Ψ_lm. Both have
    ∫_{S²} ‖Ψ_lm‖² dΩ = ∫_{S²} ‖Φ_lm‖² dΩ = l(l + 1); with ``normalized=True``
    each is divided by √(l(l + 1)), an orthonormal set of 2((l_max + 1)² − 1)
    tangent fields.

    Args:
        unit_xyz: Points on the unit sphere.
        l_max: The highest degree, at least 1.
        output: ``"cartesian"`` for 3-D vectors, or ``"enu"`` for their east
            and north components (see `enu_basis`; at exactly the poles the
            frame is chosen by atan2(y, x)).
        normalized: Divide by √(l(l + 1)).

    Returns:
        ``(psi, phi, degrees)``: Ψ and Φ with 3 (or 2 for ``"enu"``)
        components per column, (l_max + 1)² − 1 columns in the order of
        `real_spherical_harmonics` without l = 0, and the degree of each.

    Raises:
        ValidationError: If ``unit_xyz`` is not (N, 3), ``l_max`` is not an
            integer of at least 1, or ``output`` is unknown.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.sphere import vector_spherical_harmonics
        >>> xyz = jnp.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        >>> psi, phi, degrees = vector_spherical_harmonics(xyz, 2)
        >>> psi.shape, phi.shape, degrees.tolist()
        ((2, 8, 3), (2, 8, 3), [1, 1, 1, 2, 2, 2, 2, 2])
        >>> vector_spherical_harmonics(xyz, 1, output="enu")[0].shape
        (2, 3, 2)
    """
    degree = _check_degree(l_max, "l_max", minimum=1)
    if output not in _OUTPUTS:
        msg = f"'output' must be one of {list(_OUTPUTS)}, got {output!r}"
        raise ValidationError(msg)
    unit_xyz = jnp.asarray(unit_xyz)
    _check_unit_points(unit_xyz, "unit_xyz")

    def harmonics(point: Float[Array, " 3"]) -> Float[Array, " feature"]:
        row = einx.id("dim -> 1 dim", point)
        return real_spherical_harmonics(row, degree)[0, 1:]

    # the ambient gradient of each harmonic at each point
    gradient = jax.vmap(jax.jacfwd(harmonics))(unit_xyz)
    radial = einx.dot("obs feature dim, obs dim -> obs feature", gradient, unit_xyz)
    # Ψ = (I − u uᵀ) ∇Ỹ
    psi = gradient - einx.multiply(
        "obs feature, obs dim -> obs feature dim", radial, unit_xyz
    )
    # Φ = u × Ψ
    phi = jnp.cross(einx.id("obs dim -> obs 1 dim", unit_xyz), psi)

    degrees = np.asarray(harmonic_degrees(degree)[1:])
    if normalized:
        scale = jnp.asarray(1.0 / np.sqrt(degrees * (degrees + 1.0)), dtype=psi.dtype)
        psi = einx.multiply("obs feature dim, feature", psi, scale)
        phi = einx.multiply("obs feature dim, feature", phi, scale)
    if output == "enu":
        longitude = jnp.arctan2(unit_xyz[:, 1], unit_xyz[:, 0])
        latitude = jnp.arcsin(jnp.clip(unit_xyz[:, 2], -1.0, 1.0))
        east_north = _enu(longitude, latitude)[:, :2, :]
        psi = einx.dot(
            "obs feature dim, obs axis dim -> obs feature axis", psi, east_north
        )
        phi = einx.dot(
            "obs feature dim, obs axis dim -> obs feature axis", phi, east_north
        )
    return psi, phi, jnp.asarray(degrees)
