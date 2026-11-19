"""Vector spherical harmonics, a basis for tangent vector fields on the sphere.

For each real Y_ℓm with degree ℓ ≥ 1 and order m, the gradient (poloidal,
curl-free) harmonic Ψ_ℓm = ∇_S Y_ℓm and the toroidal (divergence-free) harmonic
Φ_ℓm = r̂ × ∇_S Y_ℓm, with ∇_S the surface gradient and r̂ ∈ S² the point,
together span the square-integrable tangent fields, a discrete Helmholtz
decomposition: ∇_S · Ψ_ℓm = −ℓ(ℓ + 1) Y_ℓm and ∇_S · Φ_ℓm = 0. Moved from geonnax
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
    """The gradient and toroidal vector spherical harmonics for 1 ≤ ℓ ≤ ℓ_max.

    At a point r̂ = (x, y, z) ∈ S², the surface gradient is the tangential part
    of the ambient gradient ∇_r Ỹ of any smooth extension Ỹ of Y to r ∈ ℝ³: ∇_S
    Y(r̂) = (I − r̂ r̂ᵀ) ∇_r Ỹ(r̂). `real_spherical_harmonics` is a polynomial
    in (x, y, z), so its Jacobian (``jax.jacfwd``) gives Ψ_ℓm smoothly
    everywhere, poles included, and Φ_ℓm = r̂ × Ψ_ℓm. Both have ∫_{S²} ‖Ψ_ℓm‖²
    dΩ = ∫_{S²} ‖Φ_ℓm‖² dΩ = ℓ(ℓ + 1), with dΩ the area element of S²; with
    ``normalized=True`` each is divided by √(ℓ(ℓ + 1)), an orthonormal set of
    2((ℓ_max + 1)² − 1) tangent fields.

    Algorithm:
        1. Differentiate `real_spherical_harmonics` at each point with
           ``jax.jacfwd``, the ambient gradient ∇_r Ỹ_ℓm.
        2. Project out the radial part, Ψ_ℓm = ∇_r Ỹ_ℓm − (r̂ · ∇_r Ỹ_ℓm) r̂.
        3. Rotate by a quarter turn, Φ_ℓm = r̂ × Ψ_ℓm.
        4. Optionally divide both by √(ℓ(ℓ + 1)), and project onto the east and
           north directions for ``"enu"``.

    Args:
        unit_xyz: The N points r̂ on the unit sphere.
        l_max: ℓ_max, the highest degree, at least 1.
        output: ``"cartesian"`` for 3-D vectors, or ``"enu"`` for their east
            and north components (see `enu_basis`; at exactly the poles the
            frame is chosen by atan2(y, x)).
        normalized: Divide by √(ℓ(ℓ + 1)).

    Returns:
        ``(psi, phi, degrees)``: Ψ and Φ with 3 (or 2 for ``"enu"``)
        components per column, (ℓ_max + 1)² − 1 columns in the order of
        `real_spherical_harmonics` without ℓ = 0, and the degree of each.

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
        >>> # both fields are tangent: u · Ψ = u · Φ = 0
        >>> import einx
        >>> radial = "obs feature dim, obs dim -> obs feature"
        >>> bool(jnp.allclose(einx.dot(radial, psi, xyz), 0.0, atol=1e-12))
        True
        >>> bool(jnp.allclose(einx.dot(radial, phi, xyz), 0.0, atol=1e-12))
        True
    """
    degree = _check_degree(l_max, "l_max", minimum=1)
    if output not in _OUTPUTS:
        msg = f"'output' must be one of {list(_OUTPUTS)}, got {output!r}"
        raise ValidationError(msg)
    unit_xyz = jnp.asarray(unit_xyz)
    _check_unit_points(unit_xyz, "unit_xyz")

    def harmonics(point: Float[Array, " 3"]) -> Float[Array, " feature"]:
        row = einx.id("dim -> 1 dim", point)
        values = real_spherical_harmonics(row, degree)
        # drop Y₀₀, whose gradient is zero
        return values[0, 1:]

    # --- gradient harmonics Ψ ---
    jacobian = jax.jacfwd(harmonics)
    batched_jacobian = jax.vmap(jacobian)
    # ∇_r Ỹ_ℓm, the ambient gradient of each harmonic at each point
    gradient = batched_jacobian(unit_xyz)
    # r̂ · ∇_r Ỹ_ℓm
    radial = einx.dot("obs feature dim, obs dim -> obs feature", gradient, unit_xyz)
    # (r̂ · ∇_r Ỹ_ℓm) r̂
    normal_part = einx.multiply(
        "obs feature, obs dim -> obs feature dim", radial, unit_xyz
    )
    # Ψ = (I − r̂ r̂ᵀ) ∇_r Ỹ
    psi = gradient - normal_part

    # --- toroidal harmonics Φ ---
    column = einx.id("obs dim -> obs 1 dim", unit_xyz)
    # Φ = r̂ × Ψ
    phi = jnp.cross(column, psi)

    degrees = np.asarray(harmonic_degrees(degree)[1:])
    if normalized:
        # √(ℓ(ℓ + 1))
        norm = np.sqrt(degrees * (degrees + 1.0))
        scale = jnp.asarray(1.0 / norm, dtype=psi.dtype)
        # Ψ / √(ℓ(ℓ + 1))
        psi = einx.multiply("obs feature dim, feature", psi, scale)
        # Φ / √(ℓ(ℓ + 1))
        phi = einx.multiply("obs feature dim, feature", phi, scale)
    if output == "enu":
        # longitude λ = atan2(y, x)
        longitude = jnp.arctan2(unit_xyz[:, 1], unit_xyz[:, 0])
        sin_latitude = jnp.clip(unit_xyz[:, 2], -1.0, 1.0)
        # latitude φ = arcsin z
        latitude = jnp.arcsin(sin_latitude)
        frame = _enu(longitude, latitude)
        east_north = frame[:, :2, :]
        psi = einx.dot(
            "obs feature dim, obs axis dim -> obs feature axis", psi, east_north
        )
        phi = einx.dot(
            "obs feature dim, obs axis dim -> obs feature axis", phi, east_north
        )
    return psi, phi, jnp.asarray(degrees)
