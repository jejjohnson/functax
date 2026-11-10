"""Derivatives of fields on the unit sphere: surface gradient and Laplace–Beltrami.

The operators take a field of a unit vector n ∈ S² and differentiate its
radially constant extension g(x) = f(x/‖x‖), whose Euclidean derivatives at
‖x‖ = 1 are the surface derivatives of f. They act on the unit sphere: for a
sphere of radius R, divide the surface gradient by R and the Laplace–Beltrami
operator by R². Moved from geonnax ``diffops`` at ``be5537d``.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

from collections.abc import Callable
from typing import Any

import einx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Real

from functax._src.diffops._fields import ScalarField, _check_point, laplacian
from functax._src.geometry._frames import enu_basis
from functax._src.geometry._sphere import _check_unit, _safe_norm, lonlat_to_unit
from functax.exceptions import ValidationError


__all__ = [
    "laplace_beltrami",
    "surface_grad",
    "surface_grad_enu",
]


def _unit(x: Float[Array, " 3"]) -> Float[Array, " 3"]:
    """x/‖x‖; the origin has no direction and maps to itself, with finite gradients."""
    squared = jnp.sum(x**2)
    return x / jnp.where(squared > 0.0, _safe_norm(squared), 1.0)


def _radial_extension(f: ScalarField) -> ScalarField:
    """g(x) = f(x/‖x‖), constant along rays."""

    def extension(x: Float[Array, " 3"]) -> Float[Array, ""]:
        return f(_unit(x))

    return extension


def _check_sphere_point(n: Any) -> jax.Array:
    """One 3-D point, projected onto the unit sphere."""
    n = _check_point(n, "n")
    if n.shape[0] != 3:
        msg = f"'n' must be a 3-D vector, got shape {n.shape}"
        raise ValidationError(msg)
    return _unit(n)


def surface_grad(f: ScalarField) -> Callable[[Float[Array, " 3"]], Float[Array, " 3"]]:
    """The surface gradient ∇_S f of a field on the unit sphere.

    With g(x) = f(x/‖x‖) the radially constant extension of f and n the unit
    normal, ∇_S f(n) = (I − n nᵀ) ∇g(n). The input is projected to n/‖n‖ first,
    so any non-zero vector gives the surface gradient at its direction, and
    the final projection removes round-off. The origin has no direction and
    returns a finite but meaningless value (with finite gradients, so masked
    padding is safe).

    Args:
        f: A field of a unit vector, (3,) → ().

    Returns:
        A callable (3,) → (3,) that returns the tangent vector ∇_S f in
        Cartesian components at the direction of its input.

    Raises:
        ValidationError: When the callable gets anything but one 3-D point.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.diffops import surface_grad
        >>> # f = sin(latitude): at the equator its gradient points due north
        >>> surface_grad(lambda n: n[2])(jnp.array([1.0, 0.0, 0.0])).tolist()
        [0.0, 0.0, 1.0]
    """
    extension = _radial_extension(f)

    def _surface_grad(n: Real[ArrayLike, ...]) -> Float[Array, " 3"]:
        n = _check_sphere_point(n)
        ambient = jax.grad(extension)(n)
        # (I − n nᵀ) ∇g
        return ambient - n * jnp.vdot(n, ambient)

    return _surface_grad


def surface_grad_enu(
    f: ScalarField,
    *,
    unit: str,
) -> Callable[[Float[Array, " 2"]], Float[Array, " 2"]]:
    """The east and north components of the surface gradient, at a (λ, φ).

    Projects `surface_grad` onto the local east and north unit vectors of
    `enu_basis`, (êᵀ ∇_S f, n̂ᵀ ∇_S f). On a sphere of radius R these are
    R (∂f/∂x_east, ∂f/∂y_north), the derivatives that give the geostrophic
    velocity from sea-surface height.

    Args:
        f: A field of a unit vector, (3,) → ().
        unit: ``"degrees"`` or ``"radians"``, the unit of the (λ, φ) given to
            the returned callable; no default, as for the geometry functions.

    Returns:
        A callable (2,) → (2,) taking (λ, φ) and returning the (east, north)
        components on the unit sphere.

    Raises:
        GeometryError: If ``unit`` is unknown.
        ValidationError: When the callable gets anything but one (λ, φ).

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.diffops import surface_grad_enu
        >>> # f = sin(latitude) at (30°, 0°): due north, with unit length
        >>> d = surface_grad_enu(lambda n: n[2], unit="degrees")
        >>> [round(float(v), 12) + 0.0 for v in d(jnp.array([30.0, 0.0]))]
        [0.0, 1.0]
    """
    _check_unit(unit)
    gradient_on_sphere = surface_grad(f)

    def _surface_grad_enu(lonlat: Real[ArrayLike, ...]) -> Float[Array, " 2"]:
        lonlat = _check_point(lonlat, "lonlat")
        if lonlat.shape[0] != 2:
            msg = f"'lonlat' must have shape (2,), got {lonlat.shape}"
            raise ValidationError(msg)
        points = einx.id("coordinate -> 1 coordinate", lonlat)
        n = lonlat_to_unit(points, unit=unit)[0]
        # the rows ê, n̂, û at the point
        frame = enu_basis(points, unit=unit)[0]
        return einx.dot("axis dim, dim -> axis", frame[:2], gradient_on_sphere(n))

    return _surface_grad_enu


def laplace_beltrami(
    f: ScalarField,
) -> Callable[[Float[Array, " 3"]], Float[Array, ""]]:
    """The Laplace–Beltrami operator Δ_S f of a field on the unit sphere.

    In spherical coordinates Δ = ∂_r² + (2/r) ∂_r + r⁻² Δ_S, so for the radially
    constant extension g(x) = f(x/‖x‖) the radial terms vanish and
    Δ_S f(n) = Δg(n) at ‖n‖ = 1. The spherical harmonics are its
    eigenfunctions, Δ_S Y_l^m = −l(l + 1) Y_l^m.

    Args:
        f: A field of a unit vector, (3,) → ().

    Returns:
        A callable (3,) → () that evaluates Δ_S f at the direction n/‖n‖ of its
        input (the origin returns a finite but meaningless value).

    Raises:
        ValidationError: When the callable gets anything but one 3-D point.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.diffops import laplace_beltrami
        >>> # f = z is Y_1^0 up to a constant: the eigenvalue is −l(l + 1) = −2
        >>> n = jnp.array([0.6, 0.0, 0.8])
        >>> round(float(laplace_beltrami(lambda m: m[2])(n)) / float(n[2]), 12)
        -2.0
    """
    lap = laplacian(_radial_extension(f))

    def _laplace_beltrami(n: Real[ArrayLike, ...]) -> Float[Array, ""]:
        n = _check_sphere_point(n)
        return lap(n)

    return _laplace_beltrami
