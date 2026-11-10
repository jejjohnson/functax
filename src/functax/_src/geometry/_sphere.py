"""The sphere: lon/lat to unit vectors and back, and chord and geodesic distances.

Longitude λ and latitude φ lift to u = (cos φ cos λ, cos φ sin λ, sin φ), the
convention of geonnax, so the spherical bases that move in during Wave 1 line
up. The radius a is a plain argument: the WGS84 constants and the ellipsoid
belong to geophysix, not to functax.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import math
from typing import Any

import einx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from functax._src._backend import is_tracer
from functax.exceptions import GeometryError


__all__ = [
    "chord_distance",
    "geodesic_distance",
    "lonlat_to_unit",
    "spherical_polygon_area",
    "unit_to_lonlat",
]


_UNITS: tuple[str, ...] = ("degrees", "radians")


def _check_unit(unit: str) -> None:
    if unit not in _UNITS:
        msg = f"'unit' must be one of {list(_UNITS)}, got {unit!r}"
        raise GeometryError(msg)


def _check_last_axis(array: jax.Array, size: int, name: str) -> None:
    """Raise unless ``array`` is floating with a last axis of length ``size``."""
    if array.ndim < 1 or array.shape[-1] != size:
        msg = (
            f"'{name}' must have a last axis of length {size}, got shape {array.shape}"
        )
        raise GeometryError(msg)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        msg = f"'{name}' must have a floating dtype, got {array.dtype}"
        raise GeometryError(msg)


def _check_points(array: jax.Array, name: str) -> None:
    """Raise unless ``array`` is a floating ``(points, 3)`` array of unit vectors."""
    if array.ndim != 2:
        msg = f"'{name}' must be 2-D (points, 3), got shape {array.shape}"
        raise GeometryError(msg)
    _check_last_axis(array, 3, name)


def _check_radius(radius: Any) -> None:
    """Raise when a concrete radius is not positive and finite."""
    if is_tracer(radius):
        return
    value = float(radius)
    if not (math.isfinite(value) and value > 0.0):
        msg = f"'radius' must be positive and finite, got {radius!r}"
        raise GeometryError(msg)


def _safe_norm(squared: jax.Array) -> jax.Array:
    """√s with a zero gradient at s = 0 instead of NaN (the double-where pattern)."""
    positive = squared > 0
    safe_squared = jnp.where(positive, squared, 1.0)
    return jnp.where(positive, jnp.sqrt(safe_squared), 0.0)


def lonlat_to_unit(
    lonlat: Float[Array, "*batch 2"],
    *,
    unit: str = "radians",
) -> Float[Array, "*batch 3"]:
    """Lift longitude and latitude onto the unit sphere.

    u = (cos φ cos λ, cos φ sin λ, sin φ), with λ the longitude and φ the
    latitude. Equal to geonnax's ``lonlat_to_cartesian3d``.

    Args:
        lonlat: (λ, φ) along the last axis, with any leading batch dims.
        unit: ``"radians"`` or ``"degrees"``.

    Returns:
        u, the unit vectors, with the batch dims of ``lonlat`` and a last axis
        of 3, in the dtype of ``lonlat``.

    Raises:
        GeometryError: If the last axis is not 2, the dtype is not floating,
            or ``unit`` is unknown.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import lonlat_to_unit
        >>> lonlat_to_unit(
        ...     jnp.array([[0.0, 0.0], [90.0, 0.0]]), unit="degrees"
        ... ).round(12).tolist()
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        >>> lonlat_to_unit(jnp.zeros((4, 5, 2))).shape
        (4, 5, 3)
    """
    lonlat = jnp.asarray(lonlat)
    _check_last_axis(lonlat, 2, "lonlat")
    _check_unit(unit)
    # λ, φ in radians: x · π/180 for degrees, as geonnax's deg2rad does
    angles = lonlat * (jnp.pi / 180.0) if unit == "degrees" else lonlat
    longitude = angles[..., 0]
    latitude = angles[..., 1]
    cos_latitude = jnp.cos(latitude)
    # u = (cos φ cos λ, cos φ sin λ, sin φ)
    x = cos_latitude * jnp.cos(longitude)
    y = cos_latitude * jnp.sin(longitude)
    z = jnp.sin(latitude)
    return einx.id("..., ..., ... -> ... (1 + 1 + 1)", x, y, z)


def unit_to_lonlat(
    xyz: Float[Array, "*batch 3"],
    *,
    unit: str = "radians",
) -> Float[Array, "*batch 2"]:
    """Longitude and latitude of vectors on (or off) the unit sphere.

    λ = atan2(y, x) and φ = atan2(z, √(x² + y²)), so the vector need not have
    unit length, and λ lies in (−π, π]. At the poles λ is undefined and comes
    back as 0 (or ±π, by the sign of x).

    Args:
        xyz: (x, y, z) along the last axis, with any leading batch dims.
        unit: ``"radians"`` or ``"degrees"``.

    Returns:
        (λ, φ), with the batch dims of ``xyz`` and a last axis of 2.

    Raises:
        GeometryError: If the last axis is not 3, the dtype is not floating,
            or ``unit`` is unknown.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import lonlat_to_unit, unit_to_lonlat
        >>> unit_to_lonlat(jnp.array([[0.0, 1.0, 0.0]]), unit="degrees").tolist()
        [[90.0, 0.0]]
        >>> lonlat = jnp.array([[-120.0, 45.0]])
        >>> back = unit_to_lonlat(
        ...     lonlat_to_unit(lonlat, unit="degrees"), unit="degrees"
        ... )
        >>> bool(jnp.allclose(back, lonlat))
        True
    """
    xyz = jnp.asarray(xyz)
    _check_last_axis(xyz, 3, "xyz")
    _check_unit(unit)
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    # λ = atan2(y, x), φ = atan2(z, √(x² + y²))
    longitude = jnp.arctan2(y, x)
    latitude = jnp.arctan2(z, jnp.hypot(x, y))
    lonlat = einx.id("..., ... -> ... (1 + 1)", longitude, latitude)
    if unit == "degrees":
        # x · 180/π
        lonlat = lonlat * (180.0 / jnp.pi)
    return lonlat


def chord_distance(
    u: Float[Array, "source 3"],
    v: Float[Array, "target 3"],
    *,
    radius: float | Float[Array, ""] = 1.0,
) -> Float[Array, "source target"]:
    """The straight-line distance through the sphere between every pair.

    d_chord = a ‖u − v‖ for unit vectors u and v and radius a. The norm is
    guarded at u = v, so the gradient there is zero rather than NaN.

    Args:
        u: The M source unit vectors.
        v: The N target unit vectors.
        radius: a, positive.

    Returns:
        The M × N chord distances, in the units of ``radius``.

    Raises:
        GeometryError: If ``u`` or ``v`` is not a floating (points, 3) array, or
            a concrete ``radius`` is not positive and finite.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import chord_distance
        >>> north, south = (
        ...     jnp.array([[0.0, 0.0, 1.0]]),
        ...     jnp.array([[0.0, 0.0, -1.0]]),
        ... )
        >>> chord_distance(north, south, radius=6371.0).tolist()
        [[12742.0]]
    """
    u = jnp.asarray(u)
    v = jnp.asarray(v)
    _check_points(u, "u")
    _check_points(v, "v")
    _check_radius(radius)
    # u − v for every pair
    difference = einx.subtract("source c, target c -> source target c", u, v)
    # ‖u − v‖²
    squared = einx.sum("source target [c]", difference**2)
    # d = a ‖u − v‖
    return radius * _safe_norm(squared)


def geodesic_distance(
    u: Float[Array, "source 3"],
    v: Float[Array, "target 3"],
    *,
    radius: float | Float[Array, ""] = 1.0,
) -> Float[Array, "source target"]:
    """The great-circle distance between every pair.

    d_geo = a · atan2(‖u × v‖, u · v) for unit vectors u and v and radius a.
    The atan2 form is accurate at every angle, where arccos(u · v) loses
    precision near 0 and π and has an infinite derivative at u = v. The cross
    product's norm is guarded at u = v, so the gradient there is zero.

    Args:
        u: The M source unit vectors.
        v: The N target unit vectors.
        radius: a, positive.

    Returns:
        The M × N geodesic distances, in the units of ``radius``.

    Raises:
        GeometryError: If ``u`` or ``v`` is not a floating (points, 3) array, or
            a concrete ``radius`` is not positive and finite.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import geodesic_distance
        >>> equator = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> bool(
        ...     jnp.allclose(
        ...         geodesic_distance(equator, equator),
        ...         jnp.array([[0.0, jnp.pi / 2], [jnp.pi / 2, 0.0]]),
        ...     )
        ... )
        True
    """
    u = jnp.asarray(u)
    v = jnp.asarray(v)
    _check_points(u, "u")
    _check_points(v, "v")
    _check_radius(radius)
    u_x, u_y, u_z = u[:, 0], u[:, 1], u[:, 2]
    v_x, v_y, v_z = v[:, 0], v[:, 1], v[:, 2]
    # u × v for every pair, one component at a time
    cross_x = einx.multiply("source, target -> source target", u_y, v_z)
    cross_x = cross_x - einx.multiply("source, target -> source target", u_z, v_y)
    cross_y = einx.multiply("source, target -> source target", u_z, v_x)
    cross_y = cross_y - einx.multiply("source, target -> source target", u_x, v_z)
    cross_z = einx.multiply("source, target -> source target", u_x, v_y)
    cross_z = cross_z - einx.multiply("source, target -> source target", u_y, v_x)
    # ‖u × v‖ and u · v
    cross_norm = _safe_norm(cross_x**2 + cross_y**2 + cross_z**2)
    dot = einx.dot("source c, target c -> source target", u, v)
    # d = a · atan2(‖u × v‖, u · v)
    return radius * jnp.arctan2(cross_norm, dot)


def spherical_polygon_area(
    vertices: Float[Array, "*batch vertex 3"],
    *,
    radius: float | Float[Array, ""] = 1.0,
) -> Float[Array, "*batch"]:
    """The signed area of a spherical polygon with great-circle edges.

    A = a² E with the spherical excess E = ∑ₖ αₖ − (n − 2)π of an n-gon with
    interior angles αₖ. The angle sum loses all but a few digits on small
    cells (E is the difference of numbers near (n − 2)π), so E is summed over
    the fan of triangles (u₀, uₖ, uₖ₊₁) instead, each by the formula of Van
    Oosterom and Strackee,

        tan(Eₖ / 2) = u₀ · (uₖ × uₖ₊₁) / (1 + u₀ · uₖ + uₖ · uₖ₊₁ + uₖ₊₁ · u₀),

    which is exact to rounding at every size. Eₖ carries the sign of the
    triple product, so A is positive for vertices counter-clockwise seen from
    outside the sphere, negative for clockwise ones, and exact for any simple
    polygon in an open hemisphere. A repeated vertex adds a triangle of zero
    area, so polygons with fewer vertices pad to a common count by repeating
    their last vertex.

    Args:
        vertices: u, the n ≥ 3 vertices as unit vectors, in order, with any
            leading batch dims.
        radius: a, positive.

    Returns:
        A, one area per polygon, in the units of ``radius`` squared.

    Raises:
        GeometryError: If ``vertices`` is not floating with a last axis of 3
            and at least three vertices, or a concrete ``radius`` is not
            positive and finite.

    References:
        Van Oosterom, A., Strackee, J. (1983). The solid angle of a plane
        triangle. IEEE Transactions on Biomedical Engineering, BME-30(2),
        125–126.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import icosphere, spherical_polygon_area
        >>> # the octant x, y, z ≥ 0 is an eighth of the sphere
        >>> octant = jnp.eye(3)
        >>> bool(jnp.isclose(spherical_polygon_area(octant), jnp.pi / 2))
        True
        >>> vertices, triangles = icosphere(2)
        >>> areas = spherical_polygon_area(vertices[triangles])
        >>> areas.shape, bool(jnp.isclose(areas.sum(), 4 * jnp.pi))
        ((320,), True)
    """
    vertices = jnp.asarray(vertices)
    _check_last_axis(vertices, 3, "vertices")
    if vertices.ndim < 2 or vertices.shape[-2] < 3:
        msg = (
            "'vertices' must hold at least three vertices (..., vertex, 3), got "
            f"shape {vertices.shape}"
        )
        raise GeometryError(msg)
    _check_radius(radius)
    # the fan (u₀, uₖ, uₖ₊₁) for k = 1, …, n − 2
    apex = vertices[..., 0, :]
    left = vertices[..., 1:-1, :]
    right = vertices[..., 2:, :]
    # u₀ · (uₖ × uₖ₊₁)
    triple = einx.dot("... c, ... fan c -> ... fan", apex, jnp.cross(left, right))
    # 1 + u₀ · uₖ + uₖ · uₖ₊₁ + uₖ₊₁ · u₀
    apex_left = einx.dot("... c, ... fan c -> ... fan", apex, left)
    left_right = einx.dot("... fan c, ... fan c -> ... fan", left, right)
    right_apex = einx.dot("... fan c, ... c -> ... fan", right, apex)
    denominator = 1 + apex_left + left_right + right_apex
    # Eₖ = 2 atan2(triple, denominator)
    excess = 2 * einx.sum("... [fan]", jnp.arctan2(triple, denominator))
    return radius**2 * excess
