"""Frames on the sphere and the ellipsoid: ECEF, local east/north/up, rotations.

Geographic inputs are (λ, φ) pairs, longitude first, and every function takes
their ``unit`` explicitly. The ellipsoid enters as its semi-major axis a and
flattening f: the WGS84 values belong to geophysix or the caller, not to
functax. Moved from geonnax ``_frames`` at ``be5537d``.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import math
from typing import Any

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float

from functax._src._backend import is_tracer
from functax._src.geometry._sphere import _check_unit
from functax.exceptions import GeometryError


__all__ = [
    "cartesian_to_tangent",
    "ecef_to_lonlat",
    "enu_basis",
    "local_tangent_plane",
    "lonlat_to_ecef",
    "rotation_to_pole",
    "tangent_to_cartesian",
]


_PROJECTIONS: tuple[str, ...] = ("azimuthal_equidistant", "gnomonic")


def _to_radians(lonlat: Any, unit: str) -> Float[Array, "*shape"]:
    """(λ, φ) in radians; integers become float32, as in geonnax."""
    _check_unit(unit)
    lonlat = jnp.asarray(lonlat)
    if not jnp.issubdtype(lonlat.dtype, jnp.floating):
        lonlat = lonlat.astype(jnp.float32)
    if unit == "degrees":
        # x · π/180
        return jnp.deg2rad(lonlat)
    return lonlat


def _check_lonlat(lonlat: Any, name: str = "lonlat") -> None:
    shape = jnp.shape(lonlat)
    if len(shape) != 2 or shape[-1] != 2:
        msg = f"'{name}' must have shape (N, 2), got {shape}"
        raise GeometryError(msg)


def _check_positive(value: Any, name: str) -> None:
    """Raise when a concrete ``value`` is not positive and finite."""
    if is_tracer(value):
        return
    number = float(value)
    if not (math.isfinite(number) and number > 0.0):
        msg = f"'{name}' must be positive and finite, got {value!r}"
        raise GeometryError(msg)


def _check_ellipsoid(semi_major: Any, flattening: Any) -> None:
    _check_positive(semi_major, "semi_major")
    if is_tracer(flattening):
        return
    value = float(flattening)
    if not (0.0 <= value < 1.0):
        msg = f"'flattening' must lie in [0, 1), got {flattening!r}"
        raise GeometryError(msg)


def _enu(
    longitude: Float[Array, " obs"], latitude: Float[Array, " obs"]
) -> Float[Array, "obs 3 3"]:
    """The east, north and up unit vectors at each point, as the rows of a frame."""
    stack = "obs, obs, obs -> obs (1 + 1 + 1)"
    # sin λ, cos λ, sin φ, cos φ
    sin_lon = jnp.sin(longitude)
    cos_lon = jnp.cos(longitude)
    sin_lat = jnp.sin(latitude)
    cos_lat = jnp.cos(latitude)
    zero = jnp.zeros_like(longitude)
    # ê = (−sin λ, cos λ, 0)
    east_x = -sin_lon
    east = einx.id(stack, east_x, cos_lon, zero)
    # n̂ = (−sin φ cos λ, −sin φ sin λ, cos φ)
    north_x = -sin_lat * cos_lon
    north_y = -sin_lat * sin_lon
    north = einx.id(stack, north_x, north_y, cos_lat)
    # û = (cos φ cos λ, cos φ sin λ, sin φ)
    up_x = cos_lat * cos_lon
    up_y = cos_lat * sin_lon
    up = einx.id(stack, up_x, up_y, sin_lat)
    # the frame, with rows ê, n̂, û
    return einx.id("obs dim, obs dim, obs dim -> obs (1 + 1 + 1) dim", east, north, up)


def lonlat_to_ecef(
    lonlat: Float[Array, "obs 2"],
    altitude: float | Float[Array, " obs"] = 0.0,
    *,
    semi_major: float | Float[Array, ""],
    flattening: float | Float[Array, ""] = 0.0,
    unit: str,
) -> Float[Array, "obs 3"]:
    """Geodetic (λ, φ) and height to Earth-centred, Earth-fixed coordinates.

    On the ellipsoid with semi-major axis a and flattening f, e² = f(2 − f) and
    N(φ) = a / √(1 − e² sin² φ), and

    X = (N + h) cos φ cos λ, Y = (N + h) cos φ sin λ, Z = (N(1 − e²) + h) sin φ.

    f = 0 is the sphere of radius a. The WGS84 values of a and f belong to
    geophysix or the caller; functax holds no Earth constant.

    Args:
        lonlat: (λ, φ) of each point.
        altitude: h, the height above the ellipsoid in the units of a, one
            value or one per point.
        semi_major: a.
        flattening: f, in [0, 1).
        unit: ``"degrees"`` or ``"radians"``, the unit of ``lonlat``.

    Returns:
        (X, Y, Z) of each point, in the units of a.

    Raises:
        GeometryError: If ``lonlat`` is not (N, 2), ``unit`` is unknown, or a
            concrete a is not positive or f not in [0, 1).

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import lonlat_to_ecef
        >>> lonlat_to_ecef(
        ...     jnp.array([[0.0, 0.0]]), semi_major=2.0, unit="degrees"
        ... ).tolist()
        [[2.0, 0.0, 0.0]]
        >>> # at the pole Z is the semi-minor axis b = a(1 − f)
        >>> pole = lonlat_to_ecef(
        ...     jnp.array([[0.0, 90.0]]),
        ...     semi_major=2.0,
        ...     flattening=0.25,
        ...     unit="degrees",
        ... )
        >>> bool(jnp.allclose(pole[0, 2], 1.5, rtol=1e-12))
        True
    """
    _check_lonlat(lonlat)
    _check_ellipsoid(semi_major, flattening)
    radians = _to_radians(lonlat, unit)
    longitude = radians[:, 0]
    latitude = radians[:, 1]
    height = jnp.asarray(altitude, dtype=radians.dtype)
    # e² = f(2 − f)
    eccentricity_sq = flattening * (2.0 - flattening)
    # sin φ, cos φ, sin λ, cos λ
    sin_lat = jnp.sin(latitude)
    cos_lat = jnp.cos(latitude)
    sin_lon = jnp.sin(longitude)
    cos_lon = jnp.cos(longitude)
    # 1 − e² sin² φ
    radicand = 1.0 - eccentricity_sq * sin_lat**2
    # N(φ) = a / √(1 − e² sin² φ)
    normal_radius = semi_major / jnp.sqrt(radicand)
    # X = (N + h) cos φ cos λ
    x = (normal_radius + height) * cos_lat * cos_lon
    # Y = (N + h) cos φ sin λ
    y = (normal_radius + height) * cos_lat * sin_lon
    # Z = (N(1 − e²) + h) sin φ
    z = (normal_radius * (1.0 - eccentricity_sq) + height) * sin_lat
    return einx.id("obs, obs, obs -> obs (1 + 1 + 1)", x, y, z)


def ecef_to_lonlat(
    xyz: Float[Array, "obs 3"],
    *,
    semi_major: float | Float[Array, ""],
    flattening: float | Float[Array, ""] = 0.0,
    unit: str,
) -> tuple[Float[Array, "obs 2"], Float[Array, " obs"]]:
    """Earth-centred, Earth-fixed coordinates to geodetic (λ, φ) and height.

    The inverse of `lonlat_to_ecef`, with geonnax's algorithm: Bowring's
    (1976) closed-form latitude, then two fixed refinement steps
    φ ← atan2(Z, p (1 − e² N / (N + h))) with p = √(X² + Y²), accurate to well
    under a millimetre on WGS84 for heights within ±10 km. The height is
    h = p cos φ + Z sin φ − a √(1 − e² sin² φ), which stays stable at the
    poles. With f = 0 the steps reduce to φ = atan2(Z, p) and h = ‖x‖ − a.

    Algorithm:
        1. λ = atan2(Y, X) and p = √(X² + Y²).
        2. Bowring's latitude from the parametric latitude
           θ = atan2(Z a, p b).
        3. Two refinement steps φ ← atan2(Z, p (1 − e² N / (N + h))), a fixed
           count so the work has a static shape.
        4. h at the final φ.

    Args:
        xyz: (X, Y, Z) of each point, in the units of a.
        semi_major: a.
        flattening: f, in [0, 1).
        unit: ``"degrees"`` or ``"radians"``, the unit of the returned (λ, φ).

    Returns:
        ``(lonlat, altitude)``: (λ, φ) and h of each point.

    Raises:
        GeometryError: If ``xyz`` is not (N, 3), ``unit`` is unknown, or a
            concrete a is not positive or f not in [0, 1).

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import ecef_to_lonlat, lonlat_to_ecef
        >>> ellipsoid = {"semi_major": 6.4e6, "flattening": 3e-3}
        >>> xyz = lonlat_to_ecef(
        ...     jnp.array([[10.0, 45.0]]), 100.0, unit="degrees", **ellipsoid
        ... )
        >>> lonlat, height = ecef_to_lonlat(xyz, unit="degrees", **ellipsoid)
        >>> bool(jnp.allclose(lonlat, jnp.array([[10.0, 45.0]]), atol=1e-12))
        True
        >>> bool(jnp.allclose(height, 100.0, atol=1e-6))
        True

        On the sphere (f = 0) the height is ‖x‖ − a:

        >>> point = jnp.array([[0.0, 3.0, 4.0]])
        >>> _, height = ecef_to_lonlat(point, semi_major=4.0, unit="radians")
        >>> bool(jnp.allclose(height, 1.0))
        True
    """
    xyz = jnp.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        msg = f"'xyz' must have shape (N, 3), got {xyz.shape}"
        raise GeometryError(msg)
    _check_unit(unit)
    _check_ellipsoid(semi_major, flattening)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    # λ = atan2(Y, X)
    longitude = jnp.arctan2(y, x)
    # p = √(X² + Y²)
    horizontal = jnp.hypot(x, y)
    # e² = f(2 − f), b = a(1 − f), e′² = e² / (1 − e²)
    eccentricity_sq = flattening * (2.0 - flattening)
    semi_minor = semi_major * (1.0 - flattening)
    second_eccentricity_sq = eccentricity_sq / (1.0 - eccentricity_sq)

    # --- Bowring's latitude ---
    # θ = atan2(Z a, p b)
    parametric = jnp.arctan2(z * semi_major, horizontal * semi_minor)
    sin_parametric = jnp.sin(parametric)
    cos_parametric = jnp.cos(parametric)
    # φ = atan2(Z + e′² b sin³ θ, p − e² a cos³ θ)
    numerator = z + second_eccentricity_sq * semi_minor * sin_parametric**3
    denominator = horizontal - eccentricity_sq * semi_major * cos_parametric**3
    latitude = jnp.arctan2(numerator, denominator)

    def height_at(lat: Float[Array, " obs"]) -> Float[Array, " obs"]:
        sin_lat = jnp.sin(lat)
        cos_lat = jnp.cos(lat)
        # √(1 − e² sin² φ)
        root = jnp.sqrt(1.0 - eccentricity_sq * sin_lat**2)
        # h = p cos φ + Z sin φ − a √(1 − e² sin² φ)
        return horizontal * cos_lat + z * sin_lat - semi_major * root

    # --- two fixed refinement steps, so the shape of the work is static ---
    for _ in range(2):
        sin_lat = jnp.sin(latitude)
        # N(φ) = a / √(1 − e² sin² φ)
        radicand = 1.0 - eccentricity_sq * sin_lat**2
        normal_radius = semi_major / jnp.sqrt(radicand)
        height = height_at(latitude)
        # φ ← atan2(Z, p (1 − e² N / (N + h)))
        shrink = 1.0 - eccentricity_sq * normal_radius / (normal_radius + height)
        latitude = jnp.arctan2(z, horizontal * shrink)
    height = height_at(latitude)
    lonlat = einx.id("obs, obs -> obs (1 + 1)", longitude, latitude)
    if unit == "degrees":
        # x · 180/π
        lonlat = jnp.rad2deg(lonlat)
    return lonlat, height


def enu_basis(lonlat: Float[Array, "obs 2"], *, unit: str) -> Float[Array, "obs 3 3"]:
    """The local east, north and up unit vectors at each point.

    At longitude λ and latitude φ, ê = (−sin λ, cos λ, 0),
    n̂ = (−sin φ cos λ, −sin φ sin λ, cos φ) and
    û = (cos φ cos λ, cos φ sin λ, sin φ). The frame is orthonormal and
    right-handed (ê × n̂ = û). At φ = ±90° east and north are not unique; the
    formula still applies, so the given longitude picks the frame.

    Args:
        lonlat: (λ, φ) of each point.
        unit: ``"degrees"`` or ``"radians"``, the unit of ``lonlat``.

    Returns:
        One 3 × 3 frame per point, with rows ê, n̂ and û.

    Raises:
        GeometryError: If ``lonlat`` is not (N, 2) or ``unit`` is unknown.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import enu_basis
        >>> (enu_basis(jnp.array([[0.0, 0.0]]), unit="degrees")[0] + 0.0).tolist()
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        >>> enu_basis(jnp.zeros((5, 2)), unit="radians").shape
        (5, 3, 3)
    """
    _check_lonlat(lonlat)
    radians = _to_radians(lonlat, unit)
    longitude = radians[:, 0]
    latitude = radians[:, 1]
    return _enu(longitude, latitude)


def tangent_to_cartesian(
    lonlat: Float[Array, "obs 2"],
    uv: Float[Array, "obs 2"],
    *,
    unit: str,
) -> Float[Array, "obs 3"]:
    """East and north components (u, v) to the 3-D tangent vector u ê + v n̂.

    Args:
        lonlat: (λ, φ) of each vector's base point.
        uv: The eastward and northward components.
        unit: ``"degrees"`` or ``"radians"``, the unit of ``lonlat``.

    Returns:
        The Cartesian tangent vectors.

    Raises:
        GeometryError: If ``lonlat`` is not (N, 2) or ``unit`` is unknown.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import tangent_to_cartesian
        >>> east = jnp.array([[1.0, 0.0]])
        >>> origin = jnp.array([[0.0, 0.0]])
        >>> (tangent_to_cartesian(origin, east, unit="degrees") + 0.0).tolist()
        [[0.0, 1.0, 0.0]]

        The result is tangent: it is orthogonal to û at every point:

        >>> from functax.geometry import enu_basis
        >>> lonlat = jnp.array([[10.0, 20.0], [-30.0, 60.0]])
        >>> uv = jnp.array([[1.0, 2.0], [3.0, -1.0]])
        >>> vectors = tangent_to_cartesian(lonlat, uv, unit="degrees")
        >>> up = enu_basis(lonlat, unit="degrees")[:, 2, :]
        >>> import einx
        >>> radial = einx.dot("obs dim, obs dim -> obs", vectors, up)
        >>> bool(jnp.allclose(radial, 0.0))
        True
    """
    frame = enu_basis(lonlat, unit=unit)
    # the rows ê and n̂ of each frame
    tangent_frame = frame[:, :2, :]
    components = jnp.asarray(uv)
    # v = u ê + v n̂
    return einx.dot(
        "obs component, obs component dim -> obs dim", components, tangent_frame
    )


def cartesian_to_tangent(
    lonlat: Float[Array, "obs 2"],
    vec: Float[Array, "obs 3"],
    *,
    unit: str,
) -> Float[Array, "obs 2"]:
    """3-D vectors to their east and north components (v · ê, v · n̂).

    The radial part is dropped, so on tangent vectors this inverts
    `tangent_to_cartesian`.

    Args:
        lonlat: (λ, φ) of each vector's base point.
        vec: The Cartesian vectors.
        unit: ``"degrees"`` or ``"radians"``, the unit of ``lonlat``.

    Returns:
        The eastward and northward components.

    Raises:
        GeometryError: If ``lonlat`` is not (N, 2) or ``unit`` is unknown.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import cartesian_to_tangent
        >>> up_z = jnp.array([[0.0, 0.0, 2.0]])
        >>> origin = jnp.array([[0.0, 0.0]])
        >>> (cartesian_to_tangent(origin, up_z, unit="degrees") + 0.0).tolist()
        [[0.0, 2.0]]

        It inverts `tangent_to_cartesian` on tangent vectors:

        >>> from functax.geometry import tangent_to_cartesian
        >>> lonlat = jnp.array([[10.0, 20.0], [-30.0, 60.0]])
        >>> uv = jnp.array([[1.0, 2.0], [3.0, -1.0]])
        >>> vectors = tangent_to_cartesian(lonlat, uv, unit="degrees")
        >>> back = cartesian_to_tangent(lonlat, vectors, unit="degrees")
        >>> bool(jnp.allclose(back, uv))
        True
    """
    frame = enu_basis(lonlat, unit=unit)
    # the rows ê and n̂ of each frame
    tangent_frame = frame[:, :2, :]
    vectors = jnp.asarray(vec)
    # (v · ê, v · n̂)
    return einx.dot(
        "obs component dim, obs dim -> obs component", tangent_frame, vectors
    )


def local_tangent_plane(
    lonlat: Float[Array, "obs 2"],
    origin: Float[Array, " 2"],
    *,
    radius: float | Float[Array, ""],
    projection: str = "azimuthal_equidistant",
    unit: str,
) -> Float[Array, "obs 2"]:
    """Project points onto the tangent plane at ``origin``, as (east, north).

    With ê₀, n̂₀, û₀ the frame at the origin and p a point's unit vector, let
    x = p · ê₀, y = p · n̂₀ and z = p · û₀.

    - ``"azimuthal_equidistant"`` keeps the distance and bearing from the
      origin: R (x, y) θ / sin θ, with θ = atan2(√(x² + y²), z) the
      great-circle angle.
    - ``"gnomonic"`` is the central projection R (x, y) / z, which maps great
      circles to straight lines; only the hemisphere around the origin (z > 0)
      maps to finite points.

    Args:
        lonlat: (λ, φ) of each point.
        origin: (λ, φ) of the tangent point.
        radius: R, the sphere radius; the output is in its units.
        projection: ``"azimuthal_equidistant"`` or ``"gnomonic"``.
        unit: ``"degrees"`` or ``"radians"``, the unit of ``lonlat`` and
            ``origin``.

    Returns:
        The (east, north) plane coordinates of each point.

    Raises:
        GeometryError: If ``lonlat`` is not (N, 2), ``unit`` or
            ``projection`` is unknown, or a concrete R is not positive.

    Examples:
        >>> import math
        >>> import jax.numpy as jnp
        >>> from functax.geometry import local_tangent_plane
        >>> # on a sphere of radius 180/π, one degree of arc has length 1
        >>> plane = local_tangent_plane(
        ...     jnp.array([[0.0, 1.0]]),
        ...     jnp.array([0.0, 0.0]),
        ...     radius=180 / math.pi,
        ...     unit="degrees",
        ... )
        >>> [round(float(v), 12) for v in plane[0]]
        [0.0, 1.0]

        The origin maps to (0, 0) in either projection:

        >>> origin = jnp.array([[25.0, -40.0]])
        >>> plane = local_tangent_plane(
        ...     origin, origin[0], radius=1.0, projection="gnomonic", unit="degrees"
        ... )
        >>> bool(jnp.allclose(plane, 0.0))
        True
    """
    _check_lonlat(lonlat)
    _check_positive(radius, "radius")
    if projection not in _PROJECTIONS:
        msg = f"'projection' must be one of {list(_PROJECTIONS)}, got {projection!r}"
        raise GeometryError(msg)
    radians = _to_radians(lonlat, unit)
    centre = _to_radians(origin, unit)
    # the rows ê₀, n̂₀, û₀ at the origin
    origin_frames = _enu(centre[0:1], centre[1:2])
    frame = origin_frames[0]
    # p = û, the unit vector of each point
    point_frames = _enu(radians[:, 0], radians[:, 1])
    unit_vectors = point_frames[:, 2, :]
    # (x, y, z) = (p · ê₀, p · n̂₀, p · û₀)
    local = einx.dot("obs dim, component dim -> obs component", unit_vectors, frame)
    plane = local[:, :2]
    height = local[:, 2]
    if projection == "gnomonic":
        # (x, y) / z
        central = einx.divide("obs component, obs", plane, height)
        # R (x, y) / z
        return radius * central
    # x², y²
    plane_squared = plane**2
    # s² = x² + y²
    squared = einx.sum("obs [component]", plane_squared)
    positive = squared > 0
    # s = √(x² + y²), with 1 at the origin so √ keeps a finite gradient
    safe_squared = jnp.where(positive, squared, 1.0)
    chord = jnp.sqrt(safe_squared)
    # θ = atan2(s, z)
    angle = jnp.arctan2(chord, height)
    # θ / sin θ → 1 at the origin; the where keeps its gradient finite there
    scale = jnp.where(positive, angle / chord, 1.0)
    # (x, y) θ / sin θ
    scaled = einx.multiply("obs component, obs", plane, scale)
    # R (x, y) θ / sin θ
    return radius * scaled


def rotation_to_pole(
    lonlat_centre: Float[Array, " 2"], *, unit: str
) -> Float[Array, "3 3"]:
    """The rotation R with R û(centre) = (0, 0, 1).

    Its rows are the east, north and up vectors at the centre, so
    R p = (p · ê, p · n̂, p · û): the centre goes to the north pole and its
    local north to the +y axis. The Slepian cap bases are evaluated in this
    frame.

    Args:
        lonlat_centre: (λ, φ) of the centre.
        unit: ``"degrees"`` or ``"radians"``, the unit of ``lonlat_centre``.

    Returns:
        R, orthogonal with determinant 1.

    Raises:
        GeometryError: If ``lonlat_centre`` is not (2,) or ``unit`` is
            unknown.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import rotation_to_pole
        >>> rotation = rotation_to_pole(jnp.array([0.0, 0.0]), unit="degrees")
        >>> (rotation @ jnp.array([1.0, 0.0, 0.0]) + 0.0).round(12).tolist()
        [0.0, 0.0, 1.0]

        Any centre goes to the pole:

        >>> from functax.geometry import lonlat_to_unit
        >>> centre = jnp.array([120.0, -35.0])
        >>> rotation = rotation_to_pole(centre, unit="degrees")
        >>> pole = rotation @ lonlat_to_unit(centre, unit="degrees")
        >>> bool(jnp.allclose(pole, jnp.array([0.0, 0.0, 1.0])))
        True
    """
    centre = _to_radians(lonlat_centre, unit)
    if centre.shape != (2,):
        msg = f"'lonlat_centre' must have shape (2,), got {centre.shape}"
        raise GeometryError(msg)
    frames = _enu(centre[0:1], centre[1:2])
    return frames[0]
