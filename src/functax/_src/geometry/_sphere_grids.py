"""Point sets on the unit sphere with quadrature weights for ∫_{S²} f dA.

Each grid returns ``(points, weights)`` with weights summing to 4π: the
Fibonacci lattice, the Gauss–Legendre grid, the cell-centred lon/lat grid and
the HEALPix pixel centres, with dA the area element of the unit sphere. Points
are unit vectors (``output="xyz"``) or (λ, φ) pairs of longitude and latitude
(``output="lonlat"``, which needs ``unit``). The grids are built on
the host with NumPy, as in geonnax ``be5537d``, and returned as JAX arrays.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import math
import operator
from typing import Any

import einx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from functax._src.geometry._sphere import _check_unit
from functax._src.quad._rules import gauss_legendre
from functax.exceptions import GeometryError


__all__ = [
    "fibonacci_sphere",
    "gauss_legendre_grid",
    "healpix_grid",
    "latlon_grid",
]


_OUTPUTS: tuple[str, ...] = ("lonlat", "xyz")


def _check_output(output: str, unit: str | None) -> None:
    if output not in _OUTPUTS:
        msg = f"'output' must be one of {list(_OUTPUTS)}, got {output!r}"
        raise GeometryError(msg)
    if output == "lonlat":
        if unit is None:
            msg = "'unit' is required with output='lonlat': 'degrees' or 'radians'"
            raise GeometryError(msg)
        _check_unit(unit)


def _check_count(value: Any, name: str) -> int:
    """``value`` as an int of at least 1; NumPy integers pass, floats and bools fail."""
    if isinstance(value, bool):
        msg = f"'{name}' must be an integer of at least 1, got {value!r}"
        raise GeometryError(msg)
    try:
        count = operator.index(value)
    except TypeError as err:
        msg = f"'{name}' must be an integer of at least 1, got {value!r}"
        raise GeometryError(msg) from err
    if count < 1:
        msg = f"'{name}' must be an integer of at least 1, got {value!r}"
        raise GeometryError(msg)
    return count


def _format(
    lon: Float[np.ndarray, " node"],
    lat: Float[np.ndarray, " node"],
    output: str,
    unit: str | None,
) -> Float[Array, "node dim"]:
    """The points as unit vectors or as (λ, φ) in ``unit``, on the device."""
    if output == "xyz":
        # cos φ
        cos_lat = np.cos(lat)
        # u = (cos φ cos λ, cos φ sin λ, sin φ)
        x = cos_lat * np.cos(lon)
        y = cos_lat * np.sin(lon)
        z = np.sin(lat)
        xyz = einx.id("node, node, node -> node (1 + 1 + 1)", x, y, z)
        return jnp.asarray(xyz)
    lonlat = einx.id("node, node -> node (1 + 1)", lon, lat)
    if unit == "degrees":
        # x · 180/π
        lonlat = np.degrees(lonlat)
    return jnp.asarray(lonlat)


def fibonacci_sphere(
    n: int, *, output: str = "xyz", unit: str | None = None
) -> tuple[Float[Array, "node dim"], Float[Array, " node"]]:
    """The Fibonacci lattice: n near-uniform points on the unit sphere.

    For i = 0, …, n − 1, zᵢ = 1 − 2(i + ½)/n, φᵢ = arcsin zᵢ and
    λᵢ = 4πi/(1 + √5) mod 2π (2πi over the golden ratio), wrapped to [−π, π).
    Each point carries the weight 4π/n.

    Args:
        n: The number of points.
        output: ``"xyz"`` for unit vectors, ``"lonlat"`` for (λ, φ).
        unit: ``"degrees"`` or ``"radians"``; required with
            ``output="lonlat"``, unused with ``"xyz"``.

    Returns:
        ``(points, weights)``: n points with 3 or 2 coordinates, and their n
        weights.

    Raises:
        GeometryError: If ``n`` is not a positive integer, ``output`` is
            unknown, or ``unit`` is missing or unknown with ``"lonlat"``.

    Examples:
        >>> import math
        >>> from functax.geometry import fibonacci_sphere
        >>> points, weights = fibonacci_sphere(100)
        >>> points.shape, math.isclose(float(weights.sum()), 4 * math.pi)
        ((100, 3), True)
        >>> fibonacci_sphere(4, output="lonlat", unit="degrees")[0].shape
        (4, 2)
    """
    count = _check_count(n, "n")
    _check_output(output, unit)
    index = np.arange(count, dtype=float)
    # the golden ratio (1 + √5)/2
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    # zᵢ = 1 − 2(i + ½)/n
    z = 1.0 - 2.0 * (index + 0.5) / count
    # φᵢ = arcsin zᵢ
    lat = np.arcsin(z)
    # λᵢ = 4πi/(1 + √5), wrapped to [−π, π)
    angle = 2.0 * math.pi * index / golden
    wrapped = np.mod(angle + math.pi, 2.0 * math.pi)
    lon = wrapped - math.pi
    # wᵢ = 4π/n
    weights = jnp.full((count,), 4.0 * math.pi / count)
    points = _format(lon, lat, output, unit)
    return points, weights


def gauss_legendre_grid(
    n_lat: int,
    n_lon: int | None = None,
    *,
    output: str = "xyz",
    unit: str | None = None,
) -> tuple[Float[Array, "node dim"], Float[Array, " node"]]:
    """The Gauss–Legendre grid on the sphere, with its quadrature weights.

    The sines of latitude are the n_lat Gauss–Legendre nodes tₖ on [−1, 1],
    with weights wₖ, and the longitudes are uniform, λⱼ = −π + 2πj/n_lon for
    j = 0, …, n_lon − 1; node (k, j) has weight wₖ 2π/n_lon. The rule
    integrates a band-limited function of degree up to
    min(2 n_lat − 1, n_lon − 1) exactly, so Y_ℓm Y_ℓ′m′ when
    ℓ + ℓ′ ≤ 2 n_lat − 1 and n_lon > ℓ + ℓ′.

    Args:
        n_lat: The number of latitudes.
        n_lon: The number of longitudes; ``2 * n_lat`` by default.
        output: ``"xyz"`` for unit vectors, ``"lonlat"`` for (λ, φ).
        unit: ``"degrees"`` or ``"radians"``; required with
            ``output="lonlat"``, unused with ``"xyz"``.

    Returns:
        ``(points, weights)``, n_lat · n_lon of each, latitude-major.

    Raises:
        GeometryError: If ``n_lat`` or ``n_lon`` is not a positive integer,
            ``output`` is unknown, or ``unit`` is missing or unknown with
            ``"lonlat"``.

    Examples:
        >>> import math
        >>> from functax.geometry import gauss_legendre_grid
        >>> points, weights = gauss_legendre_grid(8)
        >>> points.shape, math.isclose(float(weights.sum()), 4 * math.pi)
        ((128, 3), True)
        >>> gauss_legendre_grid(2, 3, output="lonlat", unit="radians")[0].shape
        (6, 2)
    """
    lat_count = _check_count(n_lat, "n_lat")
    lon_count = 2 * lat_count if n_lon is None else _check_count(n_lon, "n_lon")
    _check_output(output, unit)
    nodes, node_weights = gauss_legendre(lat_count)
    # φₖ = arcsin tₖ
    lat_1d = np.arcsin(nodes)
    # λⱼ = −π + 2πj/n_lon
    lon_1d = -math.pi + 2.0 * math.pi * np.arange(lon_count) / lon_count
    # latitude-major: node (k, j) at k · n_lon + j
    lat = einx.id("lat -> (lat lon)", lat_1d, lon=lon_count)
    lon = einx.id("lon -> (lat lon)", lon_1d, lat=lat_count)
    # wₖ 2π/n_lon, the weight of every node of latitude k
    band_weights = node_weights * 2.0 * math.pi / lon_count
    weights = einx.id("lat -> (lat lon)", band_weights, lon=lon_count)
    points = _format(lon, lat, output, unit)
    return points, jnp.asarray(weights)


def latlon_grid(
    n_lat: int,
    n_lon: int,
    *,
    output: str = "xyz",
    unit: str | None = None,
) -> tuple[Float[Array, "node dim"], Float[Array, " node"]]:
    """The regular cell-centred lon/lat grid, weighted by exact cell areas.

    Latitude splits into n_lat and longitude into n_lon equal intervals; each
    point is a cell centre with the cell's area Δλ (sin φ_top − sin φ_bottom)
    as its weight.

    Args:
        n_lat: The number of latitude bands.
        n_lon: The number of longitude bands.
        output: ``"xyz"`` for unit vectors, ``"lonlat"`` for (λ, φ).
        unit: ``"degrees"`` or ``"radians"``; required with
            ``output="lonlat"``, unused with ``"xyz"``.

    Returns:
        ``(points, weights)``, n_lat · n_lon of each, latitude-major.

    Raises:
        GeometryError: If ``n_lat`` or ``n_lon`` is not a positive integer,
            ``output`` is unknown, or ``unit`` is missing or unknown with
            ``"lonlat"``.

    Examples:
        >>> import math
        >>> from functax.geometry import latlon_grid
        >>> points, weights = latlon_grid(18, 36, output="lonlat", unit="degrees")
        >>> points[0].tolist(), math.isclose(float(weights.sum()), 4 * math.pi)
        ([-175.0, -85.0], True)
        >>> latlon_grid(2, 4)[0].shape
        (8, 3)
    """
    lat_count = _check_count(n_lat, "n_lat")
    lon_count = _check_count(n_lon, "n_lon")
    _check_output(output, unit)
    lat_edges = np.linspace(-math.pi / 2, math.pi / 2, lat_count + 1)
    # Δλ = 2π/n_lon
    lon_step = 2.0 * math.pi / lon_count
    # φₖ = (φ_bottom + φ_top)/2, the centre of each band
    lat_1d = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    # λⱼ = −π + Δλ (j + ½)
    lon_1d = -math.pi + lon_step * (np.arange(lon_count) + 0.5)
    # sin φ_top − sin φ_bottom
    sin_top = np.sin(lat_edges[1:])
    sin_bottom = np.sin(lat_edges[:-1])
    band_height = sin_top - sin_bottom
    # the area of each band of cells: Δλ (sin φ_top − sin φ_bottom)
    band_area = lon_step * band_height
    # latitude-major: cell (k, j) at k · n_lon + j
    lat = einx.id("lat -> (lat lon)", lat_1d, lon=lon_count)
    lon = einx.id("lon -> (lat lon)", lon_1d, lat=lat_count)
    weights = einx.id("lat -> (lat lon)", band_area, lon=lon_count)
    points = _format(lon, lat, output, unit)
    return points, jnp.asarray(weights)


# The base-face layout of HEALPix (Górski et al. 2005): for base face q, the ring
# of its southernmost-corner pixel is _JRLL[q]·nside − 1 and its longitude
# offset is _JPLL[q]·π/4.
_JRLL = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
_JPLL = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7])


def _interleave_bits(
    face_x: Int[np.ndarray, " node"], face_y: Int[np.ndarray, " node"], n_bits: int
) -> Int[np.ndarray, " node"]:
    """The Morton code of (x, y): bit b of x goes to bit 2b, of y to 2b + 1."""
    code = np.zeros_like(face_x)
    for bit in range(n_bits):
        x_bit = (face_x >> bit) & 1
        y_bit = (face_y >> bit) & 1
        code |= x_bit << (2 * bit)
        code |= y_bit << (2 * bit + 1)
    return code


def healpix_grid(
    nside: int,
    *,
    nest: bool = True,
    output: str = "xyz",
    unit: str | None = None,
) -> tuple[Float[Array, "node dim"], Float[Array, " node"]]:
    """HEALPix pixel centres, each weighted by the equal pixel area.

    HEALPix splits the sphere into 12 base faces of N_side² pixels each, so
    N = 12 N_side² pixels of weight 4π/N. The centres lie on 4 N_side − 1
    rings of constant latitude: in the polar caps (|z| > 2/3, z = sin φ) ring
    i has z = ±(1 − i²/3N_side²) and 4i pixels; in the equatorial belt
    z = 4/3 − 2i/3N_side with 4 N_side pixels per ring, alternate rings
    shifted by half a pixel. Each centre is computed in closed form from its
    face and the face-local (x, y).

    ``nest=False`` orders the pixels by ring, north to south and eastward from
    λ = 0 within a ring. ``nest=True`` is the nested order, face by face with
    the bits of (x, y) interleaved, so pixel p at N_side holds pixels 4p to
    4p + 3 at 2 N_side; it needs N_side to be a power of two. Both orders match
    ``healpy.pix2ang``. Longitudes are wrapped to [−π, π).

    Algorithm:
        1. Enumerate every pixel by its base face and face-local (x, y).
        2. Find its ring i and the ring's size, and from them z and λ in
           closed form (caps and belt apart).
        3. Number the pixels in the nested or the ring order, and sort by
           that number.

    Args:
        nside: N_side, the resolution.
        nest: The nested order (``True``) or the ring order (``False``).
        output: ``"xyz"`` for unit vectors, ``"lonlat"`` for (λ, φ).
        unit: ``"degrees"`` or ``"radians"``; required with
            ``output="lonlat"``, unused with ``"xyz"``.

    Returns:
        ``(points, weights)``, 12 N_side² of each.

    Raises:
        GeometryError: If ``nside`` is not a positive integer, or not a power
            of two with ``nest=True``, ``output`` is unknown, or ``unit`` is
            missing or unknown with ``"lonlat"``.

    Examples:
        >>> import math
        >>> from functax.geometry import healpix_grid
        >>> points, weights = healpix_grid(1, output="lonlat", unit="degrees")
        >>> points.shape, math.isclose(float(weights.sum()), 4 * math.pi)
        ((12, 2), True)
        >>> [round(v, 4) for v in points[0].tolist()]
        [45.0, 41.8103]

        The nested and ring orders hold the same pixels:

        >>> import numpy as np
        >>> nested, _ = healpix_grid(2)
        >>> ring, _ = healpix_grid(2, nest=False)
        >>> nested_z = np.sort(np.asarray(nested[:, 2]))
        >>> bool(np.allclose(nested_z, np.sort(np.asarray(ring[:, 2]))))
        True

    References:
        Górski, K. M., Hivon, E., Banday, A. J., Wandelt, B. D., Hansen,
        F. K., Reinecke, M., Bartelmann, M. (2005). HEALPix: a framework for
        high-resolution discretization and fast analysis of data distributed
        on the sphere. ApJ 622, 759.
    """
    n = _check_count(nside, "nside")
    _check_output(output, unit)
    if nest and n & (n - 1):
        msg = (
            f"'nside' must be a power of two with nest=True, got {n}; "
            "use nest=False for the ring order"
        )
        raise GeometryError(msg)
    # --- enumerate pixels by face and face-local (x, y) ---
    # N = 12 N_side²
    n_pixels = 12 * n * n
    # the pixels in the north cap, 2 N_side (N_side − 1)
    n_cap = 2 * n * (n - 1)
    face = einx.id("face -> (face x y)", np.arange(12), x=n, y=n)
    face_x = einx.id("x -> (face x y)", np.arange(n), face=12, y=n)
    face_y = einx.id("y -> (face x y)", np.arange(n), face=12, x=n)

    # --- ring and latitude ---
    # the ring number, 1 to 4 N_side − 1
    ring = _JRLL[face] * n - face_x - face_y - 1
    north = ring < n
    south = ring > 3 * n
    # the number of pixels per ring, over 4: i in the caps, N_side in the belt
    south_or_belt_size = np.where(south, 4 * n - ring, n)
    ring_size = np.where(north, ring, south_or_belt_size)
    # z = 1 − i²/(3 N_side²) in the north cap
    z_north = 1.0 - ring_size**2 / (3.0 * n * n)
    # z = i²/(3 N_side²) − 1 in the south cap
    z_south = ring_size**2 / (3.0 * n * n) - 1.0
    # z = (2 N_side − i) · 2/(3 N_side) in the belt
    z_belt = (2 * n - ring) * 2.0 / (3.0 * n)
    z_south_or_belt = np.where(south, z_south, z_belt)
    z = np.where(north, z_north, z_south_or_belt)

    # --- longitude ---
    # alternate belt rings are shifted by half a pixel
    shift = np.where(north | south, 0, (ring - n) & 1)
    # the index within the ring, 1 to 4 × ring_size
    ring_position = (_JPLL[face] * ring_size + face_x - face_y + 1 + shift) // 2
    # wrap the index into 1, …, 4 N_side
    wrapped_low = np.where(ring_position < 1, ring_position + 4 * n, ring_position)
    ring_position = np.where(ring_position > 4 * n, ring_position - 4 * n, wrapped_low)
    # λ in [0, 2π)
    lon = (ring_position - 0.5 * (shift + 1)) * (math.pi / (2.0 * ring_size))

    # --- order ---
    if nest:
        n_bits = max(n.bit_length() - 1, 0)
        morton = _interleave_bits(face_x, face_y, n_bits)
        index = face * n * n + morton
    else:
        # the pixels in the rings north of this one
        before_south = n_pixels - 2 * ring_size * (ring_size + 1)
        before_belt = n_cap + (ring - n) * 4 * n
        before_south_or_belt = np.where(south, before_south, before_belt)
        before_north = 2 * ring_size * (ring_size - 1)
        before = np.where(north, before_north, before_south_or_belt)
        index = before + ring_position - 1
    order = einx.argsort("[node]", index)
    z_sorted = z[order]
    # φ = arcsin z
    lat = np.arcsin(z_sorted)
    # λ wrapped to [−π, π)
    lon_sorted = lon[order]
    lon_wrapped = np.mod(lon_sorted + math.pi, 2.0 * math.pi)
    lon = lon_wrapped - math.pi
    # wₖ = 4π/N
    weights = jnp.full((n_pixels,), 4.0 * math.pi / n_pixels)
    points = _format(lon, lat, output, unit)
    return points, weights
