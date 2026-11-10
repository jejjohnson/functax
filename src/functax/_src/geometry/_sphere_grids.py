"""Point sets on the unit sphere with quadrature weights for ∫_{S²} f dΩ.

Each grid returns ``(points, weights)`` with weights summing to 4π: the
Fibonacci lattice, the Gauss–Legendre grid, the cell-centred lon/lat grid and
the HEALPix pixel centres. Points are unit vectors (``output="xyz"``) or
(λ, φ) pairs (``output="lonlat"``, which needs ``unit``). The grids are built on
the host with NumPy, as in geonnax ``be5537d``, and returned as JAX arrays.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import math
import operator
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from functax._src.geometry._sphere import _check_unit
from functax._src.quad._gauss_legendre import gauss_legendre
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
    lon: np.ndarray, lat: np.ndarray, output: str, unit: str | None
) -> Float[Array, "node dim"]:
    if output == "xyz":
        # u = (cos φ cos λ, cos φ sin λ, sin φ)
        cos_lat = np.cos(lat)
        xyz = np.stack(
            [cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)], axis=-1
        )
        return jnp.asarray(xyz)
    lonlat = np.stack([lon, lat], axis=-1)
    if unit == "degrees":
        lonlat = np.degrees(lonlat)
    return jnp.asarray(lonlat)


def fibonacci_sphere(
    n: int, *, output: str = "xyz", unit: str | None = None
) -> tuple[Float[Array, "node dim"], Float[Array, " node"]]:
    """The Fibonacci lattice: n near-uniform points on the unit sphere.

    zᵢ = 1 − 2(i + ½)/n, φᵢ = arcsin zᵢ and λᵢ = 2πi/ϕ mod 2π with ϕ the golden
    ratio, wrapped to [−π, π). Each point carries the weight 4π/n.

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
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    lat = np.arcsin(1.0 - 2.0 * (index + 0.5) / count)
    lon = np.mod(2.0 * math.pi * index / golden + math.pi, 2.0 * math.pi) - math.pi
    weights = jnp.full((count,), 4.0 * math.pi / count)
    return _format(lon, lat, output, unit), weights


def gauss_legendre_grid(
    n_lat: int,
    n_lon: int | None = None,
    *,
    output: str = "xyz",
    unit: str | None = None,
) -> tuple[Float[Array, "node dim"], Float[Array, " node"]]:
    """The Gauss–Legendre grid on the sphere, with its quadrature weights.

    The sines of latitude are the n_lat Gauss–Legendre nodes tₖ on [−1, 1] and
    the longitudes are uniform, λⱼ = −π + 2πj/n_lon; node (k, j) has weight
    wₖ 2π/n_lon. The rule integrates a band-limited function of degree up to
    min(2 n_lat − 1, n_lon − 1) exactly, so Y_lm Y_l′m′ when
    l + l′ ≤ 2 n_lat − 1 and n_lon > l + l′.

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
    lat_1d = np.arcsin(nodes)
    lon_1d = -math.pi + 2.0 * math.pi * np.arange(lon_count) / lon_count
    lat, lon = np.meshgrid(lat_1d, lon_1d, indexing="ij")
    weights = np.repeat(node_weights * 2.0 * math.pi / lon_count, lon_count)
    points = _format(lon.ravel(), lat.ravel(), output, unit)
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
    lon_step = 2.0 * math.pi / lon_count
    lat_1d = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_1d = -math.pi + lon_step * (np.arange(lon_count) + 0.5)
    # the area of each band of cells: Δλ (sin φ_top − sin φ_bottom)
    band_area = lon_step * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    lat, lon = np.meshgrid(lat_1d, lon_1d, indexing="ij")
    weights = np.repeat(band_area, lon_count)
    points = _format(lon.ravel(), lat.ravel(), output, unit)
    return points, jnp.asarray(weights)


# The base-face layout of HEALPix (Górski et al. 2005): for face f, the ring of
# its southernmost-corner pixel is _JRLL[f]·nside − 1 and its longitude offset
# is _JPLL[f]·π/4.
_JRLL = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
_JPLL = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7])


def _interleave_bits(ix: np.ndarray, iy: np.ndarray, n_bits: int) -> np.ndarray:
    out = np.zeros_like(ix)
    for bit in range(n_bits):
        out |= ((ix >> bit) & 1) << (2 * bit)
        out |= ((iy >> bit) & 1) << (2 * bit + 1)
    return out


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

    References:
        Górski, K. M., Hivon, E., Banday, A. J., Wandelt, B. D., Hansen,
        F. K., Reinecke, M., Bartelmann, M. (2005). HEALPix: a framework for
        high-resolution discretization and fast analysis of data distributed
        on the sphere. ApJ 622, 759.

    Examples:
        >>> import math
        >>> from functax.geometry import healpix_grid
        >>> points, weights = healpix_grid(1, output="lonlat", unit="degrees")
        >>> points.shape, math.isclose(float(weights.sum()), 4 * math.pi)
        ((12, 2), True)
        >>> [round(v, 4) for v in points[0].tolist()]
        [45.0, 41.8103]
    """
    n = _check_count(nside, "nside")
    _check_output(output, unit)
    if nest and n & (n - 1):
        msg = (
            f"'nside' must be a power of two with nest=True, got {n}; "
            "use nest=False for the ring order"
        )
        raise GeometryError(msg)
    npix = 12 * n * n
    ncap = 2 * n * (n - 1)
    face, ix, iy = (
        a.ravel()
        for a in np.meshgrid(np.arange(12), np.arange(n), np.arange(n), indexing="ij")
    )
    # the ring number, 1 to 4n − 1
    jr = _JRLL[face] * n - ix - iy - 1
    north, south = jr < n, jr > 3 * n
    # the number of pixels per ring, over 4
    nr = np.where(north, jr, np.where(south, 4 * n - jr, n))
    z = np.where(
        north,
        1.0 - nr**2 / (3.0 * n * n),
        np.where(south, nr**2 / (3.0 * n * n) - 1.0, (2 * n - jr) * 2.0 / (3.0 * n)),
    )
    kshift = np.where(north | south, 0, (jr - n) & 1)
    # the index within the ring, 1 to 4 nr
    jp = (_JPLL[face] * nr + ix - iy + 1 + kshift) // 2
    jp = np.where(jp > 4 * n, jp - 4 * n, np.where(jp < 1, jp + 4 * n, jp))
    # λ in [0, 2π)
    lon = (jp - 0.5 * (kshift + 1)) * (math.pi / (2.0 * nr))
    if nest:
        index = face * n * n + _interleave_bits(ix, iy, max(n.bit_length() - 1, 0))
    else:
        before = np.where(
            north,
            2 * nr * (nr - 1),
            np.where(south, npix - 2 * nr * (nr + 1), ncap + (jr - n) * 4 * n),
        )
        index = before + jp - 1
    order = np.argsort(index)
    lat = np.arcsin(z[order])
    lon = np.mod(lon[order] + math.pi, 2.0 * math.pi) - math.pi
    weights = jnp.full((npix,), 4.0 * math.pi / npix)
    return _format(lon, lat, output, unit), weights
