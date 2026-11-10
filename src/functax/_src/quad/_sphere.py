"""Quadrature rules on the sphere: `SphereRule` and its four constructors.

Each constructor wraps one of the point sets in `functax.geometry` (built on the
host in float64) as a rule for ∫_{S²} g dΩ, with its weights scaled by a² on a
sphere of radius a and the spherical-harmonic degree it integrates exactly.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

from typing import Any

import equinox as eqx

from functax._src.geometry._sphere import _check_radius
from functax._src.geometry._sphere_grids import (
    fibonacci_sphere,
    gauss_legendre_grid,
    healpix_grid,
    latlon_grid,
)
from functax._src.quad._rules import Rule
from functax.exceptions import ValidationError


__all__ = [
    "SphereRule",
    "sphere_fibonacci",
    "sphere_gauss_legendre",
    "sphere_healpix",
    "sphere_latlon",
]


class SphereRule(Rule):
    """A rule on a sphere of radius a: ∫_{S²} g dA ≈ ∑_q w_q g(u_q).

    A `Rule` whose nodes u_q are unit vectors, so integrix sums with it like
    any other rule and ``nodes, weights = rule`` still unpacks. The weights
    sum to 4πa². ``degree`` is the highest L for which every spherical
    harmonic Y_l^m with l ≤ L is integrated exactly; a band-limited g of
    degree at most L is then integrated exactly too.

    Attributes:
        nodes: The unit vectors u_q, one row (x, y, z) per node.
        weights: The weights w_q, in units of a².
        radius: a, the radius of the sphere the weights are scaled to.
        degree: L, the exactness degree (0 when only constants are exact).

    Raises:
        ValidationError: As `Rule`, or if ``nodes`` is not (Q, 3) or
            ``degree`` is not an integer of at least 0.

    Examples:
        >>> import math
        >>> from functax.quad import sphere_gauss_legendre
        >>> rule = sphere_gauss_legendre(4)
        >>> rule.nodes.shape, rule.degree
        ((32, 3), 7)
        >>> nodes, weights = rule
        >>> math.isclose(float(weights.sum()), 4 * math.pi)
        True
    """

    radius: float = eqx.field(static=True, kw_only=True)
    degree: int = eqx.field(static=True, kw_only=True)

    def __check_init__(self) -> None:
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            msg = (
                f"'nodes' must be (Q, 3) unit vectors on the sphere, "
                f"got shape {self.nodes.shape}"
            )
            raise ValidationError(msg)
        if isinstance(self.degree, bool) or not isinstance(self.degree, int):
            msg = f"'degree' must be an integer of at least 0, got {self.degree!r}"
            raise ValidationError(msg)
        if self.degree < 0:
            msg = f"'degree' must be an integer of at least 0, got {self.degree!r}"
            raise ValidationError(msg)


def sphere_gauss_legendre(
    n_lat: int, n_lon: int | None = None, *, radius: float = 1.0
) -> SphereRule:
    """The Gauss–Legendre × uniform-longitude rule on the sphere, exact to a degree.

    The sines of latitude are the n_lat Gauss–Legendre nodes tₖ on [−1, 1]
    (ascending, south to north) and the longitudes are uniform from −π,
    λⱼ = −π + 2πj/n_lon; node (k, j) has weight a² ωₖ 2π/n_lon. The rule
    integrates every Y_l^m with l ≤ min(2 n_lat − 1, n_lon − 1) exactly: the
    longitude sum kills every m ≠ 0 below n_lon, and Gauss–Legendre is exact
    for P_l(t) up to degree 2 n_lat − 1. Nodes are latitude-major, as in
    `gauss_legendre_grid`. spectraldiffx orders its nodes cos θ from north
    to south and its longitudes from 0, so its latitudes are these reversed.

    Algorithm:
        1. The grid and its weights on the unit sphere (`gauss_legendre_grid`).
        2. Scale the weights by a².

    Args:
        n_lat: The number of latitudes.
        n_lon: The number of longitudes; ``2 * n_lat`` by default.
        radius: a, the radius of the sphere, positive.

    Returns:
        A `SphereRule` of n_lat · n_lon nodes, ``degree`` min(2 n_lat − 1,
        n_lon − 1).

    Raises:
        GeometryError: If ``n_lat`` or ``n_lon`` is not a positive integer,
            or ``radius`` is not positive and finite.

    Examples:
        >>> import math
        >>> from functax.quad import sphere_gauss_legendre
        >>> rule = sphere_gauss_legendre(8, 9)
        >>> rule.nodes.shape, rule.degree
        ((72, 3), 8)
        >>> # the surface of a sphere of radius 2: 16π
        >>> earth_like = sphere_gauss_legendre(4, radius=2.0)
        >>> math.isclose(float(earth_like.weights.sum()), 16 * math.pi)
        True
    """
    points, weights = gauss_legendre_grid(n_lat, n_lon)
    lon_count = 2 * n_lat if n_lon is None else n_lon
    degree = min(2 * n_lat - 1, lon_count - 1)
    return _sphere_rule(points, weights, radius=radius, degree=degree)


def sphere_latlon(n_lat: int, n_lon: int, *, radius: float = 1.0) -> SphereRule:
    """The cell-centred lon/lat rule: equiangular cells weighted by their exact areas.

    Latitude splits into n_lat and longitude into n_lon equal intervals from
    −π; each node is a cell centre with the cell's area a² Δλ (sin φ_top −
    sin φ_bottom) as its weight, as in `latlon_grid`. A field constant on
    each cell is integrated exactly, but the rule is exact only to degree
    min(1, n_lon − 1): the grid's symmetry about the equator kills Y_1^0,
    and the midpoint rule in latitude misses P_2. It converges at second
    order in the cell size.

    Algorithm:
        1. The cell centres and areas on the unit sphere (`latlon_grid`).
        2. Scale the areas by a².

    Args:
        n_lat: The number of latitude bands.
        n_lon: The number of longitude bands.
        radius: a, the radius of the sphere, positive.

    Returns:
        A `SphereRule` of n_lat · n_lon nodes, latitude-major, ``degree``
        min(1, n_lon − 1).

    Raises:
        GeometryError: If ``n_lat`` or ``n_lon`` is not a positive integer,
            or ``radius`` is not positive and finite.

    Examples:
        >>> import math
        >>> from functax.quad import sphere_latlon
        >>> rule = sphere_latlon(18, 36)
        >>> rule.nodes.shape, rule.degree
        ((648, 3), 1)
        >>> math.isclose(float(rule.weights.sum()), 4 * math.pi)
        True
    """
    points, weights = latlon_grid(n_lat, n_lon)
    degree = min(1, n_lon - 1)
    return _sphere_rule(points, weights, radius=radius, degree=degree)


def sphere_healpix(nside: int, *, nest: bool = True, radius: float = 1.0) -> SphereRule:
    """The HEALPix rule: 12 N_side² pixel centres of equal weight 4πa²/Q.

    The pixels are equal-area, so every weight is the pixel area, not a
    quadrature weight fitted to the harmonics: the rule is not exact for any
    degree l ≥ 2, and its error falls with the pixel size. Its symmetry
    (north–south and a quarter turn in longitude) makes every Y_1^m sum to
    zero, so ``degree`` is 1. Pixel order and longitudes as in
    `healpix_grid`.

    Algorithm:
        1. The pixel centres and areas on the unit sphere (`healpix_grid`).
        2. Scale the areas by a².

    Args:
        nside: N_side, the resolution.
        nest: The nested order (``True``, N_side a power of two) or the ring
            order (``False``).
        radius: a, the radius of the sphere, positive.

    Returns:
        A `SphereRule` of 12 N_side² nodes, ``degree`` 1.

    Raises:
        GeometryError: If ``nside`` is not a positive integer (a power of two
            with ``nest=True``), or ``radius`` is not positive and finite.

    Examples:
        >>> import math
        >>> from functax.quad import sphere_healpix
        >>> rule = sphere_healpix(2)
        >>> rule.nodes.shape, rule.degree
        ((48, 3), 1)
        >>> math.isclose(float(rule.weights.sum()), 4 * math.pi)
        True
    """
    points, weights = healpix_grid(nside, nest=nest)
    return _sphere_rule(points, weights, radius=radius, degree=1)


def sphere_fibonacci(n: int, *, radius: float = 1.0) -> SphereRule:
    """The Fibonacci rule: n near-uniform points of equal weight 4πa²/n.

    Each point stands for an equal share of the area, so the rule is not
    exact for any degree beyond constants (``degree`` 0): it is a
    quasi-Monte Carlo rule whose error falls with n. Points as in
    `fibonacci_sphere`.

    Algorithm:
        1. The lattice and its equal weights on the unit sphere
           (`fibonacci_sphere`).
        2. Scale the weights by a².

    Args:
        n: The number of points.
        radius: a, the radius of the sphere, positive.

    Returns:
        A `SphereRule` of n nodes, ``degree`` 0.

    Raises:
        GeometryError: If ``n`` is not a positive integer, or ``radius`` is
            not positive and finite.

    Examples:
        >>> import math
        >>> from functax.quad import sphere_fibonacci
        >>> rule = sphere_fibonacci(100, radius=3.0)
        >>> rule.nodes.shape, rule.degree
        ((100, 3), 0)
        >>> math.isclose(float(rule.weights.sum()), 36 * math.pi)
        True
    """
    points, weights = fibonacci_sphere(n)
    return _sphere_rule(points, weights, radius=radius, degree=0)


def _sphere_rule(points: Any, weights: Any, *, radius: Any, degree: int) -> SphereRule:
    """A `SphereRule` with the unit-sphere weights scaled by a²."""
    _check_radius(radius)
    # a static field, so a concrete Python float
    radius = float(radius)
    # w_q a², the area element of a sphere of radius a
    scaled = weights * radius**2
    return SphereRule(points, scaled, radius=radius, degree=int(degree))
