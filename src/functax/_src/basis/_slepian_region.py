"""Slepian bases for general regions of the sphere, and a spherical polygon test.

The concentration matrix D_ij = ∫_R Y_i Y_j dΩ over the real harmonics up to
l_max has the Slepian functions as eigenvectors and their concentrations
λ_k in [0, 1] as eigenvalues; its trace, the Shannon number (l_max + 1)² A/(4π),
counts the well-concentrated ones. A region comes as a quadrature over it or as
the polar gap between two caps. Moved from geonnax ``_slepian_region`` at
``be5537d``.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import math

import einx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

from functax._src.basis._legendre import (
    _check_degree,
    associated_legendre,
    associated_legendre_indices,
)
from functax._src.basis._slepian_cap import _select_modes, _sh_index
from functax._src.basis._spherical_harmonics import (
    _check_unit_points,
    real_spherical_harmonics,
)
from functax._src.geometry._sphere import lonlat_to_unit
from functax._src.quad._gauss_legendre import gauss_legendre
from functax.exceptions import ValidationError


__all__ = [
    "SlepianBasis",
    "slepian_polar_gap_basis",
    "slepian_region_basis",
    "spherical_polygon_mask",
]


class SlepianBasis(eqx.Module):
    """The Slepian functions of a general region, in the real harmonic basis.

    Each retained function is g_k = ∑_i C_ik Y_i, evaluated in the global
    frame (no rotation, unlike `SlepianCapBasis`).

    Attributes:
        l_max: The highest spherical-harmonic degree.
        coeffs: C, one column per retained mode, sorted by decreasing
            concentration.
        eigenvalues: The concentration λ_k in [0, 1] of each mode.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.slepian import slepian_polar_gap_basis
        >>> basis = slepian_polar_gap_basis(4, 0.3, n_modes=5)
        >>> basis.evaluate(jnp.array([[1.0, 0.0, 0.0]])).shape
        (1, 5)
    """

    l_max: int = eqx.field(static=True)
    coeffs: Float[Array, "harmonic feature"]
    eigenvalues: Float[Array, " feature"]

    @property
    def num_modes(self) -> int:
        """The number of retained Slepian functions.

        Examples:
            >>> from functax.basis.slepian import slepian_polar_gap_basis
            >>> slepian_polar_gap_basis(4, 0.3, n_modes=5).num_modes
            5
        """
        return int(self.eigenvalues.shape[0])

    def evaluate(self, unit_xyz: Float[Array, "obs 3"]) -> Float[Array, "obs feature"]:
        """The retained Slepian functions g_k at points on the unit sphere.

        Examples:
            >>> import jax.numpy as jnp
            >>> from functax.basis.slepian import slepian_polar_gap_basis
            >>> basis = slepian_polar_gap_basis(4, 0.3, n_modes=5)
            >>> basis.evaluate(jnp.array([[0.0, 1.0, 0.0]])).shape
            (1, 5)
        """
        harmonics = real_spherical_harmonics(unit_xyz, self.l_max)
        # g_k = ∑_i C_ik Y_i
        return einx.dot(
            "obs harmonic, harmonic feature -> obs feature", harmonics, self.coeffs
        )


def _finish(
    l_max: int,
    matrix: Float[Array, "row col"],
    n_modes: int | None,
    eig_threshold: float | None,
) -> SlepianBasis:
    if n_modes is not None and n_modes < 1:
        msg = f"'n_modes' must be at least 1, got {n_modes!r}"
        raise ValidationError(msg)
    symmetric = 0.5 * (matrix + einx.id("row col -> col row", matrix))
    values, vectors = jnp.linalg.eigh(symmetric)
    order = jnp.argsort(values)[::-1]
    values, vectors = jnp.clip(values[order], 0.0, 1.0), vectors[:, order]
    if n_modes is None and eig_threshold is None:
        # the Shannon number, tr D
        n_modes = max(1, round(float(jnp.trace(matrix))))
    values, vectors = _select_modes(values, vectors, n_modes, eig_threshold)
    return SlepianBasis(l_max=l_max, coeffs=vectors, eigenvalues=values)


def slepian_region_basis(
    l_max: int,
    region_points_xyz: Float[Array, "node 3"],
    region_weights: Float[Array, " node"],
    *,
    n_modes: int | None = None,
    eig_threshold: float | None = None,
) -> SlepianBasis:
    """The Slepian functions of a region given by a quadrature over it.

    D_ij ≈ ∑_q w_q Y_i(x_q) Y_j(x_q) over the region's quadrature nodes. The
    basis is as accurate as that quadrature: a rule exact to degree 2 l_max on
    the region gives the exact Slepians, while a masked global grid (such as
    `gauss_legendre_grid` filtered by `spherical_polygon_mask`) has an error set
    by the grid spacing at the region's boundary.

    Args:
        l_max: The highest spherical-harmonic degree.
        region_points_xyz: The quadrature nodes inside the region, unit
            vectors.
        region_weights: Their weights, summing to the region's area.
        n_modes: The number of functions to keep; the Shannon number tr D by
            default, when ``eig_threshold`` is unset too.
        eig_threshold: Keep the functions with concentration above this value.

    Returns:
        A `SlepianBasis`.

    Raises:
        ValidationError: If ``l_max`` is not an integer of at least 0, the
            nodes are not (Q, 3), the weights not (Q,), or the mode selection
            is invalid.

    Examples:
        >>> from functax.basis.slepian import slepian_region_basis
        >>> from functax.geometry import gauss_legendre_grid
        >>> points, weights = gauss_legendre_grid(12)
        >>> # a cap of half-angle 60° by masking: Shannon number 36 (1 − cos 60°)/2
        >>> north = points[:, 2] > 0.5
        >>> slepian_region_basis(5, points[north], weights[north]).num_modes
        9
    """
    degree = _check_degree(l_max, "l_max")
    points = jnp.asarray(region_points_xyz)
    weights = jnp.asarray(region_weights)
    _check_unit_points(points, "region_points_xyz")
    if weights.shape != points.shape[:1]:
        expected = (points.shape[0],)
        msg = f"'region_weights' must have shape {expected}, got {weights.shape}"
        raise ValidationError(msg)
    harmonics = real_spherical_harmonics(points, degree)
    weighted = einx.multiply("node harmonic, node", harmonics, weights)
    # D_ij = ∑_q w_q Y_i(x_q) Y_j(x_q)
    matrix = einx.dot("node row, node col -> row col", weighted, harmonics)
    return _finish(degree, matrix, n_modes, eig_threshold)


def slepian_polar_gap_basis(
    l_max: int,
    gap_colatitude: float,
    *,
    n_modes: int | None = None,
    eig_threshold: float | None = None,
    num_quadrature: int | None = None,
) -> SlepianBasis:
    """The Slepian functions of the band between two polar caps (a polar gap).

    The region is |z| < cos Θ with Θ = ``gap_colatitude``: the sphere minus two
    caps of half-angle Θ around the poles, as for satellite data that miss the
    poles. The region is zonal, so D is block-diagonal in m with blocks
    D^m_ll′ = ∫_{−cos Θ}^{cos Θ} P̄_l^m P̄_l′^m dz, computed exactly by
    Gauss–Legendre.

    Args:
        l_max: The highest spherical-harmonic degree.
        gap_colatitude: Θ, the half-angle of each polar cap, in radians in
            (0, π/2).
        n_modes: The number of functions to keep; the Shannon number by
            default.
        eig_threshold: Keep the functions with concentration above this value.
        num_quadrature: The Gauss–Legendre order; l_max + 2 by default, exact
            for the degree-2 l_max integrands.

    Returns:
        A `SlepianBasis`.

    Raises:
        ValidationError: If ``l_max`` is not an integer of at least 0, Θ is
            outside (0, π/2), or the mode selection is invalid.

    Examples:
        >>> from functax.basis.slepian import slepian_polar_gap_basis
        >>> # the Shannon number 49 cos(0.2) ≈ 48
        >>> slepian_polar_gap_basis(6, 0.2).num_modes
        48
    """
    degree = _check_degree(l_max, "l_max")
    if not 0.0 < gap_colatitude < 0.5 * math.pi:
        msg = f"'gap_colatitude' must lie in (0, π/2) radians, got {gap_colatitude!r}"
        raise ValidationError(msg)
    n_quad = num_quadrature or (degree + 2)
    nodes, weights = gauss_legendre(n_quad)
    half = math.cos(gap_colatitude)
    z = jnp.asarray(half * nodes)
    w = jnp.asarray(half * weights)
    p_bar = associated_legendre(z, degree)
    pairs = associated_legendre_indices(degree)
    n_harmonics = (degree + 1) ** 2
    matrix = jnp.zeros((n_harmonics, n_harmonics), dtype=p_bar.dtype)
    for m_abs in range(degree + 1):
        columns = [i for i, (_, order) in enumerate(pairs) if order == m_abs]
        block_p = p_bar[:, jnp.asarray(columns)]
        weighted = einx.multiply("node row, node", block_p, w)
        block = einx.dot("node row, node col -> row col", weighted, block_p)
        for m in {m_abs, -m_abs}:
            index = jnp.asarray([_sh_index(ell, m) for ell in range(m_abs, degree + 1)])
            matrix = matrix.at[jnp.ix_(index, index)].set(block)
    return _finish(degree, matrix, n_modes, eig_threshold)


def spherical_polygon_mask(
    points_xyz: Float[Array, "obs 3"],
    polygon_lonlat: Float[Array, "vertex 2"],
    *,
    unit: str,
) -> Bool[Array, " obs"]:
    """Which points lie inside a spherical polygon.

    The edges are great-circle arcs between consecutive vertices (the ring
    closes itself). A point p is inside when the polygon winds once
    counter-clockwise around it: the signed angles
    atan2((a × b) · p, a · b − (a · p)(b · p)) over the edges (a, b) sum to +2π.
    The vertices must run counter-clockwise seen from outside the sphere
    (interior on the left, as in GeoJSON exterior rings), and the interior
    must fit in an open hemisphere, so a point and its antipode are never both
    inside.

    Args:
        points_xyz: The unit vectors to classify.
        polygon_lonlat: (λ, φ) of each vertex, at least 3.
        unit: ``"degrees"`` or ``"radians"``, the unit of ``polygon_lonlat``.

    Returns:
        Whether each point is inside.

    Raises:
        ValidationError: If ``polygon_lonlat`` is not (P ≥ 3, 2).
        GeometryError: If ``unit`` is unknown.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.slepian import spherical_polygon_mask
        >>> square = jnp.array(
        ...     [[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]]
        ... )
        >>> points = jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        >>> spherical_polygon_mask(points, square, unit="degrees").tolist()
        [True, False]
    """
    polygon = jnp.asarray(polygon_lonlat)
    if polygon.ndim != 2 or polygon.shape[-1] != 2 or polygon.shape[0] < 3:
        msg = f"'polygon_lonlat' must have shape (P ≥ 3, 2), got {polygon.shape}"
        raise ValidationError(msg)
    if not jnp.issubdtype(polygon.dtype, jnp.floating):
        polygon = polygon.astype(jnp.float32)
    # the vertices a and their successors b, as unit vectors
    a = lonlat_to_unit(polygon, unit=unit)
    b = a[np.roll(np.arange(a.shape[0]), -1)]
    p = jnp.asarray(points_xyz)
    numerator = einx.dot("obs dim, edge dim -> obs edge", p, jnp.cross(a, b))
    denominator = einx.subtract(
        "edge, obs edge -> obs edge",
        einx.dot("edge dim, edge dim -> edge", a, b),
        einx.dot("obs dim, edge dim -> obs edge", p, a)
        * einx.dot("obs dim, edge dim -> obs edge", p, b),
    )
    winding = einx.sum("obs [edge]", jnp.arctan2(numerator, denominator))
    return winding > math.pi
