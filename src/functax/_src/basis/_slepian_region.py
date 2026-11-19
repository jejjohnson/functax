"""Slepian bases for general regions of the sphere, and a spherical polygon test.

The concentration matrix D_ij = ∫_R Y_i Y_j dΩ of a region R ⊂ S² of area A,
over the real harmonics Y_i, i = 1, …, (ℓ_max + 1)², up to degree ℓ_max (dΩ the
area element of the unit sphere), has the harmonic coefficients of the Slepian
functions as eigenvectors and their concentrations λ_k in [0, 1] as
eigenvalues; its trace, the Shannon number (ℓ_max + 1)² A/(4π), counts the
well-concentrated ones. On this page λ is always a concentration, so longitude
and latitude are written out in words. A region comes as a quadrature over it
or as the polar gap between two caps. Moved from geonnax ``_slepian_region`` at
``be5537d``.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import math

import einx
import equinox as eqx
import jax.numpy as jnp
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
from functax._src.quad._rules import gauss_legendre
from functax.exceptions import ValidationError


__all__ = [
    "SlepianBasis",
    "slepian_polar_gap_basis",
    "slepian_region_basis",
    "spherical_polygon_mask",
]


class SlepianBasis(eqx.Module):
    """The Slepian functions of a general region, in the real harmonic basis.

    Each retained function is g_k = ∑_i C_ik Y_i, for k = 1, …, M with M the
    number of retained modes, evaluated in the global frame (no rotation,
    unlike `SlepianCapBasis`).

    Attributes:
        l_max: ℓ_max, the highest spherical-harmonic degree.
        coeffs: C ∈ ℝ^{(ℓ_max+1)²×M}, one column per retained mode, sorted by
            decreasing concentration.
        eigenvalues: The concentration λ_k in [0, 1] of each mode.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.slepian import slepian_polar_gap_basis
        >>> basis = slepian_polar_gap_basis(4, 0.3, n_modes=5)
        >>> basis.evaluate(jnp.array([[1.0, 0.0, 0.0]])).shape
        (1, 5)
        >>> bool((basis.eigenvalues[:-1] >= basis.eigenvalues[1:]).all())
        True
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
            >>> # orthonormal coefficients, so orthonormal functions on the sphere
            >>> bool(jnp.allclose(basis.coeffs.T @ basis.coeffs, jnp.eye(5)))
            True
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
    """Diagonalise D, sort by decreasing concentration and select the modes."""
    if n_modes is not None and n_modes < 1:
        msg = f"'n_modes' must be at least 1, got {n_modes!r}"
        raise ValidationError(msg)
    transposed = einx.id("row col -> col row", matrix)
    # D ← (D + Dᵀ)/2, symmetric against round-off
    symmetric = 0.5 * (matrix + transposed)
    # D c = λ c
    values, vectors = jnp.linalg.eigh(symmetric)
    # descending λ
    ascending = jnp.argsort(values)
    order = ascending[::-1]
    sorted_values = values[order]
    # λ ∈ [0, 1], clipped against round-off
    values = jnp.clip(sorted_values, 0.0, 1.0)
    vectors = vectors[:, order]
    if n_modes is None and eig_threshold is None:
        # the Shannon number, tr D
        trace = float(jnp.trace(matrix))
        n_modes = max(1, round(trace))
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

    D_ij ≈ ∑_q w_q Y_i(r̂_q) Y_j(r̂_q) over the region's quadrature nodes r̂_q
    ∈ S² and weights w_q, q = 1, …, Q. The basis is as accurate as that
    quadrature: a rule exact to degree 2 ℓ_max on the region gives the exact
    Slepians, while a masked global grid (such as `gauss_legendre_grid`
    filtered by `spherical_polygon_mask`) has an error set by the grid spacing
    at the region's boundary.

    Algorithm:
        1. Evaluate the real harmonics at the region's quadrature nodes.
        2. Form D = Yᵀ diag(w) Y by quadrature, with Y ∈ ℝ^{Q×(ℓ_max+1)²}
           the harmonics at the nodes, entry (q, i) being Y_i(r̂_q).
        3. Diagonalise D, sort by decreasing concentration, and keep the
           selected modes.

    Args:
        l_max: ℓ_max, the highest spherical-harmonic degree.
        region_points_xyz: The quadrature nodes r̂_q inside the region, unit
            vectors.
        region_weights: Their weights w_q, summing to the region's area.
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

        The whole sphere concentrates every harmonic fully:

        >>> import jax.numpy as jnp
        >>> whole = slepian_region_basis(3, points, weights, n_modes=16)
        >>> bool(jnp.allclose(whole.eigenvalues, 1.0))
        True
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
    # w_q Y_i(r̂_q)
    weighted = einx.multiply("node harmonic, node", harmonics, weights)
    # D_ij = ∑_q w_q Y_i(r̂_q) Y_j(r̂_q)
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

    The region is |z| < cos Θ, with z = cos θ for the colatitude θ and
    Θ = ``gap_colatitude``: the sphere minus two caps of half-angle Θ around
    the poles, as for satellite data that miss the poles. The region is zonal,
    so D is block-diagonal in the order m with blocks
    D^{(m)}_ℓℓ′ = ∫_{−cos Θ}^{cos Θ} P̄_ℓᵐ P̄_ℓ′ᵐ dz, computed exactly by
    Gauss–Legendre.

    Algorithm:
        1. Lay a Gauss–Legendre rule on z over [−cos Θ, cos Θ].
        2. For each |m|, integrate the products of the P̄_ℓᵐ by quadrature,
           and place the block at both orders ±m.
        3. Diagonalise D, sort by decreasing concentration, and keep the
           selected modes.

    Args:
        l_max: ℓ_max, the highest spherical-harmonic degree.
        gap_colatitude: Θ, the half-angle of each polar cap, in radians in
            (0, π/2).
        n_modes: The number of functions to keep; the Shannon number by
            default.
        eig_threshold: Keep the functions with concentration above this value.
        num_quadrature: The Gauss–Legendre order; ℓ_max + 2 by default, exact
            for the integrands of degree 2ℓ_max.

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
        >>> # a threshold keeps the well-concentrated ones only
        >>> basis = slepian_polar_gap_basis(6, 0.2, eig_threshold=0.5)
        >>> bool((basis.eigenvalues > 0.5).all())
        True
    """
    degree = _check_degree(l_max, "l_max")
    if not 0.0 < gap_colatitude < 0.5 * math.pi:
        msg = f"'gap_colatitude' must lie in (0, π/2) radians, got {gap_colatitude!r}"
        raise ValidationError(msg)
    n_quad = num_quadrature or (degree + 2)
    nodes, weights = gauss_legendre(n_quad)
    # cos Θ, the half-width of the band in z
    half = math.cos(gap_colatitude)
    # the nodes and weights scaled onto [−cos Θ, cos Θ]
    z = jnp.asarray(half * nodes)
    w = jnp.asarray(half * weights)
    p_bar = associated_legendre(z, degree)
    pairs = associated_legendre_indices(degree)
    n_harmonics = (degree + 1) ** 2
    matrix = jnp.zeros((n_harmonics, n_harmonics), dtype=p_bar.dtype)
    for m_abs in range(degree + 1):
        columns = [i for i, (_, order) in enumerate(pairs) if order == m_abs]
        block_p = p_bar[:, jnp.asarray(columns)]
        # w_q P̄_ℓᵐ(z_q)
        weighted = einx.multiply("node row, node", block_p, w)
        # D^{(m)}_ℓℓ′ = ∑_q w_q P̄_ℓᵐ(z_q) P̄_ℓ′ᵐ(z_q)
        block = einx.dot("node row, node col -> row col", weighted, block_p)
        for m in {m_abs, -m_abs}:
            index = jnp.asarray([_sh_index(ell, m) for ell in range(m_abs, degree + 1)])
            rows_and_cols = jnp.ix_(index, index)
            matrix = matrix.at[rows_and_cols].set(block)
    return _finish(degree, matrix, n_modes, eig_threshold)


def spherical_polygon_mask(
    points_xyz: Float[Array, "obs 3"],
    polygon_lonlat: Float[Array, "vertex 2"],
    *,
    unit: str,
) -> Bool[Array, " obs"]:
    """Which points lie inside a spherical polygon.

    The edges are great-circle arcs between consecutive vertices (the ring
    closes itself). A point r̂ ∈ S² is inside when the polygon winds once
    counter-clockwise around it: the signed angles
    atan2((a × b) · r̂, a · b − (a · r̂)(b · r̂)) over the edges, from vertex a to
    its successor b as unit vectors, sum to +2π.
    The vertices must run counter-clockwise seen from outside the sphere
    (interior on the left, as in GeoJSON exterior rings), and the interior
    must fit in an open hemisphere, so a point and its antipode are never both
    inside.

    Args:
        points_xyz: The unit vectors r̂ to classify.
        polygon_lonlat: The longitude and latitude of each vertex, at least
            3 vertices.
        unit: ``"degrees"`` or ``"radians"``, the unit of ``polygon_lonlat``.

    Returns:
        Whether each point is inside.

    Raises:
        ValidationError: If ``polygon_lonlat`` is not one (longitude,
            latitude) row per vertex, with at least 3 vertices.
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
        >>> # the same square in radians; a quarter turn east is outside
        >>> others = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> radians = jnp.deg2rad(square)
        >>> spherical_polygon_mask(others, radians, unit="radians").tolist()
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
    b = einx.roll("[edge] dim", a, shift=-1)
    p = jnp.asarray(points_xyz)
    # --- the signed angle each edge subtends at r̂ ---
    normal = jnp.cross(a, b)
    # (a × b) · r̂
    numerator = einx.dot("obs dim, edge dim -> obs edge", p, normal)
    # a · b
    a_dot_b = einx.dot("edge dim, edge dim -> edge", a, b)
    # a · r̂
    a_dot_p = einx.dot("obs dim, edge dim -> obs edge", p, a)
    # b · r̂
    b_dot_p = einx.dot("obs dim, edge dim -> obs edge", p, b)
    # (a · r̂)(b · r̂)
    projection_product = a_dot_p * b_dot_p
    # a · b − (a · r̂)(b · r̂)
    denominator = einx.subtract(
        "edge, obs edge -> obs edge", a_dot_b, projection_product
    )
    # --- the winding number ---
    angles = jnp.arctan2(numerator, denominator)
    # ∑ over the edges: 2π inside, 0 outside
    winding = einx.sum("obs [edge]", angles)
    return winding > math.pi
