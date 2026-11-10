"""Triangle meshes: the icosphere and the lumped vertex areas of any mesh.

Meshes are static geometry, built and measured on the host with NumPy in
float64 and returned as JAX arrays, as in geonnax ``be5537d``. The checks and
the corner geometry here are shared with the cotangent Laplacian eigenbasis in
`functax.basis.eigen`.
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
from jaxtyping import Array, ArrayLike, Float, Int

from functax.exceptions import GeometryError


__all__ = [
    "icosphere",
    "mesh_vertex_areas",
]


def _icosahedron() -> tuple[np.ndarray, np.ndarray]:
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            [-1, golden, 0], [1, golden, 0], [-1, -golden, 0], [1, -golden, 0],
            [0, -1, golden], [0, 1, golden], [0, -1, -golden], [0, 1, -golden],
            [golden, 0, -1], [golden, 0, 1], [-golden, 0, -1], [-golden, 0, 1],
        ],
        dtype=float,
    )  # fmt: skip
    triangles = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )  # fmt: skip
    return vertices / np.linalg.norm(vertices, axis=1, keepdims=True), triangles


def _subdivide(
    vertices: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split every triangle into four through its shared edge midpoints."""
    points = list(vertices)
    midpoint: dict[tuple[int, int], int] = {}

    def middle(first: Any, second: Any) -> int:
        key = (first, second) if first < second else (second, first)
        if key not in midpoint:
            point = points[first] + points[second]
            points.append(point / np.linalg.norm(point))
            midpoint[key] = len(points) - 1
        return midpoint[key]

    out = []
    for a, b, c in triangles:
        ab, bc, ca = middle(a, b), middle(b, c), middle(c, a)
        out += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
    return np.asarray(points), np.asarray(out, dtype=np.int64)


def icosphere(
    level: int, *, radius: float = 1.0
) -> tuple[Float[Array, "vertex 3"], Int[Array, "triangle 3"]]:
    """The subdivided icosahedron, a quasi-uniform triangle mesh of the sphere.

    From the 12-vertex icosahedron, each level splits every triangle into four
    through its edge midpoints (shared with the neighbours) and projects the
    new vertices onto the sphere. Level ℓ has V = 10 · 4^ℓ + 2 vertices,
    E = 30 · 4^ℓ edges and T = 20 · 4^ℓ triangles, ordered counter-clockwise
    seen from outside so their normals point outward.

    Args:
        level: ℓ, the number of subdivisions.
        radius: The sphere radius.

    Returns:
        ``(vertices, triangles)``: V points and T triples of vertex indices.

    Raises:
        GeometryError: If ``level`` is not an integer of at least 0.

    Examples:
        >>> from functax.geometry import icosphere
        >>> vertices, triangles = icosphere(2)
        >>> vertices.shape, triangles.shape
        ((162, 3), (320, 3))
        >>> icosphere(0)[0].shape
        (12, 3)
    """
    try:
        count = operator.index(level)
    except TypeError as err:
        msg = f"'level' must be an integer of at least 0, got {level!r}"
        raise GeometryError(msg) from err
    if isinstance(level, bool) or count < 0:
        msg = f"'level' must be an integer of at least 0, got {level!r}"
        raise GeometryError(msg)
    vertices, triangles = _icosahedron()
    for _ in range(count):
        vertices, triangles = _subdivide(vertices, triangles)
    return jnp.asarray(radius * vertices), jnp.asarray(triangles)


def _validate_mesh(vertices: Any, triangles: Any) -> tuple[np.ndarray, np.ndarray]:
    """Host copies of a mesh, checked for shape, index range and finiteness."""
    points = np.asarray(vertices, dtype=np.float64)
    corners = np.asarray(triangles)
    if points.ndim != 2 or points.shape[1] != 3:
        msg = f"'vertices' must have shape (V, 3), got {points.shape}"
        raise GeometryError(msg)
    if not np.all(np.isfinite(points)):
        msg = "'vertices' must be finite"
        raise GeometryError(msg)
    if corners.ndim != 2 or corners.shape[1] != 3 or corners.shape[0] == 0:
        msg = f"'triangles' must have shape (T, 3) with T ≥ 1, got {corners.shape}"
        raise GeometryError(msg)
    if not np.issubdtype(corners.dtype, np.integer):
        msg = f"'triangles' must hold integer indices, got {corners.dtype}"
        raise GeometryError(msg)
    n_vertices = points.shape[0]
    if corners.min() < 0 or corners.max() >= n_vertices:
        msg = (
            f"'triangles' indices must lie in [0, {n_vertices - 1}], "
            f"got [{corners.min()}, {corners.max()}]"
        )
        raise GeometryError(msg)
    return points, corners.astype(np.int64)


def _corner_geometry(
    points: np.ndarray, corners: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Twice each triangle's area, and the cotangent at each of its corners.

    ``cot[:, k]`` is the cotangent of the angle at corner k, which is opposite
    the edge (k + 1, k + 2) mod 3.

    Raises:
        GeometryError: If a triangle has zero area.
    """
    corner_points = points[corners]
    # the edges from corner k to k + 1 and to k + 2
    edge_next = np.roll(corner_points, -1, axis=1) - corner_points
    edge_prev = np.roll(corner_points, -2, axis=1) - corner_points
    # 2|t| = ‖e₁ × e₂‖, the same at every corner of a triangle
    double_area = np.linalg.norm(np.cross(edge_next, edge_prev), axis=-1)
    scale = einx.multiply(
        "triangle corner, triangle corner -> triangle corner",
        np.linalg.norm(edge_next, axis=-1),
        np.linalg.norm(edge_prev, axis=-1),
    )
    degenerate = np.any(double_area <= 1e-12 * scale, axis=1)
    if np.any(degenerate):
        bad = np.flatnonzero(degenerate)
        msg = f"{bad.size} triangle(s) have zero area, e.g. index {bad[0]}"
        raise GeometryError(msg)
    # cot θ = (e₁ · e₂) / ‖e₁ × e₂‖
    dot = einx.dot(
        "triangle corner dim, triangle corner dim -> triangle corner",
        edge_next,
        edge_prev,
    )
    cot = einx.divide("triangle corner, triangle corner", dot, double_area)
    return double_area[:, 0], cot


def _lumped_areas(
    n_vertices: int, corners: np.ndarray, double_area: np.ndarray
) -> np.ndarray:
    """a_v = ∑_{t ∋ v} |t| / 3, from twice the triangle areas."""
    areas = np.zeros(n_vertices)
    third = einx.id("triangle -> triangle corner", double_area / 6.0, corner=3)
    np.add.at(areas, corners.ravel(), third.ravel())
    return areas


def mesh_vertex_areas(
    vertices: Float[ArrayLike, "vertex 3"], triangles: Int[ArrayLike, "triangle 3"]
) -> Float[Array, " vertex"]:
    """The lumped (barycentric) vertex areas of a triangle mesh.

    Each triangle gives a third of its area to each corner,
    a_v = ∑_{t ∋ v} |t| / 3 with |t| = ½ ‖(x_q − x_p) × (x_r − x_p)‖, so ∑_v a_v
    is the surface area and M = diag(a) the lumped finite-element mass matrix.
    The areas are quadrature weights too: ∫_S g dA ≈ ∑_v a_v g(x_v). Meyer et
    al. (2003) also describe a mixed Voronoi lumping, which differs at obtuse
    triangles and converges at the same rate.

    The mesh is static geometry, so this runs on the host in float64 and does
    not trace under ``jit``.

    Args:
        vertices: The vertex positions.
        triangles: Vertex indices, three per triangle, in either orientation.

    Returns:
        a, the area of each vertex (zero for a vertex in no triangle).

    Raises:
        GeometryError: If a shape is wrong, an index is out of range, a vertex
            is not finite, or a triangle has zero area.

    References:
        Meyer, M., Desbrun, M., Schröder, P., Barr, A. H. (2003). Discrete
        differential-geometry operators for triangulated 2-manifolds.
        Visualization and Mathematics III, 35–57.

    Examples:
        >>> import math
        >>> from functax.geometry import icosphere, mesh_vertex_areas
        >>> vertices, triangles = icosphere(3)
        >>> areas = mesh_vertex_areas(vertices, triangles)
        >>> areas.shape, round(float(areas.sum()) / (4 * math.pi), 2)
        ((642,), 1.0)
    """
    points, corners = _validate_mesh(vertices, triangles)
    double_area, _ = _corner_geometry(points, corners)
    return jnp.asarray(_lumped_areas(points.shape[0], corners, double_area))
