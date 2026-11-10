"""Host-side polygon clipping: the overlap areas of quadrilateral cells.

The cells are planar or spherical.

Cell pairs that may overlap come from a k-d tree on the cells' bounding
circles (SciPy's ``cKDTree``, the ``functax[scipy]`` extra); each pair is
clipped by Sutherland–Hodgman, vectorised over pairs in NumPy, with the
target cell as the convex clip polygon. Planar cells are clipped in their
coordinates, shifted to the target cell's centre. Spherical cells have
great-circle edges and are clipped in the gnomonic tangent plane of the
target cell, where great circles are straight lines, so the planar clip is
exact; the clipped polygon goes back to the sphere and its area is the
spherical excess (`spherical_polygon_area`).
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import dataclasses

import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, Int

from functax._src._backend import import_scipy
from functax._src.geometry._sphere import spherical_polygon_area
from functax.exceptions import GeometryError


__all__ = [
    "CellOverlaps",
    "cell_overlaps",
    "clip_polygons",
    "planar_polygon_area",
]


# pairs clipped at once, so the (pair, vertex, 2) intermediates stay small
_CHUNK = 2**15

# a turn this small (relative to the cell's size squared) counts as straight
_CONVEX_RTOL = 1e-12


@dataclasses.dataclass(frozen=True)
class CellOverlaps:
    """The overlap areas |cⱼ ∩ Cᵢ| > 0 of source and target cells, as pairs.

    Attributes:
        target: i, the target cell of each pair.
        source: j, the source cell of each pair.
        areas: |cⱼ ∩ Cᵢ|, positive.
        target_areas: |Cᵢ| of every target cell, (M,).
    """

    target: Int[np.ndarray, " pair"]
    source: Int[np.ndarray, " pair"]
    areas: Float[np.ndarray, " pair"]
    target_areas: Float[np.ndarray, " target"]


def clip_polygons(
    subjects: Float[np.ndarray, "pair vertex 2"],
    counts: Int[np.ndarray, " pair"],
    clips: Float[np.ndarray, "pair corner 2"],
) -> tuple[Float[np.ndarray, "pair out 2"], Int[np.ndarray, " pair"]]:
    """Sutherland–Hodgman: each subject polygon cut to its convex clip polygon.

    Each clip polygon must be convex and counter-clockwise, so its inside is
    on the left of every edge, d(p) = (q₁ − q₀) × (p − q₀) ≥ 0. For each clip
    edge in turn, a subject edge from p to p′ keeps p′ when it is inside and
    adds the crossing p + t (p′ − p), t = d(p) / (d(p) − d(p′)), when the two
    ends lie on opposite sides. Subjects may be any simple polygon; a zero-length
    clip edge cuts nothing.

    Args:
        subjects: The subject vertices, padded past ``counts``.
        counts: The number of vertices of each subject.
        clips: The vertices of each clip polygon.

    Returns:
        The clipped vertices, padded, and their counts (0 where nothing is left).
    """
    points = subjects
    n_corners = clips.shape[1]
    for corner in range(n_corners):
        start = clips[:, corner]
        edge = clips[:, (corner + 1) % n_corners] - start
        width = points.shape[1]
        vertex = np.arange(width)
        valid = vertex < counts[:, None]
        # the previous vertex, cyclically within each polygon's own count
        safe_counts = np.maximum(counts, 1)
        previous_index = (vertex[None, :] - 1) % safe_counts[:, None]
        previous = np.take_along_axis(points, previous_index[..., None], axis=1)
        # d(p) = (q₁ − q₀) × (p − q₀)
        offset = points - start[:, None, :]
        side = edge[:, None, 0] * offset[..., 1] - edge[:, None, 1] * offset[..., 0]
        previous_side = np.take_along_axis(side, previous_index, axis=1)
        inside = side >= 0
        previous_inside = previous_side >= 0
        crosses = valid & (inside != previous_inside)
        # t = d(p) / (d(p) − d(p′)), the two of opposite sign where it crosses
        denominator = np.where(crosses, previous_side - side, 1.0)
        fraction = np.where(crosses, previous_side / denominator, 0.0)
        crossing = previous + fraction[..., None] * (points - previous)
        # two slots per vertex, the crossing first, then the vertex if kept
        slots = np.stack([crossing, points], axis=2).reshape(len(points), -1, 2)
        kept = np.stack([crosses, valid & inside], axis=2).reshape(len(points), -1)
        order = np.argsort(~kept, axis=1, kind="stable")
        points = np.take_along_axis(slots, order[..., None], axis=1)
        counts = kept.sum(axis=1)
        points = points[:, : max(int(counts.max(initial=0)), 1)]
    return points, counts


def planar_polygon_area(
    points: Float[np.ndarray, "pair vertex 2"], counts: Int[np.ndarray, " pair"]
) -> Float[np.ndarray, " pair"]:
    """The signed shoelace area ½ ∑ₖ (xₖ yₖ₊₁ − xₖ₊₁ yₖ) of padded polygons."""
    vertex = np.arange(points.shape[1])
    safe_counts = np.maximum(counts, 1)
    following_index = (vertex[None, :] + 1) % safe_counts[:, None]
    following = np.take_along_axis(points, following_index[..., None], axis=1)
    # xₖ yₖ₊₁ − xₖ₊₁ yₖ on the polygon's own edges only
    terms = points[..., 0] * following[..., 1] - following[..., 0] * points[..., 1]
    terms = np.where(vertex[None, :] < counts[:, None], terms, 0.0)
    return 0.5 * terms.sum(axis=1)


def _counter_clockwise(polygons: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """Reverse the corner order of each polygon with a negative signed area."""
    reversed_order = polygons[:, ::-1]
    return np.where((areas < 0)[:, None, None], reversed_order, polygons)


def _check_convex(polygons: np.ndarray, name: str) -> None:
    """Raise unless each counter-clockwise polygon never turns right."""
    following = np.roll(polygons, -1, axis=1)
    after = np.roll(polygons, -2, axis=1)
    first = following - polygons
    second = after - following
    turn = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    scale = np.max(np.abs(first), axis=(1, 2)) ** 2
    concave = np.any(turn < -_CONVEX_RTOL * scale[:, None], axis=1)
    if np.any(concave):
        cell = int(np.flatnonzero(concave)[0])
        msg = f"'{name}' cells must be convex to clip against, but cell {cell} is not"
        raise GeometryError(msg)


def _tangent_frames(
    centres: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors e₁, e₂ spanning the tangent plane at each centre n̂."""
    # e₁ ∥ ẑ × n̂, or x̂ × n̂ near the poles
    pole = np.abs(centres[:, 2]) > 0.9
    reference = np.where(pole[:, None], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    first = np.cross(reference, centres)
    first = first / np.linalg.norm(first, axis=1, keepdims=True)
    second = np.cross(centres, first)
    return first, second


def _gnomonic(
    points: np.ndarray,
    centre: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """(p · e₁, p · e₂) / (p · n̂), the gnomonic projection about n̂."""
    height = np.einsum("pvc,pc->pv", points, centre)
    if np.any(height <= 0):
        msg = (
            "cells must each lie within a hemisphere of the target cells they "
            "overlap, but a pair spans more"
        )
        raise GeometryError(msg)
    along_first = np.einsum("pvc,pc->pv", points, first) / height
    along_second = np.einsum("pvc,pc->pv", points, second) / height
    return np.stack([along_first, along_second], axis=-1)


def _spherical_areas(
    polygons: np.ndarray, counts: np.ndarray | None = None
) -> np.ndarray:
    """|A| of padded unit-vector polygons, zero where fewer than three vertices."""
    if counts is not None:
        # pad by repeating the last vertex, which adds zero-area triangles
        vertex = np.arange(polygons.shape[1])
        last = np.maximum(counts - 1, 0)
        index = np.minimum(vertex[None, :], last[:, None])
        polygons = np.take_along_axis(polygons, index[..., None], axis=1)
    if polygons.shape[1] < 3:
        return np.zeros(polygons.shape[0], polygons.dtype)
    areas = np.abs(np.asarray(spherical_polygon_area(jnp.asarray(polygons))))
    if counts is not None:
        areas = np.where(counts >= 3, areas, 0.0)
    return areas


def _bounding_circles(
    polygons: np.ndarray, sphere: bool
) -> tuple[np.ndarray, np.ndarray]:
    """A centre and a radius whose ball holds each cell (chords on the sphere)."""
    centres = polygons.mean(axis=1)
    if sphere:
        centres = centres / np.linalg.norm(centres, axis=1, keepdims=True)
    # every point of a convex cell (a spherical cap's, on the sphere) is no
    # farther from the centre than its farthest corner
    radii = np.linalg.norm(polygons - centres[:, None, :], axis=2).max(axis=1)
    return centres, radii


def _candidate_pairs(
    source: np.ndarray, target: np.ndarray, sphere: bool
) -> tuple[np.ndarray, np.ndarray]:
    """(target, source) pairs whose bounding balls meet, from a k-d tree."""
    spatial = import_scipy("spatial", needed_by="Conservative.plan")
    source_centres, source_radii = _bounding_circles(source, sphere)
    target_centres, target_radii = _bounding_circles(target, sphere)
    tree = spatial.cKDTree(source_centres)
    reach = target_radii + source_radii.max()
    neighbours = tree.query_ball_point(target_centres, reach)
    counts = np.array([len(row) for row in neighbours], dtype=np.int64)
    target_index = np.repeat(np.arange(len(target)), counts)
    source_index = np.fromiter(
        (j for row in neighbours for j in row), dtype=np.int64, count=counts.sum()
    )
    # ‖nᵢ − nⱼ‖ ≤ rᵢ + rⱼ, the test the tree's single reach overestimates
    gap = np.linalg.norm(
        target_centres[target_index] - source_centres[source_index], axis=1
    )
    meets = gap <= target_radii[target_index] + source_radii[source_index]
    return target_index[meets], source_index[meets]


def cell_overlaps(
    source: Float[np.ndarray, "source corner dim"],
    target: Float[np.ndarray, "target corner dim"],
    *,
    sphere: bool,
) -> CellOverlaps:
    """The overlap areas |cⱼ ∩ Cᵢ| of every pair of source and target cells.

    Algorithm:
        1. Find candidate pairs whose bounding balls meet, with a k-d tree on
           the source cells' centres.
        2. Orient each target cell counter-clockwise and check it is convex;
           on the sphere, project both cells of a pair into the gnomonic
           tangent plane at the target cell's centre.
        3. Clip each source cell to its target cell (`clip_polygons`) and
           measure the result: the shoelace area in the plane, or the
           spherical excess of the clipped polygon lifted back to the sphere.

    Args:
        source: The corners of the N source cells, (x, y) in the plane or unit
            vectors on the sphere, in order around each cell.
        target: The corners of the M target cells, the same way.
        sphere: Whether the cells are spherical, with great-circle edges.

    Returns:
        The pairs with positive overlap, and every target cell's area.

    Raises:
        GeometryError: If a target cell is not convex, or a pair spans more
            than a hemisphere.
        MissingExtraError: If SciPy (``functax[scipy]``) is missing.
    """
    dtype = np.result_type(source, target)
    if sphere:
        target_areas = _spherical_areas(target)
    else:
        target_areas = np.abs(
            planar_polygon_area(target, np.full(len(target), target.shape[1]))
        )
    target_index, source_index = _candidate_pairs(source, target, sphere)
    pair_targets: list[np.ndarray] = []
    pair_sources: list[np.ndarray] = []
    pair_areas: list[np.ndarray] = []
    for start in range(0, len(target_index), _CHUNK):
        rows = target_index[start : start + _CHUNK]
        columns = source_index[start : start + _CHUNK]
        clips = target[rows]
        subjects = source[columns]
        if sphere:
            centres, _ = _bounding_circles(clips, sphere=True)
            first, second = _tangent_frames(centres)
            clips = _gnomonic(clips, centres, first, second)
            subjects = _gnomonic(subjects, centres, first, second)
        else:
            # shift to the target's centre, so the products below lose no digits
            origin = clips.mean(axis=1, keepdims=True)
            clips = clips - origin
            subjects = subjects - origin
        n_corners = np.full(len(rows), clips.shape[1])
        clips = _counter_clockwise(clips, planar_polygon_area(clips, n_corners))
        _check_convex(clips, "target")
        n_vertices = np.full(len(rows), subjects.shape[1])
        clipped, counts = clip_polygons(subjects, n_vertices, clips)
        if sphere:
            # p = n̂ + u e₁ + v e₂, back on the sphere
            lifted = (
                centres[:, None, :]
                + clipped[..., :1] * first[:, None, :]
                + clipped[..., 1:] * second[:, None, :]
            )
            lifted = lifted / np.linalg.norm(lifted, axis=2, keepdims=True)
            areas = _spherical_areas(lifted, counts)
        else:
            areas = np.abs(planar_polygon_area(clipped, counts))
        overlapping = areas > 0
        pair_targets.append(rows[overlapping])
        pair_sources.append(columns[overlapping])
        pair_areas.append(areas[overlapping].astype(dtype))
    if not pair_targets:
        empty = np.zeros(0, np.int64)
        return CellOverlaps(empty, empty, np.zeros(0, dtype), target_areas)
    return CellOverlaps(
        np.concatenate(pair_targets),
        np.concatenate(pair_sources),
        np.concatenate(pair_areas),
        target_areas.astype(dtype),
    )
