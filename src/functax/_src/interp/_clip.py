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

import einx
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

    Source cell cⱼ and target cell Cᵢ have the areas |cⱼ| and |Cᵢ|.

    Attributes:
        target: i, the target cell of each pair.
        source: j, the source cell of each pair.
        areas: |cⱼ ∩ Cᵢ|, positive.
        target_areas: |Cᵢ| of every target cell.

    Examples:
        Two unit squares under one 2 × 1 target:

        >>> import numpy as np
        >>> from functax._src.interp._clip import cell_overlaps
        >>> square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        >>> source = np.stack([square, square + [1.0, 0.0]])
        >>> overlaps = cell_overlaps(
        ...     source, (square * [2.0, 1.0])[None], sphere=False
        ... )
        >>> overlaps.source.tolist(), overlaps.areas.tolist()
        ([0, 1], [1.0, 1.0])
        >>> overlaps.target_areas.tolist()
        [2.0]
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
    on the left of every edge q₀ → q₁: the side d(p) = (q₁ − q₀) × (p − q₀)
    of a point p is at least 0. For each clip edge in turn, a subject edge
    from p to p′ keeps p′ when it is inside and adds the crossing
    p + t (p′ − p), t = d(p) / (d(p) − d(p′)), when the two ends lie on
    opposite sides. Subjects may be any simple polygon; a zero-length
    clip edge cuts nothing.

    Algorithm:
        1. For each clip edge q₀ → q₁, find the side d(p) of every subject
           vertex and of its predecessor.
        2. Give each vertex two slots: the crossing of the edge into it, kept
           where the two ends lie on opposite sides, and the vertex itself,
           kept where it is inside.
        3. Pack the kept slots to the front of each row (a stable sort) and
           trim the padding to the longest polygon.

    Args:
        subjects: The subject vertices, padded past ``counts``.
        counts: The number of vertices of each subject.
        clips: The vertices of each clip polygon.

    Returns:
        The clipped vertices, padded, and their counts (0 where nothing is left).

    Examples:
        A unit square shifted by half its side keeps a quarter:

        >>> import numpy as np
        >>> from functax._src.interp._clip import clip_polygons, planar_polygon_area
        >>> square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        >>> points, counts = clip_polygons(
        ...     (square + 0.5)[None], np.array([4]), square[None]
        ... )
        >>> points.shape, counts.tolist()
        ((1, 4, 2), [4])
        >>> planar_polygon_area(points, counts).tolist()
        [0.25]

        A subject outside the clip polygon leaves nothing:

        >>> points, counts = clip_polygons(
        ...     (square + 2.0)[None], np.array([4]), square[None]
        ... )
        >>> counts.tolist()
        [0]
    """
    points = subjects
    n_corners = clips.shape[1]
    for corner in range(n_corners):
        # --- side of the clip edge q₀ → q₁ ---
        start = clips[:, corner]
        end = clips[:, (corner + 1) % n_corners]
        # q₁ − q₀
        edge = end - start
        width = points.shape[1]
        vertex = np.arange(width)
        valid = einx.less("vertex, pair -> pair vertex", vertex, counts)
        # the previous vertex, cyclically within each polygon's own count
        safe_counts = np.maximum(counts, 1)
        # k − 1 mod N_vertex (einx has no remainder, so both sides are broadcast first)
        shifted = einx.id("vertex -> pair vertex", vertex - 1, pair=len(counts))
        period = einx.id("pair -> pair vertex", safe_counts, vertex=width)
        previous_index = np.mod(shifted, period)
        previous = einx.get_at(
            "pair [vertex] dim, pair slot -> pair slot dim", points, previous_index
        )
        # p − q₀
        offset = einx.subtract(
            "pair vertex dim, pair dim -> pair vertex dim", points, start
        )
        # d(p) = (q₁ − q₀) × (p − q₀)
        along_y = einx.multiply(
            "pair, pair vertex -> pair vertex", edge[:, 0], offset[..., 1]
        )
        along_x = einx.multiply(
            "pair, pair vertex -> pair vertex", edge[:, 1], offset[..., 0]
        )
        side = along_y - along_x
        previous_side = einx.get_at(
            "pair [vertex], pair slot -> pair slot", side, previous_index
        )
        inside = side >= 0
        previous_inside = previous_side >= 0
        changes_side = inside != previous_inside
        crosses = valid & changes_side

        # --- crossings ---
        # d(p) − d(p′)
        side_drop = previous_side - side
        denominator = np.where(crosses, side_drop, 1.0)
        # t = d(p) / (d(p) − d(p′)), the two of opposite sign where it crosses
        ratio = previous_side / denominator
        fraction = np.where(crosses, ratio, 0.0)
        # p′ − p
        step = points - previous
        # t (p′ − p)
        advance = einx.multiply(
            "pair vertex, pair vertex dim -> pair vertex dim", fraction, step
        )
        # p + t (p′ − p)
        crossing = previous + advance

        # --- pack ---
        # two slots per vertex, the crossing first, then the vertex if kept
        slots = einx.id(
            "pair vertex dim, pair vertex dim -> pair (vertex (1 + 1)) dim",
            crossing,
            points,
        )
        keeps_vertex = valid & inside
        kept = einx.id(
            "pair vertex, pair vertex -> pair (vertex (1 + 1))", crosses, keeps_vertex
        )
        dropped = ~kept
        order = np.argsort(dropped, axis=1, kind="stable")
        points = einx.get_at(
            "pair [slot] dim, pair packed -> pair packed dim", slots, order
        )
        counts = einx.sum("pair [slot]", kept)
        # initial=0: an empty batch of pairs has no maximum
        longest = int(counts.max(initial=0))
        points = points[:, : max(longest, 1)]
    return points, counts


def planar_polygon_area(
    points: Float[np.ndarray, "pair vertex 2"], counts: Int[np.ndarray, " pair"]
) -> Float[np.ndarray, " pair"]:
    """The signed shoelace area of padded polygons.

    A = ½ ∑ₖ (xₖ yₖ₊₁ − xₖ₊₁ yₖ), the sum over the vertices (xₖ, yₖ),
    k = 0, …, N_vertex − 1, of each polygon's own N_vertex = ``counts``, with
    k + 1 taken cyclically; positive when counter-clockwise.

    Args:
        points: The vertices (x, y), padded past ``counts``.
        counts: The number of vertices of each polygon.

    Returns:
        The signed area A of each polygon.

    Examples:
        The unit square, then the same square clockwise:

        >>> import numpy as np
        >>> from functax._src.interp._clip import planar_polygon_area
        >>> square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        >>> planar_polygon_area(square[None], np.array([4])).tolist()
        [1.0]
        >>> planar_polygon_area(square[None, ::-1], np.array([4])).tolist()
        [-1.0]
    """
    width = points.shape[1]
    vertex = np.arange(width)
    safe_counts = np.maximum(counts, 1)
    # k + 1 mod N_vertex (einx has no remainder, so both sides are broadcast first)
    shifted = einx.id("vertex -> pair vertex", vertex + 1, pair=len(counts))
    period = einx.id("pair -> pair vertex", safe_counts, vertex=width)
    following_index = np.mod(shifted, period)
    following = einx.get_at(
        "pair [vertex] dim, pair slot -> pair slot dim", points, following_index
    )
    # xₖ yₖ₊₁ − xₖ₊₁ yₖ on the polygon's own edges only
    forward = points[..., 0] * following[..., 1]
    backward = following[..., 0] * points[..., 1]
    terms = forward - backward
    on_polygon = einx.less("vertex, pair -> pair vertex", vertex, counts)
    terms = np.where(on_polygon, terms, 0.0)
    # ∑ₖ (xₖ yₖ₊₁ − xₖ₊₁ yₖ)
    twice_area = einx.sum("pair [vertex]", terms)
    # A = ½ ∑ₖ (xₖ yₖ₊₁ − xₖ₊₁ yₖ)
    return 0.5 * twice_area


def _counter_clockwise(
    polygons: Float[np.ndarray, "pair corner dim"], areas: Float[np.ndarray, " pair"]
) -> Float[np.ndarray, "pair corner dim"]:
    """Reverse the corner order of each polygon with a negative signed area."""
    reversed_order = polygons[:, ::-1]
    clockwise = areas < 0
    return einx.where(
        "pair, pair corner dim, pair corner dim -> pair corner dim",
        clockwise,
        reversed_order,
        polygons,
    )


def _check_convex(polygons: Float[np.ndarray, "pair corner dim"], name: str) -> None:
    """Raise unless each counter-clockwise polygon never turns right."""
    following = einx.roll("pair [corner] dim", polygons, shift=-1)
    after = einx.roll("pair [corner] dim", polygons, shift=-2)
    # qₖ₊₁ − qₖ
    first = following - polygons
    # qₖ₊₂ − qₖ₊₁
    second = after - following
    # (qₖ₊₁ − qₖ) × (qₖ₊₂ − qₖ₊₁)
    forward = first[..., 0] * second[..., 1]
    backward = first[..., 1] * second[..., 0]
    turn = forward - backward
    # scale = (maxₖ ‖qₖ₊₁ − qₖ‖_∞)², over the corners qₖ
    magnitude = np.abs(first)
    longest = einx.max("pair [corner dim]", magnitude)
    scale = longest**2
    # a right turn below −tol · scale
    threshold = -_CONVEX_RTOL * scale
    turns_right = einx.less("pair corner, pair -> pair corner", turn, threshold)
    concave = einx.any("pair [corner]", turns_right)
    if np.any(concave):
        cell = int(np.flatnonzero(concave)[0])
        msg = f"'{name}' cells must be convex to clip against, but cell {cell} is not"
        raise GeometryError(msg)


def _tangent_frames(
    centres: Float[np.ndarray, "pair 3"],
) -> tuple[Float[np.ndarray, "pair 3"], Float[np.ndarray, "pair 3"]]:
    """Two unit vectors e₁, e₂ spanning the tangent plane at each unit centre o."""
    # e₁ ∥ e_z × o, or e_x × o near the poles, with e_x, e_z the Cartesian axes
    height = np.abs(centres[:, 2])
    pole = height > 0.9
    x_axis = np.array([1.0, 0.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    reference = einx.where("pair, dim, dim -> pair dim", pole, x_axis, z_axis)
    first = np.cross(reference, centres)
    # ‖e₁‖
    length = np.linalg.norm(first, axis=1)
    # e₁ / ‖e₁‖
    first = einx.divide("pair dim, pair -> pair dim", first, length)
    # e₂ = o × e₁
    second = np.cross(centres, first)
    return first, second


def _gnomonic(
    points: Float[np.ndarray, "pair vertex 3"],
    centre: Float[np.ndarray, "pair 3"],
    first: Float[np.ndarray, "pair 3"],
    second: Float[np.ndarray, "pair 3"],
) -> Float[np.ndarray, "pair vertex 2"]:
    """p̃ = (p · e₁, p · e₂) / (p · o), the gnomonic projection about o."""
    # p · o
    height = einx.dot("pair vertex dim, pair dim -> pair vertex", points, centre)
    if np.any(height <= 0):
        msg = (
            "cells must each lie within a hemisphere of the target cells they "
            "overlap, but a pair spans more"
        )
        raise GeometryError(msg)
    # p · e₁
    first_component = einx.dot(
        "pair vertex dim, pair dim -> pair vertex", points, first
    )
    # p · e₂
    second_component = einx.dot(
        "pair vertex dim, pair dim -> pair vertex", points, second
    )
    # p̃₁ = (p · e₁) / (p · o)
    along_first = first_component / height
    # p̃₂ = (p · e₂) / (p · o)
    along_second = second_component / height
    return einx.id(
        "pair vertex, pair vertex -> pair vertex (1 + 1)", along_first, along_second
    )


def _spherical_areas(
    polygons: Float[np.ndarray, "pair vertex 3"],
    counts: Int[np.ndarray, " pair"] | None = None,
) -> Float[np.ndarray, " pair"]:
    """|A| of padded unit-vector polygons, zero where fewer than three vertices."""
    if counts is not None:
        # pad by repeating the last vertex, which adds zero-area triangles
        vertex = np.arange(polygons.shape[1])
        # N_vertex − 1
        last_vertex = counts - 1
        last = np.maximum(last_vertex, 0)
        index = einx.minimum("vertex, pair -> pair vertex", vertex, last)
        polygons = einx.get_at(
            "pair [vertex] dim, pair slot -> pair slot dim", polygons, index
        )
    if polygons.shape[1] < 3:
        return np.zeros(polygons.shape[0], polygons.dtype)
    device_polygons = jnp.asarray(polygons)
    signed_areas = spherical_polygon_area(device_polygons)
    # |A|
    areas = np.abs(np.asarray(signed_areas))
    if counts is not None:
        is_polygon = counts >= 3
        areas = np.where(is_polygon, areas, 0.0)
    return areas


def _bounding_circles(
    polygons: Float[np.ndarray, "cell corner dim"], sphere: bool
) -> tuple[Float[np.ndarray, "cell dim"], Float[np.ndarray, " cell"]]:
    """A centre o and a radius r whose ball holds each cell (chords on the sphere)."""
    # o = (1/N_corner) ∑ₖ qₖ, over the N_corner corners qₖ
    centres = einx.mean("cell [corner] dim", polygons)
    if sphere:
        # o / ‖o‖
        length = np.linalg.norm(centres, axis=1)
        centres = einx.divide("cell dim, cell -> cell dim", centres, length)
    # every point of a convex cell (a spherical cap's, on the sphere) is no
    # farther from the centre than its farthest corner
    # qₖ − o
    offsets = einx.subtract(
        "cell corner dim, cell dim -> cell corner dim", polygons, centres
    )
    # ‖qₖ − o‖
    distances = np.linalg.norm(offsets, axis=2)
    # r = maxₖ ‖qₖ − o‖
    radii = einx.max("cell [corner]", distances)
    return centres, radii


def _candidate_pairs(
    source: Float[np.ndarray, "source corner dim"],
    target: Float[np.ndarray, "target corner dim"],
    sphere: bool,
) -> tuple[Int[np.ndarray, " pair"], Int[np.ndarray, " pair"]]:
    """(target, source) pairs whose bounding balls meet, from a k-d tree."""
    spatial = import_scipy("spatial", needed_by="Conservative.plan")
    source_centres, source_radii = _bounding_circles(source, sphere)
    target_centres, target_radii = _bounding_circles(target, sphere)
    tree = spatial.cKDTree(source_centres)
    # rᵢ + maxⱼ rⱼ, the radii of target i and source j
    largest_source = einx.max("[source]", source_radii)
    reach = target_radii + largest_source
    neighbours = tree.query_ball_point(target_centres, reach)
    counts = np.array([len(row) for row in neighbours], dtype=np.int64)
    target_index = np.repeat(np.arange(len(target)), counts)
    source_index = np.fromiter(
        (j for row in neighbours for j in row), dtype=np.int64, count=counts.sum()
    )
    # ‖oᵢ − oⱼ‖ ≤ rᵢ + rⱼ, the test the tree's single reach overestimates
    separation = target_centres[target_index] - source_centres[source_index]
    gap = np.linalg.norm(separation, axis=1)
    pair_reach = target_radii[target_index] + source_radii[source_index]
    meets = gap <= pair_reach
    return target_index[meets], source_index[meets]


def cell_overlaps(
    source: Float[np.ndarray, "source corner dim"],
    target: Float[np.ndarray, "target corner dim"],
    *,
    sphere: bool,
) -> CellOverlaps:
    """The overlap areas |cⱼ ∩ Cᵢ| of every pair of source and target cells.

    |cⱼ ∩ Cᵢ| is the area of the source cell cⱼ clipped to the target cell
    Cᵢ: the shoelace area in the plane, the spherical excess on the sphere.

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

    Examples:
        A unit square half over a 2 × 2 target, the target clockwise:

        >>> import numpy as np
        >>> from functax._src.interp._clip import cell_overlaps
        >>> square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        >>> target = (2.0 * square - [1.5, 0.0])[None, ::-1]
        >>> overlaps = cell_overlaps(square[None], target, sphere=False)
        >>> overlaps.areas.tolist(), overlaps.target_areas.tolist()
        ([0.5], [4.0])

        A concave target is rejected:

        >>> dart = np.array([[[0.0, 0.0], [2.0, 0.0], [0.5, 0.5], [0.0, 2.0]]])
        >>> cell_overlaps(
        ...     square[None], dart, sphere=False
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        GeometryError: 'target' cells must be convex to clip against
    """
    dtype = np.result_type(source, target)
    if sphere:
        target_areas = _spherical_areas(target)
    else:
        n_target_corners = np.full(len(target), target.shape[1])
        signed_target_areas = planar_polygon_area(target, n_target_corners)
        # |Cᵢ|
        target_areas = np.abs(signed_target_areas)
    target_index, source_index = _candidate_pairs(source, target, sphere)
    pair_targets: list[Int[np.ndarray, " pair"]] = []
    pair_sources: list[Int[np.ndarray, " pair"]] = []
    pair_areas: list[Float[np.ndarray, " pair"]] = []
    for start in range(0, len(target_index), _CHUNK):
        # --- project ---
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
            origin = einx.mean("pair [corner] dim", clips)
            clips = einx.subtract(
                "pair corner dim, pair dim -> pair corner dim", clips, origin
            )
            subjects = einx.subtract(
                "pair vertex dim, pair dim -> pair vertex dim", subjects, origin
            )

        # --- clip ---
        n_corners = np.full(len(rows), clips.shape[1])
        clip_areas = planar_polygon_area(clips, n_corners)
        clips = _counter_clockwise(clips, clip_areas)
        _check_convex(clips, "target")
        n_vertices = np.full(len(rows), subjects.shape[1])
        clipped, counts = clip_polygons(subjects, n_vertices, clips)

        # --- measure ---
        if sphere:
            # p̃₁ e₁
            along_first = einx.multiply(
                "pair out, pair dim -> pair out dim", clipped[..., 0], first
            )
            # p̃₂ e₂
            along_second = einx.multiply(
                "pair out, pair dim -> pair out dim", clipped[..., 1], second
            )
            # o + p̃₁ e₁
            lifted = einx.add(
                "pair dim, pair out dim -> pair out dim", centres, along_first
            )
            # p = o + p̃₁ e₁ + p̃₂ e₂, back on the sphere
            lifted = lifted + along_second
            # ‖p‖
            length = np.linalg.norm(lifted, axis=2)
            # p / ‖p‖
            lifted = einx.divide(
                "pair out dim, pair out -> pair out dim", lifted, length
            )
            areas = _spherical_areas(lifted, counts)
        else:
            signed_areas = planar_polygon_area(clipped, counts)
            # |cⱼ ∩ Cᵢ|
            areas = np.abs(signed_areas)
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
