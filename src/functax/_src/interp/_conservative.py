"""First-order conservative remapping: area-weighted cell averages as one stencil.

Target cell Cᵢ, for i over the M target cells, reads every source cell cⱼ, for
j over the N source cells, that it overlaps with the weight
wᵢⱼ = |cⱼ ∩ Cᵢ| / |Cᵢ|, where |·| is a cell's area. The remapped values
ŷᵢ = ∑ⱼ wᵢⱼ yⱼ of the samples yⱼ then keep ∑ᵢ |Cᵢ| ŷᵢ = ∑ⱼ |cⱼ| yⱼ whenever
the target cells cover the source cells. On tensor-product grids the overlap
of two cells is the product of their per-axis overlaps (`overlap_band`), so
the stencil is the tensor product of one band per axis. Curvilinear and
spherical cells are quadrilaterals whose pairwise overlaps are clipped on the
host (`_clip`).

It lives in its own module rather than in ``_schemes.py`` because it is not a
`LocalScheme`: it reads cells, not a basis at located positions, and has no
pointwise fit.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import math
from typing import Any

import einx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Bool, Float, Int

from functax._src.geometry._sphere import _check_unit, lonlat_to_unit
from functax._src.geometry._types import (
    AffineGrid,
    Axis,
    CurvilinearGrid,
    RectilinearGrid,
)
from functax._src.interp._clip import cell_overlaps
from functax._src.interp._fit import Identity
from functax._src.interp._locate import check_outside, require_no_repeats
from functax._src.interp._overlap import OverlapBand, overlap_band
from functax._src.interp._plan import Plan
from functax._src.interp._policies import Policies
from functax._src.interp._schemes import Scheme
from functax._src.interp._stencil import Stencil
from functax.exceptions import GeometryError, ValidationError


__all__ = [
    "Conservative",
]


def _cell_axes(
    geometry: Any, name: str
) -> list[tuple[Float[np.ndarray, " edge"], float | None]]:
    """The (edges, period) of each axis of a tensor-product grid of cells.

    An `AffineGrid` with transform coefficients (a, b, c, d, e, f) must be
    north-up (b = d = 0): its cells are then the tensor product of the row
    edges f + e·i and the column edges c + a·j, for row edge i and column
    edge j.
    """
    if isinstance(geometry, Axis):
        edges = np.asarray(geometry.cell_edges())
        return [(edges, geometry.period)]
    if isinstance(geometry, RectilinearGrid):
        return [(np.asarray(axis.cell_edges()), axis.period) for axis in geometry.axes]
    if isinstance(geometry, AffineGrid):
        a, b, c, d, e, f = np.asarray(geometry.transform).tolist()
        if b != 0.0 or d != 0.0:
            msg = (
                f"'{name}' must be a north-up AffineGrid (b = d = 0) for "
                f"Conservative, got b={b}, d={d}"
            )
            raise GeometryError(msg)
        rows, cols = geometry.shape
        row_index = np.arange(rows + 1, dtype=np.float64)
        col_index = np.arange(cols + 1, dtype=np.float64)
        # y = f + e·i at row edge i
        row_edges = f + e * row_index
        # x = c + a·j at column edge j
        col_edges = c + a * col_index
        return [(row_edges, None), (col_edges, None)]
    msg = (
        f"'{name}' must be an Axis, a RectilinearGrid or an AffineGrid, got "
        f"{type(geometry).__name__}"
    )
    raise GeometryError(msg)


def _axis_weights(
    band: OverlapBand, policies: Policies
) -> Float[np.ndarray, "target tap"]:
    """wᵢⱼ on one axis: |cⱼ ∩ Cᵢ| / |Cᵢ|, or over the covered length under clamp."""
    if policies.outside != "clamp":
        # wᵢⱼ = |cⱼ ∩ Cᵢ| / |Cᵢ|
        return einx.divide(
            "target tap, target -> target tap", band.lengths, band.target_lengths
        )
    # ∑ⱼ |cⱼ ∩ Cᵢ|, the length of the target cut to the source domain
    covered = einx.sum("target [tap]", band.lengths)
    has_cover = covered > 0
    safe_covered = np.where(has_cover, covered, 1.0)
    # wᵢⱼ = |cⱼ ∩ Cᵢ| / ∑ⱼ |cⱼ ∩ Cᵢ|
    return einx.divide("target tap, target -> target tap", band.lengths, safe_covered)


# a target cell whose overlaps sum to its area to this relative tolerance is
# covered; clipped areas round in the last few digits
_COVERED_RTOL = 1e-9


def _policy_stencil(
    indices: Int[np.ndarray, "target tap"],
    weights: Float[np.ndarray, "target tap"],
    covered: Bool[np.ndarray, " target"],
    touched: Bool[np.ndarray, " target"],
    n_source: int,
    policies: Policies,
) -> Stencil:
    """The stencil after the outside and missing rules, invalid rows zeroed.

    Raises:
        OutsideDomainError: Under ``outside="raise"`` when a target cell is
            not inside the source cells.
    """
    check_outside(covered, policies)
    # renormalising counts the area outside the source as missing, so a
    # partly covered cell keeps the mean of its covered, valid part
    keep_partial = (
        policies.outside in ("clamp", "extrapolate")
        or policies.missing == "renormalise"
    )
    valid = touched if keep_partial else covered
    zero = np.zeros((), weights.dtype)
    weights = einx.where("target, target tap, -> target tap", valid, weights, zero)
    device_indices = jnp.asarray(indices)
    device_weights = jnp.asarray(weights)
    device_valid = jnp.asarray(valid)
    return Stencil(
        device_indices,
        device_weights,
        n_source=n_source,
        valid=device_valid,
        check=False,
    )


def tensor_stencil_of_cells(source: Any, target: Any, *, policies: Policies) -> Stencil:
    """H of conservative remapping between tensor grids, from per-axis bands.

    Raises:
        GeometryError: If a geometry is unsupported or the axis counts differ.
        OutsideDomainError: Under ``outside="raise"`` when a target cell is
            not inside the source cells.
    """
    source_axes = _cell_axes(source, "source")
    target_axes = _cell_axes(target, "target")
    if len(source_axes) != len(target_axes):
        msg = (
            f"'target' has {len(target_axes)} axes but 'source' has {len(source_axes)}"
        )
        raise GeometryError(msg)
    bands = [
        overlap_band(source_edges, target_edges, period=period)
        for (source_edges, period), (target_edges, _) in zip(
            source_axes, target_axes, strict=True
        )
    ]
    source_shape = tuple(edges.shape[0] - 1 for edges, _ in source_axes)
    indices = bands[0].indices
    weights = _axis_weights(bands[0], policies)
    covered = bands[0].covered
    touched = bands[0].touched
    for n_cells, band in zip(source_shape[1:], bands[1:], strict=True):
        # flat index ((j₁ n₂ + j₂) n₃ + j₃) …
        scaled = indices * n_cells
        indices = einx.add("a s, b t -> (a b) (s t)", scaled, band.indices)
        # weight w₁ w₂ w₃ …
        axis_weights = _axis_weights(band, policies)
        weights = einx.multiply("a s, b t -> (a b) (s t)", weights, axis_weights)
        covered = einx.logical_and("a, b -> (a b)", covered, band.covered)
        touched = einx.logical_and("a, b -> (a b)", touched, band.touched)
    n_source = math.prod(source_shape)
    return _policy_stencil(indices, weights, covered, touched, n_source, policies)


def _node_corners(
    nodes: Float[np.ndarray, "row col dim"], sphere: bool
) -> Float[np.ndarray, "row_edge col_edge dim"]:
    """The (N_row + 1, N_col + 1, D) corners of (N_row, N_col, D) cell-centre nodes.

    The nodes x₀, …, x_{N_row−1} of a column are extended by one node on each
    side by linear extrapolation, x₋₁ = 2x₀ − x₁, and the same along the
    rows; each corner is the mean of the four nodes around it, so a tensor
    grid gets the midpoints of `Axis.cell_edges`.
    On the sphere the means are of unit vectors, normalised, so the corners
    need no unwrapped longitudes.
    """
    # --- extrapolate one ring ---
    # x₋₁ = 2x₀ − x₁, and x_{N_row} = 2x_{N_row−1} − x_{N_row−2} past the last row
    first_row = 2 * nodes[0] - nodes[1]
    last_row = 2 * nodes[-1] - nodes[-2]
    padded = einx.id(
        "col dim, row col dim, col dim -> (1 + row + 1) col dim",
        first_row,
        nodes,
        last_row,
    )
    # the same along the columns
    first_col = 2 * padded[:, 0] - padded[:, 1]
    last_col = 2 * padded[:, -1] - padded[:, -2]
    padded = einx.id(
        "row dim, row col dim, row dim -> row (1 + col + 1) dim",
        first_col,
        padded,
        last_col,
    )

    # --- corners ---
    below_left = padded[:-1, :-1]
    above_left = padded[1:, :-1]
    below_right = padded[:-1, 1:]
    above_right = padded[1:, 1:]
    # x₀₀ + x₁₀ + x₀₁ + x₁₁, the four nodes around each corner
    node_sum = below_left + above_left + below_right + above_right
    # ¼ (x₀₀ + x₁₀ + x₀₁ + x₁₁)
    corners = 0.25 * node_sum
    if sphere:
        # ‖s‖ of each corner s
        length = np.linalg.norm(corners, axis=-1)
        # s / ‖s‖
        corners = einx.divide("row col dim, row col -> row col dim", corners, length)
    return corners


def _corner_cells(
    corners: Float[np.ndarray, "row_edge col_edge dim"],
) -> Float[np.ndarray, "cell 4 dim"]:
    """(N_row N_col, 4, D) cells, row-major, from (N_row + 1, N_col + 1, D) corners."""
    below_left = corners[:-1, :-1]
    above_left = corners[1:, :-1]
    above_right = corners[1:, 1:]
    below_right = corners[:-1, 1:]
    return einx.id(
        "row col dim, row col dim, row col dim, row col dim"
        " -> (row col) (1 + 1 + 1 + 1) dim",
        below_left,
        above_left,
        above_right,
        below_right,
    )


def _polygon_cells(
    geometry: Any, name: str, unit: str | None
) -> Float[np.ndarray, "cell 4 dim"]:
    """The quadrilateral cells of a 2-D grid: (``x``, ``y``) corners, or unit vectors.

    ``unit`` is that of the longitude and latitude (λ, φ) on the sphere, and
    ``None`` in the plane.

    A `CurvilinearGrid` gives its cells by `_node_corners`; a 2-D
    `RectilinearGrid` has its axes in (``x``, ``y``) order, (λ, φ) on the
    sphere, as a target read on a curvilinear source does.
    """
    if isinstance(geometry, CurvilinearGrid):
        node_x = np.asarray(geometry.x)
        node_y = np.asarray(geometry.y)
        nodes = einx.id("row col, row col -> row col (1 + 1)", node_x, node_y)
        if unit is not None:
            device_nodes = jnp.asarray(nodes)
            unit_nodes = lonlat_to_unit(device_nodes, unit=unit)
            nodes = np.asarray(unit_nodes)
        corners = _node_corners(nodes, sphere=unit is not None)
        return _corner_cells(corners)
    if isinstance(geometry, RectilinearGrid) and len(geometry.axes) == 2:
        x_edges, y_edges = (np.asarray(axis.cell_edges()) for axis in geometry.axes)
        if unit is not None:
            # a latitude edge past the pole would wrap over it
            pole = 90.0 if unit == "degrees" else math.pi / 2
            y_edges = np.clip(y_edges, -pole, pole)
        x, y = np.meshgrid(x_edges, y_edges, indexing="ij")
        corners = einx.id("row col, row col -> row col (1 + 1)", x, y)
        if unit is not None:
            device_corners = jnp.asarray(corners)
            unit_corners = lonlat_to_unit(device_corners, unit=unit)
            corners = np.asarray(unit_corners)
        return _corner_cells(corners)
    msg = (
        f"'{name}' must be a CurvilinearGrid or a 2-D RectilinearGrid for "
        f"curvilinear or spherical remapping, got {type(geometry).__name__}"
    )
    raise GeometryError(msg)


def polygon_stencil_of_cells(
    source: Any, target: Any, *, policies: Policies, unit: str | None
) -> Stencil:
    """H of conservative remapping between quadrilateral cells, clipped pairwise.

    ``unit`` is that of the longitude and latitude (λ, φ) on the sphere, and
    ``None`` in the plane.

    Raises:
        GeometryError: If a geometry is unsupported, a target cell is not
            convex, or a pair of cells spans more than a hemisphere.
        OutsideDomainError: Under ``outside="raise"`` when a target cell is
            not inside the source cells.
        MissingExtraError: If SciPy (``functax[scipy]``) is missing.
    """
    source_cells = _polygon_cells(source, "source", unit)
    target_cells = _polygon_cells(target, "target", unit)
    for name, cells in (("source", source_cells), ("target", target_cells)):
        if not np.all(np.isfinite(cells)):
            msg = f"'{name}' must have finite coordinates"
            raise GeometryError(msg)
    overlaps = cell_overlaps(source_cells, target_cells, sphere=unit is not None)
    n_target = target_cells.shape[0]

    # --- the pairs as rows of S taps, Sᵢ in row i, padded with zero weight ---
    order = np.argsort(overlaps.target, kind="stable")
    rows = overlaps.target[order]
    columns = overlaps.source[order]
    areas = overlaps.areas[order]
    counts = np.bincount(rows, minlength=n_target)
    # initial=0: no pairs at all leaves no maximum
    most_taps = int(counts.max(initial=0))
    n_taps = max(most_taps, 1)
    # the first pair of each row, ∑_{i′ < i} Sᵢ′
    running = np.cumsum(counts)
    starts = running - counts
    pair = np.arange(len(rows))
    tap = pair - starts[rows]
    indices = np.zeros((n_target, n_taps), np.int32)
    lengths = np.zeros((n_target, n_taps), areas.dtype)
    indices[rows, tap] = columns
    lengths[rows, tap] = areas
    # padding repeats the row's first index (0 for an empty row)
    tap_index = np.arange(n_taps)
    padding = einx.greater_equal("tap, target -> target tap", tap_index, counts)
    first = indices[:, 0]
    indices = einx.where(
        "target tap, target, target tap -> target tap", padding, first, indices
    )

    # --- weights ---
    target_areas = overlaps.target_areas
    real = target_areas > 0
    safe_areas = np.where(real, target_areas, 1.0)
    # wᵢⱼ = |cⱼ ∩ Cᵢ| / |Cᵢ|
    weights = einx.divide("target tap, target -> target tap", lengths, safe_areas)
    # the covered fraction ∑ⱼ wᵢⱼ
    fraction = einx.sum("target [tap]", weights)
    full = fraction >= 1.0 - _COVERED_RTOL
    covered = real & full
    some = fraction > 0
    touched = real & some
    if policies.outside == "clamp":
        safe_fraction = np.where(touched, fraction, 1.0)
        # wᵢⱼ = |cⱼ ∩ Cᵢ| / ∑ⱼ |cⱼ ∩ Cᵢ|
        weights = einx.divide(
            "target tap, target -> target tap", weights, safe_fraction
        )
    n_source = source_cells.shape[0]
    return _policy_stencil(indices, weights, covered, touched, n_source, policies)


class Conservative(Scheme):
    """First-order conservative remapping: each target cell's area-weighted mean.

    Target cell Cᵢ, i over the M target cells, reads the source cells cⱼ,
    j over the N source cells, that it overlaps, with |·| a cell's area:

        ŷᵢ = ∑ⱼ wᵢⱼ yⱼ  with  wᵢⱼ = |cⱼ ∩ Cᵢ| / |Cᵢ|,

    so ∑ᵢ |Cᵢ| ŷᵢ = ∑ⱼ |cⱼ| yⱼ whenever the target cells cover the source
    cells: integrals (fluxes, emissions) are preserved at any factor, and an
    integer-factor coarsening is a block mean. On tensor grids the overlap is
    the product of the per-axis overlaps (`overlap_weights_1d`), so no
    polygon is clipped. Cell edges come from `Axis.cell_edges`: the axis's
    ``edges`` when given, else the midpoints between nodes; a periodic source
    axis reads targets modulo its period.

    A `CurvilinearGrid` (source or target), or ``sphere=True``, takes the
    polygon path: every cell is a quadrilateral around its node, with corners
    at the means of the four nodes around them (extrapolated past the edge),
    and the overlap of each candidate pair (a k-d tree on bounding circles,
    SciPy's ``cKDTree``) is clipped on the host by Sutherland–Hodgman against
    the target cell, which must be convex. In the plane, coordinates are
    the grid's ``x`` and ``y`` as given (a swath across the dateline needs
    unwrapped longitudes); with ``sphere=True`` they are the longitude and
    latitude (λ, φ), cells have great-circle edges (as in
    ESMF), each pair is clipped in the gnomonic tangent plane of its target
    cell, and areas are spherical excesses (`spherical_polygon_area`). A 2-D
    `RectilinearGrid` on this path has its axes in (``x``, ``y``) order, as when it is
    a target read on a curvilinear source; a target cell is covered when its
    overlaps sum to its area to 1e-9 relative.

    The policies act on cells. Under ``outside="nan"`` a target cell not
    inside the source cells is invalid; ``"raise"`` raises instead;
    ``"extrapolate"`` keeps wᵢⱼ, so a partly covered cell reads its covered
    integral over its whole area (ESMF's ``destarea``); ``"clamp"`` cuts the
    target cell to the source domain, dividing by its covered area instead.
    Under ``missing="renormalise"`` each row is divided by its weight on
    finite samples, so a cell holds the mean over its valid source area
    (a coastal mask), and the area outside the source counts as missing too:
    a partly covered cell keeps the mean of its covered part (xrtoolz's
    ``fracarea``). Cells overlapping no source are invalid under every
    policy. ``boundary`` and ``extent`` do not apply.

    Algorithm:
        1. Read the edges of each axis of both grids.
        2. On each axis, find the band of source cells every target cell
           overlaps and their lengths |cⱼ ∩ Cᵢ| (two ``searchsorted``
           sweeps, `overlap_band`).
        3. Divide by |Cᵢ| and take the tensor product of the bands:
           S = ∏_d S_d taps per target over the axes d = 1, …, D, with S_d
           the most overlaps on axis d. On the
           polygon path, S is the most source cells any target cell overlaps,
           and shorter rows are padded with zero weights.

    Args:
        sphere: Whether the cells lie on the sphere, with great-circle edges
            and (λ, φ) coordinates.
        unit: ``"degrees"`` or ``"radians"``, the unit of (λ, φ); required
            with ``sphere=True`` and ``None`` otherwise.

    Raises:
        ValidationError: If ``unit`` is given without ``sphere=True``, or is
            missing or unknown with it.
        GeometryError: When planned, if a geometry is not an `Axis`, a
            `RectilinearGrid`, a north-up `AffineGrid` or a `CurvilinearGrid`,
            the two have different axis counts, a target cell on the polygon
            path is not convex, or a pair of cells spans a hemisphere.
        MissingExtraError: When planned on the polygon path without SciPy
            (``functax[scipy]``).
        ValidationError: When planned, if the edges are traced or a target
            cell is longer than a periodic source axis's period.
        OutsideDomainError: When planned under ``outside="raise"``, if a
            target cell is not inside the source cells.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, RectilinearGrid
        >>> from functax.interp import Conservative, Policies
        >>> # 1° to 2°: each target cell is the mean of two source cells
        >>> fine = Axis(jnp.arange(0.5, 4.0), edges=jnp.arange(0.0, 4.5))
        >>> coarse = Axis(jnp.array([1.0, 3.0]), edges=jnp.array([0.0, 2.0, 4.0]))
        >>> Conservative().plan(fine, coarse)(
        ...     jnp.array([1.0, 3.0, 5.0, 7.0])
        ... ).tolist()
        [2.0, 6.0]
        >>> # any factor: the integral is kept
        >>> source = RectilinearGrid((Axis(jnp.linspace(0.5, 5.5, 6)),) * 2)
        >>> target = RectilinearGrid(
        ...     (Axis(jnp.array([1.5, 4.5]), edges=jnp.array([0.0, 3.0, 6.0])),) * 2
        ... )
        >>> field = jnp.arange(36.0)
        >>> remapped = Conservative().plan(source, target)(field)
        >>> bool(jnp.allclose(9.0 * remapped.sum(), field.sum()))
        True
        >>> # a sheared swath onto a 2 × 2 grid of (x, y) cells
        >>> from functax.geometry import CurvilinearGrid
        >>> rows, cols = jnp.meshgrid(
        ...     jnp.arange(4.0) + 0.5, jnp.arange(4.0) + 0.5, indexing="ij"
        ... )
        >>> swath = CurvilinearGrid(x=cols + 0.1 * rows, y=rows)
        >>> grid = RectilinearGrid(
        ...     (Axis(jnp.array([1.0, 3.0])), Axis(jnp.array([1.0, 3.0])))
        ... )
        >>> plan = Conservative().plan(
        ...     swath, grid, policies=Policies(outside="clamp")
        ... )
        >>> plan.read.n_taps, plan.valid.tolist()
        (6, [True, True, True, True])
    """

    sphere: bool = eqx.field(static=True, default=False)
    unit: str | None = eqx.field(static=True, default=None)

    def __check_init__(self) -> None:
        if not self.sphere:
            if self.unit is not None:
                msg = f"'unit' must be None without sphere=True, got {self.unit!r}"
                raise ValidationError(msg)
            return
        if self.unit is None:
            msg = "'unit' must be 'degrees' or 'radians' with sphere=True, got None"
            raise ValidationError(msg)
        _check_unit(self.unit)

    def plan(
        self, source: Any, target: Any, *, policies: Policies | None = None
    ) -> Plan:
        """H for the cells of ``source`` averaged onto the cells of ``target``.

        Built on the host, so the geometries must be concrete.

        Args:
            source: An `Axis`, a `RectilinearGrid`, a north-up `AffineGrid`
                or a `CurvilinearGrid`; data on it is flattened row-major.
            target: A geometry of the same kind with as many axes, or a
                `CurvilinearGrid` or 2-D `RectilinearGrid` on the polygon
                path; a grid keeps its shape as ``target_shape``, so the plan
                cuts into windows of it.
            policies: The outside and missing rules (``Policies()`` by
                default).

        Returns:
            The `Plan`, an identity fit and a `Stencil` of S taps per target
            cell.

        Raises:
            GeometryError: If the geometries are unsupported or mismatched, a
                target cell on the polygon path is not convex, or a pair of
                cells spans a hemisphere.
            ValidationError: If the edges are traced or a target cell is
                longer than a periodic source axis's period.
            OutsideDomainError: Under ``outside="raise"`` when a target cell
                is not inside the source cells.
            MissingExtraError: On the polygon path without SciPy.
        """
        policies = Policies() if policies is None else policies
        require_no_repeats(source, "Conservative")
        require_no_repeats(target, "Conservative")
        curvilinear = isinstance(source, CurvilinearGrid) or isinstance(
            target, CurvilinearGrid
        )
        if self.sphere or curvilinear:
            stencil = polygon_stencil_of_cells(
                source, target, policies=policies, unit=self.unit
            )
        else:
            stencil = tensor_stencil_of_cells(source, target, policies=policies)
        grid_target = isinstance(target, RectilinearGrid | AffineGrid | CurvilinearGrid)
        target_shape = target.shape if grid_target else None
        return Plan(Identity(), stencil, policies, None, target_shape)
