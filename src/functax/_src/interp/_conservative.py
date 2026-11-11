"""First-order conservative remapping: area-weighted cell averages as one stencil.

Target cell Cᵢ reads every source cell cⱼ it overlaps with the weight
wᵢⱼ = |cⱼ ∩ Cᵢ| / |Cᵢ|, so ∑ᵢ |Cᵢ| ŷᵢ = ∑ⱼ |cⱼ| yⱼ whenever the target cells
cover the source cells. On tensor-product grids the overlap of two cells is
the product of their per-axis overlaps (`overlap_band`), so the stencil is the
tensor product of one band per axis. Curvilinear and spherical cells are
quadrilaterals whose pairwise overlaps are clipped on the host (`_clip`).

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


def _cell_axes(geometry: Any, name: str) -> list[tuple[np.ndarray, float | None]]:
    """The (edges, period) of each axis of a tensor-product grid of cells.

    An `AffineGrid` must be north-up (b = d = 0): its cells are then the
    tensor product of row edges f + e·i and column edges c + a·j.
    """
    if isinstance(geometry, Axis):
        return [(np.asarray(geometry.cell_edges()), geometry.period)]
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
        # y = f + e·i at row edge i, x = c + a·j at column edge j
        row_edges = f + e * np.arange(rows + 1, dtype=np.float64)
        col_edges = c + a * np.arange(cols + 1, dtype=np.float64)
        return [(row_edges, None), (col_edges, None)]
    msg = (
        f"'{name}' must be an Axis, a RectilinearGrid or an AffineGrid, got "
        f"{type(geometry).__name__}"
    )
    raise GeometryError(msg)


def _axis_weights(band: OverlapBand, policies: Policies) -> np.ndarray:
    """wᵢⱼ on one axis: |cⱼ ∩ Cᵢ| / |Cᵢ|, or over the covered length under clamp."""
    if policies.outside != "clamp":
        # wᵢⱼ = |cⱼ ∩ Cᵢ| / |Cᵢ|
        return band.lengths / band.target_lengths[:, None]
    # wᵢⱼ = |cⱼ ∩ Cᵢ| / ∑ⱼ |cⱼ ∩ Cᵢ|, the target cut to the source domain
    covered = band.lengths.sum(axis=1)
    safe_covered = np.where(covered > 0, covered, 1.0)
    return band.lengths / safe_covered[:, None]


# a target cell whose overlaps sum to its area to this relative tolerance is
# covered; clipped areas round in the last few digits
_COVERED_RTOL = 1e-9


def _policy_stencil(
    indices: np.ndarray,
    weights: np.ndarray,
    covered: np.ndarray,
    touched: np.ndarray,
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
    weights = np.where(valid[:, None], weights, np.zeros((), weights.dtype))
    return Stencil(
        jnp.asarray(indices),
        jnp.asarray(weights),
        n_source=n_source,
        valid=jnp.asarray(valid),
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
        # flat index ((j₁ n₂ + j₂) n₃ + j₃) …, weight w₁ w₂ w₃ …
        indices = einx.add("a s, b t -> (a b) (s t)", indices * n_cells, band.indices)
        weights = einx.multiply(
            "a s, b t -> (a b) (s t)", weights, _axis_weights(band, policies)
        )
        covered = einx.logical_and("a, b -> (a b)", covered, band.covered)
        touched = einx.logical_and("a, b -> (a b)", touched, band.touched)
    n_source = math.prod(source_shape)
    return _policy_stencil(indices, weights, covered, touched, n_source, policies)


def _node_corners(nodes: np.ndarray, sphere: bool) -> np.ndarray:
    """The (R + 1, C + 1, D) cell corners of (R, C, D) nodes, each cell's centre a node.

    The nodes are extended by one row and column on each side by linear
    extrapolation, x₋₁ = 2x₀ − x₁, and each corner is the mean of the four
    nodes around it, so a tensor grid gets the midpoints of `Axis.cell_edges`.
    On the sphere the means are of unit vectors, normalised, so the corners
    need no unwrapped longitudes.
    """
    padded = np.concatenate(
        [2 * nodes[:1] - nodes[1:2], nodes, 2 * nodes[-1:] - nodes[-2:-1]]
    )
    padded = np.concatenate(
        [
            2 * padded[:, :1] - padded[:, 1:2],
            padded,
            2 * padded[:, -1:] - padded[:, -2:-1],
        ],
        axis=1,
    )
    # the mean of the four nodes around each corner
    corners = 0.25 * (
        padded[:-1, :-1] + padded[1:, :-1] + padded[:-1, 1:] + padded[1:, 1:]
    )
    if sphere:
        corners = corners / np.linalg.norm(corners, axis=-1, keepdims=True)
    return corners


def _corner_cells(corners: np.ndarray) -> np.ndarray:
    """(R C, 4, D) cells, row-major, from (R + 1, C + 1, D) corners."""
    cells = np.stack(
        [corners[:-1, :-1], corners[1:, :-1], corners[1:, 1:], corners[:-1, 1:]], axis=2
    )
    return cells.reshape(-1, 4, corners.shape[-1])


def _polygon_cells(geometry: Any, name: str, unit: str | None) -> np.ndarray:
    """The quadrilateral cells of a 2-D grid: (x, y) corners, or unit vectors.

    ``unit`` is that of (λ, φ) on the sphere, and ``None`` in the plane.

    A `CurvilinearGrid` gives its cells by `_node_corners`; a 2-D
    `RectilinearGrid` has its axes in (x, y) order, (λ, φ) on the sphere, as
    a target read on a curvilinear source does.
    """
    if isinstance(geometry, CurvilinearGrid):
        nodes = np.stack([np.asarray(geometry.x), np.asarray(geometry.y)], axis=-1)
        if unit is not None:
            nodes = np.asarray(lonlat_to_unit(jnp.asarray(nodes), unit=unit))
        return _corner_cells(_node_corners(nodes, sphere=unit is not None))
    if isinstance(geometry, RectilinearGrid) and len(geometry.axes) == 2:
        x_edges, y_edges = (np.asarray(axis.cell_edges()) for axis in geometry.axes)
        if unit is not None:
            # a latitude edge past the pole would wrap over it
            pole = 90.0 if unit == "degrees" else math.pi / 2
            y_edges = np.clip(y_edges, -pole, pole)
        x, y = np.meshgrid(x_edges, y_edges, indexing="ij")
        corners = np.stack([x, y], axis=-1)
        if unit is not None:
            corners = np.asarray(lonlat_to_unit(jnp.asarray(corners), unit=unit))
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

    ``unit`` is that of (λ, φ) on the sphere, and ``None`` in the plane.

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
    # --- the pairs as rows of S taps, padded with zero weight ---
    order = np.argsort(overlaps.target, kind="stable")
    rows = overlaps.target[order]
    columns = overlaps.source[order]
    areas = overlaps.areas[order]
    counts = np.bincount(rows, minlength=n_target)
    n_taps = max(int(counts.max(initial=0)), 1)
    starts = np.cumsum(counts) - counts
    tap = np.arange(len(rows)) - starts[rows]
    indices = np.zeros((n_target, n_taps), np.int32)
    lengths = np.zeros((n_target, n_taps), areas.dtype)
    indices[rows, tap] = columns
    lengths[rows, tap] = areas
    # padding repeats the row's first index (0 for an empty row)
    padding = np.arange(n_taps)[None, :] >= counts[:, None]
    indices = np.where(padding, indices[:, :1], indices)
    # wᵢⱼ = |cⱼ ∩ Cᵢ| / |Cᵢ|, and the covered fraction ∑ⱼ wᵢⱼ
    target_areas = overlaps.target_areas
    real = target_areas > 0
    safe_areas = np.where(real, target_areas, 1.0)
    weights = lengths / safe_areas[:, None]
    fraction = weights.sum(axis=1)
    covered = real & (fraction >= 1.0 - _COVERED_RTOL)
    touched = real & (fraction > 0)
    if policies.outside == "clamp":
        # wᵢⱼ = |cⱼ ∩ Cᵢ| / ∑ⱼ |cⱼ ∩ Cᵢ|
        safe_fraction = np.where(touched, fraction, 1.0)
        weights = weights / safe_fraction[:, None]
    n_source = source_cells.shape[0]
    return _policy_stencil(indices, weights, covered, touched, n_source, policies)


class Conservative(Scheme):
    """First-order conservative remapping: each target cell's area-weighted mean.

    Target cell Cᵢ reads the source cells cⱼ it overlaps,

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
    (x, y) as given (a swath across the dateline needs unwrapped longitudes);
    with ``sphere=True`` they are (λ, φ), cells have great-circle edges (as in
    ESMF), each pair is clipped in the gnomonic tangent plane of its target
    cell, and areas are spherical excesses (`spherical_polygon_area`). A 2-D
    `RectilinearGrid` on this path has its axes in (x, y) order, as when it is
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
        3. Divide by |Cᵢ| and take the tensor product of the bands: S = ∏_d
           S_d taps per target, the most overlaps on each axis. On the
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
