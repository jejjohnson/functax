"""First-order conservative remapping: area-weighted cell averages as one stencil.

Target cell Cᵢ reads every source cell cⱼ it overlaps with the weight
wᵢⱼ = |cⱼ ∩ Cᵢ| / |Cᵢ|, so ∑ᵢ |Cᵢ| ŷᵢ = ∑ⱼ |cⱼ| yⱼ whenever the target cells
cover the source cells. On tensor-product grids the overlap of two cells is
the product of their per-axis overlaps (`overlap_band`), so the stencil is the
tensor product of one band per axis.

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
import jax.numpy as jnp
import numpy as np

from functax._src.geometry._types import AffineGrid, Axis, RectilinearGrid
from functax._src.interp._fit import Identity
from functax._src.interp._locate import check_outside
from functax._src.interp._overlap import OverlapBand, overlap_band
from functax._src.interp._plan import Plan
from functax._src.interp._policies import Policies
from functax._src.interp._schemes import Scheme
from functax._src.interp._stencil import Stencil
from functax.exceptions import GeometryError


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


def conservative_stencil(
    source: Any, target: Any, *, policies: Policies
) -> tuple[Stencil, np.ndarray]:
    """H of conservative remapping between tensor grids, and each row's full cover.

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
    check_outside(covered, policies)
    # renormalising counts the area outside the source as missing, so a
    # partly covered cell keeps the mean of its covered, valid part
    keep_partial = (
        policies.outside in ("clamp", "extrapolate")
        or policies.missing == "renormalise"
    )
    valid = touched if keep_partial else covered
    weights = np.where(valid[:, None], weights, np.zeros((), weights.dtype))
    stencil = Stencil(
        jnp.asarray(indices),
        jnp.asarray(weights),
        n_source=math.prod(source_shape),
        valid=jnp.asarray(valid),
        check=False,
    )
    return stencil, covered


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
           S_d taps per target, the most overlaps on each axis.

    Raises:
        GeometryError: When planned, if a geometry is not an `Axis`, a
            `RectilinearGrid` or a north-up `AffineGrid`, or the two have
            different axis counts.
        ValidationError: When planned, if the edges are traced or a target
            cell is longer than a periodic source axis's period.
        OutsideDomainError: When planned under ``outside="raise"``, if a
            target cell is not inside the source cells.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, RectilinearGrid
        >>> from functax.interp import Conservative
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
    """

    def plan(
        self, source: Any, target: Any, *, policies: Policies | None = None
    ) -> Plan:
        """H for the cells of ``source`` averaged onto the cells of ``target``.

        Built on the host, so the geometries must be concrete.

        Args:
            source: An `Axis`, a `RectilinearGrid` or a north-up `AffineGrid`;
                data on it is flattened row-major.
            target: The same kind of geometry, with as many axes; a grid
                keeps its shape as ``target_shape``, so the plan cuts into
                windows of it.
            policies: The outside and missing rules (``Policies()`` by
                default).

        Returns:
            The `Plan`, an identity fit and a `Stencil` of S taps per target
            cell.

        Raises:
            GeometryError: If the geometries are unsupported or mismatched.
            ValidationError: If the edges are traced or a target cell is
                longer than a periodic source axis's period.
            OutsideDomainError: Under ``outside="raise"`` when a target cell
                is not inside the source cells.
        """
        policies = Policies() if policies is None else policies
        stencil, _ = conservative_stencil(source, target, policies=policies)
        grid_target = isinstance(target, RectilinearGrid | AffineGrid)
        target_shape = target.shape if grid_target else None
        return Plan(Identity(), stencil, policies, None, target_shape)
