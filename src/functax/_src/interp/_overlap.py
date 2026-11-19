"""One-axis cell overlaps, the building block of conservative remapping on tensor grids.

Source cell cⱼ = [eⱼ, eⱼ₊₁], j = 0, …, N − 1, between the N + 1 source edges e,
and target cell Cᵢ = [Eᵢ, Eᵢ₊₁], i = 0, …, M − 1, between the M + 1 target
edges E, overlap by

    |cⱼ ∩ Cᵢ| = max(0, min(eⱼ₊₁, Eᵢ₊₁) − max(eⱼ, Eᵢ)),

and on a D-dimensional tensor grid the overlap of two cells is the product of
their per-axis overlaps, so no polygon is ever clipped. Both sets of edges are
monotone, so the source cells a target overlaps are contiguous and two
``searchsorted`` sweeps find them; the result is a band of S taps per target
(S the most source cells any target overlaps) rather than a dense (M, N)
matrix. Built on the host in NumPy.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import dataclasses

import einx
import numpy as np
from jaxtyping import ArrayLike, Bool, Float, Int

from functax._src._backend import is_tracer
from functax.exceptions import ValidationError


__all__ = [
    "OverlapBand",
    "overlap_band",
    "overlap_weights_1d",
]


# a source whose edges span the period P to this relative tolerance covers the circle
_FULL_CIRCLE_RTOL = 1e-12


@dataclasses.dataclass(frozen=True)
class OverlapBand:
    """The overlaps of M target cells with the source cells on one axis.

    Row i holds the S source cells target i overlaps, padded with zero lengths
    at the row's first index (index 0 for a row with no overlap), so a NaN
    reaches a row only through a source it overlaps.

    Attributes:
        indices: j, the source cell of each tap.
        lengths: |cⱼ ∩ Cᵢ| of each tap.
        target_lengths: |Cᵢ|.
        covered: Whether Cᵢ lies inside the source cells.

    Examples:
        Two targets of length 1.5 over three unit source cells:

        >>> import numpy as np
        >>> from functax._src.interp._overlap import overlap_band
        >>> band = overlap_band(
        ...     np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.5, 3.0])
        ... )
        >>> band.indices.shape
        (2, 2)
        >>> band.lengths.tolist()
        [[1.0, 0.5], [0.5, 1.0]]

        A target past the source cells is neither covered nor touched:

        >>> band = overlap_band(
        ...     np.array([0.0, 1.0, 2.0]), np.array([1.5, 2.5, 3.5])
        ... )
        >>> band.covered.tolist(), band.touched.tolist()
        ([False, False], [True, False])
    """

    indices: Int[np.ndarray, "target tap"]
    lengths: Float[np.ndarray, "target tap"]
    target_lengths: Float[np.ndarray, " target"]
    covered: Bool[np.ndarray, " target"]

    @property
    def touched(self) -> Bool[np.ndarray, " target"]:
        """Whether Cᵢ overlaps any source cell."""
        # ∑ⱼ |cⱼ ∩ Cᵢ|
        row_length = einx.sum("target [tap]", self.lengths)
        return row_length > 0


def _host_edges(edges: ArrayLike, name: str) -> Float[np.ndarray, " edge"]:
    """``edges`` as a 1-D, floating, finite, strictly monotone NumPy array."""
    if is_tracer(edges):
        msg = (
            f"'{name}' is traced; cell overlaps are computed on the host when "
            "the plan is built, so build it outside jit"
        )
        raise ValidationError(msg)
    host = np.asarray(edges)
    if host.ndim != 1 or host.shape[0] < 2:
        msg = f"'{name}' must be 1-D with at least two edges, got shape {host.shape}"
        raise ValidationError(msg)
    if not np.issubdtype(host.dtype, np.floating):
        msg = f"'{name}' must have a floating dtype, got {host.dtype}"
        raise ValidationError(msg)
    # Δeⱼ = eⱼ₊₁ − eⱼ
    spacing = np.diff(host)
    monotone = np.all(spacing > 0) or np.all(spacing < 0)
    if not (np.all(np.isfinite(host)) and monotone):
        msg = f"'{name}' must be finite and strictly monotone, got {host!r}"
        raise ValidationError(msg)
    return host


def _intervals(
    edges: Float[np.ndarray, " edge"],
) -> tuple[Float[np.ndarray, " cell"], Float[np.ndarray, " cell"]]:
    """(lo, hi) of each cell [eⱼ, eⱼ₊₁], with lo < hi whichever way e runs."""
    # loⱼ = min(eⱼ, eⱼ₊₁)
    lower = np.minimum(edges[:-1], edges[1:])
    # hiⱼ = max(eⱼ, eⱼ₊₁)
    upper = np.maximum(edges[:-1], edges[1:])
    return lower, upper


def overlap_band(
    source_edges: ArrayLike,
    target_edges: ArrayLike,
    *,
    period: float | None = None,
) -> OverlapBand:
    """The band of overlaps |cⱼ ∩ Cᵢ| between two sets of cells on one axis.

    |cⱼ ∩ Cᵢ| = max(0, min(eⱼ₊₁, Eᵢ₊₁) − max(eⱼ, Eᵢ)), kept only on the S
    source cells j that each target i overlaps.

    Algorithm:
        1. Turn each set of edges into (lo, hi) intervals and sort the source
           ones by lo; monotone edges keep them disjoint, so both lo and hi
           are then sorted.
        2. On a periodic axis, shift each target to start in
           [e_min, e_min + P) and tile the source once more at +P, so a target
           across the seam reads both ends.
        3. For each target, the first source with hi > Eᵢ and the first with
           lo ≥ Eᵢ₊₁ (two ``searchsorted`` sweeps) bound its S taps.

    Args:
        source_edges: e, the N + 1 source edges, monotone either way.
        target_edges: E, the M + 1 target edges, monotone either way.
        period: P, the period of the source axis, or ``None``.

    Returns:
        The `OverlapBand`, in the result dtype of the two edge arrays.

    Raises:
        ValidationError: If either edge array is traced, not 1-D with two
            edges or more, not floating, or not finite and strictly monotone;
            if the source edges span more than P or a target cell is longer
            than P.

    Examples:
        Two targets of length 1.5 over three unit source cells:

        >>> import numpy as np
        >>> from functax._src.interp._overlap import overlap_band
        >>> band = overlap_band(
        ...     np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.5, 3.0])
        ... )
        >>> band.indices.tolist()
        [[0, 1], [1, 2]]

        A target across the seam of a periodic axis reads both ends:

        >>> band = overlap_band(
        ...     np.arange(0.0, 361.0, 90.0), np.array([-45.0, 45.0]), period=360.0
        ... )
        >>> band.indices.tolist(), band.lengths.tolist()
        ([[3, 0]], [[45.0, 45.0]])
    """
    # --- intervals ---
    source = _host_edges(source_edges, "source_edges")
    target = _host_edges(target_edges, "target_edges")
    dtype = np.result_type(source, target)
    source = source.astype(dtype)
    target = target.astype(dtype)
    source_lower, source_upper = _intervals(source)
    order = np.argsort(source_lower, kind="stable")
    cell_lower = source_lower[order]
    cell_upper = source_upper[order]
    cell_index = order
    target_lower, target_upper = _intervals(target)
    # |Cᵢ| = Eᵢ₊₁ − Eᵢ
    target_lengths = target_upper - target_lower
    domain_lower = cell_lower[0]
    domain_upper = cell_upper[-1]

    # --- period ---
    if period is None:
        # Eᵢ ≥ e_min
        above_lower = target_lower >= domain_lower
        # Eᵢ₊₁ ≤ e_max
        below_upper = target_upper <= domain_upper
        covered = above_lower & below_upper
    else:
        # e_max − e_min
        span = domain_upper - domain_lower
        if span > period * (1.0 + _FULL_CIRCLE_RTOL):
            msg = f"'source_edges' must span at most the period {period}, got {span}"
            raise ValidationError(msg)
        if np.any(target_lengths > period):
            msg = f"'target_edges' must hold cells no longer than the period {period}"
            raise ValidationError(msg)
        # Eᵢ − e_min
        offset = target_lower - domain_lower
        # (Eᵢ − e_min) mod P
        wrapped = np.mod(offset, period)
        # Eᵢ ↦ e_min + ((Eᵢ − e_min) mod P), the length kept
        shifted_lower = domain_lower + wrapped
        # Eᵢ₊₁ ↦ Eᵢ + |Cᵢ|
        target_upper = shifted_lower + target_lengths
        target_lower = shifted_lower
        full_circle = span >= period * (1.0 - _FULL_CIRCLE_RTOL)
        all_covered = np.full(target_lengths.shape, full_circle)
        # Eᵢ₊₁ ≤ e_max
        below_upper = target_upper <= domain_upper
        covered = all_covered | below_upper
        # the source again, one period on
        shifted_cell_lower = cell_lower + period
        shifted_cell_upper = cell_upper + period
        cell_lower = einx.id("a, b -> (a + b)", cell_lower, shifted_cell_lower)
        cell_upper = einx.id("a, b -> (a + b)", cell_upper, shifted_cell_upper)
        cell_index = einx.id("a, b -> (a + b)", cell_index, cell_index)

    # --- taps ---
    # the first cell with hi > Eᵢ, and the first with lo ≥ Eᵢ₊₁
    start = np.searchsorted(cell_upper, target_lower, side="right")
    stop = np.searchsorted(cell_lower, target_upper, side="left")
    # Sᵢ = max(stopᵢ − startᵢ, 0)
    span_taps = stop - start
    counts = np.maximum(span_taps, 0)
    # S = max(maxᵢ Sᵢ, 1)
    most_taps = einx.max("[target]", counts)
    n_taps = max(int(most_taps), 1)
    tap = np.arange(n_taps)
    # every (target, tap) pair, the taps past a row's count clipped and dropped
    last_cell = cell_lower.shape[0] - 1
    unclipped = einx.add("target, tap -> target tap", start, tap)
    position = np.minimum(unclipped, last_cell)
    in_row = einx.less("tap, target -> target tap", tap, counts)

    # --- lengths ---
    tap_upper = einx.get_at("[cell], target tap -> target tap", cell_upper, position)
    tap_lower = einx.get_at("[cell], target tap -> target tap", cell_lower, position)
    # min(eⱼ₊₁, Eᵢ₊₁)
    upper = einx.minimum("target tap, target -> target tap", tap_upper, target_upper)
    # max(eⱼ, Eᵢ)
    lower = einx.maximum("target tap, target -> target tap", tap_lower, target_lower)
    # |cⱼ ∩ Cᵢ| = min(eⱼ₊₁, Eᵢ₊₁) − max(eⱼ, Eᵢ), positive on every tap in the row
    overlap = upper - lower
    zero = np.zeros((), dtype)
    lengths = einx.where("target tap, target tap, -> target tap", in_row, overlap, zero)

    # --- indices ---
    indices = einx.get_at("[cell], target tap -> target tap", cell_index, position)
    # padding repeats the row's first index (0 for an empty row)
    has_taps = counts > 0
    first = np.where(has_taps, indices[:, 0], 0)
    padded = einx.where(
        "target tap, target tap, target -> target tap", in_row, indices, first
    )
    indices = padded.astype(np.int32)
    return OverlapBand(indices, lengths, target_lengths, covered)


def overlap_weights_1d(
    source_edges: Float[ArrayLike, " source_edge"],
    target_edges: Float[ArrayLike, " target_edge"],
    *,
    period: float | None = None,
) -> Float[np.ndarray, "target source"]:
    """The (M, N) overlap lengths |cⱼ ∩ Cᵢ| of every target and source cell.

    |cⱼ ∩ Cᵢ| = max(0, min(eⱼ₊₁, Eᵢ₊₁) − max(eⱼ, Eᵢ)) for source cell
    cⱼ = [eⱼ, eⱼ₊₁] and target cell Cᵢ = [Eᵢ, Eᵢ₊₁]; the conservative
    weight is wᵢⱼ = |cⱼ ∩ Cᵢ| / |Cᵢ|. Edges may run either way (each cell is
    read as [lo, hi]), and on a periodic source axis a target is read modulo
    P, so a target across the seam overlaps both ends. Equal to xrtoolz's
    ``overlap_weights_1d`` on a non-periodic axis. Dense, for inspection and
    tests: `Conservative` keeps the band of `overlap_band` instead.

    Args:
        source_edges: e, the N + 1 source edges.
        target_edges: E, the M + 1 target edges.
        period: P, the period of the source axis, or ``None``.

    Returns:
        The non-negative (M, N) lengths, in the result dtype of the edges.

    Raises:
        ValidationError: If either edge array is traced, not 1-D with two
            edges or more, not floating, or not finite and strictly monotone;
            if the source edges span more than P or a target cell is longer
            than P.

    Examples:
        >>> import numpy as np
        >>> from functax.interp import overlap_weights_1d
        >>> overlap_weights_1d(
        ...     np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.5, 3.0])
        ... )
        array([[1. , 0.5, 0. ],
               [0. , 0.5, 1. ]])
        >>> # a target across the seam of a periodic axis reads both ends
        >>> overlap_weights_1d(
        ...     np.arange(0.0, 361.0, 90.0), np.array([-45.0, 45.0]), period=360.0
        ... ).tolist()
        [[45.0, 0.0, 0.0, 45.0]]
    """
    band = overlap_band(source_edges, target_edges, period=period)
    n_source = np.asarray(source_edges).shape[0] - 1
    n_target = band.indices.shape[0]
    dense = np.zeros((n_target, n_source), dtype=band.lengths.dtype)
    # Wᵢⱼ = ∑_taps |cⱼ ∩ Cᵢ|; repeated indices add, as a target round the whole
    # circle reads one cell twice
    return einx.add_at(
        "target [source], target tap, target tap -> target [source]",
        dense,
        band.indices,
        band.lengths,
    )
