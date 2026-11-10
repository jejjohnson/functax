"""One-axis cell overlaps, the building block of conservative remapping on tensor grids.

Source cell cⱼ = [eⱼ, eⱼ₊₁] and target cell Cᵢ = [Eᵢ, Eᵢ₊₁] overlap by

    |cⱼ ∩ Cᵢ| = max(0, min(eⱼ₊₁, Eᵢ₊₁) − max(eⱼ, Eᵢ)),

and on a D-dimensional tensor grid the overlap of two cells is the product of
their per-axis overlaps, so no polygon is ever clipped. Both sets of edges are
monotone, so the source cells a target overlaps are contiguous and two
``searchsorted`` sweeps find them; the result is a band of S taps per target
rather than a dense (M, N) matrix. Built on the host in NumPy.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import dataclasses
from typing import Any

import numpy as np
from jaxtyping import ArrayLike, Bool, Float, Int

from functax._src._backend import is_tracer
from functax.exceptions import ValidationError


__all__ = [
    "OverlapBand",
    "overlap_band",
    "overlap_weights_1d",
]


# a source whose edges span P to this relative tolerance covers the circle
_FULL_CIRCLE_RTOL = 1e-12


@dataclasses.dataclass(frozen=True)
class OverlapBand:
    """The overlaps of M target cells with the source cells on one axis.

    Row i holds the S source cells target i overlaps, padded with zero lengths
    at the row's first index (index 0 for a row with no overlap), so a NaN
    reaches a row only through a source it overlaps.

    Attributes:
        indices: j, the source cell of each tap, (M, S).
        lengths: |cⱼ ∩ Cᵢ| of each tap, (M, S).
        target_lengths: |Cᵢ|, (M,).
        covered: Whether Cᵢ lies inside the source cells, (M,).
    """

    indices: Int[np.ndarray, "target tap"]
    lengths: Float[np.ndarray, "target tap"]
    target_lengths: Float[np.ndarray, " target"]
    covered: Bool[np.ndarray, " target"]

    @property
    def touched(self) -> Bool[np.ndarray, " target"]:
        """Whether Cᵢ overlaps any source cell, (M,)."""
        return self.lengths.sum(axis=1) > 0


def _host_edges(edges: Any, name: str) -> np.ndarray:
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
    spacing = np.diff(host)
    monotone = np.all(spacing > 0) or np.all(spacing < 0)
    if not (np.all(np.isfinite(host)) and monotone):
        msg = f"'{name}' must be finite and strictly monotone, got {host!r}"
        raise ValidationError(msg)
    return host


def _intervals(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) of each cell [eⱼ, eⱼ₊₁], with lo < hi whichever way e runs."""
    lower = np.minimum(edges[:-1], edges[1:])
    upper = np.maximum(edges[:-1], edges[1:])
    return lower, upper


def overlap_band(
    source_edges: ArrayLike,
    target_edges: ArrayLike,
    *,
    period: float | None = None,
) -> OverlapBand:
    """The band of overlaps |cⱼ ∩ Cᵢ| between two sets of cells on one axis.

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
    """
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
    if period is None:
        covered = (target_lower >= domain_lower) & (target_upper <= domain_upper)
    else:
        span = domain_upper - domain_lower
        if span > period * (1.0 + _FULL_CIRCLE_RTOL):
            msg = f"'source_edges' must span at most the period {period}, got {span}"
            raise ValidationError(msg)
        if np.any(target_lengths > period):
            msg = f"'target_edges' must hold cells no longer than the period {period}"
            raise ValidationError(msg)
        # Eᵢ ↦ e_min + ((Eᵢ − e_min) mod P), the length kept
        shifted_lower = domain_lower + np.mod(target_lower - domain_lower, period)
        target_upper = shifted_lower + target_lengths
        target_lower = shifted_lower
        full_circle = span >= period * (1.0 - _FULL_CIRCLE_RTOL)
        covered = np.full(target_lengths.shape, full_circle) | (
            target_upper <= domain_upper
        )
        # the source again, one period on
        cell_lower = np.concatenate([cell_lower, cell_lower + period])
        cell_upper = np.concatenate([cell_upper, cell_upper + period])
        cell_index = np.concatenate([cell_index, cell_index])
    # the first cell with hi > Eᵢ, and the first with lo ≥ Eᵢ₊₁
    start = np.searchsorted(cell_upper, target_lower, side="right")
    stop = np.searchsorted(cell_lower, target_upper, side="left")
    counts = np.maximum(stop - start, 0)
    n_taps = max(int(counts.max()), 1)
    tap = np.arange(n_taps)
    # every (target, tap) pair, the taps past a row's count clipped and dropped
    position = np.minimum(start[:, None] + tap, cell_lower.shape[0] - 1)
    in_row = tap < counts[:, None]
    # |cⱼ ∩ Cᵢ| = min(eⱼ₊₁, Eᵢ₊₁) − max(eⱼ, Eᵢ), positive on every tap in the row
    upper = np.minimum(cell_upper[position], target_upper[:, None])
    lower = np.maximum(cell_lower[position], target_lower[:, None])
    lengths = np.where(in_row, upper - lower, np.zeros((), dtype))
    indices = cell_index[position]
    # padding repeats the row's first index (0 for an empty row)
    first = np.where(counts > 0, indices[:, 0], 0)
    indices = np.where(in_row, indices, first[:, None]).astype(np.int32)
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
    n_target, n_taps = band.indices.shape
    dense = np.zeros((n_target, n_source), dtype=band.lengths.dtype)
    rows = np.repeat(np.arange(n_target), n_taps)
    # repeated indices add, as a target round the whole circle reads one cell twice
    np.add.at(dense, (rows, band.indices.ravel()), band.lengths.ravel())
    return dense
