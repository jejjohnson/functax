"""Locate targets on an axis, and remap the taps that run off its ends.

Every grid scheme first turns a target coordinate x into a fractional index
position u on each axis, so that node j sits at u = j. On a regular axis with
origin x₀ and spacing Δ, u = (x − x₀)/Δ; on a rectilinear axis, u = i + t
with cell i from ``searchsorted`` and t = (x − xᵢ)/(xᵢ₊₁ − xᵢ), extended
linearly past the end cells. A descending axis is located on its reversed
values; a periodic axis of period P first maps x into one period and gains a
wrap cell between its last node and the first one plus P. An axis of one node
x₀ has no spacing: u = 0 at x₀, and its edges, when it has them, sit at
u = ∓½.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from functax._src._backend import is_tracer
from functax._src.geometry._types import Axis
from functax._src.interp._policies import BOUNDARY, EXTENT, Policies
from functax.exceptions import GeometryError, OutsideDomainError, PolicyError


__all__ = [
    "check_outside",
    "domain_bounds",
    "index_positions",
    "locate",
    "remap_taps",
    "require_cell_widths",
]


def _ascending(axis: Axis) -> tuple[jax.Array, bool | jax.Array]:
    """The node values in increasing order, and whether they were reversed."""
    values = axis.values
    if is_tracer(values):
        # a traced axis cannot say which way it runs; compare its ends
        descending = values[0] > values[-1]
        ascending = jnp.where(descending, values[::-1], values)
        return ascending, descending
    # np.asarray first: under jit, indexing a concrete JAX array stages a
    # traced slice, which bool() cannot read
    ends = np.asarray(values)[[0, -1]]
    descending = bool(ends[0] > ends[1])
    return (values[::-1] if descending else values), descending


def _rectilinear_positions(
    nodes: Float[Array, " node"], positions: Float[Array, "*target"]
) -> Float[Array, "*target"]:
    """u = i + (x − xᵢ)/(xᵢ₊₁ − xᵢ) on increasing nodes, linear past the ends."""
    n_nodes = nodes.shape[0]
    # i = searchsorted(x, side="right") − 1, kept to a real cell [0, N − 2]
    cell = jnp.searchsorted(nodes, positions, side="right") - 1
    cell = jnp.clip(cell, 0, n_nodes - 2)
    left = nodes[cell]
    right = nodes[cell + 1]
    # t = (x − xᵢ)/(xᵢ₊₁ − xᵢ), outside [0, 1] beyond the end cells
    offset = (positions - left) / (right - left)
    return cell + offset


def _one_node_positions(
    axis: Axis, positions: Float[Array, "*target"]
) -> Float[Array, "*target"]:
    """u on an axis of one node x₀: 0 at x₀, ∓½ at its edges, linear in between.

    Without edges, u = sign(x − x₀), so only x₀ lies in the domain [0, 0] of
    ``extent="nodes"``. With edges [e₀, e₁] in either order, u = ±½ |x − x₀|/w
    on the side of x, w being the distance from x₀ to that side's edge, and
    u = ±1 past a side of zero width. ±∞ and NaN stay as they are, so a
    non-finite target stays invalid under every policy.
    """
    node = axis.values[0]
    offset = positions - node
    # sign(±∞) would be finite, so a non-finite target keeps its position
    finite = jnp.isfinite(offset)
    if axis.edges is None:
        return jnp.where(finite, jnp.sign(offset), offset)
    lower = jnp.minimum(axis.edges[0], axis.edges[1])
    upper = jnp.maximum(axis.edges[0], axis.edges[1])
    below = offset < 0
    # w, from x₀ to the edge on the side of x
    width = jnp.where(below, node - lower, upper - node)
    has_width = width > 0
    safe_width = jnp.where(has_width, width, jnp.ones((), width.dtype))
    distance = jnp.abs(offset)
    # ½ |x − x₀|/w, or sign |x − x₀| (0 or 1, ∞ and NaN kept) past a zero width
    past = jnp.where(finite, jnp.sign(distance), distance)
    magnitude = jnp.where(has_width, 0.5 * distance / safe_width, past)
    return jnp.where(below, -magnitude, magnitude)


def require_cell_widths(source: Any, extent: str) -> None:
    """Raise when ``extent="cells"`` meets a one-node axis without edges.

    A single node has no spacing, so the width of its cell must come from
    ``edges``; every other axis has its cells from its nodes.

    Args:
        source: The source geometry; only an `Axis` or the axes of a
            `RectilinearGrid` can have one node.
        extent: The plan's ``extent``.

    Raises:
        GeometryError: If ``extent`` is ``"cells"`` and an axis of ``source``
            has one node and no ``edges``.
    """
    if extent != "cells":
        return
    axes = (source,) if isinstance(source, Axis) else getattr(source, "axes", ())
    for index, axis in enumerate(axes):
        if axis.values.shape[0] == 1 and axis.edges is None:
            msg = (
                f"axis {index} has one node and no 'edges', so its cell has no "
                "width under extent='cells'; pass Axis(..., edges=[e0, e1])"
            )
            raise GeometryError(msg)


def index_positions(
    axis: Axis, positions: Float[Array, "*target"]
) -> Float[Array, "*target"]:
    """The fractional index position u of each target, so node j sits at u = j.

    u = (x − x₀)/Δ on a regular axis and i + (x − xᵢ)/(xᵢ₊₁ − xᵢ) on a
    rectilinear one, continued linearly past the end cells, so u < 0 or
    u > N − 1 outside. On a periodic axis x first maps into one period, and u
    lies in [0, N) on an increasing axis, the cell (N − 1, N) being the wrap
    cell between the last node and the first one plus P. On an axis of one
    node x₀, u = 0 at x₀ and ∓½ at its edges (``sign(x − x₀)`` without edges).

    Args:
        axis: The source axis, increasing or decreasing, maybe periodic.
        positions: x, the target coordinates, any shape.

    Returns:
        u, the shape of ``positions``, in its dtype promoted with the axis'.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis
        >>> from functax.interp import index_positions
        >>> index_positions(
        ...     Axis(jnp.array([0.0, 1.0, 3.0])), jnp.array([0.5, 2.0, 4.0])
        ... ).tolist()
        [0.5, 1.5, 2.5]
        >>> index_positions(
        ...     Axis(jnp.array([3.0, 2.0, 1.0])), jnp.array([2.5])
        ... ).tolist()
        [0.5]
    """
    positions = jnp.asarray(positions)
    n_nodes = axis.values.shape[0]
    nodes, descending = _ascending(axis)
    if axis.period is not None:
        origin = nodes[0]
        # x′ = x₀ + ((x − x₀) mod P), then the wrap cell x_{N−1} → x₀ + P
        wrapped = origin + jnp.mod(positions - origin, axis.period)
        extended = jnp.append(nodes, origin + axis.period)
        positions_ascending = _rectilinear_positions(extended, wrapped)
    elif n_nodes == 1:
        return _one_node_positions(axis, positions)
    elif not is_tracer(axis.values) and axis.is_regular:
        # u = (x − x₀)/Δ, with Δ from the end nodes
        spacing = (nodes[-1] - nodes[0]) / (n_nodes - 1)
        positions_ascending = (positions - nodes[0]) / spacing
    else:
        positions_ascending = _rectilinear_positions(nodes, positions)
    # a decreasing axis: u = (N − 1) − u_ascending
    reversed_positions = (n_nodes - 1) - positions_ascending
    if isinstance(descending, bool):
        return reversed_positions if descending else positions_ascending
    return jnp.where(descending, reversed_positions, positions_ascending)


def domain_bounds(n_nodes: int, extent: str) -> tuple[float, float]:
    """[lower, upper] of u inside the domain of an axis of N nodes.

    ``"nodes"`` is [0, N − 1]; ``"cells"`` is [−½, N − ½].

    Raises:
        PolicyError: If ``extent`` is unknown.
    """
    if extent not in EXTENT:
        msg = f"'extent' must be one of {list(EXTENT)}, got {extent!r}"
        raise PolicyError(msg)
    if extent == "cells":
        return (-0.5, n_nodes - 0.5)
    return (0.0, float(n_nodes - 1))


def locate(
    axis: Axis,
    positions: Float[Array, " target"],
    *,
    extent: str = "nodes",
) -> tuple[Int[Array, " target"], Float[Array, " target"], Bool[Array, " target"]]:
    """The cell index i, the offset t ∈ [0, 1) and the inside flag of each target.

    i = ⌊u⌋ and t = u − i for the index position u of `index_positions`.
    A target is inside when u lies in the domain of ``extent`` and is finite; a
    periodic axis is inside everywhere. JAX clamps out-of-range gathers
    silently, so a scheme reads ``inside`` rather than trusting i.

    Args:
        axis: The source axis.
        positions: x, the M target coordinates.
        extent: ``"nodes"`` or ``"cells"`` (see `Policies`).

    Returns:
        ``(cell, offset, inside)``, each of length M.

    Raises:
        PolicyError: If ``extent`` is unknown.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis
        >>> from functax.interp import locate
        >>> cell, offset, inside = locate(
        ...     Axis(jnp.arange(4.0)), jnp.array([1.25, 3.0, 3.25])
        ... )
        >>> cell.tolist(), offset.tolist(), inside.tolist()
        ([1, 3, 3], [0.25, 0.0, 0.25], [True, True, False])
    """
    lower, upper = domain_bounds(axis.values.shape[0], extent)
    position = index_positions(axis, positions)
    cell_float = jnp.floor(position)
    offset = position - cell_float
    cell = cell_float.astype(jnp.int32)
    finite = jnp.isfinite(position)
    if axis.period is not None:
        inside = finite
    else:
        inside = finite & (position >= lower) & (position <= upper)
    return cell, offset, inside


def remap_taps(
    taps: Int[Array, "target tap"],
    n_nodes: int,
    boundary: str,
    *,
    periodic: bool = False,
) -> tuple[Int[Array, "target tap"], Bool[Array, "target tap"]]:
    """Bring tap indices back onto [0, N − 1] by a boundary rule.

    mirror reflects about the end samples, j ↦ −j for j < 0 and
    j ↦ 2(N − 1) − j for j > N − 1, repeated with period 2(N − 1); wrap is
    j mod N; edge clips to [0, N − 1]; constant clips too, but marks the tap
    as dropped, so its weight becomes 0. A periodic axis always wraps. These
    match ``scipy.ndimage`` modes ``"mirror"``, ``"grid-wrap"``,
    ``"nearest"`` and ``"grid-constant"``.

    Args:
        taps: j, the tap indices, any integers.
        n_nodes: N, the number of nodes on the axis.
        boundary: ``"mirror"``, ``"wrap"``, ``"edge"`` or ``"constant"``.
        periodic: Whether the axis is periodic, which forces wrap.

    Returns:
        ``(indices, kept)``: indices in [0, N − 1], and False where a
        ``"constant"`` boundary dropped the tap.

    Raises:
        PolicyError: If ``boundary`` is unknown.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.interp import remap_taps
        >>> taps = jnp.array([[-2, -1, 4, 5]])
        >>> remap_taps(taps, 4, "mirror")[0].tolist()
        [[2, 1, 2, 1]]
        >>> remap_taps(taps, 4, "wrap")[0].tolist()
        [[2, 3, 0, 1]]
        >>> remap_taps(taps, 4, "constant")[1].tolist()
        [[False, False, False, False]]
    """
    if boundary not in BOUNDARY:
        msg = f"'boundary' must be one of {list(BOUNDARY)}, got {boundary!r}"
        raise PolicyError(msg)
    taps = jnp.asarray(taps)
    in_range = (taps >= 0) & (taps <= n_nodes - 1)
    all_kept = jnp.ones(taps.shape, dtype=bool)
    if periodic or boundary == "wrap":
        return jnp.mod(taps, n_nodes), all_kept
    if boundary == "edge":
        return jnp.clip(taps, 0, n_nodes - 1), all_kept
    if boundary == "constant":
        return jnp.clip(taps, 0, n_nodes - 1), in_range
    if n_nodes == 1:
        return jnp.zeros_like(taps), all_kept
    # mirror: j mod 2(N − 1), then fold the second half back
    period = 2 * (n_nodes - 1)
    folded = jnp.mod(taps, period)
    mirrored = jnp.where(folded > n_nodes - 1, period - folded, folded)
    return mirrored, all_kept


def check_outside(inside: ArrayLike, policies: Policies) -> None:
    """Enforce ``outside="raise"`` on the host; no-op for every other policy.

    Args:
        inside: The inside flags of the targets.
        policies: The plan's policies.

    Raises:
        PolicyError: Under ``outside="raise"`` on traced targets, which cannot
            be checked before they run.
        OutsideDomainError: Under ``outside="raise"`` when a target is outside.
    """
    if policies.outside != "raise":
        return
    if is_tracer(inside):
        msg = (
            "outside='raise' checks targets when the plan is built, which needs "
            "concrete targets; under jit use 'nan', 'clamp' or 'extrapolate'"
        )
        raise PolicyError(msg)
    flags = np.asarray(inside)
    if flags.all():
        return
    outside = np.flatnonzero(~flags)
    msg = (
        f"{outside.size} of {flags.size} targets lie outside the domain under "
        f"outside='raise'; the first is target {int(outside[0])}"
    )
    raise OutsideDomainError(msg)
