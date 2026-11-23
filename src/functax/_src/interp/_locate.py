"""Locate targets on an axis, and remap the taps that run off its ends.

Every grid scheme first turns a target coordinate x into a fractional index
position u on each axis of N nodes x₀, …, x_{N−1}, so that node j sits at
u = j. On a regular axis with origin x₀ and spacing Δ, u = (x − x₀)/Δ; on a
rectilinear axis, u = i + t with cell i from ``searchsorted`` and
t = (x − xᵢ)/(xᵢ₊₁ − xᵢ), extended linearly past the end cells. A descending
axis is located on its reversed values; a periodic axis of period P first maps
x into one period and gains a wrap cell between its last node and the first
one plus P. An axis of one node x₀ has no spacing: u = 0 at x₀, and its
edges, when it has them, sit at u = ∓½. On an axis with repeats, u is
right-continuous: a target at a repeated node sits at the last node of the
repeat.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from functax._src._backend import is_tracer
from functax._src.geometry._types import Axis
from functax._src.interp._policies import BOUNDARY, EXTENT, Policies
from functax.exceptions import GeometryError, OutsideDomainError, PolicyError


__all__ = [
    "check_outside",
    "clamp_positions",
    "domain_bounds",
    "hold_repeated_ends",
    "index_positions",
    "locate",
    "remap_taps",
    "require_cell_widths",
    "require_even_spacing",
    "require_no_repeats",
]


# a spacing within 10⁻³ of the mean counts as even for a wide kernel: f(x) = x is
# then read to 2 × 10⁻⁴ of a cell (measured for Cubic and BSpline(3))
_EVEN_RTOL = 1e-3


def _ascending(axis: Axis) -> tuple[Float[Array, " node"], bool | Bool[Array, ""]]:
    """The node values in increasing order, and whether they were reversed."""
    values = axis.values
    if is_tracer(values):
        # a traced axis cannot say which way it runs; compare its ends
        descending = values[0] > values[-1]
        reversed_values = values[::-1]
        ascending = jnp.where(descending, reversed_values, values)
        return ascending, descending
    # np.asarray first: under jit, indexing a concrete JAX array stages a
    # traced slice, which bool() cannot read
    host_values = np.asarray(values)
    descending = bool(host_values[0] > host_values[-1])
    ascending = values[::-1] if descending else values
    return ascending, descending


def _rectilinear_positions(
    nodes: Float[Array, " node"], positions: Float[Array, "*target"]
) -> Float[Array, "*target"]:
    """u = i + (x − xᵢ)/(xᵢ₊₁ − xᵢ) on increasing nodes, linear past the ends."""
    n_nodes = nodes.shape[0]
    # i = searchsorted(x, side="right") − 1, kept to a real cell [0, N − 2]
    insertion = jnp.searchsorted(nodes, positions, side="right")
    cell = insertion - 1
    cell = jnp.clip(cell, 0, n_nodes - 2)
    # xᵢ and xᵢ₊₁
    left = nodes[cell]
    right = nodes[cell + 1]
    # t = (x − xᵢ)/(xᵢ₊₁ − xᵢ), outside [0, 1] beyond the end cells
    offset = (positions - left) / (right - left)
    # u = i + t
    return cell + offset


def _repeated_positions(
    nodes: Float[Array, " node"], positions: Float[Array, "*target"]
) -> Float[Array, "*target"]:
    """u on increasing nodes that may repeat: right-continuous, linear past the ends.

    With i the last node such that xᵢ ≤ x, the cell [xᵢ, xᵢ₊₁] has a positive
    width and u = i + (x − xᵢ)/(xᵢ₊₁ − xᵢ), so a target at a repeat reads its
    last node. Past the ends, u continues with the width of the outermost
    cell of positive width: u = (x − x₀)/h_first below x₀ and
    u = N − 1 + (x − x_{N−1})/h_last from x_{N−1} on.
    """
    n_nodes = nodes.shape[0]
    # hᵢ = xᵢ₊₁ − xᵢ
    widths = nodes[1:] - nodes[:-1]
    # i = searchsorted(x, side="right") − 1, −1 below x₀ and N − 1 from x_{N−1}
    insertion = jnp.searchsorted(nodes, positions, side="right")
    cell = insertion - 1
    inner = jnp.clip(cell, 0, n_nodes - 2)
    left = nodes[inner]
    # xᵢ₊₁ − xᵢ for the clipped cell
    width = nodes[inner + 1] - left
    has_width = width > 0
    # a zero width is only reached below x₀ or from x_{N−1} on, which do not use it
    one = jnp.ones((), width.dtype)
    safe_width = jnp.where(has_width, width, one)
    # u = i + (x − xᵢ)/(xᵢ₊₁ − xᵢ)
    interior = inner + (positions - left) / safe_width
    # h_first and h_last, the outermost positive widths (the span is positive)
    positive = widths > 0
    first_positive = jnp.argmax(positive)
    first_width = widths[first_positive]
    positive_from_end = positive[::-1]
    last_positive = n_nodes - 2 - jnp.argmax(positive_from_end)
    last_width = widths[last_positive]
    # u = (x − x₀)/h_first below x₀
    below = (positions - nodes[0]) / first_width
    # u = N − 1 + (x − x_{N−1})/h_last from x_{N−1} on
    above = (n_nodes - 1) + (positions - nodes[-1]) / last_width
    inside_or_above = jnp.where(cell >= n_nodes - 1, above, interior)
    return jnp.where(cell < 0, below, inside_or_above)


def hold_repeated_ends(
    axis: Axis, positions: Float[Array, "*target"]
) -> Float[Array, "*target"]:
    """u held at an end of an axis with repeats whose outermost cell has zero width.

    A repeat at an end leaves a piece of one node there, which has no slope
    to extrapolate with, so ``outside="extrapolate"`` continues it as a
    constant: u is clipped to 0 below a repeated first node and to N − 1
    above a repeated last node. Every other axis returns u unchanged.

    Args:
        axis: The source axis.
        positions: u, from `index_positions` on ``axis``.

    Returns:
        u, held at the repeated ends.
    """
    n_nodes = axis.values.shape[0]
    if axis.repeats != "allow" or n_nodes == 1:
        return positions
    values = axis.values
    # in index order, which `index_positions` keeps for a decreasing axis too
    low_held = values[1] == values[0]
    high_held = values[-1] == values[-2]
    # a strict where, so the held side is a constant (zero derivative) at the
    # end node too, where min and max would split the gradient
    zero = jnp.zeros((), positions.dtype)
    last = jnp.asarray(n_nodes - 1, positions.dtype)
    # u ← 0 below a repeated first node
    positions = jnp.where(low_held & (positions < 0), zero, positions)
    # u ← N − 1 above a repeated last node
    return jnp.where(high_held & (positions >= n_nodes - 1), last, positions)


def require_no_repeats(source: Any, reader: str, *, axes: Any = None) -> None:
    """Raise when an axis that ``reader`` cannot read with repeats has one.

    A concrete axis is checked for a repeat xᵢ = xᵢ₊₁, which the message
    names; a traced axis with ``repeats="allow"`` cannot be checked when the
    plan is built, so it is refused.

    Args:
        source: The geometry: an `Axis` or a grid with ``axes``.
        reader: The scheme's name, for the message.
        axes: The indices of the axes to check; every axis when ``None``.

    Raises:
        GeometryError: If a checked axis has a repeat, or allows repeats and
            is traced.
    """
    all_axes = (source,) if isinstance(source, Axis) else getattr(source, "axes", ())
    indices = range(len(all_axes)) if axes is None else axes
    for index in indices:
        axis = all_axes[index]
        if axis.repeats != "allow" or axis.values.shape[0] == 1:
            continue
        if is_tracer(axis.values):
            msg = (
                f"{reader} cannot read an axis with repeats, and axis {index} "
                "allows them but is traced, so it cannot be checked"
            )
            raise GeometryError(msg)
        values = np.asarray(axis.values)
        # xᵢ = xᵢ₊₁
        is_repeat = values[1:] == values[:-1]
        repeated = np.flatnonzero(is_repeat)
        if repeated.size:
            first = int(repeated[0])
            msg = (
                f"{reader} cannot read repeated nodes, but axis {index} repeats "
                f"x[{first}] = x[{first + 1}] = {float(values[first])!r}"
            )
            raise GeometryError(msg)


def require_even_spacing(source: Any, reader: str, *, axes: Any = None) -> None:
    """Raise when an axis that ``reader`` reads in index space is unevenly spaced.

    With Δₖ = xₖ₊₁ − xₖ the spacings (and, on a periodic axis, the wrap gap
    from the last node to the first one plus P) and Δ̄ their mean, an axis is
    uneven when max |Δₖ − Δ̄| > τ |Δ̄|. τ is the larger of 10⁻³ and
    2 ε max(|xₖ|, P) / |Δ̄|, with ε the machine epsilon of the nodes' dtype,
    since rounding alone moves each spacing by up to ε max(|xₖ|, P). A traced
    axis cannot be checked, and passes.

    Args:
        source: The geometry: an `Axis` or a grid with ``axes``; other grids
            are read in index space by definition and pass.
        reader: The basis' name, for the message.
        axes: The indices of the axes to check; every axis when ``None``.

    Raises:
        GeometryError: If a checked axis is concrete and uneven.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis
        >>> from functax._src.interp._locate import require_even_spacing
        >>> require_even_spacing(Axis(jnp.linspace(0.0, 1.0, 11)), "CubicBasis")
        >>> require_even_spacing(
        ...     Axis(jnp.array([0.0, 1.0, 3.0])), "CubicBasis"
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        functax.exceptions.GeometryError: CubicBasis reads in index space, ...
    """
    all_axes = (source,) if isinstance(source, Axis) else getattr(source, "axes", ())
    indices = range(len(all_axes)) if axes is None else axes
    for index in indices:
        if index >= len(all_axes):
            continue
        axis = all_axes[index]
        if axis.values.shape[0] == 1 or is_tracer(axis.values):
            continue
        nodes = np.asarray(axis.values)
        # Δₖ = xₖ₊₁ − xₖ
        spacing = np.diff(nodes)
        if axis.period is not None:
            # x₀ ± P − x_{N−1}, the wrap gap, in the direction of the axis
            signed_period = np.copysign(axis.period, spacing[0])
            wrap_gap = nodes[0] + signed_period - nodes[-1]
            spacing = np.append(spacing, wrap_gap)
        # Δ̄
        mean_spacing = abs(spacing.mean())
        # max |Δₖ − Δ̄|, for spacings of one sign
        absolute_deviation = np.abs(np.abs(spacing) - mean_spacing)
        deviation = absolute_deviation.max()
        largest = max(np.abs(nodes).max(), axis.period or 0.0)
        # 2 ε max(|xₖ|, P)
        rounding = 2.0 * np.finfo(nodes.dtype).eps * largest
        if deviation > max(_EVEN_RTOL * mean_spacing, rounding):
            narrowest = float(np.abs(spacing).min())
            widest = float(np.abs(spacing).max())
            msg = (
                f"{reader} reads in index space, which does not follow the "
                f"coordinates of axis {index}: its spacing varies from "
                f"{narrowest:.6g} to {widest:.6g}. Use a scheme that follows them "
                "(Linear, or CubicSpline or Pchip on an Axis), or pass "
                "Policies(spacing='index') to read in index space"
            )
            raise GeometryError(msg)


def _one_node_positions(
    axis: Axis, positions: Float[Array, "*target"]
) -> Float[Array, "*target"]:
    """u on an axis of one node x₀: 0 at x₀, ∓½ at its edges, linear in between.

    Without edges, u = sign(x − x₀), so only x₀ lies in the domain [0, 0] of
    ``extent="nodes"``. With edges [e₀, e₁] in either order, u = ±½ |x − x₀|/h
    on the side of x, h being the distance from x₀ to that side's edge, and
    u = ±1 past a side of zero width. ±∞ and NaN stay as they are, so a
    non-finite target stays invalid under every policy.
    """
    node = axis.values[0]
    # x − x₀
    offset = positions - node
    # sign(±∞) would be finite, so a non-finite target keeps its position
    finite = jnp.isfinite(offset)
    if axis.edges is None:
        # u = sign(x − x₀)
        sign = jnp.sign(offset)
        return jnp.where(finite, sign, offset)
    lower = jnp.minimum(axis.edges[0], axis.edges[1])
    upper = jnp.maximum(axis.edges[0], axis.edges[1])
    below = offset < 0
    # h, from x₀ to the edge on the side of x
    width = jnp.where(below, node - lower, upper - node)
    has_width = width > 0
    one = jnp.ones((), width.dtype)
    safe_width = jnp.where(has_width, width, one)
    # |x − x₀|
    distance = jnp.abs(offset)
    # sign |x − x₀| (0 or 1, ∞ and NaN kept) past a zero width
    distance_sign = jnp.sign(distance)
    past = jnp.where(finite, distance_sign, distance)
    # ½ |x − x₀|/h
    scaled = 0.5 * distance / safe_width
    magnitude = jnp.where(has_width, scaled, past)
    # u = ±½ |x − x₀|/h on the side of x
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
    On an axis with repeats, u is right-continuous, so a target at a repeat
    sits at its last node, and past the ends it continues with the outermost
    cell of positive width.

    Algorithm:
        1. Sort the nodes increasing, noting whether the axis was reversed.
        2. On a periodic axis, map x into one period and append the wrap
           node x₀ + P; on one node, use ``sign(x − x₀)`` or the edges.
        3. Locate u on the increasing nodes: (x − x₀)/Δ on a regular axis,
           i + t from ``searchsorted`` otherwise (right-continuous at repeats).
        4. On a reversed axis, return (N − 1) − u.

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

        A periodic axis maps x into one period first:

        >>> periodic = Axis(jnp.arange(4.0), period=4.0)
        >>> index_positions(periodic, jnp.array([-0.5, 5.0])).tolist()
        [3.5, 1.0]
    """
    positions = jnp.asarray(positions)
    n_nodes = axis.values.shape[0]
    nodes, descending = _ascending(axis)
    if axis.period is not None:
        origin = nodes[0]
        # (x − x₀) mod P
        within_period = jnp.mod(positions - origin, axis.period)
        # x̃ = x₀ + ((x − x₀) mod P)
        wrapped = origin + within_period
        # the wrap cell x_{N−1} → x₀ + P
        extended = jnp.append(nodes, origin + axis.period)
        positions_ascending = _rectilinear_positions(extended, wrapped)
    elif n_nodes == 1:
        return _one_node_positions(axis, positions)
    elif axis.repeats == "allow":
        positions_ascending = _repeated_positions(nodes, positions)
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


def clamp_positions(
    positions: Float[Array, "*target"], lower: float, upper: float
) -> Float[Array, "*target"]:
    """u held to [lower, upper], with derivative 1 inside and at the bounds.

    ``jnp.clip`` splits the gradient at a bound (its ``maximum`` and
    ``minimum`` give ½ at a tie), which halves a derivative read exactly at
    an end node; a strict ``where`` keeps the bound inside.
    """
    positions = jnp.where(positions < lower, lower, positions)
    return jnp.where(positions > upper, upper, positions)


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
    # i = ⌊u⌋
    cell_float = jnp.floor(position)
    # t = u − i
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
    # 0 ≤ j ≤ N − 1
    in_range = (taps >= 0) & (taps <= n_nodes - 1)
    all_kept = jnp.ones(taps.shape, dtype=bool)
    if periodic or boundary == "wrap":
        # j mod N
        wrapped = jnp.mod(taps, n_nodes)
        return wrapped, all_kept
    # clip(j, 0, N − 1), for edge and constant
    clipped = jnp.clip(taps, 0, n_nodes - 1)
    if boundary == "edge":
        return clipped, all_kept
    if boundary == "constant":
        return clipped, in_range
    if n_nodes == 1:
        return jnp.zeros_like(taps), all_kept
    # mirror: j mod 2(N − 1), then fold the second half back
    period = 2 * (n_nodes - 1)
    folded = jnp.mod(taps, period)
    # j ↦ 2(N − 1) − j on the second half
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

    Examples:
        Every target inside passes, and any other policy never raises:

        >>> import jax.numpy as jnp
        >>> from functax.interp import Policies, check_outside
        >>> check_outside(jnp.array([True, True]), Policies(outside="raise"))
        >>> check_outside(jnp.array([True, False]), Policies(outside="nan"))

        Under ``outside="raise"`` a target outside is refused:

        >>> check_outside(
        ...     jnp.array([True, False]), Policies(outside="raise")
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        functax.exceptions.OutsideDomainError: 1 of 2 targets lie outside ...
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
    # full reduction, not einx: a plan may have no targets
    if flags.all():
        return
    outside = np.flatnonzero(~flags)
    msg = (
        f"{outside.size} of {flags.size} targets lie outside the domain under "
        f"outside='raise'; the first is target {int(outside[0])}"
    )
    raise OutsideDomainError(msg)
