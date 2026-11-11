"""Hermite slopes on an axis with repeats: each piece between repeats on its own.

A repeat xₖ = xₖ₊₁ splits the knots into pieces, and each piece gets the
slopes its scheme gives it as an axis of its own, with the scheme's end rules
at both of its ends. The repeats are data (they may be traced), so the pieces
are found inside the trace and every array keeps its static shape:

- the local schemes (PCHIP, Steffen, Akima) read each knot's slope off a
  window of 2r + 1 knots of its own piece, since a slope there depends on
  knots at most r away and on the piece's ends only within r;
- the cubic spline solves one tridiagonal system whose rows are the spline's
  rows on each piece, so it splits into one block per piece.

A piece of one knot gets slope 0, which no read-out uses, since no target
falls in a cell of zero width.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import functools
from collections.abc import Callable

import einx
import jax
import jax.numpy as jnp

from functax._src.interp._monotone import (
    akima_slopes,
    akima_totals,
    pchip_slopes,
    steffen_slopes,
)
from functax._src.interp._spline1d import SPLINE_CONDITIONS, solve_tridiagonal
from functax.exceptions import ValidationError


__all__ = [
    "PIECEWISE_SPLINE_SLOPES",
    "piece_bounds",
    "piecewise_akima_slopes",
    "piecewise_pchip_slopes",
    "piecewise_spline_slopes",
    "piecewise_steffen_slopes",
]


type Window = Callable[..., jax.Array]
"""slopes(x, y, *per_knot) on the L knots of one window, (L,)."""


def piece_bounds(nodes: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The first and last knot a, b of the piece that holds each knot.

    A piece runs between repeats: knot k starts one when k = 0 or
    xₖ₋₁ = xₖ, and ends one when k = N − 1 or xₖ = xₖ₊₁.

    Args:
        nodes: x, non-decreasing, (N,).

    Returns:
        ``(start, stop)``, two int32 arrays of shape (N,).
    """
    n_nodes = nodes.shape[0]
    index = jnp.arange(n_nodes, dtype=jnp.int32)
    repeat = nodes[1:] == nodes[:-1]
    first = jnp.concatenate([jnp.ones(1, dtype=bool), repeat])
    last = jnp.concatenate([repeat, jnp.ones(1, dtype=bool)])
    # a = the latest start at or before k, b = the earliest stop at or after k
    start = jax.lax.cummax(jnp.where(first, index, 0))
    stop = jax.lax.cummin(jnp.where(last, index, n_nodes - 1), reverse=True)
    return start, stop


def _window_slopes(
    slopes: Window,
    radius: int,
    nodes: jax.Array,
    values: jax.Array,
    *per_knot: jax.Array,
) -> jax.Array:
    """Each knot's slope from ``slopes`` on a window of its own piece.

    For a piece of n < 2r + 1 knots the window is the whole piece; for a
    longer one it is the 2r + 1 knots around the knot, shifted to stay inside
    the piece. A window that is not used reads safe stand-in knots, so that
    its untaken branch stays finite under `grad`.

    Args:
        slopes: The scheme's slopes on one window of L knots, (L,).
        radius: r, how far a knot's slope reaches.
        nodes: x, non-decreasing, (N,).
        values: y, (N,).
        *per_knot: Extra (N,) arrays; each window reads its knot's entry.

    Returns:
        m, (N,).
    """
    n_nodes = nodes.shape[0]
    index = jnp.arange(n_nodes, dtype=jnp.int32)
    start, stop = piece_bounds(nodes)
    length = stop - start + 1
    width = 2 * radius + 1
    slope = jnp.zeros_like(values)
    for size in range(2, min(width, n_nodes) + 1):
        if size < width:
            # the whole piece, which has exactly `size` knots
            used = length == size
            first = start
        else:
            # 2r + 1 knots around k, kept inside a piece of at least as many
            used = length >= size
            first = jnp.clip(index - radius, start, stop - size + 1)
        offsets = jnp.arange(size, dtype=jnp.int32)
        taps = jnp.clip(
            einx.add("knot, tap -> knot tap", first, offsets), 0, n_nodes - 1
        )
        # stand-ins x = 0, 1, …, y = 0 where the window is not used
        stand_in = einx.id("tap -> knot tap", offsets.astype(nodes.dtype), knot=n_nodes)
        window_nodes = einx.where(
            "knot, knot tap, knot tap -> knot tap", used, nodes[taps], stand_in
        )
        zero = jnp.zeros((), values.dtype)
        window_values = einx.where(
            "knot, knot tap, -> knot tap", used, values[taps], zero
        )
        extra = [
            jnp.where(used, entry, jnp.ones((), entry.dtype)) for entry in per_knot
        ]
        window = jax.vmap(slopes)(window_nodes, window_values, *extra)
        # knot k sits at k − first in its window
        own = window[index, index - first]
        slope = jnp.where(used, own, slope)
    return slope


def piecewise_pchip_slopes(nodes: jax.Array, values: jax.Array) -> jax.Array:
    """`pchip_slopes` on each piece between repeats."""
    return _window_slopes(pchip_slopes, 1, nodes, values)


def piecewise_steffen_slopes(nodes: jax.Array, values: jax.Array) -> jax.Array:
    """`steffen_slopes` on each piece between repeats."""
    return _window_slopes(steffen_slopes, 1, nodes, values)


def piecewise_akima_slopes(nodes: jax.Array, values: jax.Array) -> jax.Array:
    """`akima_slopes` on each piece between repeats, its threshold per piece.

    Akima's slope is defined where f₁ + f₂ exceeds 10⁻⁹ of its maximum, and
    the maximum is taken over the piece, as Akima on the piece alone would.
    """
    totals = _window_slopes(akima_totals, 2, nodes, values)
    start, _ = piece_bounds(nodes)
    # the largest f₁ + f₂ of each piece, read back at its knots
    largest = jax.ops.segment_max(totals, start, num_segments=nodes.shape[0])
    return _window_slopes(akima_slopes, 2, nodes, values, largest[start])


def _safe(divisor: jax.Array, use: jax.Array) -> jax.Array:
    """``divisor`` where ``use`` holds, 1 elsewhere: the untaken branch stays finite."""
    return jnp.where(use, divisor, jnp.ones((), divisor.dtype))


def _shift(array: jax.Array, by: int) -> jax.Array:
    """aₖ₊by at each k, 0 where k + by falls off the array."""
    zero = jnp.zeros((abs(by),), array.dtype)
    if by > 0:
        return jnp.concatenate([array[by:], zero])
    return jnp.concatenate([zero, array[:by]])


def piecewise_spline_slopes(
    nodes: jax.Array, values: jax.Array, *, condition: str
) -> jax.Array:
    """The cubic spline's slopes with end ``condition`` on each piece between repeats.

    One tridiagonal system holds every piece: a knot inside a piece has the
    interior row, hₖ mₖ₋₁ + 2(hₖ₋₁ + hₖ) mₖ + hₖ₋₁ mₖ₊₁ = 3(hₖ δₖ₋₁ + hₖ₋₁ δₖ),
    and a knot at a piece's end has the end row of ``condition``, which does
    not reach across the repeat, so the system splits into one block per
    piece. A piece of two knots is a line (mₖ = δ), as `spline_slopes` gives
    two knots, and a not-a-knot piece of three is SciPy's parabola.

    Args:
        nodes: x, non-decreasing, (N,).
        values: y, (N,).
        condition: ``"not-a-knot"``, ``"natural"`` or ``"clamped"``.

    Returns:
        m, (N,).

    Raises:
        ValidationError: If ``condition`` is unknown or ``"periodic"``.
    """
    if condition not in SPLINE_CONDITIONS or condition == "periodic":
        msg = (
            "'bc' must be 'not-a-knot', 'natural' or 'clamped' on an axis with "
            f"repeats, got {condition!r}"
        )
        raise ValidationError(msg)
    n_nodes = nodes.shape[0]
    index = jnp.arange(n_nodes, dtype=jnp.int32)
    start, stop = piece_bounds(nodes)
    length = stop - start + 1
    # hₖ and δₖ, with δ = 0 across a repeat
    widths = nodes[1:] - nodes[:-1]
    positive = widths > 0
    secants = jnp.where(
        positive, (values[1:] - values[:-1]) / _safe(widths, positive), 0.0
    )
    # around knot k: hₖ₋₁, hₖ, and one further cell each way (0 off the axis)
    zero = jnp.zeros(1, widths.dtype)
    left_width = jnp.concatenate([zero, widths])
    right_width = jnp.concatenate([widths, zero])
    left_secant = jnp.concatenate([zero, secants])
    right_secant = jnp.concatenate([secants, zero])
    far_left_width = _shift(left_width, -1)
    far_right_width = _shift(right_width, 1)
    far_left_secant = _shift(left_secant, -1)
    far_right_secant = _shift(right_secant, 1)
    ones = jnp.ones_like(values)
    zeros = jnp.zeros_like(values)

    # --- interior rows ---
    lower = right_width
    diagonal = 2.0 * (left_width + right_width)
    upper = left_width
    rhs = 3.0 * (right_width * left_secant + left_width * right_secant)

    # --- the rows at each end of a piece ---
    if condition == "natural":
        # 2 mₐ + mₐ₊₁ = 3 δₐ and m_{b−1} + 2 m_b = 3 δ_{b−1}
        head = (zeros, 2.0 * ones, ones, 3.0 * right_secant)
        tail = (ones, 2.0 * ones, zeros, 3.0 * left_secant)
    elif condition == "clamped":
        head = (zeros, ones, zeros, zeros)
        tail = (zeros, ones, zeros, zeros)
    else:
        # not-a-knot at the piece's second and second-to-last knots
        span = right_width + far_right_width
        safe_span = _safe(span, span > 0)
        head_rhs = (
            (right_width + 2.0 * span) * far_right_width * right_secant
            + right_width**2 * far_right_secant
        ) / safe_span
        head = (zeros, far_right_width, span, head_rhs)
        end_span = left_width + far_left_width
        safe_end_span = _safe(end_span, end_span > 0)
        tail_rhs = (
            left_width**2 * far_left_secant
            + (2.0 * end_span + left_width) * far_left_width * left_secant
        ) / safe_end_span
        tail = (end_span, far_left_width, zeros, tail_rhs)
        # a piece of three: mₐ + mₐ₊₁ = 2 δₐ and m_{b−1} + m_b = 2 δ_{b−1}
        three = length == 3
        head = tuple(
            jnp.where(three, short, long)
            for short, long in zip(
                (zeros, ones, ones, 2.0 * right_secant), head, strict=True
            )
        )
        tail = tuple(
            jnp.where(three, short, long)
            for short, long in zip(
                (ones, ones, zeros, 2.0 * left_secant), tail, strict=True
            )
        )
    # a piece of two is a line, mₖ = δ; a piece of one gets mₖ = 0
    two = length == 2
    head = tuple(
        jnp.where(two, line, row)
        for line, row in zip((zeros, ones, zeros, right_secant), head, strict=True)
    )
    tail = tuple(
        jnp.where(two, line, row)
        for line, row in zip((zeros, ones, zeros, left_secant), tail, strict=True)
    )
    single = (zeros, ones, zeros, zeros)

    is_single = length == 1
    is_head = (index == start) & ~is_single
    is_tail = (index == stop) & ~is_single
    rows = []
    for interior, first, last, alone in zip(
        (lower, diagonal, upper, rhs), head, tail, single, strict=True
    ):
        row = jnp.where(is_head, first, interior)
        row = jnp.where(is_tail, last, row)
        rows.append(jnp.where(is_single, alone, row))
    return solve_tridiagonal(*rows)


# one slope function per condition, made once, so plans that share a
# condition share the same static object (and the same jit cache entry)
PIECEWISE_SPLINE_SLOPES: dict[str, functools.partial[jax.Array]] = {
    condition: functools.partial(piecewise_spline_slopes, condition=condition)
    for condition in SPLINE_CONDITIONS
    if condition != "periodic"
}
