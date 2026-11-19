"""Hermite slopes on an axis with repeats: each piece between repeats on its own.

On N non-decreasing knots x₀, …, x_{N−1} with samples y, a repeat xₖ = xₖ₊₁
splits the knots into pieces, and each piece gets the slopes dₖ its scheme
gives it as an axis of its own, with the scheme's end rules at both of its
ends. The repeats are data (they may be traced), so the pieces are found
inside the trace and every array keeps its static shape:

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
from jaxtyping import Array, Bool, Float, Int

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


def piece_bounds(
    nodes: Float[Array, " node"],
) -> tuple[Int[Array, " node"], Int[Array, " node"]]:
    """The first and last knot a, b of the piece that holds each knot.

    A piece runs between repeats: knot k starts one when k = 0 or
    xₖ₋₁ = xₖ, and ends one when k = N − 1 or xₖ = xₖ₊₁.

    Algorithm:
        1. Mark the knots that start a piece and the knots that end one.
        2. a = the latest start at or before k, by a running maximum.
        3. b = the earliest stop at or after k, by a reversed running minimum.

    Args:
        nodes: x, the N non-decreasing knots.

    Returns:
        ``(start, stop)``, the int32 indices a and b of each knot's piece.

    Examples:
        A repeat at x = 1 splits five knots into pieces 0 … 1 and 2 … 4:

        >>> import jax.numpy as jnp
        >>> from functax._src.interp._pieces import piece_bounds
        >>> start, stop = piece_bounds(jnp.array([0.0, 1.0, 1.0, 2.0, 3.0]))
        >>> start.tolist(), stop.tolist()
        ([0, 0, 2, 2, 2], [1, 1, 4, 4, 4])

        Without repeats every knot is in one piece:

        >>> start, stop = piece_bounds(jnp.arange(3.0))
        >>> start.tolist(), stop.tolist()
        ([0, 0, 0], [2, 2, 2])
    """
    n_nodes = nodes.shape[0]
    index = jnp.arange(n_nodes, dtype=jnp.int32)
    # xₖ = xₖ₊₁
    repeat = nodes[1:] == nodes[:-1]
    always = jnp.ones(1, dtype=bool)
    first = einx.id("1, cell -> (1 + cell)", always, repeat)
    last = einx.id("cell, 1 -> (cell + 1)", repeat, always)
    # a = the latest start at or before k
    start_marks = jnp.where(first, index, 0)
    start = jax.lax.cummax(start_marks)
    # b = the earliest stop at or after k
    stop_marks = jnp.where(last, index, n_nodes - 1)
    stop = jax.lax.cummin(stop_marks, reverse=True)
    return start, stop


def _window_slopes(
    slopes: Window,
    radius: int,
    nodes: Float[Array, " node"],
    values: Float[Array, " node"],
    *per_knot: Float[Array, " node"],
) -> Float[Array, " node"]:
    """Each knot's slope from ``slopes`` on a window of its own piece.

    For a piece of n < 2r + 1 knots the window is the whole piece; for a
    longer one it is the 2r + 1 knots around the knot, shifted to stay inside
    the piece. A window that is not used reads safe stand-in knots, so that
    its untaken branch stays finite under `grad`.

    Args:
        slopes: The scheme's slopes on one window of L knots, (L,).
        radius: r, how far a knot's slope reaches.
        nodes: x, the N non-decreasing knots.
        values: y, the N samples.
        *per_knot: Extra arrays of one entry per knot; each window reads its
            knot's entry.

    Returns:
        d, the N knot slopes.
    """
    n_nodes = nodes.shape[0]
    index = jnp.arange(n_nodes, dtype=jnp.int32)
    start, stop = piece_bounds(nodes)
    # n = b − a + 1
    length = stop - start + 1
    # L = 2r + 1
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
            # max(a, min(k − r, b − L + 1))
            centred = index - radius
            first = jnp.clip(centred, start, stop - size + 1)
        offsets = jnp.arange(size, dtype=jnp.int32)
        # the knots first + 0 … first + L − 1 of each window
        window_taps = einx.add("knot, tap -> knot tap", first, offsets)
        taps = jnp.clip(window_taps, 0, n_nodes - 1)
        # stand-ins x = 0, 1, …, y = 0 where the window is not used
        stand_in_nodes = offsets.astype(nodes.dtype)
        stand_in = einx.id("tap -> knot tap", stand_in_nodes, knot=n_nodes)
        tap_nodes = einx.get_at("[node], knot tap -> knot tap", nodes, taps)
        window_nodes = einx.where(
            "knot, knot tap, knot tap -> knot tap", used, tap_nodes, stand_in
        )
        zero = jnp.zeros((), values.dtype)
        tap_values = einx.get_at("[node], knot tap -> knot tap", values, taps)
        window_values = einx.where(
            "knot, knot tap, -> knot tap", used, tap_values, zero
        )
        extra = [
            jnp.where(used, entry, jnp.ones((), entry.dtype)) for entry in per_knot
        ]
        window = jax.vmap(slopes)(window_nodes, window_values, *extra)
        # knot k sits at k − first in its window
        own_tap = index - first
        own = einx.get_at("knot [tap], knot -> knot", window, own_tap)
        slope = jnp.where(used, own, slope)
    return slope


def piecewise_pchip_slopes(
    nodes: Float[Array, " node"], values: Float[Array, " node"]
) -> Float[Array, " node"]:
    """`pchip_slopes` on each piece between repeats.

    Args:
        nodes: x, the N non-decreasing knots.
        values: y, the N samples.

    Returns:
        d, the N knot slopes.

    Examples:
        Each piece equals PCHIP on its own knots:

        >>> import jax.numpy as jnp
        >>> from functax._src.interp._monotone import pchip_slopes
        >>> from functax._src.interp._pieces import piecewise_pchip_slopes
        >>> nodes = jnp.array([0.0, 1.0, 2.0, 2.0, 3.0, 4.0])
        >>> values = jnp.array([0.0, 1.0, 4.0, 5.0, 4.0, 2.0])
        >>> slopes = piecewise_pchip_slopes(nodes, values)
        >>> slopes.shape
        (6,)
        >>> head = pchip_slopes(nodes[:3], values[:3])
        >>> tail = pchip_slopes(nodes[3:], values[3:])
        >>> bool(jnp.allclose(slopes, jnp.concatenate([head, tail])))
        True
    """
    return _window_slopes(pchip_slopes, 1, nodes, values)


def piecewise_steffen_slopes(
    nodes: Float[Array, " node"], values: Float[Array, " node"]
) -> Float[Array, " node"]:
    """`steffen_slopes` on each piece between repeats.

    Args:
        nodes: x, the N non-decreasing knots.
        values: y, the N samples.

    Returns:
        d, the N knot slopes.

    Examples:
        Each piece equals Steffen on its own knots:

        >>> import jax.numpy as jnp
        >>> from functax._src.interp._monotone import steffen_slopes
        >>> from functax._src.interp._pieces import piecewise_steffen_slopes
        >>> nodes = jnp.array([0.0, 1.0, 2.0, 2.0, 3.0, 4.0])
        >>> values = jnp.array([0.0, 1.0, 4.0, 5.0, 4.0, 2.0])
        >>> slopes = piecewise_steffen_slopes(nodes, values)
        >>> head = steffen_slopes(nodes[:3], values[:3])
        >>> tail = steffen_slopes(nodes[3:], values[3:])
        >>> bool(jnp.allclose(slopes, jnp.concatenate([head, tail])))
        True

        Without repeats it is Steffen on the whole axis:

        >>> line = piecewise_steffen_slopes(jnp.arange(4.0), jnp.arange(4.0))
        >>> bool(jnp.allclose(line, 1.0))
        True
    """
    return _window_slopes(steffen_slopes, 1, nodes, values)


def piecewise_akima_slopes(
    nodes: Float[Array, " node"], values: Float[Array, " node"]
) -> Float[Array, " node"]:
    """`akima_slopes` on each piece between repeats, its threshold per piece.

    Akima's slope is defined where f₁ + f₂ exceeds 10⁻⁹ of its maximum, and
    the maximum is taken over the piece, as Akima on the piece alone would.

    Algorithm:
        1. Form f₁ + f₂ at each knot from a window of its own piece.
        2. Take the largest f₁ + f₂ of each piece, read back at its knots.
        3. Form each knot's slope from a window of its own piece, against the
           piece's threshold.

    Args:
        nodes: x, the N non-decreasing knots.
        values: y, the N samples.

    Returns:
        d, the N knot slopes.

    Examples:
        Each piece equals Akima on its own knots:

        >>> import jax.numpy as jnp
        >>> from functax._src.interp._monotone import akima_slopes
        >>> from functax._src.interp._pieces import piecewise_akima_slopes
        >>> nodes = jnp.array([0.0, 1, 2, 3, 3, 4, 5, 6])
        >>> values = jnp.array([0.0, 1, 3, 2, 5, 4, 6, 7])
        >>> slopes = piecewise_akima_slopes(nodes, values)
        >>> slopes.shape
        (8,)
        >>> head = akima_slopes(nodes[:4], values[:4])
        >>> tail = akima_slopes(nodes[4:], values[4:])
        >>> bool(jnp.allclose(slopes, jnp.concatenate([head, tail])))
        True
    """
    totals = _window_slopes(akima_totals, 2, nodes, values)
    start, _ = piece_bounds(nodes)
    # the largest f₁ + f₂ of each piece, read back at its knots
    largest = jax.ops.segment_max(totals, start, num_segments=nodes.shape[0])
    piece_largest = largest[start]
    return _window_slopes(akima_slopes, 2, nodes, values, piece_largest)


def _safe(
    divisor: Float[Array, "*shape"], use: Bool[Array, "*shape"]
) -> Float[Array, "*shape"]:
    """``divisor`` where ``use`` holds, 1 elsewhere: the untaken branch is finite."""
    return jnp.where(use, divisor, jnp.ones((), divisor.dtype))


def _shift(array: Float[Array, " knot"], by: int) -> Float[Array, " knot"]:
    """vₖ₊by at each k for the array v, 0 where k + by falls off the array."""
    zero = jnp.zeros((abs(by),), array.dtype)
    if by > 0:
        return einx.id("kept, off -> (kept + off)", array[by:], zero)
    return einx.id("off, kept -> (off + kept)", zero, array[:by])


def piecewise_spline_slopes(
    nodes: Float[Array, " node"], values: Float[Array, " node"], *, condition: str
) -> Float[Array, " node"]:
    """The cubic spline's slopes with end ``condition`` on each piece between repeats.

    With widths hₖ = xₖ₊₁ − xₖ and secants δₖ = (yₖ₊₁ − yₖ)/hₖ, one
    tridiagonal system holds every piece: a knot inside a piece has the
    interior row, hₖ dₖ₋₁ + 2(hₖ₋₁ + hₖ) dₖ + hₖ₋₁ dₖ₊₁ = 3(hₖ δₖ₋₁ + hₖ₋₁ δₖ),
    and a knot at a piece's end has the end row of ``condition``, which does
    not reach across the repeat, so the system splits into one block per
    piece. A piece of two knots is a line (dₖ = δ), as `spline_slopes` gives
    two knots, and a not-a-knot piece of three is SciPy's parabola.

    Algorithm:
        1. Find each knot's piece, and the widths and secants around it, with
           δ = 0 across a repeat.
        2. Build every knot's interior row, and the head and tail rows of
           ``condition`` (special-cased for pieces of two and three knots).
        3. Pick each knot's row by its place in its piece and solve the one
           tridiagonal system.

    Args:
        nodes: x, the N non-decreasing knots.
        values: y, the N samples.
        condition: ``"not-a-knot"``, ``"natural"`` or ``"clamped"``.

    Returns:
        d, the N knot slopes.

    Raises:
        ValidationError: If ``condition`` is unknown or ``"periodic"``.

    Examples:
        Each piece equals the spline on its own knots:

        >>> import jax.numpy as jnp
        >>> from functax._src.interp._pieces import piecewise_spline_slopes
        >>> from functax._src.interp._spline1d import spline_slopes
        >>> nodes = jnp.array([0.0, 1, 2, 3, 3, 4, 5, 6])
        >>> values = jnp.array([0.0, 1, 3, 2, 5, 4, 6, 7])
        >>> slopes = piecewise_spline_slopes(nodes, values, condition="natural")
        >>> slopes.shape
        (8,)
        >>> head = spline_slopes(nodes[:4], values[:4], condition="natural")
        >>> tail = spline_slopes(nodes[4:], values[4:], condition="natural")
        >>> bool(jnp.allclose(slopes, jnp.concatenate([head, tail])))
        True

        The periodic condition has no meaning on a piece:

        >>> piecewise_spline_slopes(
        ...     nodes, values, condition="periodic"
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        functax.exceptions.ValidationError: 'bc' must be ...
    """
    if condition not in SPLINE_CONDITIONS or condition == "periodic":
        msg = (
            "'bc' must be 'not-a-knot', 'natural' or 'clamped' on an axis with "
            f"repeats, got {condition!r}"
        )
        raise ValidationError(msg)

    # --- pieces, widths and secants ---
    n_nodes = nodes.shape[0]
    index = jnp.arange(n_nodes, dtype=jnp.int32)
    start, stop = piece_bounds(nodes)
    # n = b − a + 1
    length = stop - start + 1
    # hₖ = xₖ₊₁ − xₖ
    widths = nodes[1:] - nodes[:-1]
    positive = widths > 0
    # δₖ = (yₖ₊₁ − yₖ)/hₖ, with δ = 0 across a repeat
    rises = values[1:] - values[:-1]
    ratios = rises / _safe(widths, positive)
    secants = jnp.where(positive, ratios, 0.0)
    # around knot k: hₖ₋₁, hₖ, and one further cell each way (0 off the axis)
    zero = jnp.zeros(1, widths.dtype)
    left_width = einx.id("1, cell -> (1 + cell)", zero, widths)
    right_width = einx.id("cell, 1 -> (cell + 1)", widths, zero)
    left_secant = einx.id("1, cell -> (1 + cell)", zero, secants)
    right_secant = einx.id("cell, 1 -> (cell + 1)", secants, zero)
    far_left_width = _shift(left_width, -1)
    far_right_width = _shift(right_width, 1)
    far_left_secant = _shift(left_secant, -1)
    far_right_secant = _shift(right_secant, 1)
    ones = jnp.ones_like(values)
    zeros = jnp.zeros_like(values)

    # --- interior rows ---
    # hₖ dₖ₋₁ + 2(hₖ₋₁ + hₖ) dₖ + hₖ₋₁ dₖ₊₁ = 3(hₖ δₖ₋₁ + hₖ₋₁ δₖ)
    lower = right_width
    diagonal = 2.0 * (left_width + right_width)
    upper = left_width
    rhs = 3.0 * (right_width * left_secant + left_width * right_secant)

    # --- the rows at each end of a piece ---
    if condition == "natural":
        # 2 dₐ + dₐ₊₁ = 3 δₐ and d_{b−1} + 2 d_b = 3 δ_{b−1}
        head = (zeros, 2.0 * ones, ones, 3.0 * right_secant)
        tail = (ones, 2.0 * ones, zeros, 3.0 * left_secant)
    elif condition == "clamped":
        # dₐ = 0 and d_b = 0
        head = (zeros, ones, zeros, zeros)
        tail = (zeros, ones, zeros, zeros)
    else:
        # not-a-knot at the piece's second and second-to-last knots
        # hₐ + hₐ₊₁
        span = right_width + far_right_width
        safe_span = _safe(span, span > 0)
        # ((hₐ + 2(hₐ + hₐ₊₁)) hₐ₊₁ δₐ + hₐ² δₐ₊₁) / (hₐ + hₐ₊₁)
        head_numerator = (
            right_width + 2.0 * span
        ) * far_right_width * right_secant + right_width**2 * far_right_secant
        head_rhs = head_numerator / safe_span
        head = (zeros, far_right_width, span, head_rhs)
        # h_{b−1} + h_{b−2}
        end_span = left_width + far_left_width
        safe_end_span = _safe(end_span, end_span > 0)
        # (h_{b−1}² δ_{b−2} + (2(h_{b−1} + h_{b−2}) + h_{b−1}) h_{b−2} δ_{b−1})
        #   / (h_{b−1} + h_{b−2})
        tail_numerator = (
            left_width**2 * far_left_secant
            + (2.0 * end_span + left_width) * far_left_width * left_secant
        )
        tail_rhs = tail_numerator / safe_end_span
        tail = (end_span, far_left_width, zeros, tail_rhs)
        # a piece of three: dₐ + dₐ₊₁ = 2 δₐ and d_{b−1} + d_b = 2 δ_{b−1}
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
    # a piece of two is a line, dₖ = δ; a piece of one gets dₖ = 0
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

    # --- assemble and solve ---
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
