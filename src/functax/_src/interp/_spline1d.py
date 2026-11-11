"""Interpolating cubic splines on one axis: slopes from one tridiagonal solve.

With hₖ = xₖ₊₁ − xₖ and secants δₖ = (yₖ₊₁ − yₖ)/hₖ, the slopes of the C²
cubic through the samples solve, at the interior knots,

    hₖ mₖ₋₁ + 2(hₖ₋₁ + hₖ) mₖ + hₖ₋₁ mₖ₊₁ = 3(hₖ δₖ₋₁ + hₖ₋₁ δₖ),

closed by the boundary condition in the first and last rows (natural,
clamped, not-a-knot) or by wrapping around (periodic, a cyclic system solved
with Sherman-Morrison). The rows are SciPy's ``CubicSpline``. The fit is
linear in y, so the plan is an ordinary `Plan`.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import functools

import einx
import jax
import jax.numpy as jnp

from functax.exceptions import ValidationError


__all__ = [
    "SPLINE_CONDITIONS",
    "SPLINE_SLOPES",
    "solve_tridiagonal",
    "spline_slopes",
]


SPLINE_CONDITIONS: tuple[str, ...] = ("not-a-knot", "natural", "clamped", "periodic")
"""The end conditions of `CubicSpline`, as in ``scipy.interpolate.CubicSpline``."""

# the fewest knots each end condition needs
MINIMUM_KNOTS: dict[str, int] = {
    "not-a-knot": 4,
    "natural": 2,
    "clamped": 2,
    "periodic": 4,
}


def solve_tridiagonal(
    lower: jax.Array, diagonal: jax.Array, upper: jax.Array, rhs: jax.Array
) -> jax.Array:
    """x with T x = rhs, T tridiagonal (lower[0] and upper[-1] ignored)."""
    lower = lower.at[0].set(0.0)
    upper = upper.at[-1].set(0.0)
    column = einx.id("n -> n 1", rhs)
    solution = jax.lax.linalg.tridiagonal_solve(lower, diagonal, upper, column)
    return einx.id("n 1 -> n", solution)


def _interior(widths: jax.Array, secants: jax.Array) -> tuple[jax.Array, ...]:
    """Sub-, main and super-diagonals and right-hand sides of rows 1 … N − 2."""
    previous_width = widths[:-1]
    next_width = widths[1:]
    # hₖ mₖ₋₁ + 2(hₖ₋₁ + hₖ) mₖ + hₖ₋₁ mₖ₊₁ = 3(hₖ δₖ₋₁ + hₖ₋₁ δₖ)
    lower = next_width
    diagonal = 2.0 * (previous_width + next_width)
    upper = previous_width
    rhs = 3.0 * (next_width * secants[:-1] + previous_width * secants[1:])
    return lower, diagonal, upper, rhs


def spline_slopes(nodes: jax.Array, values: jax.Array, *, condition: str) -> jax.Array:
    """The knot slopes m of the interpolating cubic spline with end ``condition``.

    Args:
        nodes: x, increasing, (N,).
        values: y, (N,); under ``"periodic"``, y₀ must equal y_{N−1}.
        condition: ``"not-a-knot"``, ``"natural"``, ``"clamped"`` (zero end
            slopes) or ``"periodic"``.

    Returns:
        m, (N,).

    Raises:
        ValidationError: If ``condition`` is unknown.
    """
    if condition not in SPLINE_CONDITIONS:
        msg = f"'bc' must be one of {list(SPLINE_CONDITIONS)}, got {condition!r}"
        raise ValidationError(msg)
    n = nodes.shape[0]
    # hₖ and δₖ
    widths = nodes[1:] - nodes[:-1]
    secants = (values[1:] - values[:-1]) / widths
    if n == 2:
        return einx.id("1, 1 -> (1 + 1)", secants, secants)
    if condition == "periodic":
        return _periodic_slopes(widths, secants)
    lower, diagonal, upper, rhs = _interior(widths, secants)
    one = jnp.ones((1,), values.dtype)
    zero = jnp.zeros((1,), values.dtype)
    if condition == "natural":
        # 2 m₀ + m₁ = 3 δ₀ and m_{N−2} + 2 m_{N−1} = 3 δ_{N−2}
        first = (zero, 2.0 * one, one, 3.0 * secants[:1])
        last = (one, 2.0 * one, zero, 3.0 * secants[-1:])
    elif condition == "clamped":
        # m₀ = 0 and m_{N−1} = 0
        first = (zero, one, zero, zero)
        last = (zero, one, zero, zero)
    else:
        # not-a-knot: the third derivative is continuous at x₁ and x_{N−2}
        span = widths[0] + widths[1]
        first_rhs = (
            (widths[0] + 2.0 * span) * widths[1] * secants[0]
            + widths[0] ** 2 * secants[1]
        ) / span
        first = (zero, widths[1:2], span[None], first_rhs[None])
        end_span = widths[-1] + widths[-2]
        last_rhs = (
            widths[-1] ** 2 * secants[-2]
            + (2.0 * end_span + widths[-1]) * widths[-2] * secants[-1]
        ) / end_span
        last = (end_span[None], widths[-2:-1], zero, last_rhs[None])
    pieces = [
        jnp.concatenate([head, body, tail])
        for head, body, tail in zip(
            first, (lower, diagonal, upper, rhs), last, strict=True
        )
    ]
    return solve_tridiagonal(*pieces)


def _periodic_slopes(widths: jax.Array, secants: jax.Array) -> jax.Array:
    """Slopes of the periodic spline: a cyclic system on m₀ … m_{N−2}, m_{N−1} = m₀."""
    n = widths.shape[0]
    # row k couples m_{k−1}, mₖ, m_{k+1} (indices mod N − 1)
    previous_width = jnp.roll(widths, 1)
    previous_secant = jnp.roll(secants, 1)
    lower = widths
    diagonal = 2.0 * (previous_width + widths)
    upper = previous_width
    rhs = 3.0 * (widths * previous_secant + previous_width * secants)
    # corners: row 0 reads m_{n−1} with lower[0], row n − 1 reads m₀ with upper[n − 1]
    corner_low = lower[0]
    corner_high = upper[n - 1]
    gamma = -diagonal[0]
    # T = A − u vᵀ with u = (γ, 0, …, 0, corner_high), v = (1, 0, …, 0, corner_low / γ)
    modified = diagonal.at[0].add(-gamma)
    modified = modified.at[n - 1].add(-corner_low * corner_high / gamma)
    u = jnp.zeros(n, widths.dtype).at[0].set(gamma).at[n - 1].set(corner_high)
    v = jnp.zeros(n, widths.dtype).at[0].set(1.0).at[n - 1].set(corner_low / gamma)
    base = solve_tridiagonal(lower, modified, upper, rhs)
    correction = solve_tridiagonal(lower, modified, upper, u)
    # x = y − (vᵀ y) / (1 + vᵀ z) z
    scale = jnp.dot(v, base) / (1.0 + jnp.dot(v, correction))
    slopes = base - scale * correction
    return jnp.append(slopes, slopes[0])


# one slope function per condition, made once, so plans that share a
# condition share the same static object (and the same jit cache entry)
SPLINE_SLOPES: dict[str, functools.partial[jax.Array]] = {
    condition: functools.partial(spline_slopes, condition=condition)
    for condition in SPLINE_CONDITIONS
}
