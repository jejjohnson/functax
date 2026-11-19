"""Interpolating cubic splines on one axis: slopes from one tridiagonal solve.

On N knots xₖ with samples yₖ, widths hₖ = xₖ₊₁ − xₖ and secants
δₖ = (yₖ₊₁ − yₖ)/hₖ, the slopes dₖ of the C² cubic through the samples solve,
at the interior knots,

    hₖ dₖ₋₁ + 2(hₖ₋₁ + hₖ) dₖ + hₖ₋₁ dₖ₊₁ = 3(hₖ δₖ₋₁ + hₖ₋₁ δₖ),

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
from jaxtyping import Array, Float

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
    lower: Float[Array, " row"],
    diagonal: Float[Array, " row"],
    upper: Float[Array, " row"],
    rhs: Float[Array, " row"],
) -> Float[Array, " row"]:
    """z with T z = rhs, T tridiagonal (lower[0] and upper[-1] ignored).

    Row r of T z = rhs reads aᵣ zᵣ₋₁ + bᵣ zᵣ + cᵣ zᵣ₊₁ = rhsᵣ.

    Args:
        lower: a, the sub-diagonal; its first entry is ignored.
        diagonal: b, the main diagonal.
        upper: c, the super-diagonal; its last entry is ignored.
        rhs: The right-hand side.

    Returns:
        z, one entry per row.

    Examples:
        A diagonal system divides:

        >>> import jax.numpy as jnp
        >>> from functax._src.interp._spline1d import solve_tridiagonal
        >>> zeros = jnp.zeros(3)
        >>> diagonal = jnp.array([2.0, 4.0, 8.0])
        >>> solution = solve_tridiagonal(zeros, diagonal, zeros, jnp.ones(3))
        >>> bool(jnp.allclose(solution, 1.0 / diagonal))
        True

        The [1, 2, 1] stencil, checked against its dense matrix:

        >>> ones = jnp.ones(4)
        >>> rhs = jnp.arange(4.0)
        >>> solution = solve_tridiagonal(ones, 2.0 * ones, ones, rhs)
        >>> dense = 2.0 * jnp.eye(4) + jnp.eye(4, k=1) + jnp.eye(4, k=-1)
        >>> bool(jnp.allclose(dense @ solution, rhs))
        True
    """
    lower = lower.at[0].set(0.0)
    upper = upper.at[-1].set(0.0)
    column = einx.id("row -> row 1", rhs)
    # z = T⁻¹ rhs, by the banded solve
    solution = jax.lax.linalg.tridiagonal_solve(lower, diagonal, upper, column)
    return einx.id("row 1 -> row", solution)


def _interior(
    widths: Float[Array, " cell"], secants: Float[Array, " cell"]
) -> tuple[
    Float[Array, " interior"],
    Float[Array, " interior"],
    Float[Array, " interior"],
    Float[Array, " interior"],
]:
    """Sub-, main and super-diagonals and right-hand sides of rows 1 … N − 2."""
    previous_width = widths[:-1]
    next_width = widths[1:]
    # hₖ dₖ₋₁ + 2(hₖ₋₁ + hₖ) dₖ + hₖ₋₁ dₖ₊₁ = 3(hₖ δₖ₋₁ + hₖ₋₁ δₖ)
    lower = next_width
    diagonal = 2.0 * (previous_width + next_width)
    upper = previous_width
    rhs = 3.0 * (next_width * secants[:-1] + previous_width * secants[1:])
    return lower, diagonal, upper, rhs


def spline_slopes(
    nodes: Float[Array, " node"], values: Float[Array, " node"], *, condition: str
) -> Float[Array, " node"]:
    """The knot slopes d of the interpolating cubic spline with end ``condition``.

    hₖ dₖ₋₁ + 2(hₖ₋₁ + hₖ) dₖ + hₖ₋₁ dₖ₊₁ = 3(hₖ δₖ₋₁ + hₖ₋₁ δₖ) at the
    interior knots, closed by the end condition.

    Algorithm:
        1. Form the widths hₖ and secants δₖ.
        2. Build the interior rows, and the first and last rows of
           ``condition``.
        3. Solve the tridiagonal system (a cyclic one with Sherman-Morrison
           under ``"periodic"``).

    Args:
        nodes: x, the N increasing knots.
        values: y, the N samples; under ``"periodic"``, y₀ must equal y_{N−1}.
        condition: ``"not-a-knot"``, ``"natural"``, ``"clamped"`` (zero end
            slopes) or ``"periodic"``.

    Returns:
        d, the N knot slopes.

    Raises:
        ValidationError: If ``condition`` is unknown.

    Examples:
        Not-a-knot reproduces a cubic, so its slopes are 3x²:

        >>> import jax.numpy as jnp
        >>> from functax._src.interp._spline1d import spline_slopes
        >>> nodes = jnp.array([0.0, 1.0, 2.5, 3.0, 4.0])
        >>> slopes = spline_slopes(nodes, nodes**3, condition="not-a-knot")
        >>> slopes.shape
        (5,)
        >>> bool(jnp.allclose(slopes, 3.0 * nodes**2))
        True

        Clamped ends have zero slope, whatever the data:

        >>> clamped = spline_slopes(nodes, nodes**3, condition="clamped")
        >>> bool(jnp.allclose(clamped[jnp.array([0, -1])], 0.0))
        True

        An unknown condition is rejected:

        >>> spline_slopes(
        ...     nodes, nodes, condition="parabolic"
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        functax.exceptions.ValidationError: 'bc' must be one of ...
    """
    if condition not in SPLINE_CONDITIONS:
        msg = f"'bc' must be one of {list(SPLINE_CONDITIONS)}, got {condition!r}"
        raise ValidationError(msg)
    n_nodes = nodes.shape[0]
    # hₖ = xₖ₊₁ − xₖ
    widths = nodes[1:] - nodes[:-1]
    # δₖ = (yₖ₊₁ − yₖ)/hₖ
    rises = values[1:] - values[:-1]
    secants = rises / widths
    if n_nodes == 2:
        return einx.id("1, 1 -> (1 + 1)", secants, secants)
    if condition == "periodic":
        return _periodic_slopes(widths, secants)
    lower, diagonal, upper, rhs = _interior(widths, secants)
    one = jnp.ones((1,), values.dtype)
    zero = jnp.zeros((1,), values.dtype)
    if condition == "natural":
        # 2 d₀ + d₁ = 3 δ₀ and d_{N−2} + 2 d_{N−1} = 3 δ_{N−2}
        first = (zero, 2.0 * one, one, 3.0 * secants[:1])
        last = (one, 2.0 * one, zero, 3.0 * secants[-1:])
    elif condition == "clamped":
        # d₀ = 0 and d_{N−1} = 0
        first = (zero, one, zero, zero)
        last = (zero, one, zero, zero)
    else:
        # not-a-knot: the third derivative is continuous at x₁ and x_{N−2}
        # h₀ + h₁
        span = widths[0] + widths[1]
        # ((h₀ + 2(h₀ + h₁)) h₁ δ₀ + h₀² δ₁) / (h₀ + h₁)
        first_numerator = (widths[0] + 2.0 * span) * widths[1] * secants[0] + widths[
            0
        ] ** 2 * secants[1]
        first_rhs = first_numerator / span
        first_span = einx.id(" -> 1", span)
        first_value = einx.id(" -> 1", first_rhs)
        first = (zero, widths[1:2], first_span, first_value)
        # h_{N−2} + h_{N−3}
        end_span = widths[-1] + widths[-2]
        # (h_{N−2}² δ_{N−3} + (2(h_{N−2} + h_{N−3}) + h_{N−2}) h_{N−3} δ_{N−2})
        #   / (h_{N−2} + h_{N−3})
        last_numerator = (
            widths[-1] ** 2 * secants[-2]
            + (2.0 * end_span + widths[-1]) * widths[-2] * secants[-1]
        )
        last_rhs = last_numerator / end_span
        last_span = einx.id(" -> 1", end_span)
        last_value = einx.id(" -> 1", last_rhs)
        last = (last_span, widths[-2:-1], zero, last_value)
    rows = [
        einx.id("1, interior, 1 -> (1 + interior + 1)", head, body, tail)
        for head, body, tail in zip(
            first, (lower, diagonal, upper, rhs), last, strict=True
        )
    ]
    return solve_tridiagonal(*rows)


def _periodic_slopes(
    widths: Float[Array, " cell"], secants: Float[Array, " cell"]
) -> Float[Array, " node"]:
    """Slopes of the periodic spline: a cyclic system on d₀ … d_{N−2}, d_{N−1} = d₀.

    Algorithm:
        1. Build the cyclic rows; row k couples d_{k−1}, dₖ, d_{k+1}
           (indices mod N − 1).
        2. Write the cyclic matrix as A = T + u vᵀ with T tridiagonal.
        3. Solve T z₁ = rhs and T z₂ = u, then
           d = z₁ − (vᵀ z₁)/(1 + vᵀ z₂) z₂ (Sherman-Morrison).
    """
    n_rows = widths.shape[0]
    # hₖ₋₁ and δₖ₋₁, indices mod N − 1
    previous_width = einx.roll("[cell]", widths, shift=1)
    previous_secant = einx.roll("[cell]", secants, shift=1)
    lower = widths
    diagonal = 2.0 * (previous_width + widths)
    upper = previous_width
    rhs = 3.0 * (widths * previous_secant + previous_width * secants)
    # corners: row 0 reads d_{N−2} with lower[0], row N − 2 reads d₀ with upper[N − 2]
    corner_low = lower[0]
    corner_high = upper[n_rows - 1]
    gamma = -diagonal[0]
    # T = A − u vᵀ with u = (γ, 0, …, 0, corner_high),
    # v = (1, 0, …, 0, corner_low / γ)
    modified = diagonal.at[0].add(-gamma)
    modified = modified.at[n_rows - 1].add(-corner_low * corner_high / gamma)
    update_left = jnp.zeros(n_rows, widths.dtype)
    update_left = update_left.at[0].set(gamma)
    update_left = update_left.at[n_rows - 1].set(corner_high)
    update_right = jnp.zeros(n_rows, widths.dtype)
    update_right = update_right.at[0].set(1.0)
    update_right = update_right.at[n_rows - 1].set(corner_low / gamma)
    # z₁ = T⁻¹ rhs and z₂ = T⁻¹ u
    base = solve_tridiagonal(lower, modified, upper, rhs)
    correction = solve_tridiagonal(lower, modified, upper, update_left)
    # vᵀ z₁ and vᵀ z₂
    projected_base = einx.dot("cell, cell ->", update_right, base)
    projected_correction = einx.dot("cell, cell ->", update_right, correction)
    # d = z₁ − (vᵀ z₁) / (1 + vᵀ z₂) z₂
    scale = projected_base / (1.0 + projected_correction)
    slopes = base - scale * correction
    # d_{N−1} = d₀
    return einx.id("cell, -> (cell + 1)", slopes, slopes[0])


# one slope function per condition, made once, so plans that share a
# condition share the same static object (and the same jit cache entry)
SPLINE_SLOPES: dict[str, functools.partial[jax.Array]] = {
    condition: functools.partial(spline_slopes, condition=condition)
    for condition in SPLINE_CONDITIONS
}
