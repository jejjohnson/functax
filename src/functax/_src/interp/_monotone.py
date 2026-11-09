"""Slopes of the shape-preserving Hermite schemes: PCHIP, Akima and Steffen.

Each depends on y nonlinearly (a harmonic mean, a weighted average with
data-dependent weights, a limited minimum), so their plans have no fixed
operator; ``linearize(y)`` gives the tangent-linear map at y. Every division
guards its untaken branch, since ``jnp.where`` differentiates both branches
and a NaN there would poison the gradient at flat segments.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import einx
import jax
import jax.numpy as jnp


__all__ = [
    "akima_slopes",
    "pchip_slopes",
    "steffen_slopes",
]


def _safe(divisor: jax.Array, use: jax.Array) -> jax.Array:
    """``divisor`` where ``use`` holds, 1 elsewhere: the untaken branch stays finite."""
    return jnp.where(use, divisor, jnp.ones((), divisor.dtype))


def _secants(nodes: jax.Array, values: jax.Array) -> tuple[jax.Array, jax.Array]:
    """hₖ = xₖ₊₁ − xₖ and δₖ = (yₖ₊₁ − yₖ)/hₖ."""
    widths = nodes[1:] - nodes[:-1]
    return widths, (values[1:] - values[:-1]) / widths


def _pchip_edge(
    width: jax.Array, next_width: jax.Array, secant: jax.Array, next_secant: jax.Array
) -> jax.Array:
    """SciPy's one-sided three-point end slope, with its shape-preserving limits."""
    # d = ((2h₀ + h₁) δ₀ − h₀ δ₁) / (h₀ + h₁)
    slope = ((2.0 * width + next_width) * secant - width * next_secant) / (
        width + next_width
    )
    wrong_sign = jnp.sign(slope) != jnp.sign(secant)
    overshoot = (jnp.sign(secant) != jnp.sign(next_secant)) & (
        jnp.abs(slope) > 3.0 * jnp.abs(secant)
    )
    limited = jnp.where(overshoot & ~wrong_sign, 3.0 * secant, slope)
    return jnp.where(wrong_sign, jnp.zeros((), slope.dtype), limited)


def pchip_slopes(nodes: jax.Array, values: jax.Array) -> jax.Array:
    """Fritsch-Carlson slopes, as SciPy's ``PchipInterpolator``.

    mₖ = (w₁ + w₂)/(w₁/δₖ₋₁ + w₂/δₖ) when δₖ₋₁ δₖ > 0 and 0 otherwise, with
    w₁ = 2hₖ + hₖ₋₁ and w₂ = hₖ + 2hₖ₋₁; one-sided three-point slopes at the
    ends.

    References:
        Fritsch, F. N. and Carlson, R. E. (1980). Monotone piecewise cubic
        interpolation. SIAM Journal on Numerical Analysis 17(2), 238-246.
    """
    widths, secants = _secants(nodes, values)
    if nodes.shape[0] == 2:
        return einx.id("1, 1 -> (1 + 1)", secants, secants)
    left, right = secants[:-1], secants[1:]
    flat = (jnp.sign(left) != jnp.sign(right)) | (left == 0) | (right == 0)
    w1 = 2.0 * widths[1:] + widths[:-1]
    w2 = widths[1:] + 2.0 * widths[:-1]
    # 1/mₖ = (w₁/δₖ₋₁ + w₂/δₖ)/(w₁ + w₂), guarded where the slope is 0
    inverse = (w1 / _safe(left, ~flat) + w2 / _safe(right, ~flat)) / (w1 + w2)
    interior = jnp.where(flat, jnp.zeros((), values.dtype), 1.0 / _safe(inverse, ~flat))
    first = _pchip_edge(widths[0], widths[1], secants[0], secants[1])
    last = _pchip_edge(widths[-1], widths[-2], secants[-1], secants[-2])
    return einx.id(", n, -> (1 + n + 1)", first, interior, last)


def akima_slopes(nodes: jax.Array, values: jax.Array) -> jax.Array:
    """Akima's slopes, as SciPy's ``Akima1DInterpolator(method="akima")``.

    The secants are extended by two on each side (m₋₁ = 2m₀ − m₁, …); with
    f₁ = |mₖ₊₁ − mₖ| and f₂ = |mₖ₋₁ − mₖ₋₂|, the slope is
    mₖ₋₁ + f₂/(f₁ + f₂) (mₖ − mₖ₋₁) where f₁ + f₂ exceeds 10⁻⁹ of its
    maximum, and the mean of the outer secants elsewhere.

    References:
        Akima, H. (1970). A new method of interpolation and smooth curve
        fitting based on local procedures. Journal of the ACM 17(4), 589-602.
    """
    _, secants = _secants(nodes, values)
    if nodes.shape[0] == 2:
        return einx.id("1, 1 -> (1 + 1)", secants, secants)
    # two extra secants at each end, continuing the end differences
    left = 2.0 * secants[0] - secants[1]
    far_left = 2.0 * left - secants[0]
    right = 2.0 * secants[-1] - secants[-2]
    far_right = 2.0 * right - secants[-1]
    extended = einx.id(
        ", , n, , -> (1 + 1 + n + 1 + 1)", far_left, left, secants, right, far_right
    )
    jumps = jnp.abs(extended[1:] - extended[:-1])
    after = jumps[2:]
    before = jumps[:-2]
    total = after + before
    defined = total > 1e-9 * jnp.max(total)
    # the fallback where the slope is not defined: the mean of the outer secants
    mean = 0.5 * (extended[3:] + extended[:-3])
    blended = extended[1:-2] + (before / _safe(total, defined)) * (
        extended[2:-1] - extended[1:-2]
    )
    return jnp.where(defined, blended, mean)


def _steffen_edge(
    width: jax.Array, next_width: jax.Array, secant: jax.Array, next_secant: jax.Array
) -> jax.Array:
    """Steffen's end slope: the parabola's, limited to [0, 2δ₀] in δ₀'s direction."""
    # p₀ = δ₀ (1 + h₀/(h₀ + h₁)) − δ₁ h₀/(h₀ + h₁)
    share = width / (width + next_width)
    parabola = secant * (1.0 + share) - next_secant * share
    opposite = parabola * secant <= 0
    too_steep = jnp.abs(parabola) > 2.0 * jnp.abs(secant)
    limited = jnp.where(too_steep, 2.0 * secant, parabola)
    return jnp.where(opposite, jnp.zeros((), parabola.dtype), limited)


def steffen_slopes(nodes: jax.Array, values: jax.Array) -> jax.Array:
    """Steffen's slopes: monotone between knots by construction.

    pₖ = (δₖ₋₁ hₖ + δₖ hₖ₋₁)/(hₖ₋₁ + hₖ) is the slope of the parabola
    through three knots, and mₖ = (sgn δₖ₋₁ + sgn δₖ) min(|δₖ₋₁|, |δₖ|, ½|pₖ|).

    References:
        Steffen, M. (1990). A simple method for monotonic interpolation in one
        dimension. Astronomy and Astrophysics 239, 443-450.
    """
    widths, secants = _secants(nodes, values)
    if nodes.shape[0] == 2:
        return einx.id("1, 1 -> (1 + 1)", secants, secants)
    left, right = secants[:-1], secants[1:]
    previous_width, next_width = widths[:-1], widths[1:]
    # pₖ = (δₖ₋₁ hₖ + δₖ hₖ₋₁)/(hₖ₋₁ + hₖ)
    parabola = (left * next_width + right * previous_width) / (
        previous_width + next_width
    )
    smallest = jnp.minimum(
        jnp.minimum(jnp.abs(left), jnp.abs(right)), 0.5 * jnp.abs(parabola)
    )
    interior = (jnp.sign(left) + jnp.sign(right)) * smallest
    first = _steffen_edge(widths[0], widths[1], secants[0], secants[1])
    last = _steffen_edge(widths[-1], widths[-2], secants[-1], secants[-2])
    return einx.id(", n, -> (1 + n + 1)", first, interior, last)
