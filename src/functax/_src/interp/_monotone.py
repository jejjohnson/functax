"""Slopes of the shape-preserving Hermite schemes: PCHIP, Akima and Steffen.

On N increasing knots xₖ with samples yₖ, widths hₖ = xₖ₊₁ − xₖ and secants
δₖ = (yₖ₊₁ − yₖ)/hₖ, each scheme gives the slope dₖ at every knot. Each
depends on y nonlinearly (a harmonic mean, a weighted average with
data-dependent weights, a limited minimum), so their plans have no fixed
operator; ``linearize(y)`` gives the tangent-linear map at y. Every division
guards its untaken branch, since ``jnp.where`` differentiates both branches
and a NaN there would poison the gradient at flat segments.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import einx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


__all__ = [
    "akima_slopes",
    "akima_totals",
    "pchip_slopes",
    "steffen_slopes",
]


def _safe(
    divisor: Float[Array, "*shape"], use: Bool[Array, "*shape"]
) -> Float[Array, "*shape"]:
    """``divisor`` where ``use`` holds, 1 elsewhere: the untaken branch is finite."""
    return jnp.where(use, divisor, jnp.ones((), divisor.dtype))


def _secants(
    nodes: Float[Array, " node"], values: Float[Array, " node"]
) -> tuple[Float[Array, " cell"], Float[Array, " cell"]]:
    """hₖ = xₖ₊₁ − xₖ and δₖ = (yₖ₊₁ − yₖ)/hₖ."""
    # hₖ = xₖ₊₁ − xₖ
    widths = nodes[1:] - nodes[:-1]
    # Δyₖ = yₖ₊₁ − yₖ
    rises = values[1:] - values[:-1]
    # δₖ = Δyₖ / hₖ
    secants = rises / widths
    return widths, secants


def _pchip_edge(
    width: Float[Array, ""],
    next_width: Float[Array, ""],
    secant: Float[Array, ""],
    next_secant: Float[Array, ""],
) -> Float[Array, ""]:
    """SciPy's one-sided three-point end slope, with its shape-preserving limits."""
    # d = ((2h₀ + h₁) δ₀ − h₀ δ₁) / (h₀ + h₁)
    numerator = (2.0 * width + next_width) * secant - width * next_secant
    slope = numerator / (width + next_width)
    # sgn d ≠ sgn δ₀: the end slope points the wrong way, so it becomes 0
    wrong_sign = jnp.sign(slope) != jnp.sign(secant)
    # sgn δ₀ ≠ sgn δ₁ and |d| > 3|δ₀|: the end cubic would overshoot
    turns = jnp.sign(secant) != jnp.sign(next_secant)
    too_steep = jnp.abs(slope) > 3.0 * jnp.abs(secant)
    overshoot = turns & too_steep
    # d ← 3δ₀ where it overshoots
    limited = jnp.where(overshoot & ~wrong_sign, 3.0 * secant, slope)
    return jnp.where(wrong_sign, jnp.zeros((), slope.dtype), limited)


def pchip_slopes(
    nodes: Float[Array, " node"], values: Float[Array, " node"]
) -> Float[Array, " node"]:
    """Fritsch-Carlson slopes, as SciPy's ``PchipInterpolator``.

    dₖ = (w₁ + w₂)/(w₁/δₖ₋₁ + w₂/δₖ) when δₖ₋₁ δₖ > 0 and 0 otherwise, with
    w₁ = 2hₖ + hₖ₋₁ and w₂ = hₖ + 2hₖ₋₁; one-sided three-point slopes at the
    ends.

    Algorithm:
        1. Form the widths hₖ and secants δₖ.
        2. At each interior knot, take the weighted harmonic mean of δₖ₋₁ and
           δₖ, or 0 where they differ in sign or one is 0.
        3. At each end, take SciPy's one-sided three-point slope, limited so
           the end cubic keeps the data's shape.

    Args:
        nodes: x, the N increasing knots.
        values: y, the N samples.

    Returns:
        d, the N knot slopes.

    Examples:
        A monotone step gets slope 0 at its flat knots:

        >>> import jax.numpy as jnp
        >>> from functax._src.interp._monotone import pchip_slopes
        >>> slopes = pchip_slopes(jnp.arange(4.0), jnp.array([0.0, 0.0, 1.0, 1.0]))
        >>> slopes.shape
        (4,)
        >>> bool(jnp.allclose(slopes, jnp.zeros(4)))
        True

        Two knots give the secant at both ends, and a line keeps its slope:

        >>> bool(
        ...     jnp.allclose(pchip_slopes(jnp.array([0.0, 2.0]), jnp.ones(2)), 0.0)
        ... )
        True
        >>> line = pchip_slopes(jnp.arange(5.0), 3.0 * jnp.arange(5.0))
        >>> bool(jnp.allclose(line, 3.0))
        True

    References:
        Fritsch, F. N. and Carlson, R. E. (1980). Monotone piecewise cubic
        interpolation. SIAM Journal on Numerical Analysis 17(2), 238-246.
    """
    widths, secants = _secants(nodes, values)
    if nodes.shape[0] == 2:
        return einx.id("1, 1 -> (1 + 1)", secants, secants)

    # --- interior knots ---
    left_secant = secants[:-1]
    right_secant = secants[1:]
    previous_width = widths[:-1]
    next_width = widths[1:]
    # δₖ₋₁ δₖ ≤ 0: a local extremum or a flat cell, so dₖ = 0
    turns = jnp.sign(left_secant) != jnp.sign(right_secant)
    flat = turns | (left_secant == 0) | (right_secant == 0)
    # w₁ = 2hₖ + hₖ₋₁
    first_weight = 2.0 * next_width + previous_width
    # w₂ = hₖ + 2hₖ₋₁
    second_weight = next_width + 2.0 * previous_width
    # 1/dₖ = (w₁/δₖ₋₁ + w₂/δₖ)/(w₁ + w₂), guarded where the slope is 0
    left_term = first_weight / _safe(left_secant, ~flat)
    right_term = second_weight / _safe(right_secant, ~flat)
    inverse = (left_term + right_term) / (first_weight + second_weight)
    # dₖ = 1 / (1/dₖ)
    harmonic = 1.0 / _safe(inverse, ~flat)
    interior = jnp.where(flat, jnp.zeros((), values.dtype), harmonic)

    # --- ends ---
    first = _pchip_edge(widths[0], widths[1], secants[0], secants[1])
    last = _pchip_edge(widths[-1], widths[-2], secants[-1], secants[-2])
    return einx.id(", knot, -> (1 + knot + 1)", first, interior, last)


def _akima_jumps(
    nodes: Float[Array, " node"], values: Float[Array, " node"]
) -> tuple[Float[Array, " extended"], Float[Array, " node"], Float[Array, " node"]]:
    """The extended secants and Akima's weights f₂ = |δₖ₋₁ − δₖ₋₂|, f₁ = |δₖ₊₁ − δₖ|.

    The secants are extended by two on each side, continuing the end
    differences (δ₋₁ = 2δ₀ − δ₁, δ₋₂ = 2δ₋₁ − δ₀, and the same at the end).
    """
    _, secants = _secants(nodes, values)
    # δ₋₁ = 2δ₀ − δ₁
    left = 2.0 * secants[0] - secants[1]
    # δ₋₂ = 2δ₋₁ − δ₀
    far_left = 2.0 * left - secants[0]
    # δ_{N−1} = 2δ_{N−2} − δ_{N−3}
    right = 2.0 * secants[-1] - secants[-2]
    # δ_N = 2δ_{N−1} − δ_{N−2}
    far_right = 2.0 * right - secants[-1]
    extended = einx.id(
        ", , cell, , -> (1 + 1 + cell + 1 + 1)",
        far_left,
        left,
        secants,
        right,
        far_right,
    )
    # |δⱼ₊₁ − δⱼ| over the extended secants
    differences = extended[1:] - extended[:-1]
    jumps = jnp.abs(differences)
    return extended, jumps[:-2], jumps[2:]


def akima_totals(
    nodes: Float[Array, " node"], values: Float[Array, " node"]
) -> Float[Array, " node"]:
    """f₁ + f₂ at each knot, whose maximum sets where Akima's slope is defined.

    Zero on two knots, where `akima_slopes` takes the secant.

    Args:
        nodes: x, the N increasing knots.
        values: y, the N samples.

    Returns:
        f₁ + f₂, one total per knot.

    Examples:
        A line has equal secants, so every total is zero:

        >>> import jax.numpy as jnp
        >>> from functax._src.interp._monotone import akima_totals
        >>> totals = akima_totals(jnp.arange(5.0), 2.0 * jnp.arange(5.0))
        >>> totals.shape
        (5,)
        >>> bool(jnp.allclose(totals, 0.0))
        True

        A kink at the middle knot shows in the totals on either side of it:

        >>> kinked = akima_totals(jnp.arange(5.0), jnp.array([0.0, 1, 2, 2, 2]))
        >>> bool(jnp.allclose(kinked, jnp.array([0.0, 1, 0, 1, 0])))
        True
    """
    if nodes.shape[0] == 2:
        return jnp.zeros(2, values.dtype)
    _, before, after = _akima_jumps(nodes, values)
    # f₁ + f₂
    return after + before


def akima_slopes(
    nodes: Float[Array, " node"],
    values: Float[Array, " node"],
    scale: Float[Array, ""] | None = None,
) -> Float[Array, " node"]:
    """Akima's slopes, as SciPy's ``Akima1DInterpolator(method="akima")``.

    The secants are extended by two on each side (δ₋₁ = 2δ₀ − δ₁, …); with
    f₁ = |δₖ₊₁ − δₖ| and f₂ = |δₖ₋₁ − δₖ₋₂|, the slope is
    dₖ = δₖ₋₁ + f₂/(f₁ + f₂) (δₖ − δₖ₋₁) where f₁ + f₂ exceeds 10⁻⁹ of its
    maximum, and the mean of the outer secants elsewhere.

    Algorithm:
        1. Extend the secants by two on each side and form f₁, f₂.
        2. Where f₁ + f₂ exceeds 10⁻⁹ of ``scale``, blend the two inner
           secants with weights f₂ and f₁.
        3. Elsewhere, take the mean of the two outer secants.

    Args:
        nodes: x, the N increasing knots.
        values: y, the N samples.
        scale: The maximum of f₁ + f₂ that the 10⁻⁹ threshold is relative
            to; that of these knots when ``None``. A piece of a longer axis
            passes the maximum over the piece.

    Returns:
        d, the N knot slopes.

    Examples:
        Akima reproduces a line:

        >>> import jax.numpy as jnp
        >>> from functax._src.interp._monotone import akima_slopes
        >>> slopes = akima_slopes(jnp.arange(5.0), 2.0 * jnp.arange(5.0))
        >>> slopes.shape
        (5,)
        >>> bool(jnp.allclose(slopes, 2.0))
        True

        A flat run after a ramp has slope 0 at its inner knot:

        >>> step = akima_slopes(jnp.arange(5.0), jnp.array([0.0, 1, 2, 2, 2]))
        >>> bool(jnp.allclose(step[3], 0.0))
        True

    References:
        Akima, H. (1970). A new method of interpolation and smooth curve
        fitting based on local procedures. Journal of the ACM 17(4), 589-602.
    """
    _, secants = _secants(nodes, values)
    if nodes.shape[0] == 2:
        return einx.id("1, 1 -> (1 + 1)", secants, secants)
    extended, before, after = _akima_jumps(nodes, values)
    # f₁ + f₂
    total = after + before
    largest = einx.max("[node]", total) if scale is None else scale
    # f₁ + f₂ > 10⁻⁹ max(f₁ + f₂)
    defined = total > 1e-9 * largest
    # the fallback where the slope is not defined: ½ (δₖ₊₁ + δₖ₋₂)
    mean = 0.5 * (extended[3:] + extended[:-3])
    # f₂/(f₁ + f₂), guarded where the slope is not defined
    share = before / _safe(total, defined)
    # δₖ − δₖ₋₁
    step = extended[2:-1] - extended[1:-2]
    # dₖ = δₖ₋₁ + f₂/(f₁ + f₂) (δₖ − δₖ₋₁)
    blended = extended[1:-2] + share * step
    return jnp.where(defined, blended, mean)


def _steffen_edge(
    width: Float[Array, ""],
    next_width: Float[Array, ""],
    secant: Float[Array, ""],
    next_secant: Float[Array, ""],
) -> Float[Array, ""]:
    """Steffen's end slope: the parabola's, limited to [0, 2δ₀] in δ₀'s direction."""
    # s = h₀/(h₀ + h₁)
    share = width / (width + next_width)
    # p₀ = δ₀ (1 + s) − δ₁ s
    parabola = secant * (1.0 + share) - next_secant * share
    # p₀ δ₀ ≤ 0: the parabola's slope points against the data, so it becomes 0
    opposite = parabola * secant <= 0
    # |p₀| > 2|δ₀|: limited to 2δ₀
    too_steep = jnp.abs(parabola) > 2.0 * jnp.abs(secant)
    limited = jnp.where(too_steep, 2.0 * secant, parabola)
    return jnp.where(opposite, jnp.zeros((), parabola.dtype), limited)


def steffen_slopes(
    nodes: Float[Array, " node"], values: Float[Array, " node"]
) -> Float[Array, " node"]:
    """Steffen's slopes: monotone between knots by construction.

    pₖ = (δₖ₋₁ hₖ + δₖ hₖ₋₁)/(hₖ₋₁ + hₖ) is the slope of the parabola
    through three knots, and dₖ = (sgn δₖ₋₁ + sgn δₖ) min(|δₖ₋₁|, |δₖ|, ½|pₖ|).

    Algorithm:
        1. Form the widths hₖ and secants δₖ.
        2. At each interior knot, take the parabola's slope pₖ and limit it
           by the neighbouring secants.
        3. At each end, take the end parabola's slope, limited to [0, 2δ].

    Args:
        nodes: x, the N increasing knots.
        values: y, the N samples.

    Returns:
        d, the N knot slopes.

    Examples:
        Steffen keeps a line:

        >>> import jax.numpy as jnp
        >>> from functax._src.interp._monotone import steffen_slopes
        >>> slopes = steffen_slopes(jnp.arange(4.0), 2.0 * jnp.arange(4.0))
        >>> slopes.shape
        (4,)
        >>> bool(jnp.allclose(slopes, 2.0))
        True

        A local maximum gets slope 0:

        >>> peak = steffen_slopes(jnp.arange(3.0), jnp.array([0.0, 1.0, 0.0]))
        >>> bool(jnp.allclose(peak[1], 0.0))
        True

    References:
        Steffen, M. (1990). A simple method for monotonic interpolation in one
        dimension. Astronomy and Astrophysics 239, 443-450.
    """
    widths, secants = _secants(nodes, values)
    if nodes.shape[0] == 2:
        return einx.id("1, 1 -> (1 + 1)", secants, secants)
    left_secant = secants[:-1]
    right_secant = secants[1:]
    previous_width = widths[:-1]
    next_width = widths[1:]
    # pₖ = (δₖ₋₁ hₖ + δₖ hₖ₋₁)/(hₖ₋₁ + hₖ)
    weighted = left_secant * next_width + right_secant * previous_width
    parabola = weighted / (previous_width + next_width)
    # min(|δₖ₋₁|, |δₖ|, ½|pₖ|)
    smaller_secant = jnp.minimum(jnp.abs(left_secant), jnp.abs(right_secant))
    smallest = jnp.minimum(smaller_secant, 0.5 * jnp.abs(parabola))
    # dₖ = (sgn δₖ₋₁ + sgn δₖ) min(…)
    direction = jnp.sign(left_secant) + jnp.sign(right_secant)
    interior = direction * smallest
    first = _steffen_edge(widths[0], widths[1], secants[0], secants[1])
    last = _steffen_edge(widths[-1], widths[-2], secants[-1], secants[-2])
    return einx.id(", knot, -> (1 + knot + 1)", first, interior, last)
