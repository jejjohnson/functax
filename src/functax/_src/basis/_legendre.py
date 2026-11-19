"""Legendre polynomials and series, and the normalised associated Legendre functions.

The one-dimensional building blocks of the spherical bases. The associated
functions run the orthonormal recurrences as one ``lax.scan`` over the degree,
carried as a mantissa and a binary exponent per order, so they stay finite to
high degree in float32 as well as float64. Moved from geonnax ``_legendre`` at
``be5537d``; Gauss–Legendre nodes moved to `functax.quad`. `legendre_series`
moved from kernellib ``functional._geo`` at ``c2d1fce``.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import math
import operator
from typing import Any

import einx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float, Int

from functax.exceptions import ValidationError


__all__ = [
    "associated_legendre",
    "associated_legendre_indices",
    "legendre_polynomials",
    "legendre_series",
]


_NORMALIZATIONS: tuple[str, ...] = ("orthonormal", "schmidt", "none")


def _check_degree(value: Any, name: str, *, minimum: int = 0) -> int:
    """``value`` as an int of at least ``minimum``; floats and bools fail."""
    message = f"'{name}' must be an integer of at least {minimum}, got {value!r}"
    if isinstance(value, bool):
        raise ValidationError(message)
    try:
        count = operator.index(value)
    except TypeError as err:
        raise ValidationError(message) from err
    if count < minimum:
        raise ValidationError(message)
    return count


def legendre_polynomials(
    t: Float[Array, "*batch"], l_max: int
) -> Float[Array, "... degree"]:
    """The Legendre polynomials P₀(t), …, P_{l_max}(t).

    P₀ = 1, P₁ = t and (l + 1) P_{l+1} = (2l + 1) t P_l − l P_{l−1}, run as a
    ``lax.scan`` over l, so the trace does not grow with ``l_max``.

    Algorithm:
        1. Start from P₀ = 1 and P₁ = t.
        2. Scan l = 1, …, l_max − 1, each step giving P_{l+1} from P_l and
           P_{l−1}.
        3. Stack P₀, P₁ and the scanned rows on a new last axis.

    Args:
        t: Points in [−1, 1], any shape.
        l_max: The highest degree, inclusive.

    Returns:
        P_l(t), with the degree l on a new last axis of length l_max + 1.

    Raises:
        ValidationError: If ``l_max`` is not an integer of at least 0.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.polynomial import legendre_polynomials
        >>> # P₂(½) = (3/4 − 1)/2 = −1/8
        >>> values = legendre_polynomials(jnp.array([0.5]), 2)
        >>> bool(jnp.allclose(values, jnp.array([[1.0, 0.5, -0.125]])))
        True
        >>> legendre_polynomials(jnp.zeros((2, 3)), 4).shape
        (2, 3, 5)
        >>> # P_l(1) = 1 for every l
        >>> bool(jnp.allclose(legendre_polynomials(jnp.ones(1), 6), 1.0))
        True
    """
    degree = _check_degree(l_max, "l_max")
    t = jnp.asarray(t)
    p0 = jnp.ones_like(t)
    if degree == 0:
        return einx.id("... -> ... 1", p0)
    if degree == 1:
        return einx.id("..., ... -> ... (1 + 1)", p0, t)

    def step(
        carry: tuple[Float[Array, "*batch"], Float[Array, "*batch"]],
        ell: Float[Array, ""],
    ) -> tuple[
        tuple[Float[Array, "*batch"], Float[Array, "*batch"]], Float[Array, "*batch"]
    ]:
        p_prev, p_curr = carry
        # P_{l+1} = ((2l + 1) t P_l − l P_{l−1}) / (l + 1)
        p_next = ((2.0 * ell + 1.0) * t * p_curr - ell * p_prev) / (ell + 1.0)
        return (p_curr, p_next), p_next

    ells = jnp.arange(1, degree, dtype=t.dtype)
    _, rest = jax.lax.scan(step, (p0, t), ells)
    return einx.id("..., ..., degree ... -> ... (1 + 1 + degree)", p0, t, rest)


def legendre_series(
    x: Float[ArrayLike, "*batch"], coeffs: Float[ArrayLike, " degree"]
) -> Float[Array, "*batch"]:
    """The Legendre series f(x) = ∑ₗ cₗ Pₗ(x), l = 0, …, L, evaluated elementwise.

    The polynomials come from the three-term recurrence P₀ = 1, P₁ = x and
    (l + 1) P_{l+1} = (2l + 1) x P_l − l P_{l−1}, run as a ``lax.scan`` over l
    that accumulates the sum as it goes: O(size(x) L) time and O(size(x))
    memory, so the (…, L + 1) table of `legendre_polynomials` is never formed,
    which matters for a Gram matrix of u · v. The value of x is clipped to
    [−1, 1] first, so rounding in a dot product of unit vectors is harmless;
    the gradient passes straight through the clip, so derivatives at
    coincident points (u · u = 1 up to rounding) are exact. The signature and
    the recurrence are kernellib's.

    Algorithm:
        1. Promote x and c to their common floating dtype, and clip the value
           (not the gradient) of x to [−1, 1].
        2. Start the sum at c₀ P₀ + c₁ P₁.
        3. Scan l = 1, …, L − 1: P_{l+1} from P_l and P_{l−1}, and add c_{l+1} P_{l+1}.

    Args:
        x: The points x, any shape; typically cosines u · v of unit vectors.
        coeffs: The L + 1 coefficients c₀, …, c_L, by degree.

    Returns:
        f(x) at every point, with the shape of ``x``.

    Raises:
        ValidationError: If ``coeffs`` is not 1-D and non-empty.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.polynomial import legendre_series
        >>> # P₂(x) = (3x² − 1)/2
        >>> values = legendre_series(
        ...     jnp.array([0.0, 0.5, 1.0]), jnp.array([0.0, 0.0, 1.0])
        ... )
        >>> bool(jnp.allclose(values, jnp.array([-0.5, -0.125, 1.0])))
        True
        >>> legendre_series(jnp.zeros((2, 3)), jnp.ones(4)).shape
        (2, 3)
        >>> # a single coefficient is the constant c₀
        >>> constant = legendre_series(jnp.array([0.3]), jnp.array([2.0]))
        >>> bool(jnp.allclose(constant, 2.0))
        True
    """
    coeffs = jnp.asarray(coeffs)
    if coeffs.ndim != 1 or coeffs.shape[0] == 0:
        msg = f"'coeffs' must be 1-D and non-empty, got shape {coeffs.shape}"
        raise ValidationError(msg)
    x = jnp.asarray(x)
    dtype = jnp.result_type(x, coeffs, 1.0)
    x = x.astype(dtype)
    # Clip the value only: ``jnp.clip`` has a zero (or half) gradient at the
    # boundary, which would corrupt the derivatives of a kernel on coincident
    # points, where u · u is 1 up to rounding.
    clipped = jnp.clip(x, -1.0, 1.0)
    correction = jax.lax.stop_gradient(clipped - x)
    x = x + correction
    coeffs = coeffs.astype(dtype)
    p0 = jnp.ones_like(x)
    # c₀ P₀
    total = coeffs[0] * p0
    if coeffs.shape[0] == 1:
        return total
    # + c₁ P₁
    total = total + coeffs[1] * x

    def step(
        carry: tuple[
            Float[Array, "*batch"], Float[Array, "*batch"], Float[Array, "*batch"]
        ],
        inputs: tuple[Float[Array, ""], Float[Array, ""]],
    ) -> tuple[
        tuple[Float[Array, "*batch"], Float[Array, "*batch"], Float[Array, "*batch"]],
        None,
    ]:
        p_prev, p_curr, accumulated = carry
        ell, coefficient = inputs
        # P_{l+1} = ((2l + 1) x P_l − l P_{l−1}) / (l + 1)
        p_next = ((2.0 * ell + 1.0) * x * p_curr - ell * p_prev) / (ell + 1.0)
        # f ← f + c_{l+1} P_{l+1}
        accumulated = accumulated + coefficient * p_next
        return (p_curr, p_next, accumulated), None

    ells = jnp.arange(1, coeffs.shape[0] - 1, dtype=dtype)
    (_, _, total), _ = jax.lax.scan(step, (p0, x, total), (ells, coeffs[2:]))
    return total


def associated_legendre_indices(l_max: int) -> tuple[tuple[int, int], ...]:
    """The (l, m) of each column of `associated_legendre`.

    Columns run with l outermost and m = 0, …, l inner, so there are
    (l_max + 1)(l_max + 2)/2 of them.

    Args:
        l_max: The highest degree, inclusive.

    Returns:
        One (l, m) pair per column.

    Raises:
        ValidationError: If ``l_max`` is not an integer of at least 0.

    Examples:
        >>> from functax.basis.polynomial import associated_legendre_indices
        >>> associated_legendre_indices(2)
        ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2))
        >>> len(associated_legendre_indices(10))
        66
        >>> associated_legendre_indices(0)
        ((0, 0),)
    """
    degree = _check_degree(l_max, "l_max")
    return tuple((ell, m) for ell in range(degree + 1) for m in range(ell + 1))


def associated_legendre(
    t: Float[Array, "*batch"],
    l_max: int,
    *,
    normalization: str = "orthonormal",
) -> Float[Array, "... feature"]:
    """The associated Legendre functions P_l^m(t) for 0 ≤ m ≤ l ≤ l_max.

    The fully normalised P̄_l^m come from the recurrences

        P̄₀⁰ = 1/√2,  P̄_m^m = −√((2m + 1)/(2m)) √(1 − t²) P̄_{m−1}^{m−1},
        P̄_{m+1}^m = √(2m + 3) t P̄_m^m,
        P̄_l^m = a_lm (t P̄_{l−1}^m − b_lm P̄_{l−2}^m),
        a_lm = √((4l² − 1)/(l² − m²)),  b_lm = √(((l − 1)² − m²)/(4(l − 1)² − 1)),

    each order carried as a mantissa and a binary exponent, so they stay finite
    and accurate to high degree (a Gram error of about 3e-5 in float32 at
    l_max = 511 on the Gauss rule).

    Algorithm:
        1. Run the sectoral recurrence for P̄_m^m over m, as mantissas and
           exponents.
        2. Run the recurrence in l for every order at once, from those seeds.
        3. Pick the (l, m) columns, and rescale them for ``"schmidt"`` or
           ``"none"``.

    The normalisations:

    - ``"orthonormal"``: ∫₋₁¹ P̄_l^m P̄_l′^m dt = δ_ll′, with the
      Condon–Shortley phase (−1)^m, the phase of `real_spherical_harmonics`:
      Y_l^0 = P̄_l^0/√(2π) and Y_l^{±m} = P̄_l^m {cos, sin}(mλ)/√π.
    - ``"schmidt"``: Schmidt semi-normalised, √((2 − δ_m0)(l − m)!/(l + m)!) P_l^m,
      without the Condon–Shortley phase (the geomagnetism convention).
    - ``"none"``: the classical P_l^m with the Condon–Shortley phase, as
      ``scipy.special.lpmv``; it overflows in float64 near l + m ≈ 340.

    Args:
        t: Points in [−1, 1], any shape. Integers become float32.
        l_max: The highest degree, inclusive.
        normalization: ``"orthonormal"``, ``"schmidt"`` or ``"none"``.

    Returns:
        P_l^m(t), with the (l, m) of `associated_legendre_indices` on a new
        last axis.

    Raises:
        ValidationError: If ``l_max`` is not an integer of at least 0, or
            ``normalization`` is unknown.

    References:
        Fukushima, T. (2012). Numerical computation of spherical harmonics of
        arbitrary degree and order by extending exponent of floating point
        numbers. Journal of Geodesy 86, 271–285.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.polynomial import associated_legendre
        >>> # P₀⁰ = 1, P₁⁰(0) = 0 and P₁¹(0) = −1, with the Condon–Shortley phase
        >>> values = associated_legendre(jnp.array([0.0]), 1, normalization="none")
        >>> bool(jnp.allclose(values, jnp.array([[1.0, 0.0, -1.0]]), atol=1e-12))
        True
        >>> associated_legendre(jnp.zeros((4,)), 3).shape
        (4, 10)
        >>> # the Schmidt P₁¹(0) is +1, without the phase
        >>> schmidt = associated_legendre(
        ...     jnp.array([0.0]), 1, normalization="schmidt"
        ... )
        >>> bool(jnp.allclose(schmidt, jnp.array([[1.0, 0.0, 1.0]]), atol=1e-12))
        True
    """
    degree = _check_degree(l_max, "l_max")
    if normalization not in _NORMALIZATIONS:
        msg = (
            f"'normalization' must be one of {list(_NORMALIZATIONS)}, "
            f"got {normalization!r}"
        )
        raise ValidationError(msg)
    t = jnp.asarray(t)
    dtype = t.dtype if jnp.issubdtype(t.dtype, jnp.floating) else jnp.float32
    t = t.astype(dtype)
    # 1 − t², clipped against round-off below 0
    one_minus_t2 = jnp.maximum(1.0 - t**2, 0.0)
    # s = √(1 − t²)
    s = jnp.sqrt(one_minus_t2)

    # The sectoral P̄_m^m as mantissa · 2^exponent: the carry keeps (1 − t²)^{m/2}
    # from underflowing near the poles, where P̄_l^m grows back to O(1) for l ≫ m
    sectoral, sectoral_exp = _sectoral_carry(s, degree)
    mantissa, exponent = _legendre_table_carry(t, sectoral, sectoral_exp, degree)
    table = _ldexp(mantissa, exponent)

    pairs = associated_legendre_indices(degree)
    ell_index = np.array([ell for ell, _ in pairs])
    m_index = np.array([order for _, order in pairs])
    # the table is indexed [l, ..., m]; pick each (l, m) and move it last
    picked = table[ell_index, ..., m_index]
    out = einx.id("feature ... -> ... feature", picked)

    if normalization == "orthonormal":
        return out
    # --- from P̄ (unit L² norm on [−1, 1]) to the classical P_l^m ---
    # log((l + m)!/(l − m)!)
    log_ratio = np.array(
        [math.lgamma(ell + m + 1) - math.lgamma(ell - m + 1) for ell, m in pairs]
    )
    # (l + m)!/(l − m)!
    ratio = np.exp(log_ratio)
    # √(2/(2l + 1) · (l + m)!/(l − m)!)
    to_classical = np.sqrt(2.0 / (2.0 * ell_index + 1.0) * ratio)
    if normalization == "none":
        classical_scale = jnp.asarray(to_classical, dtype=dtype)
        # P_l^m = √(2/(2l + 1) · (l + m)!/(l − m)!) P̄_l^m
        return einx.multiply("... feature, feature", out, classical_scale)
    # --- √((2 − δ_m0)(l − m)!/(l + m)!) P_l^m, without the Condon–Shortley phase
    # 2(2 − δ_m0)
    numerator = 2.0 * (2.0 - (m_index == 0))
    # √(2(2 − δ_m0)/(2l + 1))
    schmidt = np.sqrt(numerator / (2.0 * ell_index + 1.0))
    # (−1)^m cancels the Condon–Shortley phase
    schmidt = schmidt * (-1.0) ** m_index
    schmidt_scale = jnp.asarray(schmidt, dtype=dtype)
    return einx.multiply("... feature, feature", out, schmidt_scale)


def _sectoral_scale(l_max: int) -> Float[np.ndarray, " order"]:
    """c_m = P̄_m^m / (1 − t²)^{m/2} for m ≤ l_max.

    c₀ = 1/√2 and c_m = −√((2m + 1)/(2m)) c_{m−1}: the orthonormal sectoral
    recurrence with the factor √(1 − t²) taken out.
    """
    m = np.arange(l_max + 1)
    factor = np.ones(l_max + 1)
    # −√((2m + 1)/(2m)), m ≥ 1
    factor[1:] = -np.sqrt((2.0 * m[1:] + 1.0) / (2.0 * m[1:]))
    # ∏_{k ≤ m} of the factors
    product = np.cumprod(factor)
    # c_m = c₀ ∏_{k ≤ m} (−√((2k + 1)/(2k))), with c₀ = 1/√2
    return product / math.sqrt(2.0)


def _ldexp(
    x: Float[Array, "*batch"], e: Int[Array, "*batch"]
) -> Float[Array, "*batch"]:
    """x · 2^e, with the exact derivative 2^e everywhere.

    ``jnp.ldexp`` passes x = 0 through a ``where``, so its derivative there is
    1 rather than 2^e, and the Legendre values are exactly zero at many points
    (odd l − m at t = 0, (x + iy)^m on the axes) where their derivative is not.
    The power of two is split in two halves, and e is clipped to where both are
    finite, so x = 0 gives 0 (never 0 · ∞) and a non-zero x overflows or
    underflows only when the result does.
    """
    # the largest finite power of two
    max_half = jnp.finfo(x.dtype).maxexp - 1
    e = jnp.clip(e, -2 * max_half, 2 * max_half)
    # e = ⌊e/2⌋ + (e − ⌊e/2⌋)
    half = e // 2
    rest = e - half
    two = jnp.asarray(2.0, dtype=x.dtype)
    # 2^⌊e/2⌋
    low_power = two ** half.astype(x.dtype)
    # 2^(e − ⌊e/2⌋)
    high_power = two ** rest.astype(x.dtype)
    # x · 2^⌊e/2⌋
    scaled = x * low_power
    # x · 2^e
    return scaled * high_power


# the carry of `_sectoral_carry`: the mantissa and binary exponent of P̄_m^m
_SectoralCarry = tuple[Float[Array, "*batch"], Int[Array, "*batch"]]

# the carry of `_legendre_table_carry`: rows l − 2 and l − 1 and their exponent
_TableCarry = tuple[
    Float[Array, "*batch order"],
    Float[Array, "*batch order"],
    Int[Array, "*batch order"],
]


def _sectoral_carry(
    s: Float[Array, "*batch"], l_max: int
) -> tuple[Float[Array, "... order"], Int[Array, "... order"]]:
    """The sectoral P̄_m^m = c_m s^m as a mantissa and a binary exponent.

    Runs P̄_m^m = −√((2m + 1)/(2m)) s P̄_{m−1}^{m−1} as a ``lax.scan`` over m,
    renormalising with ``jnp.frexp`` at every step, so the mantissa stays in
    [½, 1) and s^m never underflows. Where s = 0 the mantissa is 0 for m ≥ 1,
    which is exact.
    """
    dtype = s.dtype
    mantissa0 = jnp.full_like(s, 1.0 / math.sqrt(2.0))
    exponent0 = jnp.zeros(s.shape, dtype=jnp.int32)
    if l_max == 0:
        return einx.id("... -> ... 1", mantissa0), einx.id("... -> ... 1", exponent0)
    m = np.arange(1, l_max + 1)
    # −√((2m + 1)/(2m))
    host_factors = -np.sqrt((2.0 * m + 1.0) / (2.0 * m))
    factors = jnp.asarray(host_factors, dtype=dtype)

    def step(
        carry: _SectoralCarry, factor: Float[Array, ""]
    ) -> tuple[_SectoralCarry, _SectoralCarry]:
        mantissa, exponent = carry
        # P̄_m^m = −√((2m + 1)/(2m)) s P̄_{m−1}^{m−1}, in units of 2^exponent
        value = factor * s * mantissa
        # 2^shift ≥ |value|, so the new mantissa lies in [½, 1)
        _, shift = jnp.frexp(value)
        # a zero value keeps its exponent
        shift = jnp.where(value != 0, shift, 0)
        # value · 2^−shift
        mantissa = _ldexp(value, -shift)
        exponent = exponent + shift
        return (mantissa, exponent), (mantissa, exponent)

    _, (mantissas, exponents) = jax.lax.scan(step, (mantissa0, exponent0), factors)
    mantissa = einx.id("..., order ... -> ... (1 + order)", mantissa0, mantissas)
    exponent = einx.id("..., order ... -> ... (1 + order)", exponent0, exponents)
    return mantissa, exponent


def _legendre_table_carry(
    t: Float[Array, "*batch"],
    sectoral: Float[Array, "... order"],
    sectoral_exp: Int[Array, "... order"],
    l_max: int,
) -> tuple[Float[Array, "degree ... order"], Int[Array, "degree ... order"]]:
    """Run the orthonormal recurrence in l for every order m at once.

    Given the sectoral seeds P̄_m^m (mantissas and exponents), fills
    P̄_{m+1}^m = √(2m + 3) t P̄_m^m and P̄_l^m = a_lm (t P̄_{l−1}^m − b_lm P̄_{l−2}^m)
    in one ``lax.scan`` over l. The map is linear in the seeds, so passing the
    constants of `_sectoral_scale` (without (1 − t²)^{m/2}) gives the
    pole-free P̄_l^m(t) / (1 − t²)^{m/2} that `real_spherical_harmonics` uses.

    Each order m is carried as a mantissa and one integer exponent shared by
    its two carried rows (the recurrence is linear and homogeneous per
    column); after every step both are rescaled by the power of two that
    brings the larger into [½, 1). No intermediate leaves the floating-point
    range, so float32 stays accurate where P̄_m^m ∝ (1 − t²)^{m/2} is far below
    its smallest normal number but P̄_l^m, l ≫ m, is O(1). The shift is an
    integer, so gradients flow through the mantissas only, which is exact.

    Returns:
        ``(mantissa, exponent)``, indexed [l, ..., m]; the value is
        ``_ldexp(mantissa, exponent)``, and the entries with m > l are zero.

    References:
        Fukushima, T. (2012). Numerical computation of spherical harmonics of
        arbitrary degree and order by extending exponent of floating point
        numbers. Journal of Geodesy 86, 271–285 (the "X-number" idea, here
        with one shared exponent per column).
    """
    dtype = sectoral.dtype
    n = l_max + 1
    m = np.arange(n)
    by_order = "order, ... order -> ... order"

    # --- the recurrence coefficients a_lm, b_lm (host constants) ---
    # for the rows l ≥ 1, used only where m ≤ l − 2; elsewhere the masks below
    # pick the sectoral (m = l) and first off-diagonal (m = l − 1) values
    ells = np.arange(n, dtype=float)
    orders = m.astype(float)
    # m ≤ l − 2
    general_mask = einx.less_equal("order, degree -> degree order", orders, ells - 2)
    # 4l² − 1
    a_numerator = einx.where(
        "degree order, degree, -> degree order", general_mask, 4.0 * ells**2 - 1.0, 0.0
    )
    # l² − m²
    a_difference = einx.subtract("degree, order -> degree order", ells**2, orders**2)
    a_denominator = np.where(general_mask, a_difference, 1.0)
    # a_lm = √((4l² − 1)/(l² − m²))
    a_values = np.sqrt(a_numerator / a_denominator)
    a_all = np.where(general_mask, a_values, 0.0)
    # (l − 1)² − m²
    b_difference = einx.subtract(
        "degree, order -> degree order", (ells - 1.0) ** 2, orders**2
    )
    b_numerator = np.where(general_mask, b_difference, 0.0)
    # 4(l − 1)² − 1
    b_denominator = einx.where(
        "degree order, degree, -> degree order",
        general_mask,
        4.0 * (ells - 1.0) ** 2 - 1.0,
        1.0,
    )
    # b_lm = √(((l − 1)² − m²)/(4(l − 1)² − 1))
    b_values = np.sqrt(b_numerator / b_denominator)
    b_all = np.where(general_mask, b_values, 0.0)
    # --- the masks m = l and m = l − 1 ---
    rows = np.arange(n)
    diagonal_mask = einx.equal("order, degree -> degree order", m, rows)
    diagonal = diagonal_mask.astype(float)
    subdiagonal_mask = einx.equal("order, degree -> degree order", m, rows - 1)
    subdiagonal = subdiagonal_mask.astype(float)
    # √(2m + 3)
    sub_scale = jnp.asarray(np.sqrt(2.0 * m + 3.0), dtype=dtype)

    def step(
        carry: _TableCarry,
        inputs: tuple[Float[Array, " order"], ...],
    ) -> tuple[
        _TableCarry, tuple[Float[Array, "*batch order"], Int[Array, "*batch order"]]
    ]:
        # the carried rows P̄_{l−2}^m, P̄_{l−1}^m, in units of 2^exponent
        p_lm2, p_lm1, exponent = carry
        a, b, on_diagonal, on_subdiagonal = inputs
        # t P̄_{l−1}^m
        t_times_lm1 = einx.multiply("..., ... order -> ... order", t, p_lm1)
        # b_lm P̄_{l−2}^m
        damped = einx.multiply(by_order, b, p_lm2)
        # t P̄_{l−1}^m − b_lm P̄_{l−2}^m
        difference = t_times_lm1 - damped
        # P̄_l^m = a_lm (t P̄_{l−1}^m − b_lm P̄_{l−2}^m)
        general = einx.multiply(by_order, a, difference)
        # P̄_{m+1}^m = √(2m + 3) t P̄_m^m: the previous row is the sectoral one
        first_off = einx.multiply(by_order, sub_scale, t_times_lm1)
        # --- pick the sectoral, first off-diagonal or general value per column ---
        # a column's sectoral row arrives while its carry is still zero, so its
        # exponent is still the seed's
        diagonal_part = einx.multiply(by_order, on_diagonal, sectoral)
        subdiagonal_part = einx.multiply(by_order, on_subdiagonal, first_off)
        general_weight = 1.0 - on_diagonal - on_subdiagonal
        general_part = einx.multiply(by_order, general_weight, general)
        p_l = diagonal_part + subdiagonal_part + general_part
        # --- renormalise the two carried rows by one power of two ---
        biggest = jnp.maximum(jnp.abs(p_l), jnp.abs(p_lm1))
        # 2^shift ≥ the larger row, so its mantissa lies in [½, 1)
        _, shift = jnp.frexp(biggest)
        # a zero column keeps its exponent
        shift = jnp.where(biggest > 0, shift, 0)
        new_carry = (_ldexp(p_lm1, -shift), _ldexp(p_l, -shift), exponent + shift)
        return new_carry, (p_l, exponent)

    zeros = jnp.zeros_like(sectoral)
    first_diagonal = jnp.asarray(diagonal[0], dtype=dtype)
    # P̄₀⁰ in column 0, and zeros in the columns m > 0
    p_00 = einx.multiply(by_order, first_diagonal, sectoral)
    if n == 1:
        return (
            einx.id("... order -> 1 ... order", p_00),
            einx.id("... order -> 1 ... order", sectoral_exp),
        )
    xs = tuple(
        jnp.asarray(arr[1:], dtype=dtype)
        for arr in (a_all, b_all, diagonal, subdiagonal)
    )
    _, (rows, row_exponents) = jax.lax.scan(step, (zeros, p_00, sectoral_exp), xs)
    mantissa = einx.id(
        "... order, degree ... order -> (1 + degree) ... order", p_00, rows
    )
    exponent = einx.id(
        "... order, degree ... order -> (1 + degree) ... order",
        sectoral_exp,
        row_exponents,
    )
    return mantissa, exponent
