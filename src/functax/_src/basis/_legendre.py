"""Legendre polynomials and the fully normalised associated Legendre functions.

The one-dimensional building blocks of the spherical bases. The associated
functions run the orthonormal recurrences as one ``lax.scan`` over the degree,
carried as a mantissa and a binary exponent per order, so they stay finite to
high degree in float32 as well as float64. Moved from geonnax ``_legendre`` at
``be5537d``; Gauss–Legendre nodes moved to `functax.quad`.
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
from jaxtyping import Array, Float, Int

from functax.exceptions import ValidationError


__all__ = [
    "associated_legendre",
    "associated_legendre_indices",
    "legendre_polynomials",
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
    t: Float[Array, ...], l_max: int
) -> Float[Array, "... degree"]:
    """The Legendre polynomials P₀(t), …, P_{l_max}(t).

    P₀ = 1, P₁ = t and (l + 1) P_{l+1} = (2l + 1) t P_l − l P_{l−1}, run as a
    ``lax.scan`` over l, so the trace does not grow with ``l_max``.

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
        >>> legendre_polynomials(jnp.array([0.5]), 2).tolist()
        [[1.0, 0.5, -0.125]]
        >>> legendre_polynomials(jnp.zeros((2, 3)), 4).shape
        (2, 3, 5)
    """
    degree = _check_degree(l_max, "l_max")
    t = jnp.asarray(t)
    p0 = jnp.ones_like(t)
    if degree == 0:
        return einx.id("... -> ... 1", p0)
    if degree == 1:
        return einx.id("..., ... -> ... (1 + 1)", p0, t)

    def step(
        carry: tuple[jax.Array, jax.Array], ell: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        p_prev, p_curr = carry
        # P_{l+1} = ((2l + 1) t P_l − l P_{l−1}) / (l + 1)
        p_next = ((2.0 * ell + 1.0) * t * p_curr - ell * p_prev) / (ell + 1.0)
        return (p_curr, p_next), p_next

    ells = jnp.arange(1, degree, dtype=t.dtype)
    _, rest = jax.lax.scan(step, (p0, t), ells)
    return einx.id("..., ..., degree ... -> ... (1 + 1 + degree)", p0, t, rest)


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
    """
    degree = _check_degree(l_max, "l_max")
    return tuple((ell, m) for ell in range(degree + 1) for m in range(ell + 1))


def associated_legendre(
    t: Float[Array, ...],
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
        >>> values = associated_legendre(jnp.array([0.0]), 1, normalization="none")
        >>> [round(float(v), 12) + 0.0 for v in values[0]]
        [1.0, 0.0, -1.0]
        >>> associated_legendre(jnp.zeros((4,)), 3).shape
        (4, 10)
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
    # s = √(1 − t²)
    s = jnp.sqrt(jnp.maximum(1.0 - t**2, 0.0))

    # The sectoral P̄_m^m as mantissa · 2^exponent: the carry keeps (1 − t²)^{m/2}
    # from underflowing near the poles, where P̄_l^m grows back to O(1) for l ≫ m
    sectoral, sectoral_exp = _sectoral_carry(s, degree)
    mantissa, exponent = _legendre_table_carry(t, sectoral, sectoral_exp, degree)
    table = _ldexp(mantissa, exponent)

    pairs = associated_legendre_indices(degree)
    ell_index = np.array([ell for ell, _ in pairs])
    m_index = np.array([order for _, order in pairs])
    # the table is indexed [l, ..., m]; pick each (l, m) and move it last
    out = einx.id("feature ... -> ... feature", table[ell_index, ..., m_index])

    if normalization == "orthonormal":
        return out
    # from P̄ (unit L² norm on [−1, 1]) to the classical P_l^m
    log_ratio = np.array(
        [math.lgamma(ell + m + 1) - math.lgamma(ell - m + 1) for ell, m in pairs]
    )
    to_classical = np.sqrt(2.0 / (2.0 * ell_index + 1.0) * np.exp(log_ratio))
    if normalization == "none":
        return out * jnp.asarray(to_classical, dtype=dtype)
    # √((2 − δ_m0)(l − m)!/(l + m)!) P_l^m, without the Condon–Shortley phase
    schmidt = np.sqrt(2.0 * (2.0 - (m_index == 0)) / (2.0 * ell_index + 1.0))
    schmidt = schmidt * (-1.0) ** m_index
    return out * jnp.asarray(schmidt, dtype=dtype)


def _sectoral_scale(l_max: int) -> np.ndarray:
    """c_m = P̄_m^m / (1 − t²)^{m/2} for m ≤ l_max.

    c₀ = 1/√2 and c_m = −√((2m + 1)/(2m)) c_{m−1}: the orthonormal sectoral
    recurrence with the factor √(1 − t²) taken out.
    """
    m = np.arange(l_max + 1)
    factor = np.ones(l_max + 1)
    factor[1:] = -np.sqrt((2.0 * m[1:] + 1.0) / (2.0 * m[1:]))
    return np.cumprod(factor) / math.sqrt(2.0)


def _ldexp(x: Float[Array, ...], e: Int[Array, ...]) -> Float[Array, ...]:
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
    half = e // 2
    two = jnp.asarray(2.0, dtype=x.dtype)
    return x * two ** half.astype(x.dtype) * two ** (e - half).astype(x.dtype)


def _sectoral_carry(
    s: Float[Array, ...], l_max: int
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
    factors = jnp.asarray(-np.sqrt((2.0 * m + 1.0) / (2.0 * m)), dtype=dtype)

    def step(
        carry: tuple[jax.Array, jax.Array], factor: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        mantissa, exponent = carry
        value = factor * s * mantissa
        _, shift = jnp.frexp(value)
        shift = jnp.where(value != 0, shift, 0)
        mantissa, exponent = _ldexp(value, -shift), exponent + shift
        return (mantissa, exponent), (mantissa, exponent)

    _, (mantissas, exponents) = jax.lax.scan(step, (mantissa0, exponent0), factors)
    mantissa = einx.id("..., order ... -> ... (1 + order)", mantissa0, mantissas)
    exponent = einx.id("..., order ... -> ... (1 + order)", exponent0, exponents)
    return mantissa, exponent


def _legendre_table_carry(
    t: Float[Array, ...],
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

    # The recurrence coefficients a_lm, b_lm (host constants) for the rows
    # l ≥ 1, used only where m ≤ l − 2; elsewhere the masks below pick the
    # sectoral (m = l) and first off-diagonal (m = l − 1) values instead.
    ells = np.arange(n, dtype=float)[:, None]
    orders = m[None, :].astype(float)
    general_mask = orders <= ells - 2
    a_all = np.where(
        general_mask,
        np.sqrt(
            np.where(general_mask, 4.0 * ells**2 - 1.0, 0.0)
            / np.where(general_mask, ells**2 - orders**2, 1.0)
        ),
        0.0,
    )
    b_all = np.where(
        general_mask,
        np.sqrt(
            np.where(general_mask, (ells - 1.0) ** 2 - orders**2, 0.0)
            / np.where(general_mask, 4.0 * (ells - 1.0) ** 2 - 1.0, 1.0)
        ),
        0.0,
    )
    # the masks m = l and m = l − 1
    diagonal = (m[None, :] == np.arange(n)[:, None]).astype(float)
    subdiagonal = (m[None, :] == np.arange(n)[:, None] - 1).astype(float)
    # √(2m + 3)
    sub_scale = jnp.asarray(np.sqrt(2.0 * m + 3.0), dtype=dtype)

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        inputs: tuple[jax.Array, ...],
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        # the carried rows P̄_{l−2}^m, P̄_{l−1}^m, in units of 2^exponent
        p_lm2, p_lm1, exponent = carry
        a, b, on_diagonal, on_subdiagonal = inputs
        t_times_lm1 = einx.multiply("..., ... order -> ... order", t, p_lm1)
        # P̄_l^m = a_lm (t P̄_{l−1}^m − b_lm P̄_{l−2}^m)
        general = a * (t_times_lm1 - b * p_lm2)
        # P̄_{m+1}^m = √(2m + 3) t P̄_m^m: the previous row is the sectoral one
        first_off = sub_scale * t_times_lm1
        # a column's sectoral row arrives while its carry is still zero, so its
        # exponent is still the seed's
        p_l = (
            on_diagonal * sectoral
            + on_subdiagonal * first_off
            + (1.0 - on_diagonal - on_subdiagonal) * general
        )
        biggest = jnp.maximum(jnp.abs(p_l), jnp.abs(p_lm1))
        _, shift = jnp.frexp(biggest)
        shift = jnp.where(biggest > 0, shift, 0)
        new_carry = (_ldexp(p_lm1, -shift), _ldexp(p_l, -shift), exponent + shift)
        return new_carry, (p_l, exponent)

    zeros = jnp.zeros_like(sectoral)
    p_00 = jnp.asarray(diagonal[0], dtype=dtype) * sectoral
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
