"""Real spherical harmonics Y_ℓm on the unit sphere, pole-free to high degree.

A point r̂ = (x, y, z) ∈ S² has colatitude θ, with z = cos θ, and longitude λ;
Y_ℓm has degree ℓ = 0, …, ℓ_max and order m = −ℓ, …, ℓ. The pole singularity
is avoided by pairing the pole-free polynomial Q̄_ℓᵐ(z) = P̄_ℓᵐ(z)/(1 − z²)^{m/2}
with (x + iy)ᵐ directly, with no atan2 and no division by sin θ, so each Y_ℓm
is a polynomial in (x, y, z). Moved from geonnax ``_spherical`` at ``be5537d``.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import math

import einx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, Shaped

from functax._src.basis._legendre import (
    _check_degree,
    _ldexp,
    _legendre_table_carry,
    _sectoral_scale,
)
from functax.exceptions import ValidationError


__all__ = [
    "harmonic_degrees",
    "real_spherical_harmonics",
]


def _check_unit_points(points: Shaped[Array, "*shape"], name: str) -> None:
    if points.ndim != 2 or points.shape[-1] != 3:
        msg = f"'{name}' must have shape (N, 3), got {points.shape}"
        raise ValidationError(msg)


def harmonic_degrees(l_max: int) -> tuple[int, ...]:
    """The degree ℓ of each column of `real_spherical_harmonics`.

    The columns run with ℓ outermost and m = −ℓ, …, ℓ inner, so degree ℓ
    fills 2ℓ + 1 columns and there are (ℓ_max + 1)² in all.

    Args:
        l_max: ℓ_max, the highest degree, inclusive.

    Returns:
        The degree of each column.

    Raises:
        ValidationError: If ``l_max`` is not an integer of at least 0.

    Examples:
        >>> from functax.basis.sphere import harmonic_degrees
        >>> harmonic_degrees(2)
        (0, 1, 1, 1, 2, 2, 2, 2, 2)
        >>> len(harmonic_degrees(9))
        100
    """
    degree = _check_degree(l_max, "l_max")
    return tuple(ell for ell in range(degree + 1) for _ in range(2 * ell + 1))


# the carry of `_xy_powers`: the mantissas of Re and Im, and their shared exponent
_PowerCarry = tuple[Float[Array, " obs"], Float[Array, " obs"], Int[Array, " obs"]]


def _xy_powers(
    x: Float[Array, " obs"], y: Float[Array, " obs"], m_max: int
) -> tuple[
    Float[Array, "order obs"], Float[Array, "order obs"], Int[Array, "order obs"]
]:
    """(x + iy)ᵐ for m = 0, …, m_max as mantissas and a shared binary exponent.

    One ``lax.scan`` over m. The magnitude (x² + y²)^{m/2} = sinᵐ θ underflows
    float32 near the poles, so the real and imaginary parts share an int32
    exponent and are renormalised by a power of two after every step: the
    value is (re + i im) · 2^{exp}.
    """

    def step(carry: _PowerCarry, unused: None) -> tuple[_PowerCarry, _PowerCarry]:
        re, im, exponent = carry
        # Re[(x + iy)(re + i im)] = x re − y im
        new_re = x * re - y * im
        # Im[(x + iy)(re + i im)] = x im + y re
        new_im = x * im + y * re
        # --- renormalise both parts by one power of two ---
        biggest = jnp.maximum(jnp.abs(new_re), jnp.abs(new_im))
        # 2^{shift} ≥ max(|re|, |im|), so the mantissas stay in [½, 1)
        _, shift = jnp.frexp(biggest)
        # a zero value keeps its exponent
        shift = jnp.where(biggest > 0, shift, 0)
        # re · 2^{−shift}, im · 2^{−shift}, exp + shift
        new_carry = (_ldexp(new_re, -shift), _ldexp(new_im, -shift), exponent + shift)
        return new_carry, carry

    # (x + iy)⁰ = 1 · 2⁰
    one = jnp.ones_like(x)
    zero = jnp.zeros_like(x)
    zero_exponent = jnp.zeros(x.shape, dtype=jnp.int32)
    init = (one, zero, zero_exponent)
    _, (re, im, exponent) = jax.lax.scan(step, init, None, length=m_max + 1)
    return re, im, exponent


def real_spherical_harmonics(
    unit_xyz: Float[Array, "obs 3"],
    l_max: int,
) -> Float[Array, "obs feature"]:
    """The real spherical harmonics Y_ℓm up to degree ``l_max``.

    At r̂ = (x, y, z) ∈ S² with colatitude θ and longitude λ, the tesseral real
    basis is, for ℓ = 0, …, ℓ_max,

        Y_ℓm(r̂) = N_ℓ0 P_ℓ(cos θ)                     for m = 0,
        Y_ℓm(r̂) = √2 N_ℓm P_ℓᵐ(cos θ) cos(mλ)         for m = 1, …, ℓ,
        Y_ℓm(r̂) = √2 N_ℓ|m| P_ℓ^{|m|}(cos θ) sin(|m|λ) for m = −ℓ, …, −1,

    with N_ℓm = √((2ℓ + 1)/(4π) · (ℓ − m)!/(ℓ + m)!) and the Condon–Shortley
    phase in the associated Legendre function P_ℓᵐ. The basis is orthonormal
    on the sphere, ∫_{S²} Y_ℓm Y_ℓ′m′ dΩ = δ_ℓℓ′ δ_mm′, with dΩ the area
    element of S².

    Each P_ℓᵐ(z) cos(mλ) is evaluated as Q̄_ℓᵐ(z) Re[(x + iy)ᵐ] with the
    pole-free Q̄_ℓᵐ = P̄_ℓᵐ/(1 − z²)^{m/2} (and the sine branch with Im), where
    P̄_ℓᵐ is the orthonormal associated Legendre function of
    `associated_legendre`, so every Y_ℓm is a polynomial in (x, y, z),
    differentiable at the poles too. Then Y_ℓ0 = Q̄_ℓ⁰/√(2π), and for m > 0
    Y_ℓm = Q̄_ℓᵐ Re[(x + iy)ᵐ]/√π and Y_ℓ,−m = Q̄_ℓᵐ Im[(x + iy)ᵐ]/√π.

    Algorithm:
        1. Run the forward-column recurrences of Holmes and Featherstone
           (2002) with (1 − z²)^{m/2} taken out, for every order at once in
           one ``lax.scan`` over ℓ: the trace has constant size in ℓ_max and
           no factorial ratio is ever formed.
        2. Compute (x + iy)ᵐ by a scan over m.
        3. Near the poles Q̄_ℓᵐ grows like sin⁻ᵐ θ and (x + iy)ᵐ shrinks like
           sinᵐ θ, so both are carried as mantissa and int32 exponent and
           combined by one power-of-two scaling at the end: float32 stays
           finite to ℓ_max = 511 (poles included) and within about 4e-3 of
           float64 at ℓ_max = 255.

    Args:
        unit_xyz: The N points r̂ on the unit sphere (not renormalised).
            Integers become float32.
        l_max: ℓ_max, the highest degree, inclusive; there are (ℓ_max + 1)²
            columns.

    Returns:
        Y_ℓm at each point, the columns in the order of `harmonic_degrees`
        (ℓ outer, m = −ℓ, …, ℓ inner).

    Raises:
        ValidationError: If ``unit_xyz`` is not (N, 3), or ``l_max`` is not
            an integer of at least 0.

    References:
        Holmes, S. A., Featherstone, W. E. (2002). A unified approach to the
        Clenshaw summation and the recursive computation of very high degree
        and order normalised associated Legendre functions. Journal of
        Geodesy 76(5), 279–299.

    Examples:
        >>> import math
        >>> import jax.numpy as jnp
        >>> from functax.basis.sphere import real_spherical_harmonics
        >>> xyz = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> real_spherical_harmonics(xyz, l_max=3).shape
        (2, 16)
        >>> y00 = real_spherical_harmonics(jnp.array([[0.0, 0.0, 1.0]]), 0)
        >>> bool(jnp.allclose(y00, 1 / math.sqrt(4 * math.pi)))
        True

        Orthonormal under a Gauss–Legendre rule exact to degree 2ℓ_max:

        >>> from functax.geometry import gauss_legendre_grid
        >>> points, weights = gauss_legendre_grid(5, 10)
        >>> harmonics = real_spherical_harmonics(points, 4)
        >>> import einx
        >>> weighted = einx.multiply("obs col, obs", harmonics, weights)
        >>> gram = einx.dot("obs row, obs col -> row col", harmonics, weighted)
        >>> bool(jnp.allclose(gram, jnp.eye(25), atol=1e-12))
        True
    """
    degree = _check_degree(l_max, "l_max")
    unit_xyz = jnp.asarray(unit_xyz)
    _check_unit_points(unit_xyz, "unit_xyz")
    dtype = (
        unit_xyz.dtype if jnp.issubdtype(unit_xyz.dtype, jnp.floating) else jnp.float32
    )
    unit_xyz = unit_xyz.astype(dtype)
    x = unit_xyz[:, 0]
    y = unit_xyz[:, 1]
    z = unit_xyz[:, 2]

    # --- the pole-free Q̄_ℓᵐ ---
    # the orthonormal recurrence seeded with the sectoral constants q_m instead
    # of P̄_mᵐ = q_m (1 − z²)^{m/2}
    seeds = jnp.asarray(_sectoral_scale(degree), dtype=dtype)
    ones = jnp.ones_like(z)
    sectoral = einx.multiply("obs, order -> obs order", ones, seeds)
    seed_exponent = jnp.zeros(sectoral.shape, dtype=jnp.int32)
    q_mantissa, q_exponent = _legendre_table_carry(z, sectoral, seed_exponent, degree)
    # --- (x + iy)ᵐ ---
    re, im, xy_exponent = _xy_powers(x, y, degree)

    # --- the host-side column layout: ℓ outer, m = −ℓ, …, ℓ inner ---
    ell = np.asarray(harmonic_degrees(degree))
    order = np.concatenate([np.arange(-l, l + 1) for l in range(degree + 1)])
    absolute_order = np.abs(order)
    # 1/√(2π) for m = 0, 1/√π otherwise
    scale = np.where(
        order == 0, 1.0 / math.sqrt(2.0 * math.pi), 1.0 / math.sqrt(math.pi)
    )

    # --- combine ---
    q_columns = q_mantissa[ell, :, absolute_order]
    # the m = 0 branch is exactly 1 with exponent 0
    positive = einx.id("feature -> feature 1", jnp.asarray(order > 0))
    negative = einx.id("feature -> feature 1", jnp.asarray(order < 0))
    # Im[(x + iy)^{|m|}] for m < 0, and 1 for m = 0
    sine_or_one = jnp.where(negative, im[absolute_order], 1.0)
    # Re[(x + iy)ᵐ] for m > 0
    trig = jnp.where(positive, re[absolute_order], sine_or_one)
    # the exponent of Q̄_ℓ^{|m|} (x + iy)^{|m|}
    exponent = q_exponent[ell, :, absolute_order] + xy_exponent[absolute_order]
    # mantissa of Y_ℓm = Q̄_ℓ^{|m|} {Re, Im}[(x + iy)^{|m|}] / √(2π or π)
    mantissa = einx.multiply(
        "feature obs, feature obs, feature -> feature obs",
        q_columns,
        trig,
        jnp.asarray(scale, dtype=dtype),
    )
    # Y_ℓm = mantissa · 2^{exponent}
    harmonics = _ldexp(mantissa, exponent)
    return einx.id("feature obs -> obs feature", harmonics)
