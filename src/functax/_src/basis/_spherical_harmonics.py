"""Real spherical harmonics Y_lm on the unit sphere, pole-free to high degree.

The pole singularity is avoided by pairing the pole-free polynomial
Q̄_l^m(z) = P̄_l^m(z) / (1 − z²)^{m/2} with (x + iy)^m directly, with no atan2
and no division by sin θ, so each Y_lm is a polynomial in (x, y, z). Moved from
geonnax ``_spherical`` at ``be5537d``.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import math

import einx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

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


def _check_unit_points(points: jax.Array, name: str) -> None:
    if points.ndim != 2 or points.shape[-1] != 3:
        msg = f"'{name}' must have shape (N, 3), got {points.shape}"
        raise ValidationError(msg)


def harmonic_degrees(l_max: int) -> tuple[int, ...]:
    """The degree l of each column of `real_spherical_harmonics`.

    The columns run with l outermost and m = −l, …, l inner, so degree l
    fills 2l + 1 columns and there are (l_max + 1)² in all.

    Args:
        l_max: The highest degree, inclusive.

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


def _xy_powers(
    x: Float[Array, " obs"], y: Float[Array, " obs"], m_max: int
) -> tuple[
    Float[Array, "order obs"], Float[Array, "order obs"], Int[Array, "order obs"]
]:
    """(x + iy)^m for m = 0, …, m_max as mantissas and a shared binary exponent.

    One ``lax.scan`` over m. The magnitude (x² + y²)^{m/2} = sin^m θ underflows
    float32 near the poles, so the real and imaginary parts share an int32
    exponent and are renormalised by a power of two after every step: the
    value is (re + i im) · 2^exp.
    """

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array], unused: None
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, ...]]:
        re, im, exponent = carry
        # (x + iy)(re + i im) = (x re − y im) + i (x im + y re)
        new_re, new_im = x * re - y * im, x * im + y * re
        biggest = jnp.maximum(jnp.abs(new_re), jnp.abs(new_im))
        _, shift = jnp.frexp(biggest)
        shift = jnp.where(biggest > 0, shift, 0)
        new_carry = (_ldexp(new_re, -shift), _ldexp(new_im, -shift), exponent + shift)
        return new_carry, carry

    init = (jnp.ones_like(x), jnp.zeros_like(x), jnp.zeros(x.shape, dtype=jnp.int32))
    _, (re, im, exponent) = jax.lax.scan(step, init, None, length=m_max + 1)
    return re, im, exponent


def real_spherical_harmonics(
    unit_xyz: Float[Array, "obs 3"],
    l_max: int,
) -> Float[Array, "obs feature"]:
    """The real spherical harmonics Y_lm up to degree ``l_max``.

    The tesseral real basis: Y_l^0 = N_l^0 P_l(cos θ), and for m > 0
    Y_l^m = √2 N_l^m P_l^m(cos θ) cos(mλ) and Y_l^{−m} = √2 N_l^m P_l^m(cos θ) sin(mλ),
    with N_l^m = √((2l + 1)/(4π) · (l − m)!/(l + m)!) and the Condon–Shortley
    phase in P_l^m. The basis is orthonormal on the sphere,
    ∫_{S²} Y_l^m Y_l′^m′ dΩ = δ_ll′ δ_mm′.

    Each P_l^m(z) cos(mλ) is evaluated as Q̄_l^m(z) Re[(x + iy)^m] with the
    pole-free Q̄_l^m = P̄_l^m / (1 − z²)^{m/2} (and the sine branch with Im), so
    every Y_lm is a polynomial in (x, y, z), differentiable at the poles too.
    Then Y_l^0 = Q̄_l^0/√(2π), Y_l^m = Q̄_l^m Re[(x + iy)^m]/√π and
    Y_l^{−m} = Q̄_l^m Im[(x + iy)^m]/√π.

    Algorithm:
        1. Run the forward-column recurrences of Holmes and Featherstone
           (2002) with (1 − z²)^{m/2} taken out, for every order at once in
           one ``lax.scan`` over l: the trace has constant size in ``l_max``
           and no factorial ratio is ever formed.
        2. Compute (x + iy)^m by a scan over m.
        3. Near the poles Q̄_l^m grows like sin^{−m} θ and (x + iy)^m shrinks
           like sin^m θ, so both are carried as mantissa and int32 exponent and
           combined by one power-of-two scaling at the end: float32 stays
           finite to l_max = 511 (poles included) and within about 4e-3 of
           float64 at l_max = 255.

    Args:
        unit_xyz: Points on the unit sphere (not renormalised). Integers
            become float32.
        l_max: The highest degree, inclusive; there are (l_max + 1)² columns.

    Returns:
        Y_lm at each point, the columns in the order of `harmonic_degrees`
        (l outer, m = −l, …, l inner).

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

    # the pole-free Q̄_l^m: the orthonormal recurrence seeded with the sectoral
    # constants instead of P̄_m^m = c_m (1 − z²)^{m/2}
    seeds = jnp.asarray(_sectoral_scale(degree), dtype=dtype)
    sectoral = einx.multiply("obs, order -> obs order", jnp.ones_like(z), seeds)
    q_mantissa, q_exponent = _legendre_table_carry(
        z, sectoral, jnp.zeros(sectoral.shape, dtype=jnp.int32), degree
    )
    re, im, xy_exponent = _xy_powers(x, y, degree)

    # the host-side column layout: l outer, m = −l, …, l inner
    ell = np.asarray(harmonic_degrees(degree))
    order = np.concatenate([np.arange(-l, l + 1) for l in range(degree + 1)])
    absolute_order = np.abs(order)
    scale = np.where(
        order == 0, 1.0 / math.sqrt(2.0 * math.pi), 1.0 / math.sqrt(math.pi)
    )

    q_columns = q_mantissa[ell, :, absolute_order]
    # the m = 0 branch is exactly 1 with exponent 0
    positive = einx.id("feature -> feature 1", jnp.asarray(order > 0))
    negative = einx.id("feature -> feature 1", jnp.asarray(order < 0))
    trig = jnp.where(
        positive, re[absolute_order], jnp.where(negative, im[absolute_order], 1.0)
    )
    exponent = q_exponent[ell, :, absolute_order] + xy_exponent[absolute_order]
    mantissa = einx.multiply(
        "feature obs, feature obs, feature -> feature obs",
        q_columns,
        trig,
        jnp.asarray(scale, dtype=dtype),
    )
    return einx.id("feature obs -> obs feature", _ldexp(mantissa, exponent))
