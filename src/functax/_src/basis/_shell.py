"""Product bases on a spherical shell r₀ ≤ ‖x‖ ≤ r₁.

For fields with depth or altitude, each basis function is a real spherical
harmonic in direction times a polynomial in radius,
φ_lmn(x) = Y_lm(x/‖x‖) R_n(‖x‖). Moved from geonnax ``_shell`` at ``be5537d``.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import einx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from functax._src.basis._legendre import _check_degree, legendre_polynomials
from functax._src.basis._spherical_harmonics import (
    _check_unit_points,
    harmonic_degrees,
    real_spherical_harmonics,
)
from functax.exceptions import ValidationError


__all__ = ["shell_basis"]


_RADIAL: tuple[str, ...] = ("legendre", "chebyshev")


def _chebyshev(s: Float[Array, " obs"], n_radial: int) -> Float[Array, "obs radial"]:
    """T₀(s), …, T_{n−1}(s) by T_{k+1} = 2s T_k − T_{k−1}."""
    columns = [jnp.ones_like(s)]
    if n_radial > 1:
        columns.append(s)
    for _ in range(2, n_radial):
        columns.append(2.0 * s * columns[-1] - columns[-2])
    return jnp.column_stack(columns)


def shell_basis(
    xyz: Float[Array, "obs 3"],
    l_max: int,
    n_radial: int,
    *,
    r_inner: float,
    r_outer: float,
    radial: str = "legendre",
) -> tuple[Float[Array, "obs feature"], Int[Array, "feature 2"]]:
    """Spherical harmonics times radial polynomials on a spherical shell.

    With s = 2(‖x‖ − r₀)/(r₁ − r₀) − 1 in [−1, 1]:

    - ``radial="legendre"``: R_n = √(2/(r₁ − r₀)) √((2n + 1)/2) P_n(s), so the
      basis is orthonormal in L² with the measure dΩ dr. This is not the
      volume measure r² dr dΩ; for a thin shell the two differ by roughly r².
    - ``radial="chebyshev"``: the Chebyshev polynomials T_n(s), unnormalised;
      well-conditioned, but not orthogonal under dr.

    The exact Laplacian eigenfunctions of the shell (spherical Bessel
    functions in radius) are out of scope.

    Args:
        xyz: Points with radius in [r₀, r₁].
        l_max: The highest spherical-harmonic degree.
        n_radial: The number of radial polynomials, at least 1.
        r_inner: r₀, at least 0.
        r_outer: r₁, above r₀.
        radial: ``"legendre"`` or ``"chebyshev"``.

    Returns:
        ``(values, indices)``: (l_max + 1)² n_radial columns, harmonic-major
        (the radial index varies fastest), and each column's (l, n).

    Raises:
        ValidationError: If ``xyz`` is not (N, 3), ``l_max`` or ``n_radial`` is
            not an integer in range, 0 ≤ r₀ < r₁ fails, or ``radial`` is
            unknown.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.sphere import shell_basis
        >>> xyz = jnp.array([[0.0, 0.0, 1.5], [2.0, 0.0, 0.0]])
        >>> values, indices = shell_basis(xyz, 2, 3, r_inner=1.0, r_outer=2.0)
        >>> values.shape, indices[:4].tolist()
        ((2, 27), [[0, 0], [0, 1], [0, 2], [1, 0]])
    """
    degree = _check_degree(l_max, "l_max")
    count = _check_degree(n_radial, "n_radial", minimum=1)
    if not 0.0 <= r_inner < r_outer:
        msg = f"need 0 ≤ 'r_inner' < 'r_outer', got {r_inner!r} and {r_outer!r}"
        raise ValidationError(msg)
    if radial not in _RADIAL:
        msg = f"'radial' must be one of {list(_RADIAL)}, got {radial!r}"
        raise ValidationError(msg)
    xyz = jnp.asarray(xyz)
    _check_unit_points(xyz, "xyz")
    # r = ‖x‖, and the unit direction x/‖x‖ (left as 0 at the origin)
    radius = jnp.sqrt(einx.sum("obs [dim]", xyz**2))
    unit = einx.divide("obs dim, obs", xyz, jnp.where(radius > 0, radius, 1.0))
    harmonics = real_spherical_harmonics(unit, degree)
    # s = 2(r − r₀)/(r₁ − r₀) − 1
    s = 2.0 * (radius - r_inner) / (r_outer - r_inner) - 1.0
    if radial == "legendre":
        n = np.arange(count)
        scale = np.sqrt(2.0 / (r_outer - r_inner)) * np.sqrt((2.0 * n + 1.0) / 2.0)
        radial_values = legendre_polynomials(s, count - 1) * jnp.asarray(
            scale, dtype=s.dtype
        )
    else:
        radial_values = _chebyshev(s, count)
    values = einx.multiply(
        "obs harmonic, obs radial -> obs (harmonic radial)", harmonics, radial_values
    )
    degrees = np.repeat(np.asarray(harmonic_degrees(degree)), count)
    radial_index = np.tile(np.arange(count), (degree + 1) ** 2)
    return values, jnp.asarray(np.stack([degrees, radial_index], axis=-1))
