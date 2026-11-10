"""Spherical needlets: a localised, multiscale tight frame on the sphere.

Needlets (Narcowich, Petrushev and Ward 2006) are band-limited bumps ψ_jk
centred at the nodes ξ_jk of a cubature rule at each scale j, localised in
space and frequency, with ∑_jk ⟨f, ψ_jk⟩ ψ_jk = f − f̄ for band-limited f of
degree at most B^{j_max}. Moved from geonnax ``_needlets`` at ``be5537d``.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import math
from collections.abc import Callable

import einx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from functax._src.basis._legendre import _check_degree, legendre_polynomials
from functax._src.basis._spherical_harmonics import _check_unit_points
from functax._src.geometry._sphere_grids import gauss_legendre_grid
from functax._src.quad._rules import gauss_legendre
from functax.exceptions import ValidationError


__all__ = [
    "needlet_basis",
    "needlet_window",
]


_BUMP_NODES, _BUMP_WEIGHTS = gauss_legendre(64)


def _bump(t: Float[Array, "*batch"]) -> Float[Array, "*batch"]:
    inside = jnp.abs(t) < 1.0
    safe = jnp.where(inside, t, 0.0)
    return jnp.where(inside, jnp.exp(-1.0 / (1.0 - safe**2)), 0.0)


def _smooth_step(u: Float[Array, "*batch"]) -> Float[Array, "*batch"]:
    """ψ(u) = ∫₋₁ᵘ f / ∫₋₁¹ f for the C^∞ bump f(t) = exp(−1/(1 − t²))."""
    u = jnp.clip(u, -1.0, 1.0)
    nodes = jnp.asarray(_BUMP_NODES, dtype=u.dtype)
    weights = jnp.asarray(_BUMP_WEIGHTS, dtype=u.dtype)
    half = 0.5 * (u + 1.0)
    # the Gauss–Legendre nodes mapped from [−1, 1] onto [−1, u], for every u
    scaled = einx.multiply("..., node -> ... node", half, nodes)
    t = einx.add("..., ... node", half - 1.0, scaled)
    partial = half * einx.dot("... node, node -> ...", _bump(t), weights)
    total = einx.dot("node, node ->", _bump(nodes), weights)
    return partial / total


def needlet_window(
    B: float = 2.0,
) -> Callable[[Float[Array, "*batch"]], Float[Array, "*batch"]]:
    """The needlet window b: smooth, supported in [1/B, B], a partition of unity.

    With the C^∞ step ψ (the normalised integral of exp(−1/(1 − t²))) and
    φ(t) = 1 for t ≤ 1/B, ψ(1 − 2B/(B − 1) (t − 1/B)) for 1/B ≤ t ≤ 1, and 0 for
    t ≥ 1, the window is b(ξ) = √(φ(ξ/B) − φ(ξ)). The sum
    ∑_{j=0}^{J} b²(ξ/B^j) = φ(ξ/B^{J+1}) − φ(ξ) telescopes to 1 for 1 ≤ ξ ≤ B^J.

    Args:
        B: The scale factor, above 1.

    Returns:
        A function that evaluates b elementwise (integers become float32).

    Raises:
        ValidationError: If ``B`` is not above 1.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.sphere import needlet_window
        >>> window = needlet_window(2.0)
        >>> [round(float(v), 4) for v in window(jnp.array([0.4, 1.0, 2.5]))]
        [0.0, 1.0, 0.0]
    """
    if not B > 1.0:
        msg = f"'B' must be above 1, got {B!r}"
        raise ValidationError(msg)

    def phi(t: Float[Array, "*batch"]) -> Float[Array, "*batch"]:
        ramp = _smooth_step(1.0 - 2.0 * B / (B - 1.0) * (t - 1.0 / B))
        return jnp.where(t <= 1.0 / B, 1.0, jnp.where(t >= 1.0, 0.0, ramp))

    def window(xi: Float[Array, "*batch"]) -> Float[Array, "*batch"]:
        xi = jnp.asarray(xi)
        if not jnp.issubdtype(xi.dtype, jnp.floating):
            xi = xi.astype(jnp.float32)
        # b(ξ) = √(φ(ξ/B) − φ(ξ))
        return jnp.sqrt(jnp.maximum(phi(xi / B) - phi(xi), 0.0))

    return window


def needlet_basis(
    unit_xyz: Float[Array, "obs 3"],
    *,
    B: float = 2.0,
    j_max: int,
) -> tuple[
    Float[Array, "obs feature"], Int[Array, " feature"], Float[Array, "feature 3"]
]:
    """Spherical needlets at the scales j = 0, …, j_max, plus the constant.

    At scale j, with L_j = ⌊B^{j+1}⌋ and the Gauss–Legendre grid (ξ_jk, λ_jk)
    exact to degree 2L_j (`gauss_legendre_grid` with L_j + 1 latitudes and
    2L_j + 2 longitudes),

        ψ_jk(x) = √λ_jk ∑_{l=0}^{L_j} b(l/B^j) (2l + 1)/(4π) P_l(ξ_jk · x).

    The first column is the constant Y₀₀ = 1/√(4π) (level −1, centre (0, 0, 0)),
    which the needlets omit since b(0) = 0. With it,
    ∑_m ⟨f, ψ_m⟩ ψ_m = f for every f of degree at most B^{j_max}.

    Args:
        unit_xyz: The evaluation points on the unit sphere.
        B: The scale factor, above 1.
        j_max: The finest scale, at least 0.

    Returns:
        ``(values, levels, centres)``: the needlets at each point, the scale
        of each column and each column's centre.

    Raises:
        ValidationError: If ``unit_xyz`` is not (N, 3), ``j_max`` is not an
            integer of at least 0, or ``B`` is not above 1.

    References:
        Narcowich, F. J., Petrushev, P., Ward, J. D. (2006). Localized tight
        frames on spheres. SIAM Journal on Mathematical Analysis 38(2),
        574–594.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.sphere import needlet_basis
        >>> pole = jnp.array([[0.0, 0.0, 1.0]])
        >>> values, levels, centres = needlet_basis(pole, j_max=1)
        >>> values.shape, sorted(set(levels.tolist()))
        ((1, 69), [-1, 0, 1])
    """
    finest = _check_degree(j_max, "j_max")
    unit_xyz = jnp.asarray(unit_xyz)
    _check_unit_points(unit_xyz, "unit_xyz")
    window = needlet_window(B)
    dtype = unit_xyz.dtype
    n = unit_xyz.shape[0]
    columns = [jnp.full((n, 1), 1.0 / math.sqrt(4.0 * math.pi), dtype=dtype)]
    levels = [-1]
    centres = [jnp.zeros((1, 3), dtype=dtype)]
    for scale in range(finest + 1):
        degree = math.floor(B ** (scale + 1))
        nodes, weights = gauss_legendre_grid(degree + 1, 2 * degree + 2)
        nodes, weights = nodes.astype(dtype), weights.astype(dtype)
        ells = np.arange(degree + 1)
        # b(l/B^j) (2l + 1)/(4π)
        coefficients = window(jnp.asarray(ells / B**scale, dtype=dtype)) * jnp.asarray(
            (2.0 * ells + 1.0) / (4.0 * math.pi), dtype=dtype
        )
        cosines = jnp.clip(
            einx.dot("obs dim, centre dim -> obs centre", unit_xyz, nodes), -1.0, 1.0
        )
        kernel = einx.dot(
            "obs centre degree, degree -> obs centre",
            legendre_polynomials(cosines, degree),
            coefficients,
        )
        columns.append(einx.multiply("obs centre, centre", kernel, jnp.sqrt(weights)))
        levels += [scale] * int(nodes.shape[0])
        centres.append(nodes)
    # the scales have different numbers of centres, so they are joined as a list
    return jnp.hstack(columns), jnp.asarray(levels), jnp.vstack(centres)
