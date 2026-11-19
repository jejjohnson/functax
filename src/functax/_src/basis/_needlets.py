"""Spherical needlets: a localised, multiscale tight frame on the sphere.

Needlets (Narcowich, Petrushev and Ward 2006) are band-limited bumps ψ_jk
centred at the nodes ξ_jk ∈ S² of a cubature rule at each scale
j = 0, …, j_max, with k indexing the nodes of scale j. They are localised in
space and frequency, and ∑_{j,k} ⟨f, ψ_jk⟩ ψ_jk = f − f̄ for a band-limited f
of degree at most B^{j_max}, with ⟨·, ·⟩ the L²(S²) inner product, f̄ the mean
of f over the sphere and B > 1 the scale factor. Moved from geonnax
``_needlets`` at ``be5537d``.
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
    """The C^∞ bump g(t) = exp(−1/(1 − t²)) on (−1, 1), and 0 outside."""
    inside = jnp.abs(t) < 1.0
    # t is replaced outside, so the exponent stays finite there and its gradient
    # is not NaN
    safe = jnp.where(inside, t, 0.0)
    # −1/(1 − t²)
    exponent = -1.0 / (1.0 - safe**2)
    bump = jnp.exp(exponent)
    return jnp.where(inside, bump, 0.0)


def _smooth_step(u: Float[Array, "*batch"]) -> Float[Array, "*batch"]:
    """χ(τ) = ∫_{−1}^{τ} g / ∫₋₁¹ g for the C^∞ bump g(t) = exp(−1/(1 − t²)).

    Both integrals use the 64-node Gauss–Legendre rule, nodes sₙ and weights wₙ.
    """
    u = jnp.clip(u, -1.0, 1.0)
    nodes = jnp.asarray(_BUMP_NODES, dtype=u.dtype)
    weights = jnp.asarray(_BUMP_WEIGHTS, dtype=u.dtype)
    # h = (τ + 1)/2, the half-length of [−1, τ]
    half = 0.5 * (u + 1.0)
    # --- the Gauss–Legendre nodes mapped from [−1, 1] onto [−1, τ] ---
    # h sₙ
    scaled = einx.multiply("..., node -> ... node", half, nodes)
    # tₙ = h sₙ + h − 1
    t = einx.add("..., ... node", half - 1.0, scaled)
    # --- the two integrals ---
    bump_values = _bump(t)
    # ∑ₙ wₙ g(tₙ)
    weighted_sum = einx.dot("... node, node -> ...", bump_values, weights)
    # ∫_{−1}^{τ} g ≈ h ∑ₙ wₙ g(tₙ)
    partial = half * weighted_sum
    bump_nodes = _bump(nodes)
    # ∫₋₁¹ g ≈ ∑ₙ wₙ g(sₙ)
    total = einx.dot("node, node ->", bump_nodes, weights)
    # χ(τ) = ∫_{−1}^{τ} g / ∫₋₁¹ g
    return partial / total


def needlet_window(
    B: float = 2.0,
) -> Callable[[Float[Array, "*batch"]], Float[Array, "*batch"]]:
    """The needlet window b: smooth, supported in [1/B, B], a partition of unity.

    With the C^∞ step χ (the normalised integral of exp(−1/(1 − t²))) and
    φ(t) = 1 for t ≤ 1/B, χ(1 − 2B/(B − 1) (t − 1/B)) for 1/B ≤ t ≤ 1, and 0 for
    t ≥ 1, the window is b(t) = √(φ(t/B) − φ(t)). The sum
    ∑_{j=0}^{j_max} b²(t/Bʲ) = φ(t/B^{j_max+1}) − φ(t) telescopes to 1 for
    1 ≤ t ≤ B^{j_max}.

    Args:
        B: B, the scale factor, above 1.

    Returns:
        A function that evaluates b elementwise (integers become float32).

    Raises:
        ValidationError: If ``B`` is not above 1.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.sphere import needlet_window
        >>> window = needlet_window(2.0)
        >>> values = window(jnp.array([0.4, 1.0, 2.5]))
        >>> bool(jnp.allclose(values, jnp.array([0.0, 1.0, 0.0]), atol=1e-6))
        True

        The squared windows over three scales sum to 1 on [1, B²]:

        >>> xi = jnp.linspace(1.0, 4.0, 7)
        >>> total = window(xi) ** 2 + window(xi / 2) ** 2 + window(xi / 4) ** 2
        >>> bool(jnp.allclose(total, 1.0))
        True
    """
    if not B > 1.0:
        msg = f"'B' must be above 1, got {B!r}"
        raise ValidationError(msg)

    def phi(t: Float[Array, "*batch"]) -> Float[Array, "*batch"]:
        # τ = 1 − 2B/(B − 1) (t − 1/B), from 1 at t = 1/B down to −1 at t = 1
        argument = 1.0 - 2.0 * B / (B - 1.0) * (t - 1.0 / B)
        # χ(τ)
        ramp = _smooth_step(argument)
        # φ(t) = 0 for t ≥ 1, χ(τ) between
        falling = jnp.where(t >= 1.0, 0.0, ramp)
        # φ(t) = 1 for t ≤ 1/B
        return jnp.where(t <= 1.0 / B, 1.0, falling)

    def window(xi: Float[Array, "*batch"]) -> Float[Array, "*batch"]:
        xi = jnp.asarray(xi)
        if not jnp.issubdtype(xi.dtype, jnp.floating):
            xi = xi.astype(jnp.float32)
        # φ(t/B) − φ(t)
        difference = phi(xi / B) - phi(xi)
        # clipped against round-off below 0
        clipped = jnp.maximum(difference, 0.0)
        # b(t) = √(φ(t/B) − φ(t))
        return jnp.sqrt(clipped)

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

    At scale j, with L_j = ⌊B^{j+1}⌋ and the Gauss–Legendre grid of nodes ξ_jk
    and weights w_jk exact to degree 2L_j (`gauss_legendre_grid` with L_j + 1
    latitudes and 2L_j + 2 longitudes), the needlet at r̂ ∈ S² is

        ψ_jk(r̂) = √w_jk ∑_{ℓ=0}^{L_j} b(ℓ/Bʲ) (2ℓ + 1)/(4π) P_ℓ(ξ_jk · r̂),

    with b the window of `needlet_window` and P_ℓ the Legendre polynomials.
    The first column is the constant Y₀₀ = 1/√(4π) (level −1, centre (0, 0, 0)),
    which the needlets omit since b(0) = 0. With it, the D columns ψ_d satisfy
    ∑_{d=1}^{D} ⟨f, ψ_d⟩ ψ_d = f for every f of degree at most B^{j_max}.

    Algorithm:
        1. Start with the constant column Y₀₀ = 1/√(4π).
        2. For each scale j, lay out the Gauss–Legendre grid exact to degree
           2L_j and weight the degrees by b(ℓ/Bʲ) (2ℓ + 1)/(4π).
        3. Sum the weighted Legendre series at ξ_jk · r̂ and scale each centre's
           column by √w_jk.
        4. Join the scales, whose numbers of centres differ.

    Args:
        unit_xyz: The N evaluation points r̂ on the unit sphere.
        B: B, the scale factor, above 1.
        j_max: j_max, the finest scale, at least 0.

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
        >>> # the first column is the constant Y₀₀ = 1/√(4π)
        >>> bool(jnp.allclose(values[:, 0], 1 / jnp.sqrt(4 * jnp.pi)))
        True
        >>> centres.shape
        (69, 3)
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
        # --- the degree weights ---
        # ℓ/Bʲ
        relative_degree = jnp.asarray(ells / B**scale, dtype=dtype)
        # b(ℓ/Bʲ)
        windows = window(relative_degree)
        # (2ℓ + 1)/(4π)
        addition = jnp.asarray((2.0 * ells + 1.0) / (4.0 * math.pi), dtype=dtype)
        # b(ℓ/Bʲ) (2ℓ + 1)/(4π)
        coefficients = windows * addition
        # --- the kernel at each centre ---
        # ξ_jk · r̂
        inner = einx.dot("obs dim, centre dim -> obs centre", unit_xyz, nodes)
        # clipped against round-off outside [−1, 1]
        cosines = jnp.clip(inner, -1.0, 1.0)
        polynomials = legendre_polynomials(cosines, degree)
        # ∑_ℓ b(ℓ/Bʲ) (2ℓ + 1)/(4π) P_ℓ(ξ_jk · r̂)
        kernel = einx.dot(
            "obs centre degree, degree -> obs centre", polynomials, coefficients
        )
        sqrt_weights = jnp.sqrt(weights)
        # ψ_jk = √w_jk ∑_ℓ b(ℓ/Bʲ) (2ℓ + 1)/(4π) P_ℓ(ξ_jk · r̂)
        columns.append(einx.multiply("obs centre, centre", kernel, sqrt_weights))
        levels += [scale] * int(nodes.shape[0])
        centres.append(nodes)
    # the scales have different numbers of centres, so they are joined as a list
    return jnp.hstack(columns), jnp.asarray(levels), jnp.vstack(centres)
