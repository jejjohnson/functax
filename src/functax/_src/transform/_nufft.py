"""Non-uniform Fourier transforms between point sets and a truncated rfft spectrum.

`nonuniform_rfft` takes the low Fourier modes of a field u from its values at N
arbitrary points ξ_n of the torus [0, 1)^d, n = 1, …, N, and `nonuniform_irfft`
evaluates a truncated real Fourier series at arbitrary points, the
scattered-point read-out of spectral interpolation. The forward transform takes
quadrature weights w_n, so it approximates the continuous Fourier coefficient
at each integer wavenumber vector κ (d integers) and converges as the points
grow. The spectrum is stored as the 2^{d−1} low corners of an rfft spectrum,
each (C, *n_modes) for C channels; on a uniform grid of S nodes with w_n = 1/S
the coefficients equal ``rfftn(u) / S`` on the retained modes. Moved from
geonnax ``layers/_nonuniform`` at ``be5537d``.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import itertools
import math

import einx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float, Int

from functax.exceptions import ValidationError


__all__ = [
    "nonuniform_irfft",
    "nonuniform_rfft",
]


def _corner_slices(n_modes: tuple[int, ...]) -> list[tuple[slice, ...]]:
    """The slices that pick the 2^{d−1} low-frequency corners of an rfft spectrum.

    Every axis but the last keeps its lowest positive ([0:m]) and negative
    ([−m:]) frequencies; the last, halved by the rfft, keeps [0:m] only.
    """
    per_axis = [[slice(0, m), slice(-m, None)] for m in n_modes[:-1]]
    combos = itertools.product(*per_axis) if len(n_modes) > 1 else [()]
    last = slice(0, n_modes[-1])
    return [(*combo, last) for combo in combos]


def _corner_freqs(
    n_modes: tuple[int, ...],
) -> list[tuple[Int[np.ndarray, " _mode"], ...]]:
    """The integer wavenumbers of each corner block, in `_corner_slices` order.

    Each corner holds one integer array per axis: 0, …, m − 1 or −m, …, −1 on
    an axis but the last, 0, …, m − 1 on the last.
    """
    per_axis = [[np.arange(m), np.arange(-m, 0)] for m in n_modes[:-1]]
    combos = itertools.product(*per_axis) if len(n_modes) > 1 else [()]
    last = np.arange(n_modes[-1])
    return [(*combo, last) for combo in combos]


def _modes_spec(d: int) -> str:
    """The einx axis names "m0 m1 …" of the d mode axes."""
    return " ".join(f"m{a}" for a in range(d))


def _validate_modes(n_modes: tuple[int, ...]) -> tuple[int, ...]:
    if len(n_modes) == 0:
        msg = "'n_modes' must not be empty"
        raise ValidationError(msg)
    if any(int(m) < 1 for m in n_modes):
        msg = f"every entry of 'n_modes' must be at least 1, got {n_modes}"
        raise ValidationError(msg)
    return tuple(int(m) for m in n_modes)


def _validate_points(xi: Float[Array, "*shape"], n_modes: tuple[int, ...]) -> None:
    if xi.ndim != 2 or xi.shape[-1] != len(n_modes):
        msg = (
            f"'xi' must have shape (N, {len(n_modes)}) to match n_modes={n_modes}, "
            f"got {xi.shape}"
        )
        raise ValidationError(msg)


def _corner_phases(
    xi: Float[Array, "obs dim"], n_modes: tuple[int, ...]
) -> list[Float[Array, "obs ..."]]:
    """The phases 2π κ · ξ_n of every point against each corner's modes.

    κ · ξ_n is reduced to [−½, ½] before the 2π scaling, which keeps float32
    arguments small; the gradient is unchanged.
    """
    modes = _modes_spec(len(n_modes))
    two_pi = jnp.asarray(2.0 * math.pi, dtype=xi.dtype)
    phases = []
    for frequencies in _corner_freqs(n_modes):
        grids = np.meshgrid(*frequencies, indexing="ij")
        # one d-vector κ per mode; np.stack because d varies with n_modes
        table = np.stack(grids, axis=-1)
        wavenumbers = jnp.asarray(table, dtype=xi.dtype)
        # κ · ξ_n, in turns
        turns = einx.dot(f"obs dim, {modes} dim -> obs {modes}", xi, wavenumbers)
        # the nearest integer is held constant, so the gradient is that of κ · ξ_n
        rounded = jnp.round(turns)
        nearest = jax.lax.stop_gradient(rounded)
        # κ · ξ_n − round(κ · ξ_n), in [−½, ½]
        reduced = turns - nearest
        # α = 2π (κ · ξ_n − round(κ · ξ_n))
        phases.append(two_pi * reduced)
    return phases


def nonuniform_rfft(
    u: Float[Array, "channel obs"],
    xi: Float[Array, "obs dim"],
    weights: Float[Array, " obs"],
    n_modes: tuple[int, ...],
) -> list[Complex[Array, "channel ..."]]:
    """The truncated Fourier coefficients of a field sampled at points of [0, 1)^d.

    A quadrature of the continuous Fourier coefficient on the unit torus, for
    every retained integer wavenumber vector κ (the rfft corner layout), from
    the values u_n = u(ξ_n) at the points n = 1, …, N:
    ĉ_κ = ∑_n w_n u_n exp(−2πi κ · ξ_n) ≈ ∫_{[0,1)^d} u(ξ) exp(−2πi κ · ξ) dξ.

    The weights are used as given, never normalised: they should be
    non-negative and sum to about 1 (cell areas of a mesh, or 1/N for a
    stratified cloud). A point of weight 0 contributes nothing, so ragged
    clouds can be zero-padded to one N and ``vmap``-ed, as long as the padded
    coordinates are finite. The cost is O(N N_modes C) multiply-adds for C
    channels and an N × N_modes phase table per call, with
    N_modes = 2^{d−1} ∏_a m_a retained modes and m_a the modes kept on axis a;
    for very large clouds a fast NUFFT (``jax-finufft``) computes the same
    sums and stays a user-side substitute, not a dependency.

    Algorithm:
        1. Weight each value by its quadrature weight, w_n u_n.
        2. For each corner block, form the phases α_nκ = 2π κ · ξ_n, with
           κ · ξ_n reduced to [−½, ½] first.
        3. Contract the weighted values with cos α and sin α over the points:
           ĉ_κ = ∑_n w_n u_n cos α_nκ − i ∑_n w_n u_n sin α_nκ.

    Args:
        u: The field values u_n, one row per channel.
        xi: The coordinates ξ_n in [0, 1)^d of the N points, d = len(n_modes).
        weights: w_n, the quadrature weight of each point.
        n_modes: The retained modes per axis.

    Returns:
        The 2^{d−1} corner blocks, each (C, *n_modes), lowest positive and
        negative frequencies on every axis but the last.

    Raises:
        ValidationError: If ``n_modes`` is empty or has an entry below 1, or
            the shapes of ``u``, ``xi`` and ``weights`` disagree.

    References:
        Li, Z., Huang, D. Z., Liu, B., Anandkumar, A. (2023). Fourier neural
        operator with learned deformations for PDEs on general geometries.
        JMLR 24 (the unweighted original; here the sum is a quadrature).

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from functax.transform import nonuniform_rfft
        >>> xi = jr.uniform(jr.key(0), (500, 2))
        >>> u = jnp.sin(2 * jnp.pi * xi[:, 0])[None]
        >>> blocks = nonuniform_rfft(u, xi, jnp.full((500,), 1 / 500), (4, 4))
        >>> len(blocks), blocks[0].shape
        (2, (1, 4, 4))

        In one dimension there is a single block, and on a uniform grid with
        w_n = 1/S it is the retained part of ``rfft(u) / S``:

        >>> nodes = jnp.arange(8.0) / 8
        >>> u = jnp.cos(2 * jnp.pi * nodes)[None]
        >>> (block,) = nonuniform_rfft(
        ...     u, nodes[:, None], jnp.full((8,), 1 / 8), (3,)
        ... )
        >>> bool(jnp.allclose(block, jnp.fft.rfft(u)[:, :3] / 8, atol=1e-12))
        True
    """
    n_modes = _validate_modes(n_modes)
    _validate_points(xi, n_modes)
    n = xi.shape[0]
    if u.ndim != 2 or u.shape[-1] != n:
        msg = f"'u' must have shape (C, {n}) to match xi, got {u.shape}"
        raise ValidationError(msg)
    if weights.shape != (n,):
        msg = f"'weights' must have shape ({n},) to match xi, got {weights.shape}"
        raise ValidationError(msg)
    modes = _modes_spec(len(n_modes))
    contraction = f"channel obs, obs {modes} -> channel {modes}"
    # w_n u_n
    weighted = einx.multiply("channel obs, obs", u, weights)
    blocks = []
    for phase in _corner_phases(xi, n_modes):
        cosine = jnp.cos(phase)
        sine = jnp.sin(phase)
        # Re ĉ_κ = ∑_n w_n u_n cos α_nκ
        real_part = einx.dot(contraction, weighted, cosine)
        # ∑_n w_n u_n sin α_nκ
        sine_sum = einx.dot(contraction, weighted, sine)
        # Im ĉ_κ = −∑_n w_n u_n sin α_nκ
        imag_part = -sine_sum
        blocks.append(jax.lax.complex(real_part, imag_part))
    return blocks


def nonuniform_irfft(
    coeffs: list[Complex[Array, "channel ..."]],
    xi: Float[Array, "obs dim"],
    n_modes: tuple[int, ...],
) -> Float[Array, "channel obs"]:
    """A truncated real Fourier series evaluated at points of [0, 1)^d.

    The inverse of `nonuniform_rfft` on band-limited fields, with the
    Hermitian half-spectrum handled as ``irfftn`` does:
    u(ξ) = ∑_κ γ_κ Re[ĉ_κ exp(2πi κ · ξ)] over the retained integer wavenumbers
    κ = (κ_1, …, κ_d), with the multiplicity γ_κ = 1 where the last component
    κ_d is 0 and 2 elsewhere. Taking the real part drops the imaginary part of
    the κ_d = 0 plane, as ``irfftn`` does; no Nyquist term arises because the
    modes stay below half of any grid. On the nodes ξ = j/s of a uniform grid,
    j ∈ {0, …, s − 1}^d and S = s^d nodes in all, this equals ``irfftn`` of the
    corners scattered into a spectrum, times S. The cost is
    that of `nonuniform_rfft`.

    Algorithm:
        1. Weight the real and imaginary parts of each block by γ_κ.
        2. For each corner block, form the phases α_nκ = 2π κ · ξ_n as
           `nonuniform_rfft` does.
        3. Sum Re[ĉ_κ exp(iα_nκ)] = Re ĉ_κ cos α_nκ − Im ĉ_κ sin α_nκ over the
           modes of every block.

    Args:
        coeffs: The 2^{d−1} corner blocks, each (C, *n_modes), as
            `nonuniform_rfft` returns them.
        xi: The query coordinates ξ_n in [0, 1)^d.
        n_modes: The retained modes per axis.

    Returns:
        The real field at each query point, one row per channel.

    Raises:
        ValidationError: If ``n_modes`` is empty or has an entry below 1,
            ``xi`` has the wrong number of columns, or ``coeffs`` has the
            wrong number or shape of blocks.

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from functax.transform import nonuniform_irfft, nonuniform_rfft
        >>> s = 16
        >>> g = jnp.arange(s) / s
        >>> nodes = jnp.stack(jnp.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        >>> u = jnp.cos(2 * jnp.pi * (nodes[:, 0] + 2 * nodes[:, 1]))[None]
        >>> blocks = nonuniform_rfft(u, nodes, jnp.full((s * s,), 1 / s**2), (4, 4))
        >>> query = jr.uniform(jr.key(0), (300, 2))
        >>> values = nonuniform_irfft(blocks, query, (4, 4))
        >>> exact = jnp.cos(2 * jnp.pi * (query[:, 0] + 2 * query[:, 1]))
        >>> values.shape, bool(jnp.allclose(values[0], exact, atol=1e-12))
        ((1, 300), True)

        A zero spectrum synthesises zero everywhere:

        >>> zero = [jnp.zeros((2, 4, 4), dtype=jnp.complex128)] * 2
        >>> bool(jnp.allclose(nonuniform_irfft(zero, query, (4, 4)), 0.0))
        True
    """
    n_modes = _validate_modes(n_modes)
    _validate_points(xi, n_modes)
    d = len(n_modes)
    if len(coeffs) != 2 ** (d - 1):
        msg = (
            f"'coeffs' must hold 2**(d-1) = {2 ** (d - 1)} corner blocks for "
            f"n_modes={n_modes}, got {len(coeffs)}"
        )
        raise ValidationError(msg)
    for block in coeffs:
        if block.ndim != d + 1 or tuple(block.shape[1:]) != n_modes:
            msg = (
                f"each corner block must have shape (C, *{n_modes}), got {block.shape}"
            )
            raise ValidationError(msg)
    modes = _modes_spec(d)
    last = f"m{d - 1}"
    weighting = f"channel {modes}, {last}"
    contraction = f"channel {modes}, obs {modes} -> channel obs"
    # γ_κ = 1 on the κ_d = 0 plane, 2 elsewhere
    multiplicity = np.full(n_modes[-1], 2.0)
    multiplicity[0] = 1.0
    n_channels = coeffs[0].shape[0]
    real_dtype = coeffs[0].real.dtype
    total = jnp.zeros((n_channels, xi.shape[0]), dtype=real_dtype)
    for block, phase in zip(coeffs, _corner_phases(xi, n_modes), strict=True):
        mu = jnp.asarray(multiplicity, dtype=block.real.dtype)
        # γ_κ Re ĉ_κ
        weighted_real = einx.multiply(weighting, block.real, mu)
        # γ_κ Im ĉ_κ
        weighted_imag = einx.multiply(weighting, block.imag, mu)
        cosine = jnp.cos(phase)
        sine = jnp.sin(phase)
        # ∑_κ γ_κ Re ĉ_κ cos α_nκ
        cosine_sum = einx.dot(contraction, weighted_real, cosine)
        # ∑_κ γ_κ Im ĉ_κ sin α_nκ
        sine_sum = einx.dot(contraction, weighted_imag, sine)
        # Re[ĉ_κ (cos α + i sin α)] = Re ĉ_κ cos α − Im ĉ_κ sin α
        part = cosine_sum - sine_sum
        total = total + part
    return total
