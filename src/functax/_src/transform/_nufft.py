"""Non-uniform Fourier transforms between point sets and a truncated rfft spectrum.

`nonuniform_rfft` takes the low Fourier modes of a field from its values at
arbitrary points of the torus [0, 1)^d, and `nonuniform_irfft` evaluates a
truncated real Fourier series at arbitrary points, the scattered-point
read-out of spectral interpolation. The forward transform takes quadrature
weights w_n, so it approximates the continuous Fourier coefficient and
converges as the points grow. The spectrum is stored as the 2^{d−1} low
corners of an rfft spectrum, each (C, *n_modes); on a uniform grid of S nodes
with w_n = 1/S the coefficients equal ``rfftn(u) / S`` on the retained modes.
Moved from geonnax ``layers/_nonuniform`` at ``be5537d``.
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
from jaxtyping import Array, Complex, Float

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


def _corner_freqs(n_modes: tuple[int, ...]) -> list[tuple[np.ndarray, ...]]:
    """The integer wavenumbers of each corner block, in `_corner_slices` order.

    Corner c holds one integer array per axis: 0, …, m − 1 or −m, …, −1 on an
    axis but the last, 0, …, m − 1 on the last.
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


def _validate_points(xi: jax.Array, n_modes: tuple[int, ...]) -> None:
    if xi.ndim != 2 or xi.shape[-1] != len(n_modes):
        msg = (
            f"'xi' must have shape (N, {len(n_modes)}) to match n_modes={n_modes}, "
            f"got {xi.shape}"
        )
        raise ValidationError(msg)


def _corner_phases(
    xi: Float[Array, "obs dim"], n_modes: tuple[int, ...]
) -> list[jax.Array]:
    """The phases 2π k · ξ_n of every point against each corner's modes.

    k · ξ is reduced to [−½, ½] before the 2π scaling, which keeps float32
    arguments small; the gradient is unchanged.
    """
    modes = _modes_spec(len(n_modes))
    two_pi = jnp.asarray(2.0 * math.pi, dtype=xi.dtype)
    phases = []
    for frequencies in _corner_freqs(n_modes):
        table = np.stack(np.meshgrid(*frequencies, indexing="ij"), axis=-1)
        k = jnp.asarray(table, dtype=xi.dtype)
        theta = einx.dot(f"obs dim, {modes} dim -> obs {modes}", xi, k)
        phases.append(two_pi * (theta - jax.lax.stop_gradient(jnp.round(theta))))
    return phases


def nonuniform_rfft(
    u: Float[Array, "channel obs"],
    xi: Float[Array, "obs dim"],
    weights: Float[Array, " obs"],
    n_modes: tuple[int, ...],
) -> list[Complex[Array, "channel ..."]]:
    """The truncated Fourier coefficients of a field sampled at points of [0, 1)^d.

    A quadrature of the continuous Fourier coefficient on the unit torus, for
    every retained wavenumber k (the rfft corner layout):
    ĉ_k = ∑_n w_n u_n exp(−2πi k · ξ_n) ≈ ∫_{[0,1)^d} u(ξ) exp(−2πi k · ξ) dξ.

    The weights are used as given, never normalised: they should be
    non-negative and sum to about 1 (cell areas of a mesh, or 1/N for a
    stratified cloud). A point of weight 0 contributes nothing, so ragged
    clouds can be zero-padded to one N and ``vmap``-ed, as long as the padded
    coordinates are finite. The cost is O(N |K| C) multiply-adds and an N × |K|
    phase table per call, with |K| = 2^{d−1} ∏_a m_a; for very large clouds a
    fast NUFFT (``jax-finufft``) computes the same sums and stays a user-side
    substitute, not a dependency.

    Args:
        u: The field values, one row per channel.
        xi: The coordinates in [0, 1)^d of the N points, d = len(n_modes).
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
    weighted = einx.multiply("channel obs, obs", u, weights)
    blocks = []
    for phase in _corner_phases(xi, n_modes):
        # ∑_n w_n u_n (cos − i sin)(2π k · ξ_n)
        re = einx.dot(
            f"channel obs, obs {modes} -> channel {modes}", weighted, jnp.cos(phase)
        )
        im = einx.dot(
            f"channel obs, obs {modes} -> channel {modes}", weighted, jnp.sin(phase)
        )
        blocks.append(jax.lax.complex(re, -im))
    return blocks


def nonuniform_irfft(
    coeffs: list[Complex[Array, "channel ..."]],
    xi: Float[Array, "obs dim"],
    n_modes: tuple[int, ...],
) -> Float[Array, "channel obs"]:
    """A truncated real Fourier series evaluated at points of [0, 1)^d.

    The inverse of `nonuniform_rfft` on band-limited fields, with the
    Hermitian half-spectrum handled as ``irfftn`` does:
    u(ξ) = ∑_k μ_k Re[ĉ_k exp(2πi k · ξ)], with μ_k = 1 where the last wavenumber
    k_d is 0 and 2 elsewhere. Taking the real part drops the imaginary part of
    the k_d = 0 plane, as ``irfftn`` does; no Nyquist term arises because the
    modes stay below half of any grid. On grid nodes ξ = j/s this equals
    ``irfftn`` of the corners scattered into a spectrum, times S. The cost is
    that of `nonuniform_rfft`.

    Args:
        coeffs: The 2^{d−1} corner blocks, each (C, *n_modes), as
            `nonuniform_rfft` returns them.
        xi: The query coordinates in [0, 1)^d.
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
    # μ_k = 1 on the k_d = 0 plane, 2 elsewhere
    multiplicity = np.full(n_modes[-1], 2.0)
    multiplicity[0] = 1.0
    total = jnp.zeros((coeffs[0].shape[0], xi.shape[0]), dtype=coeffs[0].real.dtype)
    for block, phase in zip(coeffs, _corner_phases(xi, n_modes), strict=True):
        mu = jnp.asarray(multiplicity, dtype=block.real.dtype)
        re = einx.multiply(f"channel {modes}, {last}", block.real, mu)
        im = einx.multiply(f"channel {modes}, {last}", block.imag, mu)
        # Re[(a + ib)(cos + i sin)] = a cos − b sin
        part = einx.dot(
            f"channel {modes}, obs {modes} -> channel obs", re, jnp.cos(phase)
        )
        part = part - einx.dot(
            f"channel {modes}, obs {modes} -> channel obs", im, jnp.sin(phase)
        )
        total = total + part
    return total
