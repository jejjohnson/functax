"""Tests for the non-uniform FFTs of `functax.transform`, copied from geonnax be5537d
with functax imports. The slow tier: round trips, grid bridges, convergence and the
jit, vmap and grad check, each a second or more (mostly building the tables).
"""

from __future__ import annotations

import math

import einx
import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from hypothesis import given, settings

from functax._src.transform._nufft import _corner_slices
from functax.transform import nonuniform_irfft, nonuniform_rfft


KEY = jr.PRNGKey(0)


GRIDS = {1: ((32,), (5,)), 2: ((16, 12), (4, 3)), 3: ((8, 8, 6), (3, 2, 2))}


def _nodes(grid: tuple[int, ...]) -> jax.Array:
    """Uniform grid nodes ``j / s`` in C order, shape ``(S, d)``."""
    axes = [np.arange(s) / s for s in grid]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return jnp.asarray(mesh.reshape(-1, len(grid)))


def _flatten(u: jax.Array) -> jax.Array:
    """``(C, *grid)`` → ``(C, S)`` in the same C order as `_nodes`."""
    spatial = " ".join(f"s{a}" for a in range(u.ndim - 1))
    return einx.id(f"c {spatial} -> c ({spatial})", u)


def _rfft_corners(u: jax.Array, n_modes: tuple[int, ...]) -> list[jax.Array]:
    """``rfftn(u) / S`` restricted to the corner blocks."""
    axes = tuple(range(1, u.ndim))
    u_ft = jnp.fft.rfftn(u, axes=axes) / math.prod(u.shape[1:])
    return [u_ft[(slice(None), *sl)] for sl in _corner_slices(n_modes)]


def _trig_poly(xi: jax.Array) -> jax.Array:
    """A real band-limited field with modes inside n_modes = (4, 4); shape (2, N)."""
    k = jnp.asarray([[0, 0], [1, 2], [-3, 1], [2, 0], [3, 3], [-1, 3]], jnp.float32)
    amp = jnp.asarray([0.7, 1.0, -0.5, 0.8, 0.3, 0.6])
    phi = jnp.asarray([0.0, 0.3, 1.1, -0.4, 2.0, 0.9])
    arg = 2 * jnp.pi * einx.dot("n d, j d -> n j", xi, k) + phi
    first = einx.dot("n j, j -> n", jnp.cos(arg), amp)
    second = einx.dot("n j, j -> n", jnp.sin(arg), amp[::-1])
    return jnp.stack([first, second])


def _jittered(key: jax.Array, s: int) -> jax.Array:
    """One uniform point per cell of an ``s × s`` grid, shape ``(s², 2)``."""
    return _nodes((s, s)) + jr.uniform(key, (s * s, 2)) / s


@pytest.mark.parametrize("nd", [1, 2, 3])
def test_nonuniform_rfft_grid_bridge(nd):
    """On grid nodes with w = 1/S the transform is ``rfftn(u) / S`` on the corners."""
    grid, n_modes = GRIDS[nd]
    s = math.prod(grid)
    u = jr.normal(KEY, (2, *grid))
    blocks = nonuniform_rfft(
        _flatten(u), _nodes(grid), jnp.full((s,), 1.0 / s), n_modes
    )
    for got, ref in zip(blocks, _rfft_corners(u, n_modes), strict=True):
        assert got.shape == (2, *n_modes)
        np.testing.assert_allclose(got, ref, atol=1e-5)


@pytest.mark.parametrize("nd", [1, 2, 3])
def test_nonuniform_irfft_matches_irfftn_on_grid(nd):
    """At grid nodes the inverse equals ``irfftn`` of the scattered corners × S."""
    grid, n_modes = GRIDS[nd]
    s = math.prod(grid)
    keys = jr.split(KEY, 2 ** (nd - 1))
    # Random, not Hermitian-consistent (irfftn's handling must match anyway),
    # scaled so the field is O(1); gate of #23: 1e-12 in float64 (geonnax used
    # a float32 tolerance of 1e-5)
    scale = 1.0 / math.sqrt(2 * len(keys) * math.prod(n_modes))
    blocks = [scale * jax.lax.complex(*jr.normal(k, (2, 3, *n_modes))) for k in keys]
    # the spectrum in the blocks' own dtype (complex128 under x64)
    spectrum = jnp.zeros((3, *grid[:-1], grid[-1] // 2 + 1), dtype=blocks[0].dtype)
    for sl, b in zip(_corner_slices(n_modes), blocks, strict=True):
        spectrum = spectrum.at[(slice(None), *sl)].set(s * b)
    ref = jnp.fft.irfftn(spectrum, s=grid, axes=tuple(range(1, nd + 1)))
    got = nonuniform_irfft(blocks, _nodes(grid), n_modes)
    assert got.shape == (3, s)
    np.testing.assert_allclose(got, _flatten(ref), atol=1e-12)


def test_round_trip_band_limited():
    """rfft on a grid then irfft at random points recovers a band-limited field."""
    grid, n_modes = (16, 16), (4, 4)
    nodes = _nodes(grid)
    s = math.prod(grid)
    blocks = nonuniform_rfft(_trig_poly(nodes), nodes, jnp.full((s,), 1.0 / s), n_modes)
    query = jr.uniform(KEY, (200, 2))
    got = nonuniform_irfft(blocks, query, n_modes)
    np.testing.assert_allclose(got, _trig_poly(query), atol=1e-5)


def test_zero_weight_padding_is_bit_identical():
    """Points with w = 0 contribute nothing: 300 points padded to 500 match exactly."""
    k1, k2, k3 = jr.split(KEY, 3)
    xi = jr.uniform(k1, (300, 2))
    u = jr.normal(k2, (3, 300))
    w = jr.uniform(k3, (300,)) / 300
    pad = 200
    xi_p = jnp.concatenate([xi, jnp.zeros((pad, 2))])
    u_p = jnp.concatenate([u, jnp.ones((3, pad))], axis=-1)
    w_p = jnp.concatenate([w, jnp.zeros((pad,))])
    ref = nonuniform_rfft(u, xi, w, (4, 4))
    got = nonuniform_rfft(u_p, xi_p, w_p, (4, 4))
    for a, b in zip(got, ref, strict=True):
        assert jnp.array_equal(a, b)


def test_weights_are_not_normalised():
    """Scaling every weight by α scales the coefficients by α."""
    k1, k2 = jr.split(KEY)
    xi = jr.uniform(k1, (64, 2))
    u = jr.normal(k2, (1, 64))
    w = jnp.full((64,), 1.0 / 64)
    ref = nonuniform_rfft(u, xi, w, (3, 3))
    got = nonuniform_rfft(u, xi, 3.0 * w, (3, 3))
    for a, b in zip(got, ref, strict=True):
        np.testing.assert_allclose(a, 3.0 * b, rtol=1e-5, atol=1e-6)


def test_convergence_with_n_for_jittered_points():
    """Stratified (jittered) quadrature with w = 1/N converges as N grows.

    The field is smooth but not band-limited, so the coefficients are a true
    quadrature. Jittered sampling converges ~N^-1 (the #75 prototype measured
    0.39 → 0.024 over N ≈ 500 → 8000); a 16× increase in N must cut the error
    by at least 4×, a loose bound well inside that rate.
    """
    n_modes = (4, 4)

    def field(xi):
        return jnp.exp(jnp.sin(2 * jnp.pi * xi[:, 0]) * jnp.cos(2 * jnp.pi * xi[:, 1]))[
            None
        ]

    fine = _nodes((128, 128))
    ref = nonuniform_rfft(field(fine), fine, jnp.full((128 * 128,), 128.0**-2), n_modes)
    ref_norm = math.sqrt(sum(float(jnp.sum(jnp.abs(r) ** 2)) for r in ref))

    def rel_err(s, key):
        xi = _jittered(key, s)
        got = nonuniform_rfft(field(xi), xi, jnp.full((s * s,), 1.0 / s**2), n_modes)
        sq = sum(
            float(jnp.sum(jnp.abs(g - r) ** 2)) for g, r in zip(got, ref, strict=True)
        )
        return math.sqrt(sq) / ref_norm

    k1, k2 = jr.split(KEY)
    coarse, finer = rel_err(16, k1), rel_err(64, k2)
    assert finer < 0.25 * coarse
    assert finer < 0.05


def test_nonuniform_jit_vmap_grad():
    """jit, vmap over a batch of padded clouds, and finite gradients w.r.t. xi."""
    n_modes = (3, 3)
    b, n = 4, 50
    k1, k2, k3 = jr.split(KEY, 3)
    xi = jr.uniform(k1, (b, n, 2))
    u = jr.normal(k2, (b, 2, n))
    w = jnp.where(jnp.arange(n) < 40, 1.0 / 40, 0.0)
    w = einx.multiply("n, b -> b n", w, jnp.ones((b,)))
    query = jr.uniform(k3, (b, 30, 2))

    def roundtrip(u, xi, w, q):
        return nonuniform_irfft(nonuniform_rfft(u, xi, w, n_modes), q, n_modes)

    assert jax.vmap(roundtrip)(u, xi, w, query).shape == (b, 2, 30)
    # jit, vmap and jit(vmap) against an eager loop over the four clouds
    assert_traces(roundtrip, u, xi, w, query, rtol=1e-12)

    def loss(xi_, q_):
        return jnp.sum(jax.jit(roundtrip)(u[0], xi_, w[0], q_) ** 2)

    g_xi, g_q = jax.grad(loss, argnums=(0, 1))(xi[0], query[0])
    assert bool(jnp.all(jnp.isfinite(g_xi))) and bool(jnp.all(jnp.isfinite(g_q)))
    assert float(jnp.abs(g_xi).max()) > 0.0


_MODES_2D = (3, 4)


# compiled once per module, so each hypothesis example reuses the kernels
_RFFT_2D = jax.jit(lambda u, xi, w: nonuniform_rfft(u, xi, w, _MODES_2D))


_IRFFT_2D = jax.jit(lambda c, xi: nonuniform_irfft(c, xi, _MODES_2D))


_RFFT_1D = jax.jit(lambda u, xi, w: nonuniform_rfft(u, xi, w, (6,))[0])


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_synthesis_is_the_weighted_adjoint_of_analysis(seed: int) -> None:
    # ∑_n w_n u_n · irfft(c)_n = ∑_κ γ_κ (Re c_κ Re ĉ_κ + Im c_κ Im ĉ_κ) with
    # ĉ = rfft(u), for any points, weights and coefficients (γ_κ = 1 on the
    # κ_d = 0 plane, 2 elsewhere)
    keys = jr.split(jr.key(seed), 5)
    xi = jr.uniform(keys[0], (40, 2))
    weights = jr.uniform(keys[1], (40,), maxval=0.05)
    u = jr.normal(keys[2], (2, 40))
    coeffs = [
        jax.lax.complex(*jr.normal(key, (2, 2, *_MODES_2D)))
        for key in jr.split(keys[3], 2)
    ]
    lhs = jnp.sum(weights * u * _IRFFT_2D(coeffs, xi))
    multiplicity = jnp.where(jnp.arange(_MODES_2D[-1]) == 0, 1.0, 2.0)
    rhs = sum(
        jnp.sum(multiplicity * (c.real * r.real + c.imag * r.imag))
        for c, r in zip(coeffs, _RFFT_2D(u, xi, weights), strict=True)
    )
    assert_matches_oracle(lhs, rhs, rtol=1e-12)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), shift=st.floats(0.0, 1.0))
def test_a_shift_of_the_points_turns_the_phases(seed: int, shift: float) -> None:
    # ĉ_κ of the points moved by s (mod 1) is ĉ_κ e^{−2πiκs}
    keys = jr.split(jr.key(seed), 3)
    xi = jr.uniform(keys[0], (30, 1))
    weights = jr.uniform(keys[1], (30,), maxval=0.05)
    u = jr.normal(keys[2], (2, 30))
    moved = _RFFT_1D(u, (xi + shift) % 1.0, weights)
    turn = jnp.exp(-2j * jnp.pi * jnp.arange(6) * shift)
    expected = _RFFT_1D(u, xi, weights) * turn
    for part in (jnp.real, jnp.imag):
        assert_matches_oracle(part(moved), part(expected), rtol=0.0, atol=1e-12)


def test_gradients_in_the_points_match_finite_differences() -> None:
    # rfft and irfft differentiated in the coordinates ξ (real and imaginary
    # parts stacked, as the Jacobian needs a real output)
    keys = jr.split(jr.key(9), 3)
    xi = jr.uniform(keys[0], (6, 2))
    u = jr.normal(keys[1], (1, 6))
    coeffs = [
        jax.lax.complex(*jr.normal(key, (2, 1, 2, 2))) for key in jr.split(keys[2], 2)
    ]
    weights = jnp.full(6, 1.0 / 6.0)

    def analysis(points: jax.Array) -> list[jax.Array]:
        blocks = nonuniform_rfft(u, points, weights, (2, 2))
        return [jnp.stack([b.real, b.imag]) for b in blocks]

    assert_grad_matches_fd(analysis, xi, rtol=1e-8)
    assert_grad_matches_fd(
        lambda points: nonuniform_irfft(coeffs, points, (2, 2)), xi, rtol=1e-8
    )
