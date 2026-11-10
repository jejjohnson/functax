"""Tests for `functax.basis.polynomial`, copied from geonnax be5537d with functax
imports. The slow tier: quadrature orthonormality, high-degree float32 checks,
gradients and frame reconstructions, each a second or more; and the gradient and
trace checks of `legendre_series` (#47).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special as sp
from _harness import assert_grad_matches_fd, assert_traces

from functax.basis.polynomial import (
    associated_legendre,
    associated_legendre_indices,
    legendre_polynomials,
    legendre_series,
)
from functax.quad import gauss_legendre


T = np.linspace(-1.0, 1.0, 41)


def _column_scale(ref):
    return np.maximum(np.max(np.abs(ref), axis=0, keepdims=True), 1.0)


def _reference_orthonormal(t, l_max):
    """P̄ₗᵐ(t) by the same recurrence in NumPy float64, with no exponent carry.

    Column order of `associated_legendre`. Exact (to rounding) wherever the
    sectoral seed stays in float64 range, which covers l_max <= 1000.
    """
    s = np.sqrt(np.clip(1.0 - t**2, 0.0, None))
    out = np.zeros((t.size, (l_max + 1) * (l_max + 2) // 2))
    col = {lm: i for i, lm in enumerate(associated_legendre_indices(l_max))}
    pmm = np.full(t.size, 1.0 / math.sqrt(2.0))
    for m in range(l_max + 1):
        if m > 0:
            pmm = -math.sqrt((2 * m + 1) / (2 * m)) * s * pmm
        p_prev, p_curr = np.zeros_like(t), pmm
        out[:, col[(m, m)]] = pmm
        for ell in range(m + 1, l_max + 1):
            if ell == m + 1:
                p_next = math.sqrt(2 * m + 3) * t * p_curr
            else:
                a = math.sqrt((4 * ell**2 - 1) / (ell**2 - m**2))
                b = math.sqrt(((ell - 1) ** 2 - m**2) / (4 * (ell - 1) ** 2 - 1))
                p_next = a * (t * p_curr - b * p_prev)
            out[:, col[(ell, m)]] = p_next
            p_prev, p_curr = p_curr, p_next
    return out


def test_classical_matches_scipy_lpmv():
    l_max = 20
    with jax.enable_x64(True):
        A = np.asarray(associated_legendre(jnp.asarray(T), l_max, normalization="none"))
    idx = associated_legendre_indices(l_max)
    ref = np.stack([sp.lpmv(m, ell, T) for ell, m in idx], axis=-1)
    # P_l^m spans many orders of magnitude, and some values are near-roots of a
    # large polynomial, so compare relative to each column's scale.
    np.testing.assert_allclose(
        A / _column_scale(ref), ref / _column_scale(ref), atol=1e-12
    )


@pytest.mark.parametrize("m", [0, 3, 50])
def test_orthonormal_by_quadrature(m):
    l_max = 200
    nodes, weights = gauss_legendre(220)
    with jax.enable_x64(True):
        P = np.asarray(associated_legendre(jnp.asarray(nodes), l_max))
    cols = [
        i for i, (_, mm) in enumerate(associated_legendre_indices(l_max)) if mm == m
    ]
    gram = (P[:, cols] * weights[:, None]).T @ P[:, cols]
    np.testing.assert_allclose(gram, np.eye(len(cols)), atol=1e-11)
    assert np.isfinite(P).all()


def test_float64_unchanged_by_exponent_carry():
    # #122: the carry rescales by exact powers of two, so float64 values equal the
    # plain recurrence to rounding: measured 4.7e-13 at l_max = 127 with
    # max |P̄| ≈ 11, the same as before the carry; 1e-12 leaves margin.
    t = np.concatenate([np.linspace(-1.0, 1.0, 21), [0.9999, -0.99999]])
    with jax.enable_x64(True):
        P = np.asarray(associated_legendre(jnp.asarray(t), 127))
    np.testing.assert_allclose(P, _reference_orthonormal(t, 127), rtol=0, atol=1e-12)


@pytest.mark.parametrize("n", [256, 512])
def test_orthonormal_float32_high_degree(n):
    # #122: without the exponent carry the sectoral seed (1 - t²)^{m/2}
    # underflowed float32 near the poles, leaving all-zero rows and a Gram error
    # of 1.2e-1 (n = 256) / 4.0e-1 (n = 512). With it: 1.2e-5 / 2.7e-5 measured;
    # 5e-5 leaves margin (float32 rounding accumulates over ~n recurrence steps).
    nodes, weights = gauss_legendre(n)
    P = np.asarray(associated_legendre(jnp.asarray(nodes, jnp.float32), n - 1))
    assert P.dtype == np.float32
    orders = np.array([m for _, m in associated_legendre_indices(n - 1)])
    err = 0.0
    for m in range(n):
        rows = P[:, orders == m].astype(np.float64).T  # (n − m, n)
        assert (np.abs(rows).max(axis=1) > 0).all(), f"zero row at m = {m}"
        gram = (rows * weights) @ rows.T
        err = max(err, float(np.abs(gram - np.eye(len(rows))).max()))
    assert err < 5e-5


def test_gradient_matches_finite_difference_at_high_degree():
    # d P̄₂₀₀¹⁵⁰ / dt through the exponent carry, against a central difference
    # (truncation ~h² · P''' ≈ 1e-6 relative here; measured agreement 3e-9).
    col = associated_legendre_indices(255).index((200, 150))

    def f(t):
        return associated_legendre(t[None], 255)[0, col]

    with jax.enable_x64(True):
        t0, h = jnp.float64(0.3), 1e-6
        grad = float(jax.grad(f)(t0))
        fd = float((f(t0 + h) - f(t0 - h)) / (2 * h))
    assert abs(grad - fd) <= 1e-6 * max(abs(fd), 1.0)


@pytest.mark.parametrize("t0", [0.0, 0.3, -0.9])
def test_gradient_matches_finite_difference_where_values_vanish(t0):
    # At t = 0 every P̄ₗᵐ with odd l − m is exactly zero but has a non-zero
    # derivative; the exponent carry must still give d/dt exactly (#122).
    # Central-difference truncation ~1e-10 at l_max = 8; measured 2e-10.
    def f(t):
        return associated_legendre(t[None], 8)[0]

    with jax.enable_x64(True):
        t, h = jnp.float64(t0), 1e-6
        jac = np.asarray(jax.jacfwd(f)(t))
        fd = np.asarray((f(t + h) - f(t - h)) / (2 * h))
    np.testing.assert_allclose(jac, fd, rtol=0, atol=1e-7)


def test_gradient_finite_inside_interval():
    t = jnp.linspace(-0.95, 0.95, 7)

    def f(t):
        return jnp.sum(associated_legendre(t, 8)) + jnp.sum(legendre_polynomials(t, 8))

    assert jnp.isfinite(jax.grad(f)(t)).all()


def test_legendre_series_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(2)
    coeffs = jnp.asarray(rng.standard_normal(21))
    x = jnp.asarray(rng.uniform(-0.95, 0.95, 6))
    assert_grad_matches_fd(lambda points: legendre_series(points, coeffs), x)
    # linear in the coefficients: ∂f/∂cₗ = Pₗ(x)
    assert_grad_matches_fd(lambda c: legendre_series(x, c), coeffs)


def test_legendre_series_traces() -> None:
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.uniform(-1.0, 1.0, (4, 9)))
    coeffs = jnp.asarray(rng.standard_normal((4, 12)))
    assert_traces(legendre_series, x, coeffs)
