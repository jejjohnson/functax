"""Tests for the needlets of `functax.basis.sphere`, copied from geonnax be5537d with
functax imports. The slow tier: quadrature orthonormality, high-degree float32
checks, gradients and frame reconstructions, each a second or more.
"""

from __future__ import annotations

import math

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special as sp
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from hypothesis import given, settings

from functax.basis.sphere import needlet_basis, needlet_window, real_spherical_harmonics
from functax.geometry import fibonacci_sphere, gauss_legendre_grid


@pytest.mark.parametrize("B", [1.5, 2.0, 3.0])
def test_window_partition_of_unity_and_support(B):
    b = needlet_window(B)
    xi = jnp.linspace(1.0, 10.0**3, 5001)
    with jax.enable_x64(True):
        xi = jnp.asarray(np.linspace(1.0, 1e3, 5001))
        j_max = int(np.ceil(np.log(1e3) / np.log(B))) + 1
        total = sum(b(xi / B**j) ** 2 for j in range(j_max + 1))
        np.testing.assert_allclose(np.asarray(total), 1.0, atol=1e-12)
        outside = b(jnp.asarray([0.0, 0.99 / B, 1.01 * B, 50.0]))
    assert np.all(np.asarray(outside) == 0.0)


def test_tight_frame_reconstruction():
    B, j_max, degree = 2.0, 3, 8  # B**j_max = 8: every degree <= 8 is covered
    with jax.enable_x64(True):
        rng = np.random.default_rng(0)
        coeffs = jnp.asarray(rng.normal(size=(degree + 1) ** 2))
        quad_points, quad_weights = gauss_legendre_grid(14, 28)  # exact to degree 26
        test_points, _ = fibonacci_sphere(50)

        def field(x):
            return real_spherical_harmonics(x, degree) @ coeffs

        psi_quad, _, _ = needlet_basis(quad_points, B=B, j_max=j_max)
        psi_test, _, _ = needlet_basis(test_points, B=B, j_max=j_max)
        inner = (quad_weights * field(quad_points)) @ psi_quad  # ⟨f, ψ_m⟩
        reconstruction = psi_test @ inner
        expected = field(test_points)
    np.testing.assert_allclose(
        np.asarray(reconstruction), np.asarray(expected), atol=1e-8
    )


def test_localisation():
    with jax.enable_x64(True):
        points, _ = fibonacci_sphere(3000)
        values, levels, centres = needlet_basis(points, j_max=4)
        column = int(np.flatnonzero(np.asarray(levels) == 4)[0])
        centre = np.asarray(centres[column])
        angle = np.arccos(np.clip(np.asarray(points) @ centre, -1.0, 1.0))
        magnitude = np.abs(np.asarray(values[:, column]))
    near = magnitude[angle < 0.05].max()
    for lo, hi in [(0.5, 1.0), (1.0, 2.0), (2.0, np.pi)]:
        band = magnitude[(angle >= lo) & (angle < hi)].max()
        assert band < 0.05 * near  # needlet decay away from its centre


def test_shapes_at_j_max_one():
    values, levels, centres = needlet_basis(jnp.array([[0.0, 0.0, 1.0]]), j_max=1)
    assert (
        values.shape == (1, 69) and levels.shape == (69,) and centres.shape == (69, 3)
    )


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), B=st.floats(1.2, 4.0))
def test_squared_windows_partition_unity_at_random_degrees(seed: int, B: float) -> None:
    # ∑_{j=0}^{J} b²(t/Bʲ) = 1 for 1 ≤ t ≤ B^J, here with J = 6 and t drawn
    # log-uniformly over [1, B⁶]
    exponent = jax.random.uniform(jax.random.key(seed), (64,), maxval=6.0)
    t = B**exponent
    window = needlet_window(B)
    total = sum(window(t / B**j) ** 2 for j in range(7))
    assert_matches_oracle(total, jnp.ones(64), rtol=1e-12)


def _points(count: int, seed: int) -> jax.Array:
    gaussian = jax.random.normal(jax.random.key(seed), (count, 3))
    return gaussian / jnp.linalg.norm(gaussian, axis=-1, keepdims=True)


def test_needlets_match_their_legendre_series() -> None:
    # ψ_jk(r̂) = √w_jk ∑_ℓ b(ℓ/Bʲ) (2ℓ + 1)/(4π) P_ℓ(ξ_jk · r̂), with SciPy's
    # Legendre polynomials and the grid and window the docstring names
    B, j_max = 2.0, 1
    points = _points(5, seed=11)
    # jitted: one compile instead of an eager compile per operation
    values, levels, centres = jax.jit(lambda x: needlet_basis(x, B=B, j_max=j_max))(
        points
    )
    window = needlet_window(B)
    expected = [np.full((5, 1), 1.0 / math.sqrt(4.0 * math.pi))]
    for scale in range(j_max + 1):
        degree = math.floor(B ** (scale + 1))
        nodes, weights = gauss_legendre_grid(degree + 1, 2 * degree + 2)
        ells = np.arange(degree + 1)
        coefficients = np.asarray(window(jnp.asarray(ells / B**scale))) * (
            (2 * ells + 1) / (4 * math.pi)
        )
        cosines = np.clip(np.asarray(points) @ np.asarray(nodes).T, -1.0, 1.0)
        legendre = sp.eval_legendre(ells[:, None, None], cosines[None])
        kernel = np.einsum("l,lnk->nk", coefficients, legendre)
        expected.append(kernel * np.sqrt(np.asarray(weights)))
        assert_matches_oracle(
            centres[np.asarray(levels) == scale], nodes, rtol=0.0, atol=0.0
        )
    assert_matches_oracle(values, np.hstack(expected), rtol=0.0, atol=1e-13)


def test_gradients_match_finite_differences() -> None:
    # needlet_basis differentiated in the points, off every centre; the window
    # in its argument, inside the ramps (1/B, 1) and (1, B)
    points = _points(3, seed=12)
    assert_grad_matches_fd(lambda x: needlet_basis(x, j_max=0)[0], points, rtol=1e-8)
    assert_grad_matches_fd(
        needlet_window(2.0), jnp.array([0.6, 0.8, 1.2, 1.7]), eps=1e-7, rtol=1e-6
    )


def test_needlets_trace() -> None:
    # one scale, as each scale compiles its own operations eagerly
    points = jnp.stack([_points(4, seed=13 + b) for b in range(3)])
    assert_traces(lambda x: needlet_basis(x, j_max=0)[0], points)
