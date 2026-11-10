"""Slow tests for `least_squares` and `galerkin`: coefficient recovery on
`functax.basis` expansions and gradients through the solves.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle

from functax.basis.fourier import fourier_basis
from functax.basis.polynomial import legendre_polynomials
from functax.project import galerkin, least_squares
from functax.quad import gauss_legendre


def _box_expansion() -> tuple[jax.Array, jax.Array]:
    """Φ of 16 Neumann box eigenfunctions at 80 random points, and known c."""
    rng = np.random.default_rng(0)
    points = jnp.asarray(rng.uniform(-1.0, 1.0, (80, 2)))
    design, _ = fourier_basis(points, 4, 1.0, boundary="neumann")
    coefficients = jnp.asarray(rng.normal(size=design.shape[1]))
    return design, coefficients


def _box_design() -> tuple[jnp.ndarray, jnp.ndarray]:
    rng = np.random.default_rng(0)
    points = jnp.asarray(rng.uniform(-1.0, 1.0, (40, 2)))
    design, _ = fourier_basis(points, 3, 1.0, boundary="neumann")
    return design, design @ jnp.asarray(rng.normal(size=design.shape[1]))


def test_least_squares_recovers_the_coefficients_of_a_basis_expansion() -> None:
    # DoD of #42: known coefficients of a functax.basis expansion to 1e-10
    rng = np.random.default_rng(0)
    points = jnp.asarray(rng.uniform(-1.0, 1.0, (80, 2)))
    design, _ = fourier_basis(points, 4, 1.0, boundary="neumann")
    coefficients = jnp.asarray(rng.normal(size=design.shape[1]))
    recovered = least_squares(design, design @ coefficients)
    assert_matches_oracle(recovered, coefficients, rtol=0, atol=1e-10)


def test_galerkin_recovers_legendre_coefficients_exactly() -> None:
    # Gauss–Legendre with 12 nodes integrates degree 23 exactly, so ΦᵀWΦ and
    # ΦᵀWy are exact for a degree-6 expansion
    nodes, weights = (jnp.asarray(a) for a in gauss_legendre(12))
    design = legendre_polynomials(nodes, 6)
    coefficients = jnp.asarray(np.random.default_rng(1).normal(size=7))
    y = design @ coefficients
    recovered = galerkin(design, y, weights=weights)
    assert_matches_oracle(recovered, coefficients, rtol=0, atol=1e-12)


@pytest.mark.parametrize("ridge", [0.0, 0.5])
def test_least_squares_gradient_in_y_matches_finite_differences(ridge: float) -> None:
    design, y = _box_design()
    assert_grad_matches_fd(lambda data: least_squares(design, data, ridge=ridge), y)


def test_galerkin_gradient_in_y_and_weights_matches_finite_differences() -> None:
    design, y = _box_design()
    # off the span, so the projection depends on the weights
    y = y + jnp.cos(3.0 * jnp.arange(y.shape[0]))
    weights = jnp.linspace(0.5, 1.5, design.shape[0])
    assert_grad_matches_fd(lambda data: galerkin(design, data, weights=weights), y)
    assert_grad_matches_fd(lambda w: galerkin(design, y, weights=w), weights)


def test_ridge_least_squares_equals_its_normal_equations() -> None:
    design, coefficients = _box_expansion()
    y = design @ coefficients + 0.1 * jnp.sin(jnp.arange(design.shape[0]))
    ridge = 0.3
    phi = np.asarray(design)
    expected = np.linalg.solve(
        phi.T @ phi + ridge * np.eye(phi.shape[1]), phi.T @ np.asarray(y)
    )
    assert_matches_oracle(least_squares(design, y, ridge=ridge), expected, rtol=1e-12)


@pytest.mark.parametrize("ridge", [0.0, 1e-3])
def test_least_squares_batches_over_leading_dims(ridge: float) -> None:
    design, coefficients = _box_expansion()
    stacked = jnp.stack([coefficients, 2.0 * coefficients, -coefficients])
    y = jnp.stack([design @ c for c in stacked])
    batched = least_squares(design, y, ridge=ridge)
    looped = jnp.stack([least_squares(design, row, ridge=ridge) for row in y])
    assert batched.shape == stacked.shape
    assert_matches_oracle(batched, looped, rtol=1e-12, atol=1e-14)


def test_galerkin_projects_onto_the_span() -> None:
    # the L² projection of x³ onto {P₀, P₁} on [−1, 1] is (3/5) x
    nodes, weights = (jnp.asarray(a) for a in gauss_legendre(8))
    design = legendre_polynomials(nodes, 1)
    recovered = galerkin(design, nodes**3, weights=weights)
    assert_matches_oracle(recovered, np.array([0.0, 0.6]), rtol=0, atol=1e-14)


def test_least_squares_keeps_float32() -> None:
    design, coefficients = _box_expansion()
    y = design @ coefficients
    result = least_squares(design.astype(jnp.float32), y.astype(jnp.float32))
    assert result.dtype == jnp.float32
