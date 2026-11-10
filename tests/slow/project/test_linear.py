"""Slow tests for `least_squares` and `galerkin`: coefficient recovery on
`functax.basis` expansions and gradients through the solves.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle

from functax.basis.fourier import fourier_basis
from functax.basis.polynomial import legendre_polynomials
from functax.project import galerkin, least_squares
from functax.quad import gauss_legendre


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
