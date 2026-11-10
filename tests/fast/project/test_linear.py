"""Tests for `least_squares` and `galerkin`: recovery of known coefficients of a
`functax.basis` expansion, the ridge closed form, batching and validation.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax import ValidationError
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


@pytest.mark.parametrize(
    ("design", "y", "ridge", "match"),
    [
        (jnp.ones(4), jnp.ones(4), 0.0, "design_matrix"),
        (jnp.ones((4, 2), dtype=jnp.int32), jnp.ones(4), 0.0, "design_matrix"),
        (jnp.ones((4, 2)), jnp.ones(3), 0.0, "'y'"),
        (jnp.ones((4, 2)), jnp.ones(4), -1.0, "ridge"),
        (jnp.ones((4, 2)), jnp.ones(4), float("nan"), "ridge"),
        (jnp.ones((4, 2)), jnp.ones(4), jnp.array(1.0), "ridge"),
    ],
)
def test_least_squares_validates(design, y, ridge, match) -> None:
    with pytest.raises(ValidationError, match=match):
        inspect.unwrap(least_squares)(design, y, ridge=ridge)


@pytest.mark.parametrize(
    ("weights", "match"),
    [(jnp.ones(3), "weights"), (jnp.array([1.0, 0.0, 1.0, 1.0]), "positive")],
)
def test_galerkin_validates_weights(weights, match) -> None:
    with pytest.raises(ValidationError, match=match):
        inspect.unwrap(galerkin)(jnp.ones((4, 2)), jnp.ones(4), weights=weights)
