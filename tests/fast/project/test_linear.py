"""Tests for `least_squares` and `galerkin`: recovery of known coefficients of a
`functax.basis` expansion, the ridge closed form, batching and validation.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import pytest

from functax import ValidationError
from functax.project import galerkin, least_squares


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
