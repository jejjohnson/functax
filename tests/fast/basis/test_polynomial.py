"""Tests for `functax.basis.polynomial`, copied from geonnax be5537d (and, for
`legendre_series`, kernellib c2d1fce) with functax imports. The fast tier: the index
layout and validation.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import pytest

from functax.basis.polynomial import (
    associated_legendre,
    associated_legendre_indices,
    legendre_polynomials,
    legendre_series,
)
from functax.exceptions import ValidationError


def test_indices_order_and_count():
    idx = associated_legendre_indices(3)
    assert len(idx) == 10
    assert idx[:4] == ((0, 0), (1, 0), (1, 1), (2, 0))


def test_invalid_normalization_raises():
    with pytest.raises(ValueError, match="normalization"):
        associated_legendre(jnp.zeros(3), 2, normalization="bad")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("l_max", [True, 2.0])
def test_degrees_must_be_integers(l_max: object) -> None:
    with pytest.raises(ValidationError, match="'l_max' must be an integer"):
        inspect.unwrap(legendre_polynomials)(jnp.zeros(3), l_max)


def test_legendre_polynomials_rejects_a_negative_degree() -> None:
    with pytest.raises(ValueError, match="l_max"):
        legendre_polynomials(jnp.zeros((2, 3)), -1)


@pytest.mark.parametrize("shape", [(2, 2), (0,)])
def test_legendre_series_coefficients_must_be_one_non_empty_vector(
    shape: tuple[int, ...],
) -> None:
    # unwrapped, so the module's own check runs rather than the type hook
    with pytest.raises(ValidationError, match="'coeffs' must be 1-D and non-empty"):
        inspect.unwrap(legendre_series)(jnp.zeros(3), jnp.ones(shape))
