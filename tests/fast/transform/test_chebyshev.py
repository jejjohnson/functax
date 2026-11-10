"""Tests for `ChebyshevTransform1D` and `ChebyshevTransform2D`, copied from
spectraldiffx d68f8f6 (``test_chebyshev_new_features.py``, ``test_chebyshev_grid.py``
and ``test_chebyshev_extensions.py``) with functax imports, plus the testing list
of #46. The fast tier: closed forms, unit vectors and validation.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import pytest

from functax import ValidationError
from functax.transform import ChebyshevTransform1D, ChebyshevTransform2D


def test_shapes_and_sizes() -> None:
    assert ChebyshevTransform1D(4).n_nodes == 5
    assert ChebyshevTransform1D(4, kind="gauss").n_nodes == 4
    assert ChebyshevTransform2D(4, 2).shape == (3, 5)
    assert ChebyshevTransform2D(4, 2, kind="gauss").shape == (2, 4)


@pytest.mark.parametrize("n", [0, -2, True])
def test_degrees_must_be_positive_integers(n: object) -> None:
    with pytest.raises(ValidationError, match="'n' must be an integer of at least 1"):
        ChebyshevTransform1D(n)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValidationError, match="'ny' must be an integer"):
        ChebyshevTransform2D(3, n)  # ty: ignore[invalid-argument-type]


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValidationError, match="'kind' must be one of"):
        ChebyshevTransform1D(4, kind="gauss-lobatto")


def test_values_must_fit_the_nodes() -> None:
    transform = ChebyshevTransform1D(4)
    to_spectral = inspect.unwrap(ChebyshevTransform1D.to_spectral)
    from_spectral = inspect.unwrap(ChebyshevTransform1D.from_spectral)
    with pytest.raises(
        ValidationError, match=r"'u' must have trailing axes of shape \(5,\)"
    ):
        to_spectral(transform, jnp.ones(4))
    with pytest.raises(ValidationError, match="'a' must have trailing axes"):
        from_spectral(transform, jnp.ones(()))
    with pytest.raises(ValidationError, match="'u' must be real"):
        to_spectral(transform, jnp.ones(5, dtype=jnp.complex128))
    grid = ChebyshevTransform2D(2, 3)
    with pytest.raises(
        ValidationError, match=r"'u' must have trailing axes of shape \(4, 3\)"
    ):
        inspect.unwrap(ChebyshevTransform2D.to_spectral)(grid, jnp.ones((3, 4)))
