"""Tests for `functax.basis.fourier`, copied from geonnax be5537d with functax imports.
The fast tier: validation and the empty eigenvalue list.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from functax.basis.fourier import (
    fourier_basis,
    fourier_basis_1d,
    fourier_eigenvalues,
    fourier_eigenvalues_1d,
)


L = 1.5


def test_validation_names_the_argument() -> None:
    from functax import ValidationError

    with pytest.raises(ValidationError, match="'num_basis' must be at least 1"):
        fourier_basis_1d(jnp.zeros(3), 0, 1.0)
    with pytest.raises(ValidationError, match="'L' must be positive"):
        fourier_eigenvalues_1d(3, -1.0)
    with pytest.raises(ValidationError, match="'num_basis_per_dim' must have length 2"):
        fourier_basis(jnp.zeros((3, 2)), (2, 3, 4), 1.0)
    with pytest.raises(ValidationError, match="'boundary' must be one of"):
        fourier_basis_1d(jnp.zeros(3), 2, 1.0, boundary="robin")


def test_fourier_basis_rejects_an_unknown_boundary() -> None:
    with pytest.raises(ValueError, match="boundary"):
        fourier_basis(jnp.zeros((1, 2)), 3, L, boundary="robin")  # ty: ignore[invalid-argument-type]


def test_no_axes_gives_no_eigenvalues() -> None:
    assert fourier_eigenvalues(2, 1.0, 0).shape == (0,)
