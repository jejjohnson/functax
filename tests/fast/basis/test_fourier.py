"""Tests for `functax.basis.fourier`, copied from geonnax be5537d with functax imports.
The fast tier: shapes, layouts and validation.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from functax.basis.fourier import (
    fourier_basis,
    fourier_basis_1d,
    fourier_eigenvalues,
    fourier_eigenvalues_1d,
)
from functax.quad import gauss_legendre


BOUNDARIES = ["dirichlet", "neumann", "periodic"]


L = 1.5


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_orthonormal_on_interval(boundary):
    nodes, weights = gauss_legendre(80)
    with jax.enable_x64(True):
        phi = np.asarray(
            fourier_basis_1d(jnp.asarray(L * nodes), 9, L, boundary=boundary)
        )
    gram = (phi * (L * weights)[:, None]).T @ phi
    np.testing.assert_allclose(gram, np.eye(9), atol=1e-12)


def test_dirichlet_default_unchanged():
    x = jnp.linspace(-L, L, 7)
    j = np.arange(1, 5)
    expected = np.sin(np.outer(np.asarray(x) + L, j) * math.pi / (2 * L)) / math.sqrt(L)
    np.testing.assert_allclose(
        np.asarray(fourier_basis_1d(x, 4, L)), expected, atol=1e-6
    )
    np.testing.assert_array_equal(
        np.asarray(fourier_basis_1d(x, 4, L)),
        np.asarray(fourier_basis_1d(x, 4, L, boundary="dirichlet")),
    )


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


def test_no_axes_gives_no_eigenvalues() -> None:
    assert fourier_eigenvalues(2, 1.0, 0).shape == (0,)
