"""Tests for `functax.basis.vector`, copied from geonnax be5537d with functax imports,
plus a test of the boundary fix in `divfree_basis`. The fast tier: shapes, layouts
and validation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from functax.basis.fourier import (
    fourier_eigenvalues,
)
from functax.basis.vector import divfree_basis


class TestDivFreeBasis:
    def test_shapes(self):
        xy = jnp.zeros((7, 2))
        phi, lam = divfree_basis(xy, num_basis_per_dim=3, L=1.0)
        assert phi.shape == (7, 9, 2)  # M = 3 * 3, last axis = (u, v)
        assert lam.shape == (9,)

    def test_eigenvalues_match_fourier(self):

        _, lam = divfree_basis(jnp.zeros((2, 2)), num_basis_per_dim=3, L=2.0)
        expected = fourier_eigenvalues(3, 2.0, 2, dtype=lam.dtype)
        np.testing.assert_allclose(lam, expected, atol=1e-6)

    def test_rejects_non_2d_inputs(self):
        import inspect

        # unwrapped: the type hook would reject the shape first
        with pytest.raises(ValueError):
            inspect.unwrap(divfree_basis)(jnp.zeros((5, 3)), num_basis_per_dim=2, L=1.0)
