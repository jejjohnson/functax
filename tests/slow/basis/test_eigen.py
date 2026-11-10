"""Tests for the EOFs of `functax.basis.eigen`, copied from geonnax be5537d with
functax imports. The slow tier: a second or more each (mostly compiling the SVD).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from functax.basis.eigen import eof_basis


class TestEOFBasis:
    def test_shapes_and_orthonormal_columns(self):
        X = jax.random.normal(jax.random.PRNGKey(0), (20, 8))
        phi, s = eof_basis(X, n_modes=5)
        assert phi.shape == (8, 5)
        assert s.shape == (5,)
        np.testing.assert_allclose(phi.T @ phi, jnp.eye(5), atol=1e-5)

    def test_singular_values_descending(self):
        X = jax.random.normal(jax.random.PRNGKey(1), (30, 10))
        _, s = eof_basis(X, n_modes=10)
        assert bool(jnp.all(jnp.diff(s) <= 1e-6))
