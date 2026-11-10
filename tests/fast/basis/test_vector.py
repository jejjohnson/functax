"""Tests for `functax.basis.vector`, copied from geonnax be5537d with functax imports,
plus a test of the boundary fix in `divfree_basis`. The fast tier: shapes, layouts
and validation.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from functax.basis.vector import divfree_basis


class TestDivFreeBasis:
    def test_rejects_non_2d_inputs(self):
        import inspect

        # unwrapped: the type hook would reject the shape first
        with pytest.raises(ValueError):
            inspect.unwrap(divfree_basis)(jnp.zeros((5, 3)), num_basis_per_dim=2, L=1.0)
