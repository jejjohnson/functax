"""Tests for the needlets of `functax.basis.sphere`, copied from geonnax be5537d with
functax imports. The fast tier: validation.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from functax.basis.sphere import needlet_basis, needlet_window


def test_window_rejects_a_dilation_of_one() -> None:
    with pytest.raises(ValueError, match="'B' must"):
        needlet_window(1.0)


def test_basis_rejects_a_negative_j_max() -> None:
    with pytest.raises(ValueError, match="j_max"):
        needlet_basis(jnp.zeros((1, 3)), j_max=-1)
