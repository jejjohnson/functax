"""Tests for `functax.basis.slepian`, the region tests copied from geonnax be5537d with
functax imports and the cap tests written for functax. The fast tier: the polygon
test and validation.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from functax.basis.slepian import (
    slepian_polar_gap_basis,
    slepian_region_basis,
    spherical_polygon_mask,
)


def test_validation():
    with pytest.raises(ValueError, match="gap_colatitude"):
        slepian_polar_gap_basis(3, 2.0)
    with pytest.raises(ValueError, match="polygon"):
        spherical_polygon_mask(jnp.zeros((1, 3)), jnp.zeros((2, 2)), unit="degrees")
    # unwrapped: the type hook would reject the mismatched lengths first
    import inspect

    with pytest.raises(ValueError, match="region_weights"):
        inspect.unwrap(slepian_region_basis)(2, jnp.zeros((3, 3)), jnp.zeros(2))
