"""Tests for `real_spherical_harmonics` and `harmonic_degrees`, copied from geonnax
be5537d with functax imports. The fast tier: closed forms and validation, each well
under a second.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from functax.basis.sphere import real_spherical_harmonics


def test_input_validation():
    with pytest.raises(ValueError, match="l_max"):
        real_spherical_harmonics(jnp.zeros((2, 3)), -1)
    # unwrapped: the type hook would reject the shape before the function's check
    import inspect

    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        inspect.unwrap(real_spherical_harmonics)(jnp.zeros((2, 2)), 2)
