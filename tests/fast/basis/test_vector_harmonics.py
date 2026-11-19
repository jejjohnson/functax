"""Tests for `vector_spherical_harmonics`, copied from geonnax be5537d with functax
imports. The fast tier: validation.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from functax import ValidationError
from functax.basis.sphere import vector_spherical_harmonics


NORTH = jnp.array([[0.0, 0.0, 1.0]])


def test_rejects_degree_zero() -> None:
    with pytest.raises(ValueError, match="l_max"):
        vector_spherical_harmonics(NORTH, 0)


def test_rejects_an_unknown_output() -> None:
    with pytest.raises(ValidationError, match="'output' must be one of"):
        vector_spherical_harmonics(NORTH, 1, output="polar")
