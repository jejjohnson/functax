"""Tests for `shell_basis`, copied from geonnax be5537d with functax imports. The fast
tier: validation.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from functax import ValidationError
from functax.basis.sphere import shell_basis


R0, R1 = 1.0, 1.8


POINTS = jnp.array([[0.0, 0.0, R0], [0.0, 0.0, R1]])


def test_rejects_an_inner_radius_above_the_outer() -> None:
    with pytest.raises(ValueError, match="r_inner"):
        shell_basis(POINTS, 1, 2, r_inner=2.0, r_outer=1.0)


def test_rejects_zero_radial_modes() -> None:
    with pytest.raises(ValueError, match="n_radial"):
        shell_basis(POINTS, 1, 0, r_inner=R0, r_outer=R1)


def test_rejects_an_unknown_radial_family() -> None:
    with pytest.raises(ValidationError, match="'radial' must be one of"):
        shell_basis(POINTS, 1, 2, r_inner=R0, r_outer=R1, radial="bessel")
