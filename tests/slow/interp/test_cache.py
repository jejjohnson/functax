"""Tests for caching plans: `plan_key`, `Plan.serialize` and `Plan.deserialize`. The
slow tier: oracle sweeps, dot-product tests across policies, gradient and trace
checks, each a second or more (mostly einx compiling new shapes).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from functax import ValidationError
from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import (
    Linear,
    Plan,
)


GRID = RectilinearGrid((Axis(jnp.arange(3.0)), Axis(jnp.linspace(0.0, 1.0, 5))))


class TestSerialize:
    targets = Points(jnp.array([[0.5, 0.25], [1.5, 0.8], [2.4, -0.3], [9.0, 0.5]]))
    field = jnp.array([1.0, 2.0, jnp.nan, 4.0, 8.0] * 3)

    def test_rejects_a_template_of_another_shape(self) -> None:
        data = Linear().plan(GRID, self.targets).serialize()
        other = Linear().plan(GRID, Points(jnp.zeros((2, 2))))
        with pytest.raises(ValidationError, match="'data' does not match 'like'"):
            Plan.deserialize(data, like=other)
