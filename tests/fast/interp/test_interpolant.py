"""Tests for `Interpolant` and `LocalScheme.fit`: moving targets inside the trace. The
fast tier: behaviour, validation and small closed-form checks, each well under a
second.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from functax import ValidationError
from functax.geometry import Axis, RectilinearGrid
from functax.interp import (
    Linear,
)


GRID = RectilinearGrid(
    (Axis(jnp.linspace(0.0, 5.0, 8)), Axis(jnp.linspace(-1.0, 2.0, 9)))
)


class TestDerivativesAndPolicies:
    def test_rejects_data_of_the_wrong_size(self) -> None:
        with pytest.raises(
            ValidationError, match="'data' must have a last axis of length 72"
        ):
            Linear().fit(GRID, jnp.ones(70))
