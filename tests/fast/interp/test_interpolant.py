"""Tests for `Interpolant` and `LocalScheme.fit`: moving targets inside the trace. The
fast tier: behaviour, validation and small closed-form checks, each well under a
second.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from functax import PolicyError, ValidationError
from functax.geometry import Axis, RectilinearGrid
from functax.interp import (
    BSpline,
    Linear,
    Policies,
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

    def test_a_derivative_of_a_derivative_raises_its_order_instead(self) -> None:
        slope = Linear().fit(Axis(jnp.arange(3.0)), jnp.arange(3.0)).derivative(axis=0)
        with pytest.raises(ValidationError, match="raise 'order' instead"):
            slope.derivative(axis=0)

    def test_renormalise_needs_a_local_plan(self) -> None:
        with pytest.raises(PolicyError, match="needs a local plan"):
            BSpline(3).fit(
                Axis(jnp.arange(4.0)),
                jnp.ones(4),
                policies=Policies(missing="renormalise"),
            )
