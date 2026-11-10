"""Tests for `GaussianPlan` and `BrownianBridge`: linear interpolation's variance. The
fast tier: behaviour, validation and small closed-form checks, each well under a
second.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest
from _harness import (
    assert_matches_oracle,
)

from functax import GeometryError, PolicyError, ValidationError
from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import (
    BrownianBridge,
    GaussianPlan,
    Identity,
    Linear,
    Plan,
    Policies,
    Tensor,
)


RATE = 2.5


class TestPlan:
    axis = Axis(jnp.arange(4.0))
    targets = Points(jnp.array([[-1.0], [0.5], [2.25], [5.0]]))
    data = jnp.array([1.0, 2.0, 4.0, 8.0])

    def test_without_a_rate_the_plan_is_plain(self) -> None:
        plan = Linear().plan(self.axis, self.targets)
        assert type(plan) is Plan
        assert not hasattr(plan, "variance")

    def test_outside_raise_is_supported(self) -> None:
        plan = Linear(prior_variance_rate=RATE).plan(
            self.axis,
            Points(jnp.array([[0.5]])),
            policies=Policies(outside="raise"),
        )
        assert_matches_oracle(plan.variance(), jnp.array([RATE / 4]), rtol=1e-15)


class TestValidation:
    axis = Axis(jnp.arange(4.0))
    targets = Points(jnp.array([[0.5]]))

    @pytest.mark.parametrize("rate", [0.0, -1.0, math.inf, math.nan, True])
    def test_rejects_a_bad_rate(self, rate: float | int) -> None:
        with pytest.raises(ValidationError, match="'prior_variance_rate' must be"):
            Linear(prior_variance_rate=rate)

    def test_an_int_rate_is_accepted(self) -> None:
        plan = Linear(prior_variance_rate=4).plan(self.axis, self.targets)
        assert plan.variance().tolist() == [1.0]

    @pytest.mark.parametrize(
        ("policies", "field"),
        [
            (Policies(outside="clamp"), "outside"),
            (Policies(outside="extrapolate"), "outside"),
            (Policies(extent="cells"), "extent"),
            (Policies(missing="renormalise"), "missing"),
        ],
        ids=["clamp", "extrapolate", "cells", "renormalise"],
    )
    def test_rejects_policies_with_another_mean(
        self, policies: Policies, field: str
    ) -> None:
        with pytest.raises(PolicyError, match=field):
            Linear(prior_variance_rate=RATE).plan(
                self.axis, self.targets, policies=policies
            )

    def test_rejects_a_grid_source(self) -> None:
        grid = RectilinearGrid((self.axis, self.axis))
        with pytest.raises(GeometryError, match="needs a 1-D Axis source"):
            Linear(prior_variance_rate=RATE).plan(grid, Points(jnp.zeros((1, 2))))

    def test_tensor_rejects_a_rate(self) -> None:
        with pytest.raises(ValidationError, match="must not set prior_variance_rate"):
            Tensor((Linear(prior_variance_rate=RATE),))

    def test_bridge_rejects_bad_arrays(self) -> None:
        cell = jnp.zeros(3, dtype=jnp.int32)
        offset = jnp.zeros(3)
        with pytest.raises(ValidationError, match="'offset' must be 1-D"):
            BrownianBridge(jnp.zeros((1, 3), jnp.int32), jnp.zeros((1, 3)), offset)
        with pytest.raises(ValidationError, match="'cell' must have the shape"):
            BrownianBridge(cell[:2], offset, offset)
        with pytest.raises(ValidationError, match="'scale' must have the shape"):
            BrownianBridge(cell, offset, offset[:2])
        with pytest.raises(ValidationError, match="'cell' must have an integer"):
            BrownianBridge(offset, offset, offset)
        with pytest.raises(ValidationError, match="'scale' must have a floating"):
            BrownianBridge(cell, offset, cell)

    def test_gaussian_plan_rejects_a_posterior_of_another_size(self) -> None:
        plan = Linear(prior_variance_rate=RATE).plan(self.axis, self.targets)
        assert isinstance(plan, GaussianPlan)
        bridge = BrownianBridge(jnp.zeros(2, jnp.int32), jnp.zeros(2), jnp.ones(2))
        with pytest.raises(ValidationError, match="'posterior' must have one target"):
            GaussianPlan(Identity(), plan.read, posterior=bridge)
