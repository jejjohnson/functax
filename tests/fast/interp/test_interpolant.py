"""Tests for `Interpolant` and `LocalScheme.fit`: moving targets inside the trace. The
fast tier: behaviour, validation and small closed-form checks, each well under a
second.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax import PolicyError, ValidationError
from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import (
    BSpline,
    Linear,
    Policies,
)


GRID = RectilinearGrid(
    (Axis(jnp.linspace(0.0, 5.0, 8)), Axis(jnp.linspace(-1.0, 2.0, 9)))
)


FIELD = jax.random.normal(jax.random.key(0), (72,))


def _targets(count: int, seed: int) -> jax.Array:
    rng = np.random.default_rng(seed)
    return jnp.asarray(
        np.column_stack([rng.uniform(-0.5, 5.5, count), rng.uniform(-1.3, 2.3, count)])
    )


class TestDerivativesAndPolicies:
    def test_renormalise(self) -> None:
        data = FIELD.at[10].set(jnp.nan)
        policies = Policies(missing="renormalise")
        targets = _targets(12, seed=10)
        field = Linear().fit(GRID, data, policies=policies)
        expected = Linear().plan(GRID, Points(targets), policies=policies)(data)
        assert_matches_oracle(field(targets), expected, rtol=1e-14, atol=1e-15)
        with pytest.raises(PolicyError, match="needs a local plan"):
            BSpline(3).fit(GRID, data, policies=policies)

    def test_rejects_data_of_the_wrong_size(self) -> None:
        with pytest.raises(
            ValidationError, match="'data' must have a last axis of length 72"
        ):
            Linear().fit(GRID, jnp.ones(70))
