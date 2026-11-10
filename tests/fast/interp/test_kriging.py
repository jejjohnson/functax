"""Tests for `Kriging` and `KernelCovariance`. The fast tier: the weights of
ordinary kriging, zero variance at the sources, the linear trend of universal
kriging, padding, validation and configs.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from functax import ValidationError
from functax.geometry import Points
from functax.interp import (
    Kriging,
    Scheme,
    Wendland,
)


def _gaussian(targets: jax.Array, sources: jax.Array) -> jax.Array:
    """A squared-exponential covariance, σ² = 2 and ℓ² = 0.08."""
    difference = targets[:, None, :] - sources[None, :, :]
    return 2.0 * jnp.exp(-jnp.sum(difference**2, axis=-1) / 0.16)


def _scattered(seed: int, n_source: int = 15, n_target: int = 8):
    """Random sources and targets in the unit square, and a field on the sources."""
    rng = np.random.default_rng(seed)
    sources = rng.uniform(size=(n_source, 2))
    targets = rng.uniform(size=(n_target, 2))
    data = np.cos(2.0 * sources[:, 0]) - sources[:, 1]
    return jnp.asarray(sources), jnp.asarray(targets), jnp.asarray(data)


class TestValidationAndConfig:
    @pytest.mark.parametrize(
        ("arguments", "field"),
        [
            ({"drift": "linear"}, "'drift'"),
            ({"drift": -1}, "'drift'"),
            ({"drift": True}, "'drift'"),
            ({"nugget": -0.1}, "'nugget'"),
        ],
    )
    def test_rejects_bad_fields(self, arguments: dict, field: str) -> None:
        with pytest.raises(ValidationError, match=field):
            Kriging(Wendland(2), **arguments)

    def test_rejects_a_drift_of_the_wrong_shape(self) -> None:
        sources, targets, _ = _scattered(seed=8)
        with pytest.raises(ValidationError, match="'drift'"):
            Kriging(_gaussian, drift=lambda x: jnp.ones(x.shape[0])).plan(
                Points(sources), Points(targets)
            )

    @pytest.mark.parametrize(
        "scheme",
        [
            Kriging(Wendland(4, support=0.5)),
            Kriging(Wendland(2), drift=None, nugget=0.1),
            Kriging(Wendland(2), drift=2),
        ],
        ids=["ordinary", "simple", "universal"],
    )
    def test_config_round_trips_through_strict_json(self, scheme: Kriging) -> None:
        text = json.dumps(scheme.get_config(), allow_nan=False)
        assert Scheme.from_config(json.loads(text)) == scheme
