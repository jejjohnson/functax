"""Tests for `Kriging` and `KernelCovariance`. The fast tier: validation of the
fields, the drift and the posterior's inputs, and configs.
"""

from __future__ import annotations

import inspect
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from functax import ValidationError
from functax.geometry import Points
from functax.interp import (
    KernelCovariance,
    KernelSolve,
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

    def test_covariance_checks_its_inputs(self) -> None:
        # the check reads only the fit's sizes: 3 sources and a one-term drift,
        # so 4 coefficients per read-out row, with no factorisation to build
        fit = KernelSolve.__new__(KernelSolve)
        object.__setattr__(fit, "n_source", 3)
        object.__setattr__(fit, "n_terms", 1)
        check = inspect.unwrap(KernelCovariance.__check_init__)
        wrong = KernelCovariance.__new__(KernelCovariance)
        object.__setattr__(wrong, "fit", fit)
        object.__setattr__(wrong, "read", jnp.zeros((2, 4)))
        object.__setattr__(wrong, "targets", jnp.zeros((1, 2)))
        with pytest.raises(ValidationError, match="'targets'"):
            check(wrong)
        object.__setattr__(wrong, "targets", jnp.zeros((2, 2)))
        object.__setattr__(wrong, "read", jnp.zeros((2, 3)))
        with pytest.raises(ValidationError, match="'read'"):
            check(wrong)

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
