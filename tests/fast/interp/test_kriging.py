"""Tests for `Kriging` and `KernelCovariance`. The fast tier: the weights of
ordinary kriging, zero variance at the sources, the linear trend of universal
kriging, padding, validation and configs.
"""

from __future__ import annotations

import inspect
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax import ValidationError
from functax.geometry import Points
from functax.interp import (
    GaussianPlan,
    KernelCovariance,
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


class TestSimpleKriging:
    def test_plan_is_a_gaussian_plan_with_a_kernel_covariance(self) -> None:
        sources, targets, _ = _scattered(seed=1)
        plan = Kriging(_gaussian, drift=None).plan(Points(sources), Points(targets))
        assert isinstance(plan, GaussianPlan)
        assert isinstance(plan.posterior, KernelCovariance)
        assert plan.posterior.n_target == 8


class TestOrdinaryAndUniversal:
    def test_ordinary_kriging_weights_sum_to_one(self) -> None:
        sources, targets, _ = _scattered(seed=2)
        plan = Kriging(_gaussian, nugget=0.01).plan(Points(sources), Points(targets))
        # row k of the plan's matrix holds the kriging weights λₖ of target k
        weights = plan.operator().as_matrix()
        assert_matches_oracle(jnp.sum(weights, axis=1), jnp.ones(8), rtol=1e-12)

    def test_ordinary_variance_exceeds_simple_variance(self) -> None:
        # estimating the mean adds uᵀ (Fᵀ K⁻¹ F)⁻¹ u ≥ 0 to the variance
        sources, targets, _ = _scattered(seed=3)
        simple = Kriging(_gaussian, drift=None).plan(Points(sources), Points(targets))
        ordinary = Kriging(_gaussian).plan(Points(sources), Points(targets))
        assert bool(jnp.all(ordinary.variance() >= simple.variance() - 1e-12))

    @pytest.mark.parametrize(
        "drift", ["constant", None, 1], ids=["ordinary", "simple", "universal"]
    )
    def test_variance_is_zero_at_noiseless_sources(self, drift) -> None:
        sources, _, data = _scattered(seed=4)
        points = Points(sources)
        plan = Kriging(_gaussian, drift=drift).plan(points, points)
        assert_matches_oracle(plan.variance(), jnp.zeros(15), rtol=0.0, atol=1e-10)
        assert_matches_oracle(plan.mean(data), data, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize(
        "drift",
        [1, lambda x: jnp.column_stack([jnp.ones(x.shape[0]), x])],
        ids=["degree", "callable"],
    )
    @pytest.mark.parametrize("nugget", [0.0, 0.1], ids=["noiseless", "nugget"])
    def test_universal_kriging_reproduces_a_linear_trend(
        self, drift, nugget: float
    ) -> None:
        sources, targets, _ = _scattered(seed=5)
        trend = 3.0 - 1.5 * sources[:, 0] + 0.25 * sources[:, 1]
        plan = Kriging(_gaussian, drift=drift, nugget=nugget).plan(
            Points(sources), Points(targets)
        )
        expected = 3.0 - 1.5 * targets[:, 0] + 0.25 * targets[:, 1]
        assert_matches_oracle(plan(trend), expected, rtol=1e-10, atol=1e-10)

    def test_padding_targets_have_nan_variance(self) -> None:
        sources, targets, data = _scattered(seed=6)
        valid = jnp.arange(8) != 2
        padded = jnp.where(valid[:, None], targets, jnp.nan)
        plan = Kriging(_gaussian).plan(Points(sources), Points(padded, valid=valid))
        assert bool(jnp.all(jnp.isnan(plan.variance()) == ~valid))
        assert bool(jnp.all(jnp.isnan(plan(data)) == ~valid))
        draw = plan.sample(data, key=jax.random.key(0))
        assert bool(jnp.all(jnp.isfinite(draw[valid])))


class TestValidationAndConfig:
    def test_a_negative_variance_beyond_rounding_raises(self) -> None:
        # inconsistent on purpose: the 1 × 1 prior variance is half of what
        # the matrices on the sources say, so σ² = −k** at a source
        def inconsistent(targets: jax.Array, sources: jax.Array) -> jax.Array:
            scale = 1.0 if targets.shape[0] == 1 else 2.0
            return scale * _gaussian(targets, sources)

        sources, _, _ = _scattered(seed=7)
        plan = Kriging(inconsistent, drift=None).plan(Points(sources), Points(sources))
        with pytest.raises(ValidationError, match="non-negative"):
            plan.variance()

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
        sources, targets, _ = _scattered(seed=9)
        plan = Kriging(_gaussian).plan(Points(sources), Points(targets))
        posterior = plan.posterior
        assert isinstance(posterior, KernelCovariance)
        check = inspect.unwrap(KernelCovariance.__check_init__)
        wrong_targets = KernelCovariance.__new__(KernelCovariance)
        object.__setattr__(wrong_targets, "fit", posterior.fit)
        object.__setattr__(wrong_targets, "read", posterior.read)
        object.__setattr__(wrong_targets, "targets", targets[:3])
        with pytest.raises(ValidationError, match="'targets'"):
            check(wrong_targets)
        object.__setattr__(wrong_targets, "targets", targets)
        object.__setattr__(wrong_targets, "read", posterior.read[:, :4])
        with pytest.raises(ValidationError, match="'read'"):
            check(wrong_targets)

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
