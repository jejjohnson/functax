"""Tests for `Kriging` and `KernelCovariance`. The slow tier: the dense GP
posterior oracle, the Monte Carlo covariance of `sample`, gradient and trace
checks, each a second or more.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces

from functax import ValidationError
from functax.geometry import Points
from functax.interp import (
    GaussianPlan,
    KernelCovariance,
    Kriging,
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
    @pytest.mark.parametrize("nugget", [0.0, 0.05], ids=["noiseless", "nugget"])
    def test_equals_the_dense_gp_posterior(self, nugget: float) -> None:
        sources, targets, data = _scattered(seed=0)
        plan = Kriging(_gaussian, drift=None, nugget=nugget).plan(
            Points(sources), Points(targets)
        )
        # the GP posterior of the latent field, with plain dense solves
        gram = _gaussian(sources, sources) + nugget * jnp.eye(15)
        cross = _gaussian(targets, sources)
        mean = cross @ jnp.linalg.solve(gram, data)
        covariance = _gaussian(targets, targets) - cross @ jnp.linalg.solve(
            gram, cross.T
        )
        assert_matches_oracle(plan.mean(data), mean, rtol=1e-10, atol=1e-10)
        assert_matches_oracle(
            plan.variance(), jnp.diag(covariance), rtol=1e-10, atol=1e-10
        )
        assert_matches_oracle(plan.covariance(), covariance, rtol=1e-10, atol=1e-10)

    def test_plan_is_a_gaussian_plan_with_a_kernel_covariance(self) -> None:
        sources, targets, _ = _scattered(seed=1)
        plan = Kriging(_gaussian, drift=None).plan(Points(sources), Points(targets))
        assert isinstance(plan, GaussianPlan)
        assert isinstance(plan.posterior, KernelCovariance)
        assert plan.posterior.n_target == 8


class TestSample:
    @pytest.mark.parametrize("drift", [None, "constant"], ids=["simple", "ordinary"])
    def test_draws_have_the_posterior_covariance(self, drift) -> None:
        sources, targets, data = _scattered(seed=10, n_target=5)
        plan = Kriging(_gaussian, drift=drift, nugget=0.05).plan(
            Points(sources), Points(targets)
        )
        n_draws = 20_000
        fields = jnp.broadcast_to(data, (n_draws, 15))
        draws = plan.sample(fields, key=jax.random.key(1))
        # deviations from the known mean, so Ĉ = εᵀε / S is unbiased
        deviations = draws - plan.mean(data)
        empirical = deviations.T @ deviations / n_draws
        covariance = plan.covariance()
        variance = jnp.diag(covariance)
        # Var(εᵢ εⱼ) = Σᵢⱼ² + Σᵢᵢ Σⱼⱼ for a Gaussian, so Ĉᵢⱼ has standard error
        # √((Σᵢⱼ² + Σᵢᵢ Σⱼⱼ) / S); six of them bound all 15 entries with a
        # family-wise false-alarm rate below 15 · 2e-9
        standard_error = jnp.sqrt(
            (covariance**2 + jnp.outer(variance, variance)) / n_draws
        )
        assert bool(jnp.all(jnp.abs(empirical - covariance) <= 6.0 * standard_error))

    def test_draws_at_a_noiseless_source_equal_the_sample(self) -> None:
        sources, targets, data = _scattered(seed=11, n_target=4)
        targets = targets.at[0].set(sources[3])
        plan = Kriging(_gaussian).plan(Points(sources), Points(targets))
        draws = plan.sample(jnp.broadcast_to(data, (50, 15)), key=jax.random.key(2))
        # Σ* is singular there; √λ₊ of a rounding-size eigenvalue is ~1e-8
        assert_matches_oracle(draws[:, 0], jnp.full(50, data[3]), rtol=0.0, atol=1e-6)


class TestTransforms:
    def test_mean_traces_under_jit_and_vmap(self) -> None:
        sources, targets, data = _scattered(seed=12)
        plan = Kriging(_gaussian, nugget=0.01).plan(Points(sources), Points(targets))
        batch = jnp.stack([data, -data, 2.0 * data])
        assert_traces(lambda values: plan(values), batch)

    def test_mean_gradient_in_the_data_matches_finite_differences(self) -> None:
        sources, targets, data = _scattered(seed=13, n_target=4)
        plan = Kriging(_gaussian, drift=1, nugget=0.01).plan(
            Points(sources), Points(targets)
        )
        assert_grad_matches_fd(plan, data)


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
