"""Tests for `GaussianPlan` and `BrownianBridge`: linear interpolation's variance. The
slow tier: oracle sweeps, dot-product tests across policies, gradient and trace
checks, each a second or more (mostly einx compiling new shapes).
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from _harness import (
    assert_adjoint,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)

from functax.geometry import Axis, Points
from functax.interp import (
    GaussianPlan,
    Linear,
    Policies,
)


RATE = 2.5


def _knots(count: int, seed: int) -> np.ndarray:
    """Increasing knots with random, uneven spacing."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.uniform(0.3, 1.7, size=count))


def _points(coordinates: np.ndarray) -> Points:
    """1-D targets as (M, 1) points."""
    return Points(einx.id("target -> target 1", jnp.asarray(coordinates)))


def _targets(knots: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Random interior targets, every knot, and one repeated target."""
    rng = np.random.default_rng(seed)
    interior = rng.uniform(knots[0], knots[-1], size=count)
    return np.concatenate([interior, knots, interior[:1]])


def _brownian_gp_posterior(
    knots: np.ndarray, targets: np.ndarray, data: np.ndarray, rate: float
) -> tuple[jax.Array, jax.Array]:
    """The dense GP posterior of Brownian motion conditioned on exact knots.

    k(x, x′) = σ² min(x − x₀, x′ − x₀), with the origin x₀ one unit left of the
    first knot so that K_XX is positive definite; the posterior is
    m = K*X K_XX⁻¹ y and Σ* = K** − K*X K_XX⁻¹ KX*, solved with lineax.
    """
    origin = knots[0] - 1.0
    knot_time = jnp.asarray(knots - origin)
    target_time = jnp.asarray(targets - origin)

    def kernel(row: jax.Array, col: jax.Array) -> jax.Array:
        # σ² min(s, s′)
        return rate * einx.minimum("row, col -> row col", row, col)

    knot_gram = kernel(knot_time, knot_time)
    cross = kernel(target_time, knot_time)
    target_gram = kernel(target_time, target_time)
    operator = lx.MatrixLinearOperator(knot_gram, lx.positive_semidefinite_tag)

    def solve(rhs: jax.Array) -> jax.Array:
        return lx.linear_solve(operator, rhs, solver=lx.Cholesky()).value

    # K_XX⁻¹ y and K_XX⁻¹ KX*, one solve per column
    weights = solve(jnp.asarray(data))
    cross_t = einx.id("target knot -> knot target", cross)
    gain = jax.vmap(solve, in_axes=1, out_axes=1)(cross_t)
    mean = einx.dot("target knot, knot -> target", cross, weights)
    reduction = einx.dot("target knot, knot col -> target col", cross, gain)
    return mean, target_gram - reduction


class TestOracle:
    @pytest.mark.parametrize(
        "descending", [False, True], ids=["ascending", "descending"]
    )
    def test_equals_the_dense_gp_posterior(self, descending: bool) -> None:
        knots = _knots(9, seed=0)
        targets = _targets(knots, 40, seed=1)
        data = np.random.default_rng(2).normal(size=knots.size)
        gp_mean, gp_covariance = _brownian_gp_posterior(knots, targets, data, RATE)
        axis_values = knots[::-1] if descending else knots
        axis_data = data[::-1] if descending else data
        plan = Linear(prior_variance_rate=RATE).plan(
            Axis(jnp.asarray(axis_values)), _points(targets)
        )
        # the GP's subtraction K** − K*X K⁻¹ KX* rounds at ε σ² (x − x₀) ≈ 1e-14,
        # so the zeros at the knots need an absolute floor
        atol = 1e-12
        assert_matches_oracle(
            plan.mean(jnp.asarray(axis_data)), gp_mean, rtol=1e-12, atol=atol
        )
        assert_matches_oracle(
            plan.variance(), jnp.diagonal(gp_covariance), rtol=1e-12, atol=atol
        )
        assert_matches_oracle(plan.covariance(), gp_covariance, rtol=1e-12, atol=atol)


class TestBridge:
    axis = Axis(jnp.array([0.0, 1.0, 3.0, 4.0]))

    def test_covariance_is_symmetric_positive_semi_definite(self) -> None:
        knots = _knots(7, seed=3)
        targets = _targets(knots, 30, seed=4)
        plan = Linear(prior_variance_rate=RATE).plan(
            Axis(jnp.asarray(knots)), _points(targets)
        )
        covariance = plan.covariance()
        transposed = einx.id("row col -> col row", covariance)
        assert bool(jnp.all(covariance == transposed))
        eigenvalues = jnp.linalg.eigvalsh(covariance)
        # eigenvalues of a PSD matrix in float64 sit above −ε ‖Σ*‖ M
        floor = -1e-13 * float(jnp.max(eigenvalues))
        assert float(jnp.min(eigenvalues)) >= floor

    def test_cells_are_independent(self) -> None:
        plan = Linear(prior_variance_rate=RATE).plan(
            self.axis, Points(jnp.array([[0.5], [0.75], [2.0]]))
        )
        covariance = plan.covariance()
        # σ² Δ t₁ (1 − t₂) in cell 0; 0 across cells
        assert_matches_oracle(covariance[0, 1], RATE * 0.5 * 0.25, rtol=1e-15)
        assert covariance[0, 2] == 0.0 and covariance[1, 2] == 0.0

    def test_periodic_wrap_cell(self) -> None:
        longitude = Axis(jnp.arange(0.0, 360.0, 10.0), period=360.0)
        reversed_longitude = Axis(jnp.arange(350.0, -10.0, -10.0), period=360.0)
        targets = Points(jnp.array([[355.0], [-5.0], [5.0], [360.0]]))
        # mid-cell of the 10° wrap cell and of cell 0, then a knot
        expected = [RATE * 10.0 / 4] * 3 + [0.0]
        for axis in (longitude, reversed_longitude):
            variance = Linear(prior_variance_rate=RATE).plan(axis, targets).variance()
            assert_matches_oracle(variance, jnp.array(expected), rtol=1e-12, atol=1e-12)

    def test_float32_stays_float32(self) -> None:
        axis = Axis(jnp.arange(4.0, dtype=jnp.float32))
        targets = Points(jnp.array([[0.5], [2.25]], dtype=jnp.float32))
        plan = Linear(prior_variance_rate=RATE).plan(axis, targets)
        assert plan.variance().dtype == jnp.float32
        assert plan.covariance().dtype == jnp.float32
        data = jnp.ones((2, 4), dtype=jnp.float32)
        assert plan.sample(data, key=jax.random.key(0)).dtype == jnp.float32

    def test_zero_at_the_knots_and_a_quarter_at_mid_cell(self) -> None:
        knots = [[0.0], [1.0], [3.0], [4.0]]
        mids = [[0.5], [2.0], [3.5]]
        plan = Linear(prior_variance_rate=RATE).plan(
            self.axis, Points(jnp.array(knots + mids))
        )
        variance = plan.variance()
        assert variance[:4].tolist() == [0.0, 0.0, 0.0, 0.0]
        # σ² Δᵢ / 4 for Δ = 1, 2, 1
        expected = [RATE * width / 4 for width in (1.0, 2.0, 1.0)]
        assert_matches_oracle(variance[4:], jnp.array(expected), rtol=1e-15)


class TestPlan:
    axis = Axis(jnp.arange(4.0))
    targets = Points(jnp.array([[-1.0], [0.5], [2.25], [5.0]]))
    data = jnp.array([1.0, 2.0, 4.0, 8.0])

    def test_invalid_rows_have_nan_variance_and_the_fill_value(self) -> None:
        plan = Linear(prior_variance_rate=RATE).plan(
            self.axis, self.targets, policies=Policies(fill_value=-9.0)
        )
        variance = plan.variance()
        assert bool(jnp.isnan(variance[0])) and bool(jnp.isnan(variance[3]))
        covariance = plan.covariance()
        assert bool(jnp.all(jnp.isnan(covariance[0])))
        assert bool(jnp.all(jnp.isnan(covariance[:, 3])))
        assert not bool(jnp.any(jnp.isnan(covariance[1:3, 1:3])))
        draws = plan.sample(jnp.stack([self.data] * 3), key=jax.random.key(2))
        assert draws[:, 0].tolist() == [-9.0] * 3
        assert draws[:, 3].tolist() == [-9.0] * 3

    def test_sample_is_the_mean_at_the_knots(self) -> None:
        plan = Linear(prior_variance_rate=RATE).plan(
            self.axis, Points(jnp.array([[0.0], [1.5], [3.0], [2.0]]))
        )
        draws = plan.sample(jnp.stack([self.data] * 5), key=jax.random.key(3))
        assert draws[:, 0].tolist() == [1.0] * 5
        assert draws[:, 2].tolist() == [8.0] * 5
        assert draws[:, 3].tolist() == [4.0] * 5
        # one field gives one draw; the same key gives the same draw
        single = plan.sample(self.data, key=jax.random.key(4))
        assert single.shape == (4,)
        again = plan.sample(self.data, key=jax.random.key(4))
        assert single.tolist() == again.tolist()

    def test_monte_carlo_matches_the_variance(self) -> None:
        knots = _knots(6, seed=5)
        rng = np.random.default_rng(6)
        # two targets per cell, so the within-cell covariance is checked too
        cells = np.repeat(np.arange(5), 2)
        offsets = rng.uniform(0.05, 0.95, size=10)
        targets = knots[cells] + offsets * (knots[cells + 1] - knots[cells])
        plan = Linear(prior_variance_rate=RATE).plan(
            Axis(jnp.asarray(knots)), _points(targets)
        )
        n_draws = 100_000
        data = jnp.asarray(rng.normal(size=knots.size))
        fields = einx.id("source -> draw source", data, draw=n_draws)
        draws = np.asarray(plan.sample(fields, key=jax.random.key(7)))
        variance = np.asarray(plan.variance())
        covariance = np.asarray(plan.covariance())
        mean = np.asarray(plan.mean(data))
        # a sample variance of n Gaussian draws has relative standard error
        # √(2 / (n − 1)) ≈ 0.45 %, so the 2 % bound is about 4.5 standard errors
        empirical = np.cov(draws, rowvar=False)
        assert np.all(np.abs(np.diag(empirical) / variance - 1.0) < 0.02)
        # a sample mean is within 5 standard errors √(var / n) of the mean
        mean_error = np.abs(einx.mean("[draw] target", draws) - mean)
        assert np.all(mean_error < 5.0 * np.sqrt(variance / n_draws))
        # a sample covariance has standard error √((σ_a² σ_b² + c²) / n); 5 of them
        spread = np.sqrt((np.outer(variance, variance) + covariance**2) / n_draws)
        assert np.all(np.abs(empirical - covariance) < 5.0 * spread)

    def test_mean_adjoint_and_operator_are_the_plain_plan(self) -> None:
        plain = Linear().plan(self.axis, self.targets)
        gaussian = Linear(prior_variance_rate=RATE).plan(self.axis, self.targets)
        assert isinstance(gaussian, GaussianPlan)
        assert_matches_oracle(gaussian.mean(self.data), plain(self.data), rtol=0.0)
        residual = jnp.array([1.0, 2.0, 3.0, 4.0])
        assert_matches_oracle(
            gaussian.adjoint(residual), plain.adjoint(residual), rtol=0.0
        )
        operator = gaussian.operator()
        assert_adjoint(
            lambda y: operator.mv(y), gaussian.adjoint, 4, 4, key=jax.random.key(1)
        )


class TestTraces:
    knots = _knots(6, seed=8)

    def test_variance_with_the_plan_built_inside_the_trace(self) -> None:
        axis = Axis(jnp.asarray(self.knots))
        rng = np.random.default_rng(9)
        targets = rng.uniform(self.knots[0], self.knots[-1], size=(3, 7, 1))

        def variance(positions: jax.Array) -> jax.Array:
            return (
                Linear(prior_variance_rate=RATE)
                .plan(axis, Points(positions))
                .variance()
            )

        assert_traces(variance, jnp.asarray(targets))

    def test_sample_under_jit_equals_eager(self) -> None:
        axis = Axis(jnp.asarray(self.knots))
        targets = _points(_targets(self.knots, 8, seed=10))
        plan = Linear(prior_variance_rate=RATE).plan(axis, targets)
        fields = jax.random.normal(jax.random.key(11), (4, 6))
        key = jax.random.key(12)
        eager = plan.sample(fields, key=key)
        jitted = jax.jit(lambda y: plan.sample(y, key=key))(fields)
        assert_matches_oracle(jitted, eager, rtol=1e-12)

    def test_grad_of_the_variance_in_the_target_positions(self) -> None:
        axis = Axis(jnp.asarray(self.knots))
        rng = np.random.default_rng(13)
        # inside cells, away from the kinks at the knots
        cells = np.arange(5)
        offsets = rng.uniform(0.1, 0.9, size=5)
        widths = self.knots[cells + 1] - self.knots[cells]
        positions = _points(self.knots[cells] + offsets * widths).xy

        def variance(xy: jax.Array) -> jax.Array:
            return Linear(prior_variance_rate=RATE).plan(axis, Points(xy)).variance()

        assert_grad_matches_fd(variance, positions)


def test_a_gaussian_plan_keeps_its_geometry_and_target_shape() -> None:
    # built on the ordinary plan, so derivative() and windows still work
    axis = Axis(jnp.arange(5.0))
    target = Axis(jnp.array([0.5, 1.5, 2.5]))
    plan = Linear(prior_variance_rate=2.0).plan(axis, target)
    assert plan.local is not None
    slope = plan.derivative(axis=0)(jnp.arange(5.0) * 3.0)
    assert slope.tolist() == [3.0, 3.0, 3.0]
