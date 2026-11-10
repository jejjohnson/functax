"""Slow tests for `Variational`: the normal equations against a dense solve, the
adjoint, use as a plan's fit, implicit gradients through the CG solve against
finite differences, and jit/vmap agreement.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from _harness import (
    assert_adjoint,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)

from functax.geometry import Axis, Points
from functax.interp import Linear, Plan
from functax.project import Variational


TIGHT = lx.CG(rtol=1e-13, atol=1e-13)


def _first_differences(n_cell: int) -> jax.Array:
    """D, (N − 1, N), with (Dx)ₖ = xₖ₊₁ − xₖ."""
    return jnp.asarray(np.diff(np.eye(n_cell), axis=0))


def _problem() -> tuple[Plan, jax.Array, jax.Array]:
    """A linear plan from 12 grid nodes to 7 points, a diagonal R⁻¹ and y."""
    grid = Axis(jnp.linspace(0.0, 1.0, 12))
    rng = np.random.default_rng(0)
    sites = Points(jnp.asarray(rng.uniform(0.0, 1.0, (7, 1))))
    plan = Linear().plan(grid, sites)
    precision = jnp.asarray(rng.uniform(0.5, 2.0, 7))
    y = jnp.asarray(rng.normal(size=7))
    return plan, precision, y


def _fit(plan: Plan, precision: jax.Array, weight: float = 0.7) -> Variational:
    roughness = lx.MatrixLinearOperator(_first_differences(12))
    return Variational(
        plan.operator(),
        roughness,
        observation_precision=lx.DiagonalLinearOperator(precision),
        smoothing_weight=weight,
        solver=TIGHT,
    )


def test_the_fit_solves_the_normal_equations() -> None:
    plan, precision, y = _problem()
    fit = _fit(plan, precision)
    # dense (HᵀR⁻¹H + λDᵀD) x = HᵀR⁻¹y
    observation = np.asarray(plan.read.to_dense())
    differences = np.asarray(_first_differences(12))
    weighted = observation.T * np.asarray(precision)
    normal = weighted @ observation + 0.7 * differences.T @ differences
    expected = np.linalg.solve(normal, weighted @ np.asarray(y))
    assert_matches_oracle(fit(y), expected, rtol=1e-10, atol=1e-12)


def test_the_default_solver_reaches_root_epsilon() -> None:
    plan, precision, y = _problem()
    tight = _fit(plan, precision)
    default = Variational(
        tight.observation,
        tight.roughness,
        observation_precision=tight.observation_precision,
        smoothing_weight=0.7,
    )
    scale = float(jnp.max(jnp.abs(tight(y))))
    assert_matches_oracle(default(y), tight(y), rtol=0, atol=1e-6 * scale)


def test_apply_adjoint_passes_the_dot_product_test() -> None:
    plan, precision, _ = _problem()
    fit = _fit(plan, precision)
    # a solve to 1e-13 relative bounds the inner products to about that
    assert_adjoint(
        fit.apply, fit.apply_adjoint, 7, 12, key=jax.random.key(0), rtol=1e-10
    )


def test_the_fit_batches_over_leading_dims() -> None:
    plan, precision, y = _problem()
    fit = _fit(plan, precision)
    stacked = jnp.stack([y, 2.0 * y, -y])
    assert fit.apply(stacked).shape == (3, 12)
    assert_matches_oracle(fit.apply(stacked)[1], 2.0 * fit(y), rtol=1e-10, atol=1e-13)


def test_a_variational_fit_sits_in_a_plan() -> None:
    plan, precision, y = _problem()
    fit = _fit(plan, precision)
    read = Linear().plan(plan.local.source, Points(jnp.array([[0.25], [0.8]])))
    analysis = Plan(fit, read.read)
    assert_matches_oracle(analysis(y), read(fit(y)), rtol=1e-12)
    assert analysis.operator().in_structure().shape == (7,)


def _parts() -> tuple[lx.AbstractLinearOperator, lx.AbstractLinearOperator, jax.Array]:
    grid = Axis(jnp.linspace(0.0, 1.0, 10))
    rng = np.random.default_rng(0)
    sites = Points(jnp.asarray(rng.uniform(0.0, 1.0, (6, 1))))
    observation = Linear().plan(grid, sites).operator()
    roughness = lx.MatrixLinearOperator(jnp.asarray(np.diff(np.eye(10), axis=0)))
    return observation, roughness, jnp.asarray(rng.normal(size=6))


def test_gradient_in_y_matches_finite_differences() -> None:
    # gate of #42: implicit gradients through the solve to 1e-6
    observation, roughness, y = _parts()
    fit = Variational(observation, roughness, smoothing_weight=0.3, solver=TIGHT)
    assert_grad_matches_fd(fit, y, rtol=1e-6)


def test_gradient_in_the_smoothing_weight_matches_finite_differences() -> None:
    observation, roughness, y = _parts()

    def analysis(weight: jax.Array) -> jax.Array:
        fit = Variational(observation, roughness, smoothing_weight=weight, solver=TIGHT)
        return fit(y)

    assert_grad_matches_fd(analysis, jnp.asarray(0.3), rtol=1e-6)


def test_jit_and_vmap_agree_with_a_loop() -> None:
    observation, roughness, y = _parts()
    fit = Variational(observation, roughness, smoothing_weight=0.3, solver=TIGHT)
    batch = jnp.stack([y, jnp.cos(y), y**2])
    assert_traces(lambda data: fit(data), batch, rtol=1e-10)


def test_the_default_solver_works_under_jit() -> None:
    observation, roughness, y = _parts()
    fit = Variational(observation, roughness, smoothing_weight=0.3)
    jitted = jax.jit(lambda data: fit(data))
    assert_matches_oracle(jitted(y), fit(y), rtol=1e-12, atol=1e-14)
