"""Tests for `Variational`: H = I with λ = 0 returns y, and validation. The solves
against dense references, the adjoint and the gradients are in tests/slow.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax import ValidationError
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


def test_identity_observation_without_smoothing_returns_y() -> None:
    y = jnp.asarray(np.random.default_rng(3).normal(size=9))
    identity = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((9,), y.dtype))
    roughness = lx.MatrixLinearOperator(_first_differences(9))
    fit = Variational(identity, roughness, smoothing_weight=0.0, solver=TIGHT)
    assert_matches_oracle(fit(y), y, rtol=1e-12)


def test_the_smoothing_weight_is_a_leaf() -> None:
    plan, precision, _ = _problem()
    fit = _fit(plan, precision)
    assert isinstance(fit.smoothing_weight, jax.Array)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"roughness": lx.MatrixLinearOperator(jnp.ones((3, 5)))}, "roughness"),
        (
            {"observation_precision": lx.DiagonalLinearOperator(jnp.ones(3))},
            "observation_precision",
        ),
        ({"smoothing_weight": -1.0}, "non-negative"),
        ({"smoothing_weight": jnp.ones(2)}, "scalar"),
    ],
)
def test_variational_validates(kwargs, match) -> None:
    plan, _, _ = _problem()
    arguments = {
        "observation": plan.operator(),
        "roughness": lx.MatrixLinearOperator(_first_differences(12)),
    }
    with pytest.raises(ValidationError, match=match):
        Variational(**(arguments | kwargs))


def test_apply_checks_the_data_length() -> None:
    plan, precision, _ = _problem()
    with pytest.raises(ValidationError, match="'data'"):
        _fit(plan, precision).apply(jnp.ones(5))
    with pytest.raises(ValidationError, match="'coefficients'"):
        _fit(plan, precision).apply_adjoint(jnp.ones(5))


def test_float32_is_kept() -> None:
    structure = jax.ShapeDtypeStruct((4,), jnp.float32)
    identity = lx.IdentityLinearOperator(structure)
    fit = Variational(identity, identity, smoothing_weight=0.5)
    assert fit(jnp.ones(4, dtype=jnp.float32)).dtype == jnp.float32
