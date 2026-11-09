"""Tests for `Plan.derivative`, `DerivativePlan` and the sum of plans. The fast tier:
behaviour, validation and small closed-form checks, each well under a second.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import pytest

from functax import PolicyError, ValidationError
from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import (
    Linear,
    Plan,
    Policies,
    Readout,
)


class TestPolicies:
    def test_outside_rows_stay_invalid(self) -> None:
        plan = Linear().plan(Axis(jnp.arange(5.0)), Points(jnp.array([[1.5], [9.0]])))
        derivative = plan.derivative(axis=0)
        assert derivative.valid.tolist() == [True, False]
        assert bool(jnp.isnan(derivative(jnp.arange(5.0))[1]))

    def test_raise_is_enforced_once(self) -> None:
        plan = Linear().plan(
            Axis(jnp.arange(5.0)),
            Points(jnp.array([[1.5]])),
            policies=Policies(outside="raise"),
        )
        assert plan.derivative(axis=0)(jnp.arange(5.0) ** 2).tolist() == [3.0]

    def test_renormalise_has_no_derivative(self) -> None:
        plan = Linear().plan(
            Axis(jnp.arange(5.0)),
            Points(jnp.array([[1.5]])),
            policies=Policies(missing="renormalise"),
        )
        with pytest.raises(
            PolicyError, match="not defined under missing='renormalise'"
        ):
            plan.derivative(axis=0)


class TestValidation:
    plan = Linear().plan(
        RectilinearGrid((Axis(jnp.arange(4.0)), Axis(jnp.arange(5.0)))),
        Points(jnp.array([[1.5, 2.5]])),
    )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"axis": 2}, r"'axis' must lie in \[-2, 2\)"),
            ({"axis": -3}, r"'axis' must lie in \[-2, 2\)"),
            ({"axis": 0, "order": 0}, "'order' must be a positive integer"),
        ],
    )
    def test_rejects_bad_arguments(self, kwargs: dict[str, int], match: str) -> None:
        with pytest.raises(ValidationError, match=match):
            self.plan.derivative(**kwargs)

    def test_rejects_non_integer_arguments_behind_the_type_hook(self) -> None:
        from functax._src.interp._derivative import derivative_stencil

        # the type hook would reject these before the function's own check
        raw = inspect.unwrap(derivative_stencil)
        local = self.plan.local
        with pytest.raises(ValidationError, match="'axis' must be an integer"):
            raw(local, self.plan.read, self.plan.policies, axis=True, order=1)
        with pytest.raises(ValidationError, match="'order' must be a positive integer"):
            raw(local, self.plan.read, self.plan.policies, axis=0, order=1.5)

    def test_a_hand_built_plan_has_no_derivative(self) -> None:
        bare = Plan(self.plan.fit, self.plan.read, self.plan.policies)
        with pytest.raises(
            ValidationError, match="needs a plan built by a local scheme"
        ):
            bare.derivative(axis=0)

    def test_sums_need_matching_plans(self) -> None:
        other = Linear().plan(Axis(jnp.arange(7.0)), Points(jnp.array([[1.5]])))
        with pytest.raises(ValidationError, match="plans add on one source"):
            self.plan + other
        shifted = Linear().plan(
            RectilinearGrid((Axis(jnp.arange(4.0)), Axis(jnp.arange(5.0)))),
            Points(jnp.array([[1.5, 2.5]])),
            policies=Policies(outside="clamp"),
        )
        with pytest.raises(ValidationError, match="share the fit and the policies"):
            self.plan + shifted
        assert self.plan.__add__(object()) is NotImplemented

    def test_sums_need_stencil_read_outs(self) -> None:
        class Dense(Readout):
            matrix: jax.Array
            valid: jax.Array

            def apply(self, coefficients):  # type: ignore[no-untyped-def]
                return coefficients @ self.matrix

            def apply_adjoint(self, residual):  # type: ignore[no-untyped-def]
                return self.matrix @ residual

            def operator(self):  # type: ignore[no-untyped-def]
                return None

        dense = Plan(
            self.plan.fit,
            Dense(jnp.ones((20, 1)), jnp.ones(1, dtype=bool)),
            self.plan.policies,
        )
        with pytest.raises(ValidationError, match="both read-outs are stencils"):
            self.plan + dense
