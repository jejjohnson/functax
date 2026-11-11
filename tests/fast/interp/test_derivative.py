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


class TestEndNodes:
    """A derivative read exactly at an end node comes from inside the axis (#128)."""

    @pytest.mark.parametrize("outside", ["nan", "clamp", "extrapolate"])
    @pytest.mark.parametrize("boundary", ["mirror", "edge", "wrap", "constant"])
    def test_linear_reads_the_end_cells_slope(
        self, outside: str, boundary: str
    ) -> None:
        # secants 1, 2, 3
        data = jnp.array([1.0, 2.0, 4.0, 7.0])
        plan = Linear().plan(
            Axis(jnp.arange(4.0)),
            Points(jnp.array([[0.0], [2.5], [3.0]])),
            policies=Policies(outside=outside, boundary=boundary),
        )
        assert plan(data).tolist() == [1.0, 5.5, 7.0]
        assert plan.derivative(0)(data).tolist() == [1.0, 3.0, 3.0]

    def test_a_decreasing_axis(self) -> None:
        data = jnp.array([7.0, 4.0, 2.0, 1.0])
        plan = Linear().plan(
            Axis(jnp.array([3.0, 2.0, 1.0, 0.0])), Points(jnp.array([[0.0], [3.0]]))
        )
        assert plan.derivative(0)(data).tolist() == [1.0, 3.0]

    def test_clamp_holds_a_constant_past_the_ends(self) -> None:
        data = jnp.array([1.0, 2.0, 4.0, 7.0])
        plan = Linear().plan(
            Axis(jnp.arange(4.0)),
            Points(jnp.array([[-1.0], [4.0]])),
            policies=Policies(outside="clamp"),
        )
        assert plan.derivative(0)(data).tolist() == [0.0, 0.0]
