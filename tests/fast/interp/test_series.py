"""Tests for the 1-D series schemes and `NonlinearPlan`. The fast tier: behaviour,
validation and small closed-form checks, each well under a second.
"""

from __future__ import annotations

import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from functax import (
    GeometryError,
    NonlinearSchemeError,
    OutsideDomainError,
    PolicyError,
    ValidationError,
)
from functax.geometry import Axis, Points
from functax.interp import (
    CubicSpline,
    DerivativePlan,
    NonlinearDerivativePlan,
    Pchip,
    Policies,
    SeriesScheme,
)


class TestLinearize:
    def test_operator_and_adjoint_refuse(self) -> None:
        plan = Pchip().plan(Axis(jnp.arange(5.0)), Points(jnp.array([[1.5]])))
        with pytest.raises(NonlinearSchemeError, match="no fixed operator"):
            plan.operator()
        with pytest.raises(NonlinearSchemeError, match="no fixed adjoint"):
            plan.adjoint(jnp.ones(1))
        with pytest.raises(NonlinearSchemeError, match="no adjoint"):
            plan.fit.apply_adjoint(jnp.ones(10))
        with pytest.raises(ValidationError, match="needs the data"):
            plan.linearize()
        with pytest.raises(ValidationError, match=r"shape \(5,\)"):
            plan.linearize(jnp.ones(4))


class TestPoliciesAndValidation:
    knots = Axis(jnp.arange(6.0))

    def test_rejects_bad_sources_and_conditions(self) -> None:
        with pytest.raises(GeometryError, match="read an Axis"):
            Pchip().plan(Points(jnp.zeros((3, 1))), Points(jnp.zeros((1, 1))))
        with pytest.raises(ValidationError, match="needs at least 4 knots"):
            CubicSpline().plan(Axis(jnp.arange(3.0)), Points(jnp.zeros((1, 1))))
        with pytest.raises(ValidationError, match="'bc' must be one of"):
            CubicSpline("parabolic")
        with pytest.raises(ValidationError, match="non-periodic Axis"):
            Pchip().plan(Axis(jnp.arange(5.0), period=10.0), Points(jnp.zeros((1, 1))))

    def test_outside_raise_rejects_targets_off_the_axis(self) -> None:
        targets = Points(jnp.array([[-1.0], [2.5], [7.0]]))
        with pytest.raises(OutsideDomainError, match="outside the domain"):
            Pchip().plan(self.knots, targets, policies=Policies(outside="raise"))

    def test_renormalise_needs_a_local_plan(self) -> None:
        values = jnp.arange(6.0).at[2].set(jnp.nan)
        targets = Points(jnp.array([[4.5]]))
        with pytest.raises(PolicyError, match="needs a local plan"):
            Pchip().plan(self.knots, targets, policies=Policies(missing="renormalise"))(
                values
            )

    def test_periodic_spline_names_the_repeat(self) -> None:
        axis = Axis(jnp.array([0.0, 1.0, 1.0, 2.0, 3.0]), repeats="allow")
        with pytest.raises(GeometryError, match=r"bc='periodic'.* x\[1\] = x\[2\]"):
            CubicSpline(bc="periodic").plan(axis, Points(jnp.array([[0.5]])))

    def test_spline_slopes_reject_an_unknown_condition(self) -> None:
        from functax._src.interp._spline1d import spline_slopes

        with pytest.raises(ValidationError, match="'bc' must be one of"):
            inspect.unwrap(spline_slopes)(
                jnp.arange(4.0), jnp.arange(4.0), condition="parabolic"
            )


class TestDerivatives:
    """Closed-form derivatives of the series read-outs (#116)."""

    def test_a_series_plan_has_a_derivative(self) -> None:
        plan = Pchip().plan(Axis(jnp.arange(4.0)), Points(jnp.array([[1.5]])))
        derivative = plan.derivative(0)
        assert isinstance(derivative, NonlinearDerivativePlan)
        with pytest.raises(NonlinearSchemeError, match="no fixed operator"):
            derivative.operator()
        with pytest.raises(NonlinearSchemeError, match="no fixed adjoint"):
            derivative.adjoint(jnp.ones(1))
        spline = CubicSpline().plan(Axis(jnp.arange(5.0)), Points(jnp.array([[1.5]])))
        assert type(spline.derivative(0)) is DerivativePlan

    def test_rejects_bad_orders_axes_and_nesting(self) -> None:
        field = Pchip().fit(Axis(jnp.arange(4.0)), jnp.arange(4.0))
        with pytest.raises(ValidationError, match="'order' must be a positive"):
            field.derivative(order=0)
        with pytest.raises(ValidationError, match="'axis' must be 0"):
            field.derivative(axis=1)
        with pytest.raises(ValidationError, match="raise 'order' instead"):
            field.derivative().derivative()


class _CountedSlopes(SeriesScheme):
    """Centred secants, which count how often they are traced."""

    calls: list[int] = eqx.field(static=True, default_factory=list)

    def _slopes(self):  # type: ignore[no-untyped-def]
        def slopes(nodes: jax.Array, values: jax.Array) -> jax.Array:
            self.calls.append(1)
            return jnp.gradient(values, nodes)

        return slopes

    _piecewise_slopes = _slopes


class TestFit:
    """The slopes are computed once, at the fit (#162)."""

    def test_a_read_computes_no_slopes(self) -> None:
        scheme = _CountedSlopes()
        knots = jnp.array([0.0, 1.0, 2.5, 3.0])
        field = scheme.fit(Axis(knots), knots**2)
        assert len(scheme.calls) == 1
        targets = jnp.array([[0.5], [2.0]])
        field(targets)
        field.derivative()(targets)
        eqx.filter_jit(lambda f, x: f(x))(field, targets)
        assert len(scheme.calls) == 1

    def test_a_decreasing_field_reads_the_same_through_jit(self) -> None:
        # the fit sees a concrete decreasing axis and reverses it; inside jit
        # the axis is traced, and the read must still follow that order
        knots = jnp.array([4.0, 3.0, 1.5, 0.0])
        field = Pchip().fit(Axis(knots), jnp.array([1.0, 0.0, 2.0, 5.0]))
        targets = jnp.array([[0.2], [2.0], [3.7]])
        for read in (field, field.derivative()):
            jitted = eqx.filter_jit(lambda f, x: f(x))(read, targets)
            assert jnp.allclose(jitted, read(targets), rtol=1e-14)

    def test_a_field_fitted_on_a_traced_decreasing_axis(self) -> None:
        knots = jnp.array([4.0, 3.0, 1.5, 0.0])
        values = jnp.array([1.0, 0.0, 2.0, 5.0])
        targets = jnp.array([[0.2], [2.0], [3.7]])
        # fitted in the trace, where the direction is unknown, and read outside
        field = eqx.filter_jit(lambda x, y: Pchip().fit(Axis(x), y))(knots, values)
        expected = Pchip().plan(Axis(knots), Points(targets))(values)
        assert jnp.allclose(field(targets), expected, rtol=1e-14)

    def test_keeps_the_samples_in_the_order_of_the_source(self) -> None:
        values = jnp.array([[1.0, 0.0, 2.0], [3.0, 5.0, 4.0]])
        for knots in (jnp.array([0.0, 1.0, 3.0]), jnp.array([3.0, 1.0, 0.0])):
            field = Pchip().fit(Axis(knots), values)
            assert jnp.array_equal(field.data, values)

    @pytest.mark.parametrize("outside", ["nan", "clamp", "extrapolate"])
    def test_outside_rules_equal_the_plan(self, outside: str) -> None:
        axis = Axis(jnp.array([0.0, 1.0, 2.5, 3.0]))
        values = jnp.array([1.0, 0.0, 2.0, 5.0])
        targets = jnp.array([[-0.5], [1.7], [3.4]])
        policies = Policies(outside=outside)
        expected = Pchip().plan(axis, Points(targets), policies=policies)(values)
        read = Pchip().fit(axis, values, policies=policies)(targets)
        assert jnp.allclose(read, expected, rtol=1e-14, equal_nan=True)

    def test_missing_samples_are_filled_before_the_slopes(self) -> None:
        values = jnp.arange(6.0).at[2].set(jnp.nan)
        policies = Policies(missing="fill", fill_value=0.0)
        targets = jnp.array([[1.5], [4.5]])
        field = Pchip().fit(Axis(jnp.arange(6.0)), values, policies=policies)
        plan = Pchip().plan(Axis(jnp.arange(6.0)), Points(targets), policies=policies)
        assert jnp.allclose(field(targets), plan(values), rtol=1e-14)

    def test_rejects_short_axes_bad_data_and_policies(self) -> None:
        axis = Axis(jnp.arange(6.0))
        with pytest.raises(ValidationError, match="needs at least 4 knots"):
            CubicSpline().fit(Axis(jnp.arange(3.0)), jnp.zeros(3))
        with pytest.raises(ValidationError, match="last axis of length 6"):
            Pchip().fit(axis, jnp.zeros((6, 2)))
        with pytest.raises(PolicyError, match="needs a local plan"):
            Pchip().fit(axis, jnp.zeros(6), policies=Policies(missing="renormalise"))
