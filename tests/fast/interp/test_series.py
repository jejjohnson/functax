"""Tests for the 1-D series schemes and `NonlinearPlan`. The fast tier: behaviour,
validation and small closed-form checks, each well under a second.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.interpolate
from _harness import (
    assert_matches_oracle,
)

from functax import GeometryError, NonlinearSchemeError, PolicyError, ValidationError
from functax.geometry import Axis, Points
from functax.interp import (
    Akima,
    CubicSpline,
    Pchip,
    Policies,
    SeriesInterpolant,
    Steffen,
)


def _axis_and_data(seed: int, n: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Random uneven increasing knots and random values."""
    rng = np.random.default_rng(seed)
    knots = np.cumsum(rng.uniform(0.2, 1.5, n))
    return knots, rng.normal(size=n)


def _steffen_reference(
    knots: np.ndarray, values: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """A direct, loop-based port of Steffen (1990), eqs. 11, 13, 26 and 27."""
    n = knots.size
    widths = np.diff(knots)
    secants = np.diff(values) / widths
    slopes = np.zeros(n)
    for k in range(1, n - 1):
        parabola = (secants[k - 1] * widths[k] + secants[k] * widths[k - 1]) / (
            widths[k - 1] + widths[k]
        )
        slopes[k] = (np.sign(secants[k - 1]) + np.sign(secants[k])) * min(
            abs(secants[k - 1]), abs(secants[k]), 0.5 * abs(parabola)
        )
    for end, (h0, h1, d0, d1) in (
        (0, (widths[0], widths[1], secants[0], secants[1])),
        (n - 1, (widths[-1], widths[-2], secants[-1], secants[-2])),
    ):
        parabola = d0 * (1 + h0 / (h0 + h1)) - d1 * h0 / (h0 + h1)
        if parabola * d0 <= 0:
            slopes[end] = 0.0
        elif abs(parabola) > 2 * abs(d0):
            slopes[end] = 2 * d0
        else:
            slopes[end] = parabola
    return scipy.interpolate.CubicHermiteSpline(knots, values, slopes)(targets)


class TestOracles:
    @pytest.mark.parametrize(
        "descending", [False, True], ids=["ascending", "descending"]
    )
    def test_steffen_equals_the_paper(self, descending: bool) -> None:
        knots, values = _axis_and_data(seed=3)
        targets = np.random.default_rng(4).uniform(knots[0], knots[-1], size=60)
        expected = _steffen_reference(knots, values, targets)
        source_knots, source_values = (
            (knots[::-1], values[::-1]) if descending else (knots, values)
        )
        plan = Steffen().plan(
            Axis(jnp.asarray(source_knots)), Points(jnp.asarray(targets)[:, None])
        )
        assert_matches_oracle(
            plan(jnp.asarray(source_values)), expected, rtol=1e-12, atol=1e-12
        )

    def test_two_knots_are_linear(self) -> None:
        for scheme in (CubicSpline("natural"), Pchip(), Akima(), Steffen()):
            plan = scheme.plan(Axis(jnp.array([0.0, 2.0])), Points(jnp.array([[0.5]])))
            assert_matches_oracle(
                plan(jnp.array([1.0, 3.0])), jnp.array([1.5]), rtol=1e-15
            )


class TestReproduction:
    def test_not_a_knot_reproduces_cubics(self) -> None:
        knots, _ = _axis_and_data(seed=7)
        values = 0.3 * knots**3 - knots**2 + 2.0 * knots - 4.0
        targets = np.random.default_rng(8).uniform(knots[0], knots[-1], size=40)
        plan = CubicSpline().plan(
            Axis(jnp.asarray(knots)), Points(jnp.asarray(targets)[:, None])
        )
        expected = 0.3 * targets**3 - targets**2 + 2.0 * targets - 4.0
        assert_matches_oracle(
            plan(jnp.asarray(values)), expected, rtol=1e-11, atol=1e-11
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


class TestFitInTheTrace:
    def test_equals_the_plan(self) -> None:
        knots, values = _axis_and_data(seed=16)
        targets = jnp.asarray(
            np.random.default_rng(17).uniform(knots[0], knots[-1], 9)
        )[:, None]
        for scheme in (CubicSpline(), Pchip(), Akima(), Steffen()):
            field = scheme.fit(Axis(jnp.asarray(knots)), jnp.asarray(values))
            assert isinstance(field, SeriesInterpolant)
            expected = scheme.plan(Axis(jnp.asarray(knots)), Points(targets))(
                jnp.asarray(values)
            )
            assert_matches_oracle(field(targets), expected, rtol=0.0)
            assert_matches_oracle(
                field.plan(Points(targets))(jnp.asarray(values)), expected, rtol=0.0
            )


class TestPoliciesAndValidation:
    knots = Axis(jnp.arange(6.0))

    def test_outside_policies(self) -> None:
        targets = Points(jnp.array([[-1.0], [2.5], [7.0]]))
        values = jnp.arange(6.0) ** 2
        nan = Pchip().plan(self.knots, targets)(values)
        assert bool(jnp.isnan(nan[0])) and bool(jnp.isnan(nan[2]))
        clamp = Pchip().plan(self.knots, targets, policies=Policies(outside="clamp"))(
            values
        )
        assert clamp[0] == 0.0 and clamp[2] == 25.0
        with pytest.raises(Exception, match="outside the domain"):
            Pchip().plan(self.knots, targets, policies=Policies(outside="raise"))

    def test_missing_policies(self) -> None:
        values = jnp.arange(6.0).at[2].set(jnp.nan)
        targets = Points(jnp.array([[4.5]]))
        filled = CubicSpline().plan(
            self.knots, targets, policies=Policies(missing="fill", fill_value=0.0)
        )(values)
        assert bool(jnp.isfinite(filled[0]))
        with pytest.raises(PolicyError, match="needs a local plan"):
            Pchip().plan(self.knots, targets, policies=Policies(missing="renormalise"))(
                values
            )

    def test_rejects_bad_sources_and_conditions(self) -> None:
        with pytest.raises(GeometryError, match="read an Axis"):
            Pchip().plan(Points(jnp.zeros((3, 1))), Points(jnp.zeros((1, 1))))
        with pytest.raises(ValidationError, match="needs at least 4 knots"):
            CubicSpline().plan(Axis(jnp.arange(3.0)), Points(jnp.zeros((1, 1))))
        with pytest.raises(ValidationError, match="'bc' must be one of"):
            CubicSpline("parabolic")
        with pytest.raises(ValidationError, match="non-periodic Axis"):
            Pchip().plan(Axis(jnp.arange(5.0), period=10.0), Points(jnp.zeros((1, 1))))

    def test_spline_slopes_reject_an_unknown_condition(self) -> None:
        from functax._src.interp._spline1d import spline_slopes

        with pytest.raises(ValidationError, match="'bc' must be one of"):
            inspect.unwrap(spline_slopes)(
                jnp.arange(4.0), jnp.arange(4.0), condition="parabolic"
            )
