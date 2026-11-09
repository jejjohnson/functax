"""Tests for the 1-D series schemes and `NonlinearPlan`. The slow tier: oracle sweeps,
dot-product tests across policies, gradient and trace checks, each a second or more
(mostly einx compiling new shapes).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.interpolate
from _harness import (
    assert_adjoint,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_transpose_matches,
)

from functax.geometry import Axis, Points
from functax.interp import (
    Akima,
    CubicSpline,
    NonlinearPlan,
    Pchip,
    Plan,
    Policies,
    Steffen,
)


def _axis_and_data(seed: int, n: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Random uneven increasing knots and random values."""
    rng = np.random.default_rng(seed)
    knots = np.cumsum(rng.uniform(0.2, 1.5, n))
    return knots, rng.normal(size=n)


def _scipy(scheme: object, knots: np.ndarray, values: np.ndarray):  # type: ignore[no-untyped-def]
    """The SciPy interpolator matching ``scheme`` (increasing knots)."""
    if isinstance(scheme, CubicSpline):
        return scipy.interpolate.CubicSpline(knots, values, bc_type=scheme.bc)
    if isinstance(scheme, Pchip):
        return scipy.interpolate.PchipInterpolator(knots, values)
    return scipy.interpolate.Akima1DInterpolator(knots, values, extrapolate=True)


ORACLE_SCHEMES = [
    CubicSpline("not-a-knot"),
    CubicSpline("natural"),
    CubicSpline("clamped"),
    CubicSpline("periodic"),
    Pchip(),
    Akima(),
]


ORACLE_IDS = ["not-a-knot", "natural", "clamped", "periodic", "pchip", "akima"]


class TestOracles:
    @pytest.mark.parametrize("scheme", ORACLE_SCHEMES, ids=ORACLE_IDS)
    @pytest.mark.parametrize(
        "descending", [False, True], ids=["ascending", "descending"]
    )
    def test_equals_scipy(self, scheme, descending: bool) -> None:  # type: ignore[no-untyped-def]
        knots, values = _axis_and_data(
            seed=len(scheme.__class__.__name__) + 7 * descending
        )
        if isinstance(scheme, CubicSpline) and scheme.bc == "periodic":
            values[-1] = values[0]
        rng = np.random.default_rng(1)
        targets = rng.uniform(knots[0] - 0.5, knots[-1] + 0.5, size=60)
        expected = _scipy(scheme, knots, values)(targets)
        source_knots, source_values = (
            (knots[::-1], values[::-1]) if descending else (knots, values)
        )
        plan = scheme.plan(
            Axis(jnp.asarray(source_knots)),
            Points(jnp.asarray(targets)[:, None]),
            policies=Policies(outside="extrapolate"),
        )
        assert_matches_oracle(
            plan(jnp.asarray(source_values)), expected, rtol=1e-12, atol=1e-12
        )

    def test_batched_data(self) -> None:
        knots, _ = _axis_and_data(seed=5)
        data = jax.random.normal(jax.random.key(6), (3, 4, 11))
        targets = Points(jnp.linspace(knots[0], knots[-1], 7)[:, None])
        for scheme in (CubicSpline(), Pchip()):
            values = scheme.plan(Axis(jnp.asarray(knots)), targets)(data)
            assert values.shape == (3, 4, 7)
            single = scheme.plan(Axis(jnp.asarray(knots)), targets)(data[1, 2])
            assert_matches_oracle(values[1, 2], single, rtol=1e-14)


class TestReproduction:
    def test_clamped_reproduces_cubics_with_flat_ends(self) -> None:
        # f = 2x³ − 3x² has f′(0) = f′(1) = 0
        knots = np.sort(
            np.concatenate(
                [[0.0, 1.0], np.random.default_rng(9).uniform(0.05, 0.95, 8)]
            )
        )
        values = 2.0 * knots**3 - 3.0 * knots**2
        targets = np.linspace(0.0, 1.0, 31)
        plan = CubicSpline("clamped").plan(
            Axis(jnp.asarray(knots)), Points(jnp.asarray(targets)[:, None])
        )
        assert_matches_oracle(
            plan(jnp.asarray(values)),
            2.0 * targets**3 - 3.0 * targets**2,
            rtol=1e-11,
            atol=1e-12,
        )


class TestMonotone:
    @pytest.mark.parametrize("scheme", [Pchip(), Steffen()], ids=["pchip", "steffen"])
    def test_monotone_data_gives_monotone_curves_in_10000_cases(self, scheme) -> None:  # type: ignore[no-untyped-def]
        # 5 uneven axes × 2,000 random non-decreasing data sets, plateaus included
        rng = np.random.default_rng(10)
        for seed in range(5):
            knots, _ = _axis_and_data(seed=100 + seed, n=9)
            steps = rng.exponential(size=(2_000, 8)) * (rng.random((2_000, 8)) < 0.7)
            data = np.concatenate(
                [np.zeros((2_000, 1)), np.cumsum(steps, axis=1)], axis=1
            )
            targets = np.linspace(knots[0], knots[-1], 400)
            values = scheme.plan(
                Axis(jnp.asarray(knots)), Points(jnp.asarray(targets)[:, None])
            )(jnp.asarray(data))
            assert float(jnp.min(jnp.diff(values, axis=-1))) >= -1e-12

    def test_gradients_are_finite_at_flat_segments(self) -> None:
        knots = jnp.arange(6.0)
        flat = jnp.array([0.0, 1.0, 1.0, 1.0, 2.0, 2.0])
        targets = Points(jnp.linspace(0.0, 5.0, 23)[:, None])
        for scheme in (Pchip(), Akima(), Steffen()):
            plan = scheme.plan(Axis(knots), targets)
            gradient = jax.grad(lambda y, plan=plan: jnp.sum(plan(y)))(flat)
            assert bool(jnp.all(jnp.isfinite(gradient))), type(scheme).__name__


class TestLinearize:
    @pytest.mark.parametrize(
        "scheme", [Pchip(), Akima(), Steffen()], ids=["pchip", "akima", "steffen"]
    )
    @pytest.mark.parametrize(
        "descending", [False, True], ids=["ascending", "descending"]
    )
    def test_equals_the_dense_jacobian(self, scheme, descending: bool) -> None:  # type: ignore[no-untyped-def]
        knots, values = _axis_and_data(seed=11)
        if descending:
            knots, values = knots[::-1], values[::-1]
        targets = Points(
            jnp.asarray(
                np.random.default_rng(12).uniform(knots.min(), knots.max(), 25)
            )[:, None]
        )
        plan = scheme.plan(Axis(jnp.asarray(knots)), targets)
        assert isinstance(plan, NonlinearPlan)
        tangent = plan.linearize(jnp.asarray(values))
        dense = jax.jacfwd(lambda y: plan(y))(jnp.asarray(values))
        assert tangent.n_taps == sum(scheme.window) + 1
        assert_matches_oracle(tangent.to_dense(), dense, rtol=1e-12, atol=1e-14)


class TestCubicSplineIsLinear:
    @pytest.mark.parametrize("bc", ["not-a-knot", "natural", "clamped", "periodic"])
    @pytest.mark.parametrize(
        "descending", [False, True], ids=["ascending", "descending"]
    )
    def test_dot_product(self, bc: str, descending: bool) -> None:
        knots, _ = _axis_and_data(seed=13)
        if descending:
            knots = knots[::-1]
        targets = Points(
            jnp.asarray(
                np.random.default_rng(14).uniform(knots.min(), knots.max(), 17)
            )[:, None]
        )
        plan = CubicSpline(bc).plan(Axis(jnp.asarray(knots)), targets)
        assert type(plan) is Plan
        operator = plan.operator()
        assert operator.in_size() == 11
        assert_adjoint(
            lambda y: operator.mv(y), plan.adjoint, 11, 17, key=jax.random.key(15)
        )
        assert_transpose_matches(lambda y: operator.mv(y), plan.adjoint, jnp.zeros(11))


class TestFitInTheTrace:
    @pytest.mark.parametrize(
        "scheme",
        [CubicSpline(), Pchip(), Akima(), Steffen()],
        ids=["spline", "pchip", "akima", "steffen"],
    )
    def test_gradients_in_the_data_and_the_knots(self, scheme) -> None:  # type: ignore[no-untyped-def]
        knots, values = _axis_and_data(seed=18, n=8)
        # targets away from the knots, where the curve is smooth in both
        targets = jnp.asarray(0.5 * (knots[1:] + knots[:-1]))[:, None]

        def in_data(y: jax.Array) -> jax.Array:
            return scheme.fit(Axis(jnp.asarray(knots)), y)(targets)

        def in_knots(x: jax.Array) -> jax.Array:
            return scheme.fit(Axis(x), jnp.asarray(values))(targets)

        assert_grad_matches_fd(in_data, jnp.asarray(values))
        assert_grad_matches_fd(in_knots, jnp.asarray(knots))
