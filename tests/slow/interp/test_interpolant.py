"""Tests for `Interpolant` and `LocalScheme.fit`: moving targets inside the trace. The
slow tier: oracle sweeps, dot-product tests across policies, gradient and trace
checks, each a second or more (mostly einx compiling new shapes).
"""

from __future__ import annotations

import functools

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from hypothesis import given, settings

from functax.geometry import AffineGrid, Axis, Points, RectilinearGrid
from functax.interp import (
    BSpline,
    Cubic,
    Interpolant,
    Linear,
    Nearest,
    Policies,
    Tensor,
)


GRID = RectilinearGrid(
    (Axis(jnp.linspace(0.0, 5.0, 8)), Axis(jnp.linspace(-1.0, 2.0, 9)))
)


FIELD = jax.random.normal(jax.random.key(0), (72,))


def _targets(count: int, seed: int) -> jax.Array:
    rng = np.random.default_rng(seed)
    return jnp.asarray(
        np.column_stack([rng.uniform(-0.5, 5.5, count), rng.uniform(-1.3, 2.3, count)])
    )


SCHEMES = [
    Nearest(),
    Linear(),
    Cubic(),
    *(BSpline(order) for order in range(6)),
    Tensor((Nearest(), BSpline(3))),
]


SCHEME_IDS = [
    "nearest",
    "linear",
    "cubic",
    *(f"bspline{k}" for k in range(6)),
    "tensor",
]


class TestEquivalence:
    @pytest.mark.parametrize("scheme", SCHEMES, ids=SCHEME_IDS)
    def test_equals_the_plan(self, scheme) -> None:  # type: ignore[no-untyped-def]
        targets = _targets(40, seed=1)
        field = scheme.fit(GRID, FIELD)
        expected = scheme.plan(GRID, Points(targets))(FIELD)
        assert isinstance(field, Interpolant)
        assert_matches_oracle(field(targets), expected, rtol=1e-14, atol=1e-15)
        assert_matches_oracle(field(Points(targets)), expected, rtol=1e-14, atol=1e-15)

    def test_on_a_periodic_axis_and_an_affine_grid(self) -> None:
        longitude = Axis(jnp.arange(0.0, 360.0, 15.0), period=360.0)
        samples = jnp.sin(jnp.deg2rad(longitude.values))
        positions = jnp.array([[350.0], [-10.0], [725.0]])
        field = BSpline(3).fit(longitude, samples)
        assert_matches_oracle(
            field(positions),
            BSpline(3).plan(longitude, Points(positions))(samples),
            rtol=1e-14,
        )
        grid = AffineGrid([2.0, 0.0, 10.0, 0.0, -2.0, 30.0], (6, 7))
        data = jax.random.normal(jax.random.key(2), (42,))
        xy = jnp.array([[13.0, 25.0], [20.5, 21.25]])
        assert_matches_oracle(
            Linear().fit(grid, data)(xy),
            Linear().plan(grid, Points(xy))(data),
            rtol=1e-14,
        )

    def test_batched_coefficients(self) -> None:
        fields = jax.random.normal(jax.random.key(3), (2, 3, 72))
        targets = _targets(5, seed=4)
        values = BSpline(3).fit(GRID, fields)(targets)
        assert values.shape == (2, 3, 5)
        assert_matches_oracle(
            values,
            BSpline(3).plan(GRID, Points(targets))(fields),
            rtol=1e-14,
            atol=1e-15,
        )

    def test_plan_freezes_the_targets(self) -> None:
        targets = Points(_targets(6, seed=5))
        field = Cubic().fit(GRID, FIELD)
        plan = field.plan(targets)
        assert_matches_oracle(plan(FIELD), Cubic().plan(GRID, targets)(FIELD), rtol=0.0)


class TestTraces:
    def test_vmap_over_target_sets(self) -> None:
        field = BSpline(3).fit(GRID, FIELD)
        batches = jnp.stack([_targets(7, seed=s) for s in range(3)])
        assert_traces(lambda xy: field(xy), batches)

    def test_gradient_in_the_positions_and_the_data(self) -> None:
        # differentiates the moving targets, then the samples through the fit
        # (the prefilter); offsets mid-cell, clear of the knots
        rng = np.random.default_rng(11)
        rows = jnp.linspace(0.0, 5.0, 8)[rng.integers(1, 6, 6)] + rng.uniform(
            0.1, 0.6, 6
        )
        cols = jnp.linspace(-1.0, 2.0, 9)[rng.integers(1, 7, 6)] + rng.uniform(
            0.05, 0.3, 6
        )
        xy = jnp.stack([rows, cols], axis=-1)
        scheme = BSpline(3)
        field = scheme.fit(GRID, FIELD)
        assert_grad_matches_fd(lambda positions: field(positions), xy)
        assert_grad_matches_fd(lambda data: scheme.fit(GRID, data)(xy), FIELD)


@functools.cache
def _compiled_fit_and_read(scheme):  # type: ignore[no-untyped-def]
    """``(data, targets) ↦ scheme.fit(GRID, data)(targets)``, compiled once."""
    return jax.jit(lambda data, targets: scheme.fit(GRID, data)(targets))


class TestProperties:
    @pytest.mark.parametrize(
        "scheme",
        [Nearest(), Linear(), Cubic(), BSpline(3), Tensor((Nearest(), BSpline(3)))],
        ids=["nearest", "linear", "cubic", "bspline3", "tensor"],
    )
    @settings(max_examples=20, deadline=None)
    @given(
        seed=st.integers(0, 2**31 - 1),
        scale=st.floats(-3.0, 3.0),
        level=st.floats(-10.0, 10.0),
    )
    def test_linear_in_the_data_and_reproduces_constants(
        self, scheme, seed: int, scale: float, level: float
    ) -> None:  # type: ignore[no-untyped-def]
        # f̂[a y₁ + y₂ + c](x) = a f̂[y₁](x) + f̂[y₂](x) + c at moving targets inside
        rng = np.random.default_rng(seed)
        first, second = rng.normal(size=(2, 72))
        data = jnp.asarray(np.stack([scale * first + second + level, first, second]))
        targets = jnp.asarray(
            np.column_stack([rng.uniform(0.0, 5.0, 16), rng.uniform(-1.0, 2.0, 16)])
        )
        values = _compiled_fit_and_read(scheme)(data, targets)
        expected = scale * values[1] + values[2] + level
        assert_matches_oracle(values[0], expected, rtol=1e-12, atol=1e-12)


class TestDerivativesAndPolicies:
    def test_derivative_interpolant_equals_the_derivative_plan(self) -> None:
        targets = _targets(9, seed=8)
        field = BSpline(3).fit(GRID, FIELD)
        slope = field.derivative(axis=1)
        expected = BSpline(3).plan(GRID, Points(targets)).derivative(axis=1)(FIELD)
        assert_matches_oracle(slope(targets), expected, rtol=1e-12, atol=1e-13)
        assert_matches_oracle(
            slope.plan(Points(targets))(FIELD), expected, rtol=1e-12, atol=1e-13
        )

    def test_fill_runs_at_the_fit(self) -> None:
        data = FIELD.at[10].set(jnp.nan)
        policies = Policies(missing="fill", fill_value=0.0)
        targets = _targets(12, seed=9)
        for scheme in (Linear(), BSpline(3)):
            field = scheme.fit(GRID, data, policies=policies)
            expected = scheme.plan(GRID, Points(targets), policies=policies)(data)
            assert_matches_oracle(field(targets), expected, rtol=1e-14, atol=1e-15)

    def test_renormalise_equals_the_plan(self) -> None:
        data = FIELD.at[10].set(jnp.nan)
        policies = Policies(missing="renormalise")
        targets = _targets(12, seed=10)
        field = Linear().fit(GRID, data, policies=policies)
        expected = Linear().plan(GRID, Points(targets), policies=policies)(data)
        assert_matches_oracle(field(targets), expected, rtol=1e-14, atol=1e-15)
