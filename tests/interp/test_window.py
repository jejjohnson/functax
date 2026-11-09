"""Tests for `WindowedPlan` (``plan[window]``) and `Plan.to_gaussx`."""

from __future__ import annotations

import inspect

import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_adjoint, assert_matches_oracle

from functax import PolicyError, ValidationError
from functax.geometry import AffineGrid, Axis, Points, RectilinearGrid
from functax.interp import (
    Fit,
    Identity,
    Linear,
    Plan,
    Policies,
    Readout,
    Stencil,
    WindowedPlan,
)


# a 10 × 12 source raster and an 8 × 9 destination raster, shifted and finer,
# so its right and bottom edges fall off the source (invalid rows)
SOURCE = AffineGrid([1.0, 0.0, 0.0, 0.0, -1.0, 10.0], (10, 12))
DESTINATION = AffineGrid([1.4, 0.0, 1.3, 0.0, -1.3, 9.6], (8, 9))
CHIPS = [
    (slice(0, 3), slice(0, 4)),
    (slice(3, 8), slice(4, 9)),
    (slice(None), slice(2, 3)),
    (slice(1, 7, 2), slice(None, None, 3)),
]


def _warp(policies: Policies | None = None) -> Plan:
    return Linear().plan(SOURCE, DESTINATION, policies=policies)


def _as_raster(values: jax.Array) -> np.ndarray:
    """The flat destination rows as an (…, 8, 9) raster."""
    return np.asarray(einx.id("... (row col) -> ... row col", values, row=8))


class TestWindow:
    @pytest.mark.slow
    @pytest.mark.parametrize("chip", CHIPS, ids=lambda w: str(w))
    def test_chips_equal_the_full_warp_bit_for_bit(self, chip: tuple) -> None:
        warp = _warp()
        assert not bool(jnp.all(warp.valid))
        fields = jax.random.normal(jax.random.key(0), (3, 120))
        windowed = warp[chip]
        assert isinstance(windowed, WindowedPlan)
        assert windowed.window == chip
        full = _as_raster(warp(fields))[(slice(None), *chip)]
        assert windowed.grid_shape == full.shape[1:]
        chip_values = einx.id(
            "... (row col) -> ... row col", windowed(fields), row=full.shape[1]
        )
        np.testing.assert_array_equal(chip_values, full)
        np.testing.assert_array_equal(
            einx.id("(row col) -> row col", windowed.valid, row=full.shape[1]),
            _as_raster(warp.valid)[chip],
        )

    @pytest.mark.slow
    def test_chips_under_renormalise_are_bit_for_bit(self) -> None:
        warp = _warp(Policies(missing="renormalise"))
        field = jax.random.normal(jax.random.key(1), (120,)).at[::7].set(jnp.nan)
        full = _as_raster(warp(field))
        for chip in CHIPS:
            np.testing.assert_array_equal(
                einx.id(
                    "(row col) -> row col", warp[chip](field), row=full[chip].shape[0]
                ),
                full[chip],
            )

    @pytest.mark.slow
    def test_adjoint_is_the_full_adjoint_of_the_chip(self) -> None:
        warp = _warp()
        chip = warp[2:6, 1:5]
        residual = jax.random.normal(jax.random.key(2), (16,))
        padded = (
            jnp.zeros((8, 9))
            .at[2:6, 1:5]
            .set(einx.id("(row col) -> row col", residual, row=4))
        )
        expected = warp.adjoint(einx.id("row col -> (row col)", padded))
        assert_matches_oracle(chip.adjoint(residual), expected, rtol=1e-12, atol=1e-14)
        operator = chip.operator()
        assert_adjoint(
            lambda y: operator.mv(y), chip.adjoint, 120, 16, key=jax.random.key(3)
        )

    @pytest.mark.slow
    def test_a_flat_target_takes_one_slice(self) -> None:
        plan = Linear().plan(
            Axis(jnp.arange(6.0)), Points(jnp.linspace(0.0, 5.0, 9)[:, None])
        )
        field = jax.random.normal(jax.random.key(4), (6,))
        assert plan.target_shape is None
        assert plan.grid_shape == (9,)
        np.testing.assert_array_equal(plan[2:7](field), plan(field)[2:7])
        np.testing.assert_array_equal(plan[(slice(1, 3),)](field), plan(field)[1:3])

    def test_a_window_of_a_window(self) -> None:
        warp = _warp()
        field = jax.random.normal(jax.random.key(5), (120,))
        nested = warp[1:7, 2:9][1:3, 0:2]
        np.testing.assert_array_equal(nested(field), warp[2:4, 2:4](field))

    def test_a_window_of_a_composed_plan(self) -> None:
        warp = _warp(Policies(outside="clamp"))
        smooth = Linear().plan(DESTINATION, DESTINATION)
        composed = smooth @ warp
        assert composed.target_shape == (8, 9)
        field = jax.random.normal(jax.random.key(6), (120,))
        np.testing.assert_array_equal(
            composed[0:4, 3:6](field), _as_raster(composed(field))[0:4, 3:6].ravel()
        )

    def test_rejects_bad_windows(self) -> None:
        warp = _warp()
        with pytest.raises(ValidationError, match=r"one slice per target axis \(2\)"):
            warp[0:2]
        with pytest.raises(ValidationError, match="selects no targets"):
            warp[3:3, :]
        # the type hook would reject a non-slice first, so skip it
        init = inspect.unwrap(WindowedPlan.__init__)
        with pytest.raises(ValidationError, match="one slice per target axis"):
            init(object.__new__(WindowedPlan), warp, (0, slice(None)))

    def test_needs_a_stencil_read_out(self) -> None:
        class Dense(Readout):
            valid: jax.Array

            def apply(self, coefficients):  # type: ignore[no-untyped-def]
                return coefficients

            def apply_adjoint(self, residual):  # type: ignore[no-untyped-def]
                return residual

            def operator(self):  # type: ignore[no-untyped-def]
                return None

        plan = Plan(Identity(), Dense(jnp.ones(3, dtype=bool)))
        with pytest.raises(PolicyError, match="needs a plan with a stencil read-out"):
            plan[0:2]


class TestTargetShape:
    def test_grid_targets_keep_their_shape(self) -> None:
        assert _warp().target_shape == (8, 9)

    def test_rejects_a_shape_that_does_not_hold_the_rows(self) -> None:
        stencil = Stencil(
            jnp.zeros((6, 1), dtype=jnp.int32), jnp.ones((6, 1)), n_source=2
        )
        assert Plan(Identity(), stencil, target_shape=(2, 3)).grid_shape == (2, 3)
        with pytest.raises(ValidationError, match=r"'target_shape' \(4, 2\) must hold"):
            Plan(Identity(), stencil, target_shape=(4, 2))


class TestToGaussx:
    def test_round_trip_through_a_dense_matrix(self) -> None:
        warp = _warp()
        indices, weights = warp.to_gaussx()
        assert indices.shape == weights.shape == (72, 4)
        # W from gaussx's layout: row k adds wₖₛ at column iₖₛ
        rows = einx.id("target -> target tap", jnp.arange(72), tap=4)
        dense = jnp.zeros((72, 120)).at[rows, indices].add(weights)
        np.testing.assert_array_equal(dense, warp.read.to_dense())
        np.testing.assert_array_equal(dense, warp.operator().as_matrix())

    def test_invalid_rows_are_zeroed(self) -> None:
        stencil = Stencil(
            jnp.array([[1, 2], [2, 3]]),
            jnp.array([[0.5, 0.5], [0.25, 0.75]]),
            n_source=4,
            valid=jnp.array([True, False]),
        )
        indices, weights = Plan(Identity(), stencil).to_gaussx()
        assert indices.tolist() == [[1, 2], [0, 0]]
        assert weights.tolist() == [[0.5, 0.5], [0.0, 0.0]]

    def test_a_composed_plan_exports_its_product(self) -> None:
        warp = _warp(Policies(outside="clamp"))
        smooth = Linear().plan(DESTINATION, DESTINATION)
        indices, weights = (smooth @ warp).to_gaussx()
        assert indices.shape == (72, 16)
        assert_matches_oracle(weights, (smooth @ warp).read.weights, rtol=0.0)

    def test_rejects_plans_without_fixed_stencil_weights(self) -> None:
        with pytest.raises(PolicyError, match="depend on the data"):
            _warp(Policies(missing="renormalise")).to_gaussx()

        class Doubled(Fit):
            def apply(self, data):  # type: ignore[no-untyped-def]
                return 2.0 * data

            def apply_adjoint(self, coefficients):  # type: ignore[no-untyped-def]
                return 2.0 * coefficients

        with pytest.raises(PolicyError, match="to_gaussx\\(\\) needs a local plan"):
            Plan(Doubled(), _warp().read).to_gaussx()


def test_a_derivative_plan_keeps_the_grid_target_and_windows() -> None:
    # derivative() keeps the plan's target shape, so its rows window too
    source = RectilinearGrid((Axis(jnp.arange(6.0)), Axis(jnp.arange(7.0))))
    target = RectilinearGrid(
        (Axis(jnp.linspace(0.5, 4.5, 5)), Axis(jnp.linspace(0.5, 5.5, 6)))
    )
    derivative = Linear().plan(source, target).derivative(axis=1)
    assert derivative.target_shape == (5, 6)
    window = derivative[slice(1, 3), slice(2, 5)]
    field = jnp.tile(jnp.arange(7.0) * 2.0, 6)
    assert window(field).tolist() == [2.0] * 6
