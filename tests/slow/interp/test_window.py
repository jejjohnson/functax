"""Tests for `WindowedPlan` (``plan[window]``) and `Plan.to_gaussx`. The slow tier:
oracle sweeps, dot-product tests across policies, gradient and trace checks, each a
second or more (mostly einx compiling new shapes).
"""

from __future__ import annotations

import inspect

import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_adjoint, assert_matches_oracle

from functax import ValidationError
from functax.geometry import AffineGrid, Axis, Points, RectilinearGrid
from functax.interp import (
    Linear,
    Plan,
    Policies,
    WindowedPlan,
)


SOURCE = AffineGrid([1.0, 0.0, 0.0, 0.0, -1.0, 10.0], (10, 12))


DESTINATION = AffineGrid([1.4, 0.0, 1.3, 0.0, -1.3, 9.6], (8, 9))


def _warp(policies: Policies | None = None) -> Plan:
    return Linear().plan(SOURCE, DESTINATION, policies=policies)


def _as_raster(values: jax.Array) -> np.ndarray:
    """The flat destination rows as an (…, 8, 9) raster."""
    return np.asarray(einx.id("... (row col) -> ... row col", values, row=8))


class TestWindow:
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

    def test_a_composed_plan_exports_its_product(self) -> None:
        warp = _warp(Policies(outside="clamp"))
        smooth = Linear().plan(DESTINATION, DESTINATION)
        indices, weights = (smooth @ warp).to_gaussx()
        assert indices.shape == (72, 16)
        assert_matches_oracle(weights, (smooth @ warp).read.weights, rtol=0.0)


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


class TestTargetShape:
    def test_grid_targets_keep_their_shape(self) -> None:
        assert _warp().target_shape == (8, 9)
