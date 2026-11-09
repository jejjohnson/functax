"""Tests for `WindowedPlan` (``plan[window]``) and `Plan.to_gaussx`. The fast tier:
behaviour, validation and small closed-form checks, each well under a second.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import pytest
from _harness import assert_matches_oracle

from functax import PolicyError, ValidationError
from functax.geometry import AffineGrid
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


SOURCE = AffineGrid([1.0, 0.0, 0.0, 0.0, -1.0, 10.0], (10, 12))


DESTINATION = AffineGrid([1.4, 0.0, 1.3, 0.0, -1.3, 9.6], (8, 9))


def _warp(policies: Policies | None = None) -> Plan:
    return Linear().plan(SOURCE, DESTINATION, policies=policies)


class TestWindow:
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
