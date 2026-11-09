"""Tests for `Cubic`, `BSpline` and the `BandedPrefilter`. The fast tier: behaviour,
validation and small closed-form checks, each well under a second.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from _harness import (
    assert_matches_oracle,
)

from functax import PolicyError, ValidationError
from functax.geometry import Axis, Points
from functax.interp import (
    BandedPrefilter,
    BSpline,
    BSplineBasis,
    Cubic,
    Identity,
    Linear,
    Policies,
    spline_poles,
)


class TestCubic:
    def test_the_parameter_reaches_the_weights(self) -> None:
        axis = Axis(jnp.arange(6.0))
        targets = Points(jnp.array([[2.3]]))
        sharper = Cubic(a=-0.75).plan(axis, targets).read.weights
        default = Cubic().plan(axis, targets).read.weights
        assert not bool(jnp.allclose(sharper, default))
        assert_matches_oracle(jnp.sum(sharper), 1.0, rtol=1e-15)


class TestMixedAndPolicies:
    def test_low_orders_need_no_prefilter(self) -> None:
        axis = Axis(jnp.arange(5.0))
        targets = Points(jnp.array([[1.25], [3.5]]))
        for order in (0, 1):
            assert isinstance(BSpline(order).plan(axis, targets).fit, Identity)
        linear = Linear().plan(axis, targets)(jnp.arange(5.0) ** 2)
        assert_matches_oracle(
            BSpline(1).plan(axis, targets)(jnp.arange(5.0) ** 2), linear, rtol=1e-15
        )

    def test_the_prefilter_needs_mirror_or_wrap(self) -> None:
        with pytest.raises(PolicyError, match="mirror or the wrap boundary"):
            BSpline(3).plan(
                Axis(jnp.arange(5.0)),
                Points(jnp.zeros((1, 1))),
                policies=Policies(boundary="edge"),
            )
        # wrap on a non-periodic axis prefilters periodically
        plan = BSpline(3).plan(
            Axis(jnp.arange(5.0)),
            Points(jnp.zeros((1, 1))),
            policies=Policies(boundary="wrap"),
        )
        assert plan.fit.periodic == (True,)

    def test_renormalise_needs_a_local_plan(self) -> None:
        plan = BSpline(3).plan(
            Axis(jnp.arange(5.0)),
            Points(jnp.zeros((1, 1))),
            policies=Policies(missing="renormalise"),
        )
        with pytest.raises(PolicyError, match="needs a local plan"):
            plan(jnp.ones(5))

    def test_fill_runs_before_the_prefilter(self) -> None:
        policies = Policies(missing="fill", fill_value=0.0)
        field = jnp.array([1.0, jnp.nan, 3.0, 4.0, 5.0])
        plan = BSpline(3).plan(
            Axis(jnp.arange(5.0)), Points(jnp.array([[2.0]])), policies=policies
        )
        assert_matches_oracle(plan(field), jnp.array([3.0]), rtol=1e-12)


class TestBasisShape:
    @pytest.mark.parametrize(
        ("order", "support"),
        [
            (0, (0, 1)),
            (1, (0, 1)),
            (2, (-1, 2)),
            (3, (-1, 2)),
            (4, (-2, 3)),
            (5, (-2, 3)),
        ],
    )
    def test_taps_and_support(self, order: int, support: tuple[int, int]) -> None:
        basis = BSplineBasis(order)
        assert basis.taps == order + 1
        assert basis.support == support
        taps, _ = basis.stencil(jnp.linspace(0.0, 0.999, 9))
        assert int(jnp.min(taps)) >= support[0]
        assert int(jnp.max(taps)) <= support[1]

    def test_extrapolate_keeps_every_tap_on_the_axis(self) -> None:
        axis = Axis(jnp.arange(8.0))
        plan = BSpline(3).plan(
            axis,
            Points(jnp.array([[-2.0], [11.0]])),
            policies=Policies(outside="extrapolate"),
        )
        assert bool(jnp.all(plan.valid))
        # cells clipped to [1, N − 3]: the taps never leave [0, N − 1]
        assert int(jnp.min(plan.read.indices)) == 0
        assert int(jnp.max(plan.read.indices)) == 7


class TestValidation:
    @pytest.mark.parametrize("order", [-1, 6])
    def test_rejects_orders_outside_0_to_5(self, order: int) -> None:
        with pytest.raises(
            ValidationError, match=r"'order' must be an integer in \[0, 5\]"
        ):
            BSpline(order)
        with pytest.raises(
            ValidationError, match=r"'order' must be an integer in \[0, 5\]"
        ):
            spline_poles(order)

    def test_basis_rejects_a_bool_order(self) -> None:
        basis = object.__new__(BSplineBasis)
        object.__setattr__(basis, "order", True)
        with pytest.raises(ValidationError, match="'order' must be an integer"):
            basis.__check_init__()

    def test_prefilter_rejects_mismatched_axes(self) -> None:
        with pytest.raises(ValidationError, match="one entry per axis"):
            BandedPrefilter((3,), (5, 6), (False, False))

    def test_a_one_node_axis_is_left_alone(self) -> None:
        prefilter = BandedPrefilter((None, 3), (3, 1), (False, False))
        assert prefilter.apply(jnp.arange(3.0)).tolist() == [0.0, 1.0, 2.0]
