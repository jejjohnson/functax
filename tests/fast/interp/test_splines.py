"""Tests for `Cubic`, `BSpline` and the `BandedPrefilter`. The fast tier: behaviour,
validation and small closed-form checks, each well under a second.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle
from scipy.interpolate import BSpline as ScipyBSpline

from functax import PolicyError, ValidationError
from functax.geometry import Axis, Points
from functax.interp import (
    BandedPrefilter,
    BSpline,
    BSplineBasis,
    Policies,
    bspline,
    spline_poles,
)


class TestMixedAndPolicies:
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


def _centred_basis(order: int) -> ScipyBSpline:
    """SciPy's B-spline basis element on the k + 2 unit knots centred on zero."""
    knots = np.arange(order + 2) - (order + 1) / 2
    return ScipyBSpline.basis_element(knots, extrapolate=False)


class TestBSplineFunction:
    @pytest.mark.parametrize("order", range(6))
    def test_equals_scipy_basis_element(self, order: int) -> None:
        half_width = (order + 1) / 2
        x = np.random.default_rng(order).uniform(
            -half_width - 1.0, half_width + 1.0, size=200
        )
        # SciPy writes NaN off the support, where βᵏ is zero; the truncated
        # powers cancel to about 2e-13 just past x = (k + 1)/2 at k = 5
        expected = np.nan_to_num(_centred_basis(order)(x), nan=0.0)
        assert_matches_oracle(
            bspline(jnp.asarray(x), order), expected, rtol=1e-12, atol=1e-12
        )


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
