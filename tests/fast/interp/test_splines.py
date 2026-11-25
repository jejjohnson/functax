"""Tests for `Cubic`, `BSpline` and the `BandedPrefilter`. The fast tier: behaviour,
validation and small closed-form checks, each well under a second.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle
from scipy.interpolate import BSpline as ScipyBSpline

from functax import GeometryError, PolicyError, ValidationError
from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import (
    BandedPrefilter,
    BSpline,
    BSplineBasis,
    Cubic,
    Linear,
    LocalScheme,
    Nearest,
    Policies,
    Tensor,
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


class TestUnevenSpacing:
    """Kernels wider than a cell read in index space, so they refuse uneven axes
    unless the policies ask for it (#163)."""

    uneven = Axis(jnp.array([0.0, 0.1, 0.3, 1.0, 1.2, 2.5, 3.0, 4.0]))
    targets = Points(jnp.linspace(0.3, 2.5, 41)[:, None])

    @pytest.mark.parametrize(
        "scheme", [Cubic(), BSpline(2), BSpline(3)], ids=["cubic", "b2", "b3"]
    )
    def test_refuses_an_uneven_axis(self, scheme: LocalScheme) -> None:
        with pytest.raises(GeometryError, match=r"axis 0: .* from 0\.1 to 1\.3"):
            scheme.plan(self.uneven, self.targets)
        with pytest.raises(GeometryError, match="CubicSpline or Pchip"):
            scheme.fit(self.uneven, self.uneven.values)

    @pytest.mark.parametrize("scheme", [Cubic(), BSpline(3)], ids=["cubic", "b3"])
    def test_index_space_is_an_opt_in(self, scheme: LocalScheme) -> None:
        policies = Policies(spacing="index")
        values = scheme.fit(self.uneven, self.uneven.values, policies=policies)(
            self.targets
        )
        # away from the ends f(x) = x is not reproduced: measured 0.066 and 0.082
        error = float(jnp.max(jnp.abs(values - self.targets.xy[:, 0])))
        assert 0.05 < error < 0.1

    @pytest.mark.parametrize(
        "scheme",
        [Nearest(), Linear(), BSpline(0), BSpline(1)],
        ids=["nearest", "linear", "b0", "b1"],
    )
    def test_kernels_within_a_cell_follow_the_coordinates(
        self, scheme: LocalScheme
    ) -> None:
        values = scheme.plan(self.uneven, self.targets)(self.uneven.values)
        if isinstance(scheme, Linear) or scheme == BSpline(1):
            assert_matches_oracle(values, self.targets.xy[:, 0], rtol=1e-14)

    def test_names_the_uneven_axis_of_a_grid(self) -> None:
        grid = RectilinearGrid((Axis(jnp.arange(5.0)), self.uneven))
        targets = Points(jnp.array([[1.5, 1.1]]))
        with pytest.raises(GeometryError, match="axis 1"):
            Cubic().plan(grid, targets)
        # Tensor checks each axis with its own scheme
        Tensor((Cubic(), Linear())).plan(grid, targets)

    @pytest.mark.parametrize(
        "nodes",
        [
            jnp.linspace(0.0, 1.0, 10_001),
            jnp.linspace(0.0, 1.0, 100, dtype=jnp.float32),
            jnp.asarray(1.7e9 + 3600.0 * np.arange(48), dtype=jnp.float32),
            jnp.asarray(np.float32(0.1 * np.arange(3600)), dtype=jnp.float64),
            jnp.linspace(4.0, 0.0, 9),
            jnp.array([0.0, 3.0]),
        ],
        ids=["long", "float32", "epoch", "float32-in-64", "descending", "two"],
    )
    def test_rounding_is_not_uneven(self, nodes: jax.Array) -> None:
        # Axis.is_regular, to 1e-12, is false for the first four
        Cubic().plan(Axis(nodes), Points(nodes[:1, None]))

    def test_a_periodic_axis_counts_its_wrap_gap(self) -> None:
        targets = Points(jnp.array([[355.0]]))
        for step in (10.0, -10.0):
            even = Axis(jnp.arange(0.0, 36 * step, step), period=360.0)
            Cubic().plan(even, targets)
        # one node short of the circle, so closed with a gap of two spacings
        gap = Axis(jnp.arange(0.0, 350.0, 10.0), period=360.0)
        with pytest.raises(GeometryError, match="from 10 to 20"):
            Cubic().plan(gap, targets)

    def test_an_open_axis_has_no_wrap_gap(self) -> None:
        region = Axis(jnp.arange(0.0, 340.0, 10.0), period=360.0)
        Cubic().plan(region, Points(jnp.array([[-35.0]])))

    def test_a_traced_axis_is_read_in_index_space(self) -> None:
        def read(nodes: jax.Array) -> jax.Array:
            return Cubic().plan(Axis(nodes), self.targets)(nodes)

        policies = Policies(spacing="index")
        plan = Cubic().plan(self.uneven, self.targets, policies=policies)
        expected = plan(self.uneven.values)
        assert jnp.allclose(jax.jit(read)(self.uneven.values), expected, rtol=1e-14)

    def test_an_unknown_option_is_refused(self) -> None:
        with pytest.raises(PolicyError, match="'spacing' must be one of"):
            Policies(spacing="uneven")


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
