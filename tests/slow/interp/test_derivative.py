"""Tests for `Plan.derivative`, `DerivativePlan` and the sum of plans. The slow tier:
oracle sweeps, dot-product tests across policies, gradient and trace checks, each a
second or more (mostly einx compiling new shapes).
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.interpolate
from _harness import assert_adjoint, assert_matches_oracle, assert_transpose_matches
from scipy.ndimage import spline_filter1d

from functax.geometry import AffineGrid, Axis, Points, RectilinearGrid
from functax.interp import (
    BSpline,
    Cubic,
    DerivativePlan,
    Linear,
    Nearest,
    Plan,
    Policies,
)


def _scipy_spline(samples: np.ndarray, order: int) -> scipy.interpolate.BSpline:
    """SciPy's evaluation of the same spline: its own mirror-prefiltered coefficients.

    f(u) = ∑ⱼ cⱼ βᵏ(u − j) on unit knots, with c from ``spline_filter1d`` and
    extended past the ends by the mirror rule c₋ⱼ = cⱼ, c_{N−1+j} = c_{N−1−j}.
    """
    n = samples.size
    coefficients = spline_filter1d(samples, order=order, mode="mirror")
    pad = order + 2
    left = coefficients[1 : pad + 1][::-1]
    right = coefficients[-pad - 1 : -1][::-1]
    extended = np.concatenate([left, coefficients, right])
    # centred knots: the basis βᵏ(u − j) has knots j − (k + 1)/2 + l
    first_centre = -pad
    knots = np.arange(extended.size + order + 1) + first_centre - (order + 1) / 2
    del n
    return scipy.interpolate.BSpline(knots, extended, order, extrapolate=False)


class TestOracle:
    @pytest.mark.parametrize("order", [3, 4, 5])
    @pytest.mark.parametrize("nu", [1, 2])
    def test_bspline_derivatives_equal_scipy(self, order: int, nu: int) -> None:
        rng = np.random.default_rng(order + 10 * nu)
        samples = rng.normal(size=14)
        targets = rng.uniform(0.0, 13.0, size=40)
        axis = Axis(jnp.arange(14.0))
        plan = BSpline(order).plan(axis, Points(jnp.asarray(targets)[:, None]))
        derivative = plan.derivative(axis=0, order=nu)
        expected = _scipy_spline(samples, order).derivative(nu)(targets)
        assert_matches_oracle(
            derivative(jnp.asarray(samples)), expected, rtol=1e-10, atol=1e-11
        )

    def test_the_chain_rule_through_a_regular_spacing(self) -> None:
        rng = np.random.default_rng(1)
        samples = rng.normal(size=12)
        spacing = 0.25
        nodes = 3.0 + spacing * np.arange(12.0)
        targets = rng.uniform(nodes[0], nodes[-1], size=20)
        plan = BSpline(3).plan(
            Axis(jnp.asarray(nodes)), Points(jnp.asarray(targets)[:, None])
        )
        index_targets = (targets - 3.0) / spacing
        # d/dx = (1/Δ) d/du
        expected = _scipy_spline(samples, 3).derivative(1)(index_targets) / spacing
        assert_matches_oracle(
            plan.derivative(axis=0)(jnp.asarray(samples)),
            expected,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_linear_slope_is_the_cell_slope_on_a_rectilinear_axis(self) -> None:
        nodes = np.array([0.0, 1.0, 3.0, 7.0])
        samples = np.array([2.0, 4.0, 5.0, 13.0])
        plan = Linear().plan(
            Axis(jnp.asarray(nodes)), Points(jnp.array([[0.5], [2.0], [5.0]]))
        )
        assert plan.derivative(axis=0)(jnp.asarray(samples)).tolist() == [2.0, 0.5, 2.0]

    def test_derivatives_of_reproduced_polynomials_are_exact(self) -> None:
        grid = RectilinearGrid(
            (Axis(jnp.linspace(-1.0, 2.0, 7)), Axis(jnp.linspace(0.0, 3.0, 9)))
        )
        mesh_y, mesh_x = np.meshgrid(
            np.linspace(-1.0, 2.0, 7), np.linspace(0.0, 3.0, 9), indexing="ij"
        )
        targets = np.column_stack(
            [
                np.random.default_rng(2).uniform(0.0, 1.0, size=15),
                np.random.default_rng(3).uniform(0.6, 2.4, size=15),
            ]
        )
        # linear: an affine field has constant slopes
        affine = 1.5 * mesh_y - 0.75 * mesh_x + 2.0
        linear = Linear().plan(grid, Points(jnp.asarray(targets)))
        flat = einx.id("y x -> (y x)", jnp.asarray(affine))
        assert_matches_oracle(
            linear.derivative(axis=0)(flat), np.full(15, 1.5), rtol=1e-12
        )
        assert_matches_oracle(
            linear.derivative(axis=-1)(flat), np.full(15, -0.75), rtol=1e-12
        )
        # cubic: a quadratic in x has slope 2x, and second derivative 2
        quadratic = einx.id("y x -> (y x)", jnp.asarray(mesh_x**2))
        cubic = Cubic().plan(grid, Points(jnp.asarray(targets)))
        assert_matches_oracle(
            cubic.derivative(axis=1)(quadratic),
            2.0 * targets[:, 1],
            rtol=1e-12,
            atol=1e-12,
        )
        assert_matches_oracle(
            cubic.derivative(axis=1, order=2)(quadratic),
            np.full(15, 2.0),
            rtol=1e-11,
            atol=1e-11,
        )
        assert_matches_oracle(
            cubic.derivative(axis=0)(quadratic), np.zeros(15), rtol=0.0, atol=1e-12
        )

    def test_a_rotated_affine_grid_differentiates_in_map_coordinates(self) -> None:
        # rotated by 30°, so ∂/∂x mixes rows and columns
        angle = np.pi / 6
        transform = [
            np.cos(angle),
            -np.sin(angle),
            5.0,
            np.sin(angle),
            np.cos(angle),
            -2.0,
        ]
        grid = AffineGrid(transform, (8, 9))
        rows, cols = np.meshgrid(
            np.arange(8.0) + 0.5, np.arange(9.0) + 0.5, indexing="ij"
        )
        x = transform[0] * cols + transform[1] * rows + transform[2]
        y = transform[3] * cols + transform[4] * rows + transform[5]
        field = einx.id("r c -> (r c)", jnp.asarray(3.0 * x - 2.0 * y))
        centre = Points(jnp.array([[x[4, 4], y[4, 4]], [x[3, 5], y[3, 5]]]) + 0.1)
        plan = Linear().plan(grid, centre)
        assert_matches_oracle(
            plan.derivative(axis=0)(field), np.full(2, 3.0), rtol=1e-12
        )
        assert_matches_oracle(
            plan.derivative(axis=1)(field), np.full(2, -2.0), rtol=1e-12
        )


class TestFiniteDifferences:
    @pytest.mark.parametrize(
        "scheme",
        [Cubic(), BSpline(3), BSpline(5)],
        ids=["cubic", "bspline3", "bspline5"],
    )
    def test_agrees_with_central_differences_away_from_knots(self, scheme) -> None:  # type: ignore[no-untyped-def]
        grid = RectilinearGrid((Axis(jnp.arange(7.0)), Axis(jnp.linspace(0.0, 4.0, 9))))
        field = jax.random.normal(jax.random.key(4), (63,))
        rng = np.random.default_rng(5)
        # offsets in (0.1, 0.9) of a cell: away from the knots
        targets = np.column_stack(
            [
                rng.integers(1, 5, 10) + rng.uniform(0.1, 0.9, 10),
                0.5 * (rng.integers(1, 7, 10) + rng.uniform(0.1, 0.9, 10)),
            ]
        )
        step = 1e-6
        for axis in (0, 1):
            plan = scheme.plan(grid, Points(jnp.asarray(targets)))
            shift = np.zeros(2)
            shift[axis] = step
            ahead = scheme.plan(grid, Points(jnp.asarray(targets + shift)))(field)
            behind = scheme.plan(grid, Points(jnp.asarray(targets - shift)))(field)
            central = (ahead - behind) / (2.0 * step)
            assert_matches_oracle(
                plan.derivative(axis=axis)(field), central, rtol=1e-6, atol=1e-6
            )

    def test_grad_through_traced_targets_equals_the_derivative_plan(self) -> None:
        grid = RectilinearGrid((Axis(jnp.arange(6.0)), Axis(jnp.arange(7.0))))
        field = jax.random.normal(jax.random.key(6), (42,))
        xy = jnp.asarray(np.random.default_rng(7).uniform(1.2, 4.6, size=(5, 2)))
        scheme = BSpline(3)

        def read(positions: jax.Array) -> jax.Array:
            return jnp.sum(scheme.plan(grid, Points(positions))(field))

        gradient = jax.jit(jax.grad(read))(xy)
        plan = scheme.plan(grid, Points(xy))
        for axis in (0, 1):
            expected = plan.derivative(axis=axis)(field)
            assert_matches_oracle(gradient[:, axis], expected, rtol=1e-12, atol=1e-13)


class TestLinearity:
    @pytest.mark.parametrize(
        ("scheme", "order"),
        [
            (Linear(), 1),
            (Cubic(), 1),
            (Cubic(), 2),
            (BSpline(3), 1),
            (BSpline(3), 2),
            (BSpline(5), 3),
        ],
        ids=["linear1", "cubic1", "cubic2", "bspline3-1", "bspline3-2", "bspline5-3"],
    )
    def test_dot_product(self, scheme, order: int) -> None:  # type: ignore[no-untyped-def]
        grid = RectilinearGrid((Axis(jnp.arange(6.0)), Axis(jnp.arange(7.0))))
        rng = np.random.default_rng(8)
        targets = np.column_stack(
            [rng.uniform(-0.5, 5.5, 25), rng.uniform(-0.5, 6.5, 25)]
        )
        derivative = scheme.plan(grid, Points(jnp.asarray(targets))).derivative(
            axis=1, order=order
        )
        assert isinstance(derivative, DerivativePlan)
        assert (derivative.axis, derivative.order) == (1, order)
        operator = derivative.operator()
        assert_adjoint(
            lambda y: operator.mv(y), derivative.adjoint, 42, 25, key=jax.random.key(9)
        )
        assert_transpose_matches(
            lambda y: operator.mv(y), derivative.adjoint, jnp.zeros(42)
        )

    def test_the_laplacian_is_a_sum_of_plans(self) -> None:
        grid = RectilinearGrid(
            (Axis(jnp.linspace(0.0, 2.0, 9)), Axis(jnp.linspace(0.0, 3.0, 10)))
        )
        mesh_y, mesh_x = np.meshgrid(
            np.linspace(0.0, 2.0, 9), np.linspace(0.0, 3.0, 10), indexing="ij"
        )
        # ∇²(x² + 3y²) = 2 + 6
        field = einx.id("y x -> (y x)", jnp.asarray(mesh_x**2 + 3.0 * mesh_y**2))
        targets = Points(
            jnp.asarray(
                np.random.default_rng(10).uniform([0.5, 0.6], [1.5, 2.4], size=(12, 2))
            )
        )
        plan = Cubic().plan(grid, targets)
        laplacian = plan.derivative(axis=-1, order=2) + plan.derivative(
            axis=-2, order=2
        )
        assert isinstance(laplacian, Plan)
        assert laplacian.read.n_taps == 2 * plan.read.n_taps
        assert_matches_oracle(
            laplacian(field), np.full(12, 8.0), rtol=1e-10, atol=1e-10
        )
        operator = laplacian.operator()
        assert_adjoint(
            lambda y: operator.mv(y), laplacian.adjoint, 90, 12, key=jax.random.key(11)
        )


class TestPolicies:
    def test_nearest_has_zero_derivative(self) -> None:
        plan = Nearest().plan(Axis(jnp.arange(5.0)), Points(jnp.array([[1.3], [2.8]])))
        assert plan.derivative(axis=0)(jnp.arange(5.0) ** 2).tolist() == [0.0, 0.0]

    def test_clamp_gives_zero_slope_outside(self) -> None:
        plan = Linear().plan(
            Axis(jnp.arange(5.0)),
            Points(jnp.array([[9.0]])),
            policies=Policies(outside="clamp"),
        )
        assert plan.derivative(axis=0)(jnp.arange(5.0) ** 2).tolist() == [0.0]
