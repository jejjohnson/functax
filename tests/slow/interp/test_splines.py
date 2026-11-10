"""Tests for `Cubic`, `BSpline` and the `BandedPrefilter`. The slow tier: oracle
sweeps, dot-product tests across policies, gradient and trace checks, each a second
or more (mostly einx compiling new shapes).
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import (
    assert_adjoint,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
    assert_transpose_matches,
)
from scipy.ndimage import map_coordinates, spline_filter1d

from functax import PolicyError
from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import (
    BandedPrefilter,
    BSpline,
    BSplineBasis,
    Cubic,
    Identity,
    Linear,
    Nearest,
    Policies,
    Tensor,
)


def _index_grid(shape: tuple[int, ...], *, period: bool = False) -> RectilinearGrid:
    """Nodes at the integers, so coordinates are index positions."""
    axes = tuple(
        Axis(jnp.arange(float(n)), period=float(n) if period else None) for n in shape
    )
    return RectilinearGrid(axes)


def _flat(field: np.ndarray) -> jax.Array:
    return einx.id("... -> (...)", jnp.asarray(field))


class TestBSplineOracle:
    @pytest.mark.parametrize("order", range(6))
    @pytest.mark.parametrize(
        "shape", [(11,), (7, 8), (6, 5, 7)], ids=["1d", "2d", "3d"]
    )
    def test_equals_map_coordinates_mirror(
        self, order: int, shape: tuple[int, ...]
    ) -> None:
        rng = np.random.default_rng(order + 10 * len(shape))
        field = rng.normal(size=shape)
        targets = np.column_stack([rng.uniform(0.0, n - 1.0, size=150) for n in shape])
        plan = BSpline(order).plan(_index_grid(shape), Points(jnp.asarray(targets)))
        expected = map_coordinates(field, targets.T, order=order, mode="mirror")
        assert_matches_oracle(plan(_flat(field)), expected, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("order", [2, 3, 5])
    def test_a_periodic_axis_equals_grid_wrap(self, order: int) -> None:
        rng = np.random.default_rng(order)
        field = rng.normal(size=12)
        targets = rng.uniform(0.0, 12.0, size=(60, 1))
        axis = Axis(jnp.arange(12.0), period=12.0)
        plan = BSpline(order).plan(axis, Points(jnp.asarray(targets)))
        expected = map_coordinates(field, targets.T, order=order, mode="grid-wrap")
        assert_matches_oracle(
            plan(jnp.asarray(field)), expected, rtol=1e-10, atol=1e-12
        )

    def test_prefilter_equals_spline_filter(self) -> None:
        field = np.random.default_rng(3).normal(size=(4, 9))
        for order in range(2, 6):
            prefilter = BandedPrefilter((order,), (9,), (False,))
            expected = spline_filter1d(field, order=order, axis=-1, mode="mirror")
            assert_matches_oracle(
                prefilter.apply(jnp.asarray(field)), expected, rtol=1e-12, atol=1e-13
            )

    @pytest.mark.parametrize("order", range(2, 6))
    def test_reads_the_samples_back_at_the_knots(self, order: int) -> None:
        shape = (6, 7)
        field = np.random.default_rng(order).normal(size=shape)
        grid = _index_grid(shape)
        plan = BSpline(order).plan(grid, grid)
        assert_matches_oracle(plan(_flat(field)), _flat(field), rtol=1e-12, atol=1e-12)

    def test_cubic_spline_reproduces_cubics_far_from_the_ends(self) -> None:
        nodes = np.arange(60.0)
        field = 0.01 * nodes**3 - 0.4 * nodes**2 + 2.0 * nodes - 1.0
        targets = np.random.default_rng(4).uniform(25.0, 35.0, size=(40, 1))
        plan = BSpline(3).plan(Axis(jnp.asarray(nodes)), Points(jnp.asarray(targets)))
        expected = (
            0.01 * targets[:, 0] ** 3
            - 0.4 * targets[:, 0] ** 2
            + 2.0 * targets[:, 0]
            - 1.0
        )
        # the mirror ends perturb the fit by about |z|²⁵ ≈ 5e-15 of the field here
        assert_matches_oracle(
            plan(jnp.asarray(field)), expected, rtol=1e-10, atol=1e-10
        )


class TestCubic:
    def test_reproduces_quadratics_inside(self) -> None:
        nodes = np.linspace(-2.0, 3.0, 11)
        field = 1.5 * nodes**2 - nodes + 0.25
        # interior: every tap i − 1 … i + 2 is a real node
        targets = np.random.default_rng(5).uniform(nodes[1], nodes[-2], size=(50, 1))
        plan = Cubic().plan(Axis(jnp.asarray(nodes)), Points(jnp.asarray(targets)))
        expected = 1.5 * targets[:, 0] ** 2 - targets[:, 0] + 0.25
        assert_matches_oracle(
            plan(jnp.asarray(field)), expected, rtol=1e-12, atol=1e-12
        )

    def test_the_parameter_reaches_the_weights(self) -> None:
        axis = Axis(jnp.arange(6.0))
        targets = Points(jnp.array([[2.3]]))
        sharper = Cubic(a=-0.75).plan(axis, targets).read.weights
        default = Cubic().plan(axis, targets).read.weights
        assert not bool(jnp.allclose(sharper, default))
        assert_matches_oracle(jnp.sum(sharper), 1.0, rtol=1e-15)


@pytest.mark.parametrize(
    "scheme",
    [Cubic(), BSpline(2), BSpline(3), BSpline(4), BSpline(5)],
    ids=["cubic", "bspline2", "bspline3", "bspline4", "bspline5"],
)
class TestAdjointAndTraces:
    def test_dot_product(self, scheme) -> None:  # type: ignore[no-untyped-def]
        grid = _index_grid((6, 7))
        rng = np.random.default_rng(6)
        targets = np.column_stack([rng.uniform(-1.0, n, size=30) for n in (6, 7)])
        plan = scheme.plan(grid, Points(jnp.asarray(targets)))
        operator = plan.operator()
        assert_adjoint(
            lambda y: operator.mv(y), plan.adjoint, 42, 30, key=jax.random.key(7)
        )
        assert_transpose_matches(lambda y: operator.mv(y), plan.adjoint, jnp.zeros(42))

    def test_traces_and_gradients(self, scheme) -> None:  # type: ignore[no-untyped-def]
        grid = _index_grid((6, 7))
        field = jax.random.normal(jax.random.key(8), (42,))
        rng = np.random.default_rng(9)
        xy = jnp.asarray(
            np.column_stack([rng.uniform(1.2, n - 2.2, size=8) for n in (6, 7)])
        )

        def read(positions: jax.Array) -> jax.Array:
            return scheme.plan(grid, Points(positions))(field)

        assert_traces(read, einx.id("(b m) d -> b m d", xy, b=2))
        assert_grad_matches_fd(read, xy)
        fields = jax.random.normal(jax.random.key(10), (3, 42))
        plan = scheme.plan(grid, Points(xy))
        assert_traces(lambda y: plan(y), fields)


class TestMixedAndPolicies:
    def test_tensor_prefilters_only_its_spline_axes(self) -> None:
        shape = (4, 9)
        field = np.random.default_rng(11).normal(size=shape)
        rng = np.random.default_rng(12)
        frames = rng.integers(0, 4, size=20).astype(float)
        positions = rng.uniform(0.0, 8.0, size=20)
        targets = np.column_stack([frames, positions])
        plan = Tensor((Nearest(), BSpline(3))).plan(
            _index_grid(shape), Points(jnp.asarray(targets))
        )
        assert plan.fit.orders == (None, 3)
        filtered = spline_filter1d(field, order=3, axis=-1, mode="mirror")
        expected = [
            map_coordinates(
                filtered[int(f)], [[p]], order=3, mode="mirror", prefilter=False
            )[0]
            for f, p in zip(frames, positions, strict=True)
        ]
        assert_matches_oracle(
            plan(_flat(field)), np.array(expected), rtol=1e-10, atol=1e-12
        )

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

    def test_fill_runs_before_the_prefilter(self) -> None:
        policies = Policies(missing="fill", fill_value=0.0)
        field = jnp.array([1.0, jnp.nan, 3.0, 4.0, 5.0])
        plan = BSpline(3).plan(
            Axis(jnp.arange(5.0)), Points(jnp.array([[2.0]])), policies=policies
        )
        assert_matches_oracle(plan(field), jnp.array([3.0]), rtol=1e-12)

    def test_renormalise_needs_a_local_plan(self) -> None:
        plan = BSpline(3).plan(
            Axis(jnp.arange(5.0)),
            Points(jnp.zeros((1, 1))),
            policies=Policies(missing="renormalise"),
        )
        with pytest.raises(PolicyError, match="needs a local plan"):
            plan(jnp.ones(5))


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
    def test_a_one_node_axis_is_left_alone(self) -> None:
        prefilter = BandedPrefilter((None, 3), (3, 1), (False, False))
        assert prefilter.apply(jnp.arange(3.0)).tolist() == [0.0, 1.0, 2.0]
