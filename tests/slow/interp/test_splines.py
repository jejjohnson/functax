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

from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import (
    BandedPrefilter,
    BSpline,
    Cubic,
    Nearest,
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
