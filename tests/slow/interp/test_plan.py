"""Tests for `Plan`, `Nearest`, `Linear` and `Tensor`. The slow tier:
oracle sweeps, dot-product tests across policies, gradient and trace checks, each a
second or more (mostly einx compiling new shapes).
"""

from __future__ import annotations

import itertools

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
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

from functax.geometry import AffineGrid, Axis, Points, RectilinearGrid
from functax.interp import Linear, Nearest, Policies, Tensor


def _rectilinear(
    shape: tuple[int, ...], seed: int
) -> tuple[RectilinearGrid, list[np.ndarray]]:
    """A grid with random, uneven spacing on every axis."""
    rng = np.random.default_rng(seed)
    nodes = [np.cumsum(rng.uniform(0.5, 1.5, size=n)) for n in shape]
    return RectilinearGrid(tuple(Axis(jnp.asarray(values)) for values in nodes)), nodes


def _interior(nodes: list[np.ndarray], count: int, seed: int) -> np.ndarray:
    """Random targets strictly inside every axis."""
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(n[0], n[-1], size=count) for n in nodes])


def _policy_grid() -> list[Policies]:
    """Every outside × boundary × extent combination except raise."""
    combinations = itertools.product(
        ("nan", "clamp", "extrapolate"),
        ("mirror", "wrap", "edge", "constant"),
        ("nodes", "cells"),
    )
    return [Policies(outside=o, boundary=b, extent=e) for o, b, e in combinations]


class TestOracle:
    @pytest.mark.parametrize(
        ("scheme", "method"),
        [(Linear(), "linear"), (Nearest(), "nearest")],
        ids=["linear", "nearest"],
    )
    @pytest.mark.parametrize(
        "shape", [(9,), (6, 7), (5, 4, 6), (4, 3, 5, 3)], ids=["1d", "2d", "3d", "4d"]
    )
    def test_equals_regular_grid_interpolator(
        self, scheme, method: str, shape: tuple[int, ...]
    ) -> None:  # type: ignore[no-untyped-def]
        grid, nodes = _rectilinear(shape, seed=len(shape))
        field = np.random.default_rng(10 + len(shape)).normal(size=shape)
        targets = _interior(nodes, 200, seed=20 + len(shape))
        plan = scheme.plan(grid, Points(jnp.asarray(targets)))
        flat = einx.id("... -> (...)", jnp.asarray(field))
        expected = RegularGridInterpolator(tuple(nodes), field, method=method)(targets)
        assert_matches_oracle(plan(flat), expected, rtol=1e-12, atol=1e-14)

    def test_linear_reproduces_affine_functions(self) -> None:
        grid, nodes = _rectilinear((5, 6, 4), seed=50)
        slope = np.array([0.7, -1.3, 2.1])
        mesh = np.stack(np.meshgrid(*nodes, indexing="ij"), axis=-1)
        field = mesh @ slope + 4.0
        targets = _interior(nodes, 100, seed=51)
        plan = Linear().plan(grid, Points(jnp.asarray(targets)))
        flat = einx.id("... -> (...)", jnp.asarray(field))
        assert_matches_oracle(plan(flat), targets @ slope + 4.0, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize(
        ("scheme", "order"), [(Linear(), 1), (Nearest(), 0)], ids=["linear", "nearest"]
    )
    @pytest.mark.parametrize(
        "shape", [(9,), (6, 7), (5, 4, 6), (4, 3, 5, 3)], ids=["1d", "2d", "3d", "4d"]
    )
    def test_equals_map_coordinates(
        self, scheme, order: int, shape: tuple[int, ...]
    ) -> None:  # type: ignore[no-untyped-def]
        # integer node coordinates, so grid coordinates are index positions
        grid = RectilinearGrid(tuple(Axis(jnp.arange(float(n))) for n in shape))
        field = np.random.default_rng(30 + len(shape)).normal(size=shape)
        rng = np.random.default_rng(40 + len(shape))
        targets = np.column_stack([rng.uniform(0.0, n - 1.0, size=200) for n in shape])
        plan = scheme.plan(grid, Points(jnp.asarray(targets)))
        expected = map_coordinates(field, targets.T, order=order)
        flat = einx.id("... -> (...)", jnp.asarray(field))
        assert_matches_oracle(plan(flat), expected, rtol=1e-12, atol=1e-14)


class TestAdjoint:
    @pytest.mark.parametrize("scheme", [Linear(), Nearest()], ids=["linear", "nearest"])
    @pytest.mark.parametrize(
        "policies", _policy_grid(), ids=lambda p: f"{p.outside}-{p.boundary}-{p.extent}"
    )
    def test_dot_product_for_every_policy(self, scheme, policies: Policies) -> None:  # type: ignore[no-untyped-def]
        grid, nodes = _rectilinear((5, 6), seed=60)
        rng = np.random.default_rng(61)
        # targets inside, in the half-cell border and outside
        targets = np.column_stack(
            [rng.uniform(n[0] - 2.0, n[-1] + 2.0, size=40) for n in nodes]
        )
        plan = scheme.plan(grid, Points(jnp.asarray(targets)), policies=policies)
        operator = plan.operator()
        assert_adjoint(
            lambda y: operator.mv(y), plan.adjoint, 30, 40, key=jax.random.key(62)
        )

    def test_adjoint_equals_the_lineax_transpose(self) -> None:
        grid, nodes = _rectilinear((4, 5), seed=63)
        plan = Linear().plan(grid, Points(jnp.asarray(_interior(nodes, 12, seed=64))))
        operator = plan.operator()
        assert_transpose_matches(lambda y: operator.mv(y), plan.adjoint, jnp.zeros(20))

    def test_invalid_rows_ignore_their_residual(self) -> None:
        plan = Linear().plan(Axis(jnp.arange(4.0)), Points(jnp.array([[1.5], [9.0]])))
        assert plan.valid.tolist() == [True, False]
        # the NaN the forward pass wrote at the invalid row does not spread
        gradient = plan.adjoint(jnp.array([1.0, jnp.nan]))
        assert gradient.tolist() == [0.0, 0.5, 0.5, 0.0]


class TestTraces:
    def test_apply_over_a_batch_of_fields(self) -> None:
        grid, nodes = _rectilinear((4, 5), seed=70)
        plan = Linear().plan(grid, Points(jnp.asarray(_interior(nodes, 9, seed=71))))
        fields = jax.random.normal(jax.random.key(72), (3, 20))
        assert_traces(lambda y: plan(y), fields)
        assert plan(jnp.ones((2, 3, 20))).shape == (2, 3, 9)

    def test_plan_built_inside_the_trace(self) -> None:
        grid, nodes = _rectilinear((4, 5), seed=73)
        field = jax.random.normal(jax.random.key(74), (20,))
        targets = jnp.asarray(_interior(nodes, 3 * 6, seed=75).reshape(3, 6, 2))

        def read(xy: jax.Array) -> jax.Array:
            return Linear().plan(grid, Points(xy))(field)

        assert_traces(read, targets)

    def test_nearest_has_zero_gradient(self) -> None:
        grid, nodes = _rectilinear((4, 5), seed=79)
        field = jax.random.normal(jax.random.key(80), (20,))
        xy = jnp.asarray(_interior(nodes, 4, seed=81))
        gradient = jax.grad(lambda p: jnp.sum(Nearest().plan(grid, Points(p))(field)))(
            xy
        )
        assert bool(jnp.all(gradient == 0))

    def test_grad_in_the_target_positions(self) -> None:
        grid, nodes = _rectilinear((4, 5), seed=76)
        field = jax.random.normal(jax.random.key(77), (20,))
        xy = jnp.asarray(_interior(nodes, 6, seed=78))

        def read(positions: jax.Array) -> jax.Array:
            return Linear().plan(grid, Points(positions))(field)

        assert_grad_matches_fd(read, xy)


class TestSchemes:
    def test_tensor_is_nearest_in_time_then_bilinear_in_space(self) -> None:
        time = Axis(jnp.arange(4.0))
        grid, nodes = _rectilinear((5, 6), seed=90)
        cube = RectilinearGrid((time, *grid.axes))
        field = jax.random.normal(jax.random.key(91), (4, 5, 6))
        rng = np.random.default_rng(92)
        space = _interior(nodes, 30, seed=93)
        when = rng.uniform(0.0, 3.0, size=30)
        targets = jnp.asarray(np.column_stack([when, space]))
        mixed = Tensor((Nearest(), Linear(), Linear())).plan(cube, Points(targets))
        values = mixed(einx.id("t y x -> (t y x)", field))
        # nearest frame (ties down), then bilinear in that frame
        frame = np.where(
            when - np.floor(when) > 0.5, np.ceil(when), np.floor(when)
        ).astype(int)
        bilinear = Linear().plan(grid, Points(jnp.asarray(space)))
        expected = [
            bilinear(einx.id("y x -> (y x)", field[f]))[k] for k, f in enumerate(frame)
        ]
        assert_matches_oracle(values, jnp.stack(expected), rtol=1e-12, atol=1e-14)

    def test_periodic_longitude_reads_across_the_dateline(self) -> None:
        longitude = Axis(jnp.arange(0.0, 360.0, 10.0), period=360.0)
        field = jnp.arange(36.0)
        plan = Linear().plan(longitude, Points(jnp.array([[355.0], [-5.0], [715.0]])))
        # halfway between node 35 (350°) and node 0 (360° ≡ 0°)
        assert plan(field).tolist() == [17.5, 17.5, 17.5]

    def test_targets_on_grids_and_axes(self) -> None:
        grid, nodes = _rectilinear((3, 4), seed=95)
        field = jax.random.normal(jax.random.key(96), (12,))
        # a grid read at its own nodes is the field itself
        assert_matches_oracle(
            Linear().plan(grid, grid)(field), field, rtol=1e-14, atol=1e-15
        )
        axis = Axis(jnp.asarray(nodes[0]))
        assert_matches_oracle(
            Linear().plan(axis, axis)(field[:3]), field[:3], rtol=1e-14, atol=1e-15
        )
        affine = AffineGrid([2.0, 0.0, 10.0, 0.0, -2.0, 30.0], (3, 4))
        assert_matches_oracle(
            Linear().plan(affine, affine)(field), field, rtol=1e-14, atol=1e-15
        )


class TestMissing:
    axis = Axis(jnp.arange(4.0))
    field = jnp.array([1.0, jnp.nan, 4.0, 8.0])
    targets = Points(jnp.array([[0.25], [2.5], [1.0]]))

    def test_renormalise(self) -> None:
        plan = Linear().plan(
            self.axis, self.targets, policies=Policies(missing="renormalise")
        )
        values = plan(self.field)
        # 0.25: the NaN tap is dropped and the row divided by the weight left
        assert values[:2].tolist() == [1.0, 6.0]
        # 1.0 sits on the missing node, so no weight is left: the fill value
        assert bool(jnp.isnan(values[2]))

    def test_renormalise_with_every_tap_missing_is_invalid(self) -> None:
        field = jnp.array([jnp.nan, jnp.nan, 4.0, 8.0])
        plan = Linear().plan(
            self.axis,
            Points(jnp.array([[0.5], [2.5]])),
            policies=Policies(missing="renormalise", fill_value=-1.0),
        )
        assert plan(field).tolist() == [-1.0, 6.0]

    def test_renormalise_over_a_batch(self) -> None:
        plan = Linear().plan(
            self.axis, self.targets, policies=Policies(missing="renormalise")
        )
        batch = jnp.stack([self.field, jnp.array([1.0, 2.0, 4.0, 8.0])])
        values = plan(batch)
        assert values.shape == (2, 3)
        assert values[1].tolist() == [1.25, 6.0, 2.0]


class TestOutside:
    axis = Axis(jnp.arange(4.0))
    field = jnp.array([1.0, 2.0, 4.0, 8.0])
    targets = Points(jnp.array([[-1.0], [1.5], [5.0]]))

    def test_nan_writes_the_fill_value(self) -> None:
        plan = Linear().plan(self.axis, self.targets)
        values = plan(self.field)
        assert bool(jnp.isnan(values[0])) and bool(jnp.isnan(values[2]))
        filled = Linear().plan(
            self.axis, self.targets, policies=Policies(fill_value=-9.0)
        )
        assert filled(self.field).tolist() == [-9.0, 3.0, -9.0]

    def test_non_finite_targets_are_invalid_under_every_policy(self) -> None:
        targets = Points(jnp.array([[jnp.nan], [1.0]]))
        for outside in ("nan", "clamp", "extrapolate"):
            plan = Linear().plan(self.axis, targets, policies=Policies(outside=outside))
            assert plan.valid.tolist() == [False, True]
