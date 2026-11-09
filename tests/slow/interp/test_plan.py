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
    assert_matches_oracle,
    assert_traces,
)
from scipy.interpolate import RegularGridInterpolator

from functax.geometry import Axis, Points, RectilinearGrid
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
