"""Tests for `locate_curvilinear` and `Linear` and `Nearest` on a `CurvilinearGrid`."""

from __future__ import annotations

import inspect
import itertools

import hypothesis.strategies as st
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
from hypothesis import given, settings

from functax import GeometryError, OutsideDomainError, PolicyError, ValidationError
from functax.geometry import (
    AffineGrid,
    Axis,
    CurvilinearGrid,
    Points,
    RectilinearGrid,
    from_pixel,
    pixel_centres,
)
from functax.interp import Linear, Nearest, Policies, locate_curvilinear


# a rotated, sheared affine raster, 6 × 7 pixels
AFFINE = AffineGrid([2.0, 0.6, 100.0, 0.4, -1.5, 50.0], (6, 7))


def _affine_in_disguise() -> CurvilinearGrid:
    """`AFFINE` with its pixel centres given node by node."""
    x, y = pixel_centres(AFFINE)
    return CurvilinearGrid(x=jnp.asarray(x), y=jnp.asarray(y))


def _warped_nodes(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """A rotated, sheared and gently curved grid, the x and y of every node."""
    rows, cols = np.meshgrid(
        np.arange(float(shape[0])), np.arange(float(shape[1])), indexing="ij"
    )
    angle = 0.4
    x = np.cos(angle) * cols - np.sin(angle) * rows + 0.3 * rows
    x = x + 0.15 * np.sin(0.7 * rows + 0.3 * cols)
    y = np.sin(angle) * cols + np.cos(angle) * rows + 0.013 * cols**2
    return x, y


def _warped(shape: tuple[int, int] = (8, 9)) -> CurvilinearGrid:
    x, y = _warped_nodes(shape)
    return CurvilinearGrid(x=jnp.asarray(x), y=jnp.asarray(y))


def _bilinear(
    nodes: np.ndarray, row: np.ndarray, col: np.ndarray, s: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """x(s, t) of cell (row, col), in NumPy."""
    return (
        (1 - s) * (1 - t) * nodes[row, col]
        + s * (1 - t) * nodes[row + 1, col]
        + (1 - s) * t * nodes[row, col + 1]
        + s * t * nodes[row + 1, col + 1]
    )


def _warped_targets(
    shape: tuple[int, int], count: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Targets made from known (row + s, col + t), and those index positions."""
    x, y = _warped_nodes(shape)
    rng = np.random.default_rng(seed)
    row = rng.integers(0, shape[0] - 1, size=count)
    col = rng.integers(0, shape[1] - 1, size=count)
    s = rng.uniform(0.0, 1.0, size=count)
    t = rng.uniform(0.0, 1.0, size=count)
    targets = np.column_stack(
        [_bilinear(x, row, col, s, t), _bilinear(y, row, col, s, t)]
    )
    return targets, np.column_stack([row + s, col + t])


def _mixed_affine_targets(count: int, seed: int) -> jax.Array:
    """(x, y) of targets inside AFFINE, in its half-pixel border and outside."""
    rng = np.random.default_rng(seed)
    rowcol = np.column_stack(
        [rng.uniform(-3.0, 8.0, size=count), rng.uniform(-3.0, 9.0, size=count)]
    )
    return from_pixel(AFFINE.transform, jnp.asarray(rowcol))


def _policy_grid() -> list[Policies]:
    """Every outside × boundary × extent combination except raise."""
    combinations = itertools.product(
        ("nan", "clamp", "extrapolate"),
        ("mirror", "wrap", "edge", "constant"),
        ("nodes", "cells"),
    )
    return [Policies(outside=o, boundary=b, extent=e) for o, b, e in combinations]


class TestInDisguise:
    def _assert_equal_plans(self, scheme, policies: Policies) -> None:  # type: ignore[no-untyped-def]
        targets = Points(_mixed_affine_targets(300, seed=0))
        field = jax.random.normal(jax.random.key(1), (42,))
        affine = scheme.plan(AFFINE, targets, policies=policies)
        swath = scheme.plan(_affine_in_disguise(), targets, policies=policies)
        np.testing.assert_array_equal(swath.valid, affine.valid)
        assert_matches_oracle(swath(field), affine(field), rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("scheme", [Linear(), Nearest()], ids=["linear", "nearest"])
    def test_an_affine_grid_with_default_policies(self, scheme) -> None:  # type: ignore[no-untyped-def]
        self._assert_equal_plans(scheme, Policies())

    @pytest.mark.slow
    @pytest.mark.parametrize("scheme", [Linear(), Nearest()], ids=["linear", "nearest"])
    @pytest.mark.parametrize(
        "policies", _policy_grid(), ids=lambda p: f"{p.outside}-{p.boundary}-{p.extent}"
    )
    def test_an_affine_grid_under_every_policy(
        self, scheme, policies: Policies
    ) -> None:  # type: ignore[no-untyped-def]
        self._assert_equal_plans(scheme, policies)

    @pytest.mark.parametrize("scheme", [Linear(), Nearest()], ids=["linear", "nearest"])
    @pytest.mark.parametrize(
        "outside",
        [
            "nan",
            pytest.param("clamp", marks=pytest.mark.slow),
            pytest.param("extrapolate", marks=pytest.mark.slow),
        ],
    )
    def test_a_rectilinear_grid(self, scheme, outside: str) -> None:  # type: ignore[no-untyped-def]
        rng = np.random.default_rng(2)
        first = np.cumsum(rng.uniform(0.5, 1.5, size=5))
        second = np.cumsum(rng.uniform(0.5, 1.5, size=6))
        grid = RectilinearGrid((Axis(jnp.asarray(first)), Axis(jnp.asarray(second))))
        # x varies down the rows and y along the columns, as the axes do
        x, y = np.meshgrid(first, second, indexing="ij")
        swath = CurvilinearGrid(x=jnp.asarray(x), y=jnp.asarray(y))
        targets = Points(
            jnp.asarray(
                np.column_stack(
                    [
                        rng.uniform(first[0] - 1.0, first[-1] + 1.0, size=200),
                        rng.uniform(second[0] - 1.0, second[-1] + 1.0, size=200),
                    ]
                )
            )
        )
        policies = Policies(outside=outside)
        field = jax.random.normal(jax.random.key(3), (30,))
        expected = scheme.plan(grid, targets, policies=policies)
        actual = scheme.plan(swath, targets, policies=policies)
        np.testing.assert_array_equal(actual.valid, expected.valid)
        assert_matches_oracle(actual(field), expected(field), rtol=1e-12, atol=1e-12)

    def test_a_swath_gridded_onto_a_rectilinear_grid(self) -> None:
        swath = _affine_in_disguise()
        target = RectilinearGrid(
            (Axis(jnp.linspace(95.0, 115.0, 5)), Axis(jnp.linspace(40.0, 52.0, 4)))
        )
        mesh_x, mesh_y = jnp.meshgrid(target.axes[0].values, target.axes[1].values)
        points = Points(jnp.column_stack([jnp.ravel(mesh_x.T), jnp.ravel(mesh_y.T)]))
        field = jax.random.normal(jax.random.key(4), (42,))
        gridded = Linear().plan(swath, target)(field)
        assert_matches_oracle(
            gridded, Linear().plan(AFFINE, points)(field), rtol=1e-12, atol=1e-12
        )


class TestWarped:
    def test_recovers_the_index_position(self) -> None:
        targets, truth = _warped_targets((8, 9), 500, seed=10)
        cell, offset, inside = locate_curvilinear(_warped(), jnp.asarray(targets))
        assert bool(jnp.all(inside))
        assert_matches_oracle(cell + offset, truth, rtol=1e-12, atol=1e-12)

    def test_the_found_position_maps_back_to_the_target(self) -> None:
        x, y = _warped_nodes((8, 9))
        rng = np.random.default_rng(11)
        # any point within two units of the central cell, not built from a cell
        centre = [_bilinear(nodes, 3, 4, 0.5, 0.5) for nodes in (x, y)]
        targets = np.asarray(centre) + rng.uniform(-2.0, 2.0, size=(200, 2))
        cell, offset, inside = locate_curvilinear(_warped(), jnp.asarray(targets))
        assert bool(jnp.all(inside))
        row, col = np.asarray(cell).T
        s, t = np.asarray(offset).T
        # on the far edge node u = N − 1 sits in cell N − 1; read the last cell
        row_cell = np.minimum(row, 6)
        col_cell = np.minimum(col, 7)
        s = s + (row - row_cell)
        t = t + (col - col_cell)
        mapped = np.column_stack(
            [
                _bilinear(x, row_cell, col_cell, s, t),
                _bilinear(y, row_cell, col_cell, s, t),
            ]
        )
        assert_matches_oracle(mapped, targets, rtol=1e-12, atol=1e-12)

    def test_linear_reproduces_affine_functions_of_the_index(self) -> None:
        targets, truth = _warped_targets((8, 9), 200, seed=12)
        rows, cols = np.meshgrid(np.arange(8.0), np.arange(9.0), indexing="ij")
        # f = 2 − 0.7 r + 1.3 c at the nodes
        field = 2.0 - 0.7 * rows + 1.3 * cols
        plan = Linear().plan(_warped(), Points(jnp.asarray(targets)))
        expected = 2.0 - 0.7 * truth[:, 0] + 1.3 * truth[:, 1]
        assert_matches_oracle(
            plan(jnp.asarray(field.ravel())), expected, rtol=1e-12, atol=1e-12
        )

    @pytest.mark.slow
    @settings(max_examples=25, deadline=None)
    @given(seed=st.integers(0, 2**31 - 1))
    def test_weights_sum_to_one(self, seed: int) -> None:
        targets, _ = _warped_targets((8, 9), 16, seed=seed)
        plan = Linear().plan(_warped(), Points(jnp.asarray(targets)))
        assert_matches_oracle(plan(jnp.ones(72)), np.ones(16), rtol=1e-13)

    def test_the_walk_leaves_the_nearest_cell(self) -> None:
        targets, truth = _warped_targets((8, 9), 500, seed=13)
        grid = _warped()
        _, _, without_walk = locate_curvilinear(grid, jnp.asarray(targets), n_walks=0)
        cell, offset, inside = locate_curvilinear(grid, jnp.asarray(targets))
        # some targets do not sit in the cell with the nearest centre
        assert int(jnp.sum(without_walk)) < 500
        assert bool(jnp.all(inside))
        assert_matches_oracle(cell + offset, truth, rtol=1e-12, atol=1e-12)

    def test_too_few_newton_steps_find_nothing(self) -> None:
        targets, _ = _warped_targets((8, 9), 20, seed=14)
        _, offset, inside = locate_curvilinear(
            _warped(), jnp.asarray(targets), n_steps=1
        )
        assert not bool(jnp.any(inside))
        assert bool(jnp.all(jnp.isnan(offset)))

    def test_a_degenerate_grid_finds_nothing(self) -> None:
        rows, cols = jnp.meshgrid(jnp.arange(3.0), jnp.arange(3.0), indexing="ij")
        # every node on one line: det J = 0 in every cell
        line = CurvilinearGrid(x=rows + cols, y=rows + cols)
        plan = Linear().plan(line, Points(jnp.array([[1.0, 1.0]])))
        assert plan.valid.tolist() == [False]

    def test_float32_stays_float32(self) -> None:
        x, y = _warped_nodes((8, 9))
        grid = CurvilinearGrid(
            x=jnp.asarray(x, dtype=jnp.float32), y=jnp.asarray(y, dtype=jnp.float32)
        )
        targets, truth = _warped_targets((8, 9), 50, seed=15)
        cell, offset, inside = locate_curvilinear(
            grid, jnp.asarray(targets, dtype=jnp.float32)
        )
        assert offset.dtype == jnp.float32
        assert bool(jnp.all(inside))
        # float32 coordinates near 10 resolve the index to about 1e-5
        assert_matches_oracle(cell + offset, truth, rtol=0.0, atol=1e-4)


class TestOutside:
    def test_targets_outside_are_invalid(self) -> None:
        targets = jnp.asarray(
            from_pixel(
                AFFINE.transform, jnp.array([[2.5, 3.5], [-1.0, 2.0], [3.0, 9.0]])
            )
        )
        swath = _affine_in_disguise()
        _, _, inside = locate_curvilinear(swath, targets)
        assert inside.tolist() == [True, False, False]
        plan = Linear().plan(swath, Points(targets), policies=Policies(fill_value=-1.0))
        assert plan.valid.tolist() == [True, False, False]
        assert plan(jnp.ones(42)).tolist()[1:] == [-1.0, -1.0]

    def test_the_half_cell_border_is_inside_under_cells(self) -> None:
        targets = from_pixel(AFFINE.transform, jnp.array([[-0.4, 3.0], [5.4, 6.4]]))
        swath = _affine_in_disguise()
        _, _, nodes = locate_curvilinear(swath, targets)
        _, _, cells = locate_curvilinear(swath, targets, extent="cells")
        assert nodes.tolist() == [False, False]
        assert cells.tolist() == [True, True]

    def test_raise(self) -> None:
        targets = Points(
            from_pixel(AFFINE.transform, jnp.array([[2.0, 2.0], [9.0, 1.0]]))
        )
        with pytest.raises(OutsideDomainError, match="1 of 2 targets"):
            Linear().plan(
                _affine_in_disguise(), targets, policies=Policies(outside="raise")
            )

    def test_non_finite_targets_are_invalid_under_every_policy(self) -> None:
        targets = Points(jnp.array([[jnp.nan, 45.0], [104.0, 46.0]]))
        for outside in ("nan", "clamp", "extrapolate"):
            plan = Linear().plan(
                _affine_in_disguise(), targets, policies=Policies(outside=outside)
            )
            assert plan.valid.tolist() == [False, True]


class TestAdjoint:
    @pytest.mark.slow
    @pytest.mark.parametrize("scheme", [Linear(), Nearest()], ids=["linear", "nearest"])
    @pytest.mark.parametrize("outside", ["nan", "clamp", "extrapolate"])
    def test_dot_product(self, scheme, outside: str) -> None:  # type: ignore[no-untyped-def]
        x, y = _warped_nodes((8, 9))
        rng = np.random.default_rng(20)
        # inside, near the edges and outside
        targets = np.column_stack(
            [
                rng.uniform(x.min() - 1.0, x.max() + 1.0, size=60),
                rng.uniform(y.min() - 1.0, y.max() + 1.0, size=60),
            ]
        )
        plan = scheme.plan(
            _warped(), Points(jnp.asarray(targets)), policies=Policies(outside=outside)
        )
        operator = plan.operator()
        assert_adjoint(
            lambda field: operator.mv(field),
            plan.adjoint,
            72,
            60,
            key=jax.random.key(21),
        )

    def test_adjoint_equals_the_lineax_transpose(self) -> None:
        targets, _ = _warped_targets((8, 9), 30, seed=22)
        plan = Linear().plan(_warped(), Points(jnp.asarray(targets)))
        operator = plan.operator()
        assert_transpose_matches(
            lambda field: operator.mv(field), plan.adjoint, jnp.zeros(72)
        )


class TestTraces:
    def test_plan_built_inside_the_trace(self) -> None:
        grid = _warped()
        field = jax.random.normal(jax.random.key(30), (72,))
        targets, _ = _warped_targets((8, 9), 3 * 5, seed=31)
        batched = jnp.asarray(targets.reshape(3, 5, 2))

        def read(xy: jax.Array) -> jax.Array:
            return Linear().plan(grid, Points(xy))(field)

        assert_traces(read, batched)

    def test_grad_in_the_target_positions(self) -> None:
        grid = _warped()
        field = jax.random.normal(jax.random.key(32), (72,))
        targets, _ = _warped_targets((8, 9), 6, seed=33)

        def read(xy: jax.Array) -> jax.Array:
            return Linear().plan(grid, Points(xy))(field)

        assert_grad_matches_fd(read, jnp.asarray(targets))


class TestValidation:
    def test_rejects_bad_arguments(self) -> None:
        grid = _warped()
        targets = jnp.zeros((2, 2))
        raw = inspect.unwrap(locate_curvilinear)
        with pytest.raises(ValidationError, match="'positions' must be"):
            raw(grid, jnp.zeros((2, 3)))
        with pytest.raises(ValidationError, match="'n_steps' must be"):
            locate_curvilinear(grid, targets, n_steps=0)
        with pytest.raises(ValidationError, match="'n_walks' must be"):
            locate_curvilinear(grid, targets, n_walks=-1)
        with pytest.raises(PolicyError, match="'extent' must be one of"):
            locate_curvilinear(grid, targets, extent="pixels")

    def test_rejects_targets_with_one_coordinate(self) -> None:
        with pytest.raises(GeometryError, match="'target' has 1 coordinates per point"):
            Linear().plan(_warped(), Points(jnp.zeros((2, 1))))
