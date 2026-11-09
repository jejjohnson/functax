"""Tests for `locate_curvilinear` and `Linear` and `Nearest` on a `CurvilinearGrid`.
The fast tier: behaviour, validation and small closed-form checks, each well under a
second.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import (
    assert_matches_oracle,
)

from functax import GeometryError, OutsideDomainError, PolicyError, ValidationError
from functax.geometry import (
    AffineGrid,
    CurvilinearGrid,
    Points,
    from_pixel,
    pixel_centres,
)
from functax.interp import Linear, Policies, locate_curvilinear


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


class TestWarped:
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


class TestOutside:
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
