"""Tests for `locate_curvilinear` and `Linear` and `Nearest` on a `CurvilinearGrid`.
The fast tier: behaviour, validation and small closed-form checks, each well under a
second.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from functax import GeometryError, PolicyError, ValidationError
from functax.geometry import (
    CurvilinearGrid,
    Points,
)
from functax.interp import Linear, locate_curvilinear


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
