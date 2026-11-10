"""Tests for the CRS-free geometry types in `functax.geometry`. The slow tier:
sweeps, oracle and dot-product checks, gradient and trace checks, anything over
0.3 s on CI (mostly einx and XLA compiling new shapes).
"""

from __future__ import annotations

import jax.numpy as jnp

from functax.geometry import Axis, RectilinearGrid


class TestRectilinearGrid:
    def test_shape_follows_the_axes(self) -> None:
        latitude = Axis(jnp.linspace(-90.0, 90.0, 73))
        longitude = Axis(jnp.linspace(0.0, 357.5, 144), period=360.0)
        grid = RectilinearGrid([latitude, longitude])
        assert isinstance(grid.axes, tuple)
        assert grid.shape == (73, 144)
