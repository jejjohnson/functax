"""functax's affine kernels against geoarray's, pinned at geotoolz 1e7308b: to_pixel
bit for bit and window exactly (tests/_geoarray_reference.py).
"""

from __future__ import annotations

import _geoarray_reference as geoarray
import numpy as np
from _harness import assert_matches_oracle

from functax.geometry import AffineGrid, to_pixel, window


NORTH_UP = np.array([20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0])


def _random_transforms(count: int, seed: int) -> np.ndarray:
    """Random rotated, sheared and scaled transforms with |det A| ≥ 0.1."""
    rng = np.random.default_rng(seed)
    transforms = []
    while len(transforms) < count:
        a, b, d, e = rng.uniform(-30.0, 30.0, size=4)
        if abs(a * e - b * d) < 0.1:
            continue
        c, f = rng.uniform(-1e6, 1e6, size=2)
        transforms.append([a, b, c, d, e, f])
    return np.array(transforms)


class TestToPixel:
    def test_equals_geoarray_bit_for_bit_on_its_doctest(self) -> None:
        xy = np.array([[600010.0, 3549990.0]])
        np.testing.assert_array_equal(
            to_pixel(NORTH_UP, xy), geoarray.to_pixel(NORTH_UP, xy)
        )

    def test_equals_geoarray_on_random_points_and_transforms(self) -> None:
        rng = np.random.default_rng(0)
        for transform in _random_transforms(20, seed=1):
            xy = rng.uniform(-1e6, 1e6, size=(50, 2))
            assert_matches_oracle(
                to_pixel(transform, xy), geoarray.to_pixel(transform, xy), rtol=1e-12
            )


class TestWindow:
    def test_equals_geoarray_on_its_doctest(self) -> None:
        grid = AffineGrid(NORTH_UP, (600, 600))
        rows, cols = window(grid, (600100, 3549000, 600400, 3549900))
        assert (rows, cols) == (slice(5, 50), slice(5, 20))
        col_off, row_off, width, height = geoarray.window(
            NORTH_UP, (600100, 3549000, 600400, 3549900)
        )
        assert (rows.start, rows.stop) == (row_off, row_off + height)
        assert (cols.start, cols.stop) == (col_off, col_off + width)

    def test_equals_geoarray_on_random_north_up_bounds(self) -> None:
        rng = np.random.default_rng(12)
        grid = AffineGrid(NORTH_UP, (600, 600))
        for _ in range(500):
            x = np.sort(rng.uniform(599000.0, 613000.0, size=2))
            y = np.sort(rng.uniform(3537000.0, 3551000.0, size=2))
            # half the cases land exactly on pixel edges, to exercise the snap
            if rng.random() < 0.5:
                x = 600000.0 + 20.0 * np.round((x - 600000.0) / 20.0)
                y = 3550000.0 - 20.0 * np.round((3550000.0 - y) / 20.0)
            bounds = (x[0], y[0], x[1], y[1])
            rows, cols = window(grid, bounds)
            col_off, row_off, width, height = geoarray.window(NORTH_UP, bounds)
            assert (rows.start, rows.stop, cols.start, cols.stop) == (
                row_off,
                row_off + height,
                col_off,
                col_off + width,
            )
