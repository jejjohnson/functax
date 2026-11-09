"""Tests for the affine raster kernels in `functax.geometry.affine`."""

from __future__ import annotations

import subprocess
import sys
import types

import _geoarray_reference as geoarray
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces

from functax import GeometryError
from functax.geometry import AffineGrid, from_pixel, pixel_centres, to_pixel, window


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

    def test_accepts_a_tuple_transform_and_batch_dims(self) -> None:
        xy = np.broadcast_to(np.array([600010.0, 3549990.0]), (2, 3, 2))
        rowcol = to_pixel(tuple(NORTH_UP), xy)
        assert rowcol.shape == (2, 3, 2)
        np.testing.assert_array_equal(rowcol, 0.0)

    def test_rejects_a_singular_transform(self) -> None:
        with pytest.raises(GeometryError, match="'transform' must be invertible"):
            to_pixel(np.array([1.0, 2.0, 0.0, 2.0, 4.0, 0.0]), np.zeros((1, 2)))


class TestNamespace:
    def test_numpy_in_numpy_out(self) -> None:
        assert type(to_pixel(NORTH_UP, np.zeros((1, 2)))) is np.ndarray
        assert type(from_pixel(NORTH_UP, np.zeros((1, 2)))) is np.ndarray

    @pytest.mark.parametrize(
        ("transform", "points"),
        [
            (jnp.asarray(NORTH_UP), np.zeros((1, 2))),
            (NORTH_UP, jnp.zeros((1, 2))),
            (jnp.asarray(NORTH_UP), jnp.zeros((1, 2))),
        ],
        ids=["jax-transform", "jax-points", "both-jax"],
    )
    def test_any_jax_input_gives_jax_out(
        self, transform: object, points: object
    ) -> None:
        assert isinstance(to_pixel(transform, points), jax.Array)
        assert isinstance(from_pixel(transform, points), jax.Array)

    def test_jax_equals_numpy(self) -> None:
        xy = np.random.default_rng(2).uniform(-1e6, 1e6, size=(30, 2))
        assert_matches_oracle(
            to_pixel(jnp.asarray(NORTH_UP), jnp.asarray(xy)),
            to_pixel(NORTH_UP, xy),
            rtol=1e-12,
        )


class TestRoundTrip:
    @pytest.mark.parametrize("seed", [3, 4, 5])
    def test_from_pixel_inverts_to_pixel(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        for transform in _random_transforms(10, seed=seed):
            xy = rng.uniform(-1e6, 1e6, size=(40, 2))
            # |x| reaches 1e6 while cancellation in A⁻¹ costs digits, so
            # compare in map units at 1e-12 of the coordinate scale
            assert_matches_oracle(
                from_pixel(transform, to_pixel(transform, xy)),
                xy,
                rtol=1e-12,
                atol=1e-12 * 1e6,
            )

    def test_from_pixel_puts_centres_at_integers(self) -> None:
        corner = from_pixel(NORTH_UP, np.array([[-0.5, -0.5]]))
        np.testing.assert_array_equal(corner, [[600000.0, 3550000.0]])


class TestTraces:
    def test_to_pixel_under_jit_and_vmap(self) -> None:
        transforms = jnp.asarray(_random_transforms(4, seed=6))
        points = jnp.asarray(
            np.random.default_rng(7).uniform(-1e3, 1e3, size=(4, 5, 2))
        )
        assert_traces(to_pixel, transforms, points)

    def test_from_pixel_under_jit_and_vmap(self) -> None:
        transforms = jnp.asarray(_random_transforms(4, seed=8))
        rowcol = jnp.asarray(
            np.random.default_rng(9).uniform(0.0, 100.0, size=(4, 5, 2))
        )
        assert_traces(from_pixel, transforms, rowcol)

    @pytest.mark.slow
    def test_grad_in_the_points_and_the_transform(self) -> None:
        # rotated and sheared, with det A = −3.15, so central differences
        # with ε = 1e-6 are accurate to about 1e-10
        transform = jnp.asarray([2.0, 0.5, 10.0, -0.3, -1.5, 20.0])
        xy = jnp.asarray([[12.0, -7.0], [3.0, 4.0]])
        assert_grad_matches_fd(lambda p: to_pixel(transform, p), xy)
        assert_grad_matches_fd(lambda t: to_pixel(t, xy), transform)

    def test_a_concrete_transform_closed_over_under_jit(self) -> None:
        transform = jnp.asarray(NORTH_UP)
        rowcol = jax.jit(lambda p: to_pixel(transform, p))(jnp.zeros((1, 2)))
        assert rowcol.shape == (1, 2)

    def test_grad_matches_the_inverse_matrix(self) -> None:
        transform = jnp.asarray(NORTH_UP)
        # row = (−d·Δx + a·Δy) / det A and col = (e·Δx − b·Δy) / det A; for
        # north-up (a = 20, e = −20, det A = −400) ∂(row, col)/∂(x, y) is
        # [[0, −1/20], [1/20, 0]]
        jacobian = jax.jacrev(lambda p: to_pixel(transform, p))(jnp.zeros(2))
        np.testing.assert_allclose(jacobian, [[0.0, -0.05], [0.05, 0.0]], rtol=1e-15)


class TestPixelCentres:
    def test_north_up(self) -> None:
        grid = AffineGrid([10.0, 0.0, 0.0, 0.0, -10.0, 100.0], (2, 3))
        x, y = pixel_centres(grid)
        assert isinstance(x, jax.Array)
        np.testing.assert_array_equal(x, [[5.0, 15.0, 25.0], [5.0, 15.0, 25.0]])
        np.testing.assert_array_equal(y, [[95.0, 95.0, 95.0], [85.0, 85.0, 85.0]])

    def test_centres_map_to_integer_pixels(self) -> None:
        transform = _random_transforms(1, seed=11)[0]
        grid = types.SimpleNamespace(transform=transform, shape=(4, 6))
        x, y = pixel_centres(grid)
        assert x.shape == y.shape == (4, 6)
        rowcol = to_pixel(transform, np.stack([x, y], axis=-1))
        rows, cols = np.meshgrid(np.arange(4.0), np.arange(6.0), indexing="ij")
        assert_matches_oracle(rowcol[..., 0], rows, rtol=0.0, atol=1e-9)
        assert_matches_oracle(rowcol[..., 1], cols, rtol=0.0, atol=1e-9)


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

    def test_is_not_clipped_to_the_grid(self) -> None:
        grid = AffineGrid(NORTH_UP, (600, 600))
        rows, cols = window(grid, (599960.0, 3549980.0, 600020.0, 3550040.0))
        assert (rows, cols) == (slice(-2, 1), slice(-2, 1))

    @pytest.mark.parametrize(
        "bounds", [(0.0, 0.0, 1.0), (2.0, 0.0, 1.0, 1.0), (0.0, 2.0, 1.0, 1.0)]
    )
    def test_rejects_bad_bounds(self, bounds: tuple[float, ...]) -> None:
        grid = AffineGrid(NORTH_UP, (600, 600))
        with pytest.raises(GeometryError, match="'bounds' must be"):
            window(grid, bounds)

    def test_rejects_a_singular_transform(self) -> None:
        grid = types.SimpleNamespace(
            transform=(1.0, 2.0, 0.0, 2.0, 4.0, 0.0), shape=(2, 2)
        )
        with pytest.raises(GeometryError, match="'transform' must be invertible"):
            window(grid, (0.0, 0.0, 1.0, 1.0))


def test_the_kernels_never_import_jax() -> None:
    code = (
        "import sys, functax.geometry as g, numpy as np\n"
        "g.to_pixel(np.zeros(6) + [1, 0, 0, 0, -1, 0], np.zeros((1, 2)))\n"
        "g.from_pixel(np.zeros(6) + [1, 0, 0, 0, -1, 0], np.zeros((1, 2)))\n"
        "import types\n"
        "grid = types.SimpleNamespace(\n"
        "    transform=(1.0, 0, 0, 0, -1.0, 0), shape=(2, 2)\n"
        ")\n"
        "g.pixel_centres(grid)\n"
        "g.window(grid, (0.0, -1.0, 1.0, 0.0))\n"
        "assert 'jax' not in sys.modules, 'jax was imported'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
