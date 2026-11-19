"""Tests for the affine raster kernels in `functax.geometry.affine`. The fast tier:
behaviour, validation and small closed-form checks, each well under a second.
"""

from __future__ import annotations

import types

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle
from hypothesis import given, settings

from functax import GeometryError
from functax.geometry import (
    AffineGrid,
    from_pixel,
    pixel_centres,
    slice_transform,
    to_pixel,
    window,
)


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

    @settings(max_examples=20, deadline=None)
    @given(seed=st.integers(0, 2**31 - 1))
    def test_to_pixel_inverts_from_pixel(self, seed: int) -> None:
        # (row, col) → (x, y) → (row, col) on a random rotated, sheared transform
        rng = np.random.default_rng(seed)
        transform = _random_transforms(1, seed=seed)[0]
        rowcol = rng.uniform(-1e3, 1e3, size=(40, 2))
        assert_matches_oracle(
            to_pixel(transform, from_pixel(transform, rowcol)),
            rowcol,
            rtol=0.0,
            atol=1e-6,
        )

    def test_from_pixel_puts_centres_at_integers(self) -> None:
        corner = from_pixel(NORTH_UP, np.array([[-0.5, -0.5]]))
        np.testing.assert_array_equal(corner, [[600000.0, 3550000.0]])


class TestBatchedFromPixel:
    def test_each_transform_maps_its_own_pixels(self) -> None:
        transforms = _random_transforms(3, seed=20)
        rowcol = np.random.default_rng(21).uniform(-5.0, 50.0, size=(3, 4, 5, 2))
        batched = from_pixel(transforms, rowcol)
        assert batched.shape == (3, 4, 5, 2)
        for index, transform in enumerate(transforms):
            np.testing.assert_array_equal(
                batched[index], from_pixel(transform, rowcol[index])
            )

    def test_a_batch_of_one_pixel_each(self) -> None:
        transforms = _random_transforms(4, seed=22)
        rowcol = np.zeros((4, 2))
        batched = from_pixel(jnp.asarray(transforms), jnp.asarray(rowcol))
        assert isinstance(batched, jax.Array)
        pairs = zip(transforms, rowcol, strict=True)
        expected = np.stack([from_pixel(t, r) for t, r in pairs])
        np.testing.assert_array_equal(batched, expected)

    def test_rejects_pixels_without_the_batch_dims(self) -> None:
        transforms = np.zeros((2, 3, 6))
        with pytest.raises(GeometryError, match="'rowcol' must lead with"):
            from_pixel(transforms, np.zeros((2, 2)))

    def test_rejects_a_transform_that_does_not_end_in_six(self) -> None:
        with pytest.raises(GeometryError, match=r"shape \(\*batch, 6\)"):
            from_pixel(np.zeros((2, 5)), np.zeros((2, 1, 2)))


class TestSliceTransform:
    def test_corners_of_the_slice_are_corners_of_the_raster(self) -> None:
        # corner (i, j) of the slice is corner (r₀ + sᵣ i, c₀ + s_c j) of the
        # raster; a corner sits at pixel − ½ in the centre convention
        rows, cols = np.meshgrid(np.arange(4.0), np.arange(5.0), indexing="ij")
        slice_pixels = np.stack([rows, cols], axis=-1) - 0.5
        for transform in _random_transforms(5, seed=23):
            chip = slice_transform(transform, 7, -3, row_step=2, col_step=3)
            raster_pixels = np.stack([7 + 2 * rows, -3 + 3 * cols], axis=-1) - 0.5
            assert_matches_oracle(
                to_pixel(chip, from_pixel(transform, raster_pixels)),
                slice_pixels,
                rtol=0.0,
                atol=1e-9,
            )

    def test_a_unit_step_slice_samples_the_raster_pixels(self) -> None:
        transform = _random_transforms(1, seed=26)[0]
        chip = slice_transform(transform, 3, 8)
        assert_matches_oracle(
            from_pixel(chip, np.array([[0.0, 0.0], [2.0, 1.0]])),
            from_pixel(transform, np.array([[3.0, 8.0], [5.0, 9.0]])),
            rtol=1e-12,
        )

    @settings(max_examples=20, deadline=None)
    @given(seed=st.integers(0, 2**31 - 1))
    def test_a_slice_of_a_slice_composes_offsets_and_steps(self, seed: int) -> None:
        # slicing [i₀::sᵢ] then [i₁::tᵢ] is slicing [i₀ + sᵢ i₁ :: sᵢ tᵢ]
        rng = np.random.default_rng(seed)
        transform = _random_transforms(1, seed=seed)[0]
        row0, col0, row1, col1 = rng.integers(-50, 50, size=4).tolist()
        row_step, col_step, row_step1, col_step1 = rng.integers(1, 5, size=4).tolist()
        first = slice_transform(transform, row0, col0, row_step, col_step)
        twice = slice_transform(first, row1, col1, row_step1, col_step1)
        once = slice_transform(
            transform,
            row0 + row_step * row1,
            col0 + col_step * col1,
            row_step * row_step1,
            col_step * col_step1,
        )
        # c and f reach 1e6, so compare at 1e-12 of that scale
        assert_matches_oracle(twice, once, rtol=1e-12, atol=1e-12 * 1e6)

    def test_unit_steps_at_the_origin_are_the_identity(self) -> None:
        np.testing.assert_array_equal(slice_transform(NORTH_UP, 0, 0), NORTH_UP)

    def test_accepts_a_tuple_transform(self) -> None:
        chip = slice_transform(tuple(NORTH_UP), 1, 2)
        assert type(chip) is np.ndarray
        np.testing.assert_array_equal(chip, slice_transform(NORTH_UP, 1, 2))

    def test_a_batch_of_transforms_and_offsets(self) -> None:
        transforms = _random_transforms(3, seed=24)
        row0 = np.array([0, 4, 9])
        col0 = np.array([2, 0, 5])
        batched = slice_transform(transforms, row0, col0, row_step=2)
        assert batched.shape == (3, 6)
        for index, transform in enumerate(transforms):
            np.testing.assert_array_equal(
                batched[index],
                slice_transform(transform, row0[index], col0[index], row_step=2),
            )

    def test_rejects_a_transform_that_does_not_end_in_six(self) -> None:
        with pytest.raises(GeometryError, match=r"shape \(\*batch, 6\)"):
            slice_transform(np.zeros(5), 0, 0)

    @pytest.mark.parametrize("backend", [np, jnp], ids=["numpy", "jax"])
    def test_integer_offsets_keep_a_float32_transform(self, backend: object) -> None:
        transforms = backend.asarray(_random_transforms(3, seed=27), dtype="float32")
        offsets = backend.asarray([0, 4, 9], dtype="int64")
        steps = backend.asarray([1, 2, 3], dtype="int64")
        chip = slice_transform(transforms, offsets, offsets, steps, steps)
        assert chip.dtype == np.float32


class TestTraces:
    def test_a_concrete_transform_closed_over_under_jit(self) -> None:
        transform = jnp.asarray(NORTH_UP)
        rowcol = jax.jit(lambda p: to_pixel(transform, p))(jnp.zeros((1, 2)))
        assert rowcol.shape == (1, 2)


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
    def test_is_not_clipped_to_the_grid(self) -> None:
        grid = AffineGrid(NORTH_UP, (600, 600))
        rows, cols = window(grid, (599960.0, 3549980.0, 600020.0, 3550040.0))
        assert (rows, cols) == (slice(-2, 1), slice(-2, 1))

    @settings(max_examples=20, deadline=None)
    @given(seed=st.integers(0, 2**31 - 1))
    def test_the_bounds_of_a_pixel_block_give_back_the_block(self, seed: int) -> None:
        # a north-up grid: the corners of rows [r₀, r₁) and columns [c₀, c₁)
        rng = np.random.default_rng(seed)
        width, height = rng.uniform(0.1, 100.0, size=2)
        left, top = rng.uniform(-1e6, 1e6, size=2)
        transform = (width, 0.0, left, 0.0, -height, top)
        grid = types.SimpleNamespace(transform=transform, shape=(600, 600))
        row0, col0 = rng.integers(-20, 500, size=2).tolist()
        row1, col1 = row0 + int(rng.integers(1, 80)), col0 + int(rng.integers(1, 80))
        bounds = (
            left + width * col0,
            top - height * row1,
            left + width * col1,
            top - height * row0,
        )
        assert window(grid, bounds) == (slice(row0, row1), slice(col0, col1))

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
