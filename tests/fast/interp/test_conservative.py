"""Conservative remapping on tensor grids (#40): overlaps, conservation, policies."""

from __future__ import annotations

import inspect
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax.exceptions import (
    GeometryError,
    OutsideDomainError,
    ValidationError,
)
from functax.geometry import AffineGrid, Axis, Points, RectilinearGrid
from functax.interp import Conservative, Policies, Scheme, overlap_weights_1d


def random_edges(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    inner = np.sort(rng.uniform(lo, hi, n - 1))
    return np.concatenate([[lo], inner, [hi]])


def axis_from_edges(edges: np.ndarray, *, period: float | None = None) -> Axis:
    centres = 0.5 * (edges[:-1] + edges[1:])
    return Axis(jnp.asarray(centres), edges=jnp.asarray(edges), period=period)


def grid_from_edges(*edges: np.ndarray) -> RectilinearGrid:
    return RectilinearGrid(tuple(axis_from_edges(e) for e in edges))


def cell_areas(*edges: np.ndarray) -> np.ndarray:
    lengths = [np.abs(np.diff(e)) for e in edges]
    return np.prod(np.meshgrid(*lengths, indexing="ij"), axis=0).ravel()


class TestOverlapWeights:
    def test_closed_form(self) -> None:
        dense = overlap_weights_1d(
            np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.5, 2.5, 4.0])
        )
        np.testing.assert_array_equal(dense, [[0.5, 1.0, 0.5], [0.0, 0.0, 0.5]])

    def test_columns_sum_to_the_source_lengths_when_covered(self) -> None:
        rng = np.random.default_rng(0)
        source = random_edges(rng, 20, 0.0, 1.0)
        target = random_edges(rng, 7, -0.1, 1.1)
        dense = overlap_weights_1d(source, target)
        assert_matches_oracle(dense.sum(axis=0), np.diff(source), rtol=1e-14)

    def test_descending_edges_read_as_intervals(self) -> None:
        source = np.array([0.0, 1.0, 2.0, 3.0])
        target = np.array([0.5, 2.5, 4.0])
        ascending = overlap_weights_1d(source, target)
        np.testing.assert_array_equal(
            overlap_weights_1d(source[::-1].copy(), target), ascending[:, ::-1]
        )
        np.testing.assert_array_equal(
            overlap_weights_1d(source, target[::-1].copy()), ascending[::-1]
        )

    def test_a_periodic_source_reads_targets_modulo_the_period(self) -> None:
        source = np.arange(0.0, 361.0, 30.0)
        shifted = overlap_weights_1d(source, np.array([350.0, 370.0]), period=360.0)
        wrapped = overlap_weights_1d(source, np.array([-10.0, 10.0]), period=360.0)
        np.testing.assert_array_equal(shifted, wrapped)
        assert wrapped[0, 0] == 10.0
        assert wrapped[0, -1] == 10.0

    def test_keeps_float32(self) -> None:
        dense = overlap_weights_1d(
            np.array([0.0, 1.0, 2.0], np.float32), np.array([0.0, 2.0], np.float32)
        )
        assert dense.dtype == np.float32

    @pytest.mark.parametrize(
        ("source", "target", "match"),
        [
            (np.array([0.0]), np.array([0.0, 1.0]), "'source_edges' must be 1-D"),
            (np.array([0.0, 2.0, 1.0]), np.array([0.0, 1.0]), "strictly monotone"),
            (np.array([0.0, 1.0]), np.array([0.0, np.nan]), "'target_edges' must be"),
        ],
    )
    def test_rejects_bad_edges(
        self, source: np.ndarray, target: np.ndarray, match: str
    ) -> None:
        with pytest.raises(ValidationError, match=match):
            inspect.unwrap(overlap_weights_1d)(source, target)

    def test_rejects_cells_longer_than_the_period(self) -> None:
        with pytest.raises(ValidationError, match="no longer than the period"):
            overlap_weights_1d(
                np.array([0.0, 180.0, 360.0]), np.array([0.0, 400.0]), period=360.0
            )
        with pytest.raises(ValidationError, match="span at most the period"):
            overlap_weights_1d(
                np.array([0.0, 200.0, 400.0]), np.array([0.0, 10.0]), period=360.0
            )

    def test_rejects_traced_edges(self) -> None:
        with pytest.raises(ValidationError, match="is traced"):
            jax.jit(lambda e: overlap_weights_1d(e, np.array([0.0, 1.0])))(
                jnp.array([0.0, 1.0])
            )


class TestPolicies:
    source = Axis(jnp.arange(4.0) + 0.5)
    # [−1, 1] is half outside, [1, 3] and [3, 4] inside, [4, 6] off the source
    target = Axis(
        jnp.array([0.0, 2.0, 3.5, 5.0]), edges=jnp.array([-1.0, 1.0, 3.0, 4.0, 6.0])
    )
    field = jnp.array([1.0, 2.0, 3.0, 4.0])

    def test_nan_marks_partly_covered_cells_invalid(self) -> None:
        plan = Conservative().plan(self.source, self.target)
        assert plan.valid.tolist() == [False, True, True, False]
        assert_matches_oracle(plan(self.field), [np.nan, 2.5, 4.0, np.nan], rtol=1e-15)

    def test_raise(self) -> None:
        with pytest.raises(OutsideDomainError, match="2 of 4 targets"):
            Conservative().plan(
                self.source, self.target, policies=Policies(outside="raise")
            )

    def test_extrapolate_divides_by_the_whole_cell(self) -> None:
        plan = Conservative().plan(
            self.source, self.target, policies=Policies(outside="extrapolate")
        )
        assert plan.valid.tolist() == [True, True, True, False]
        assert_matches_oracle(plan(self.field), [0.5, 2.5, 4.0, np.nan], rtol=1e-15)

    def test_clamp_divides_by_the_covered_part(self) -> None:
        plan = Conservative().plan(
            self.source, self.target, policies=Policies(outside="clamp")
        )
        assert_matches_oracle(plan(self.field), [1.0, 2.5, 4.0, np.nan], rtol=1e-15)

    def test_renormalise_averages_the_valid_covered_area(self) -> None:
        plan = Conservative().plan(
            self.source, self.target, policies=Policies(missing="renormalise")
        )
        masked = self.field.at[2].set(jnp.nan)
        # a coastal mask: cell [1, 3] reads only its valid half
        assert_matches_oracle(plan(masked), [1.0, 2.0, 4.0, np.nan], rtol=1e-15)

    def test_fill_value(self) -> None:
        plan = Conservative().plan(
            self.source, self.target, policies=Policies(fill_value=-9.0)
        )
        assert plan(self.field).tolist() == [-9.0, 2.5, 4.0, -9.0]


class TestGeometries:
    def test_default_edges_are_midpoints(self) -> None:
        # source cells [−½, ½], [½, 2], [2, 4]
        source = Axis(jnp.array([0.0, 1.0, 3.0]))
        target = Axis(jnp.array([0.0, 3.0]), edges=jnp.array([-0.5, 1.0, 4.0]))
        weights = Conservative().plan(source, target).read.to_dense()
        expected = [[1.0 / 1.5, 0.5 / 1.5, 0.0], [0.0, 1.0 / 3.0, 2.0 / 3.0]]
        assert_matches_oracle(weights, expected, rtol=1e-15)

    def test_descending_axes(self) -> None:
        rng = np.random.default_rng(8)
        edges = random_edges(rng, 6, 0.0, 1.0)
        target_edges = np.array([0.0, 0.3, 1.0])
        field = jnp.asarray(rng.normal(size=6))
        ascending = Conservative().plan(
            axis_from_edges(edges), axis_from_edges(target_edges)
        )(field)
        descending = Conservative().plan(
            axis_from_edges(edges[::-1].copy()),
            axis_from_edges(target_edges[::-1].copy()),
        )(field[::-1])
        assert_matches_oracle(descending, ascending[::-1], rtol=1e-14)

    def test_a_north_up_affine_grid_equals_its_rectilinear_grid(self) -> None:
        source = AffineGrid([2.0, 0.0, 10.0, 0.0, -1.0, 8.0], (8, 6))
        target = AffineGrid([3.0, 0.0, 10.0, 0.0, -2.0, 8.0], (4, 4))
        rows = grid_from_edges(8.0 - np.arange(9.0), 10.0 + 2.0 * np.arange(7.0))
        target_rows = grid_from_edges(
            8.0 - 2.0 * np.arange(5.0), 10.0 + 3.0 * np.arange(5.0)
        )
        field = jnp.asarray(np.random.default_rng(9).normal(size=48))
        affine = Conservative().plan(source, target)
        rectilinear = Conservative().plan(rows, target_rows)
        assert affine.target_shape == (4, 4)
        assert_matches_oracle(affine(field), rectilinear(field), rtol=1e-15)

    def test_rejects_a_rotated_affine_grid(self) -> None:
        rotated = AffineGrid([1.0, 0.5, 0.0, 0.0, -1.0, 0.0], (2, 2))
        with pytest.raises(GeometryError, match="north-up"):
            Conservative().plan(rotated, rotated)

    def test_rejects_points_and_mismatched_axes(self) -> None:
        grid = RectilinearGrid((Axis(jnp.arange(3.0)), Axis(jnp.arange(3.0))))
        with pytest.raises(GeometryError, match="'target' must be an Axis"):
            Conservative().plan(grid, Points(jnp.zeros((1, 2))))
        with pytest.raises(GeometryError, match="'target' has 1 axes"):
            Conservative().plan(grid, Axis(jnp.arange(3.0)))

    def test_keeps_float32_weights(self) -> None:
        axis = Axis(jnp.arange(4.0, dtype=jnp.float32))
        plan = Conservative().plan(axis, Axis(jnp.array([0.5, 2.5], jnp.float32)))
        assert plan.read.weights.dtype == jnp.float32


def test_config_round_trips() -> None:
    text = json.dumps(Conservative().get_config(), allow_nan=False)
    assert Scheme.from_config(json.loads(text)) == Conservative()
