"""The host-side polygon clipper of conservative remapping (#41)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax._src.interp._clip import (
    cell_overlaps,
    clip_polygons,
    planar_polygon_area,
)
from functax.exceptions import GeometryError
from functax.geometry import lonlat_to_unit, spherical_polygon_area


UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def clipped_area(subject: np.ndarray, clip: np.ndarray) -> float:
    points, counts = clip_polygons(subject[None], np.array([len(subject)]), clip[None])
    return float(planar_polygon_area(points, counts)[0])


class TestClipPolygons:
    def test_offset_squares_overlap_in_a_quarter(self) -> None:
        assert clipped_area(UNIT_SQUARE + 0.5, UNIT_SQUARE) == 0.25

    def test_a_rotated_square_clipped_to_an_octagon(self) -> None:
        # the unit square rotated 45° about its centre: the overlap is a regular
        # octagon of area 2(√2 − 1)
        angles = np.pi / 4 + np.arange(4) * np.pi / 2
        diamond = 0.5 + np.sqrt(0.5) * np.stack(
            [np.cos(angles + np.pi / 4), np.sin(angles + np.pi / 4)], axis=1
        )
        points, counts = clip_polygons(diamond[None], np.array([4]), UNIT_SQUARE[None])
        assert counts.tolist() == [8]
        area = planar_polygon_area(points, counts)
        assert_matches_oracle(area, [2 * (np.sqrt(2) - 1)], rtol=1e-14)

    def test_disjoint_and_degenerate_subjects_have_no_area(self) -> None:
        points, counts = clip_polygons(
            np.stack([UNIT_SQUARE + 3.0, np.zeros((4, 2)) + 0.5]),
            np.array([4, 4]),
            np.stack([UNIT_SQUARE, UNIT_SQUARE]),
        )
        assert counts[0] == 0
        assert planar_polygon_area(points, counts).tolist() == [0.0, 0.0]

    def test_a_subject_inside_is_kept(self) -> None:
        small = 0.25 + 0.5 * UNIT_SQUARE
        assert clipped_area(small, UNIT_SQUARE) == 0.25
        assert clipped_area(UNIT_SQUARE, small) == 0.25

    def test_a_clockwise_subject_gives_a_negative_area(self) -> None:
        assert clipped_area((UNIT_SQUARE + 0.5)[::-1], UNIT_SQUARE) == -0.25


class TestCellOverlaps:
    def test_planar_pairs_partition_each_source_cell(self) -> None:
        shift = np.array([1.0, 0.0])
        source = np.stack([UNIT_SQUARE, UNIT_SQUARE + shift])
        target = 2.0 * UNIT_SQUARE[None] - np.array([0.0, 0.5])
        overlaps = cell_overlaps(source, target, sphere=False)
        assert overlaps.target.tolist() == [0, 0]
        assert overlaps.source.tolist() == [0, 1]
        assert_matches_oracle(overlaps.areas, [1.0, 1.0], rtol=1e-15)
        assert overlaps.target_areas.tolist() == [4.0]

    def test_spherical_cells_with_great_circle_edges(self) -> None:
        # two 10° cells at the equator, and the cell over both
        def cell(lon0: float, lon1: float) -> np.ndarray:
            lonlat = jnp.array([[lon0, 0.0], [lon1, 0.0], [lon1, 10.0], [lon0, 10.0]])
            return np.asarray(lonlat_to_unit(lonlat, unit="degrees"))

        source = np.stack([cell(0.0, 10.0), cell(10.0, 20.0)])
        target = cell(0.0, 20.0)[None]
        overlaps = cell_overlaps(source, target, sphere=True)
        # the 20° cell's top edge bulges past the two 10° cells', so each lies
        # inside it whole, and together they cover it only to a part in 10²
        source_areas = np.asarray(spherical_polygon_area(jnp.asarray(source)))
        assert_matches_oracle(overlaps.areas, source_areas, rtol=1e-12)
        assert 0.99 < overlaps.areas.sum() / overlaps.target_areas[0] < 1.0

    def test_rejects_a_concave_target(self) -> None:
        dart = np.array([[0.0, 0.0], [2.0, 1.0], [0.0, 2.0], [0.5, 1.0]])
        with pytest.raises(GeometryError, match="'target' cells must be convex"):
            cell_overlaps(UNIT_SQUARE[None], dart[None], sphere=False)
