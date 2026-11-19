"""The host-side polygon clipper of conservative remapping (#41). The fast tier:
closed-form clips of one pair, and validation. Every clip here is of one 4-gon
against one 4-gon, so einx compiles that shape once for the whole file.
"""

from __future__ import annotations

import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax._src.interp._clip import (
    cell_overlaps,
    clip_polygons,
    planar_polygon_area,
)
from functax.exceptions import GeometryError


UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def clipped_area(subject: np.ndarray, clip: np.ndarray) -> float:
    points, counts = clip_polygons(subject[None], np.array([len(subject)]), clip[None])
    return float(planar_polygon_area(points, counts)[0])


class TestClipPolygons:
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
        # one pair per call, the shape the other clips here compile
        disjoint = UNIT_SQUARE + 3.0
        points, counts = clip_polygons(disjoint[None], np.array([4]), UNIT_SQUARE[None])
        assert counts.tolist() == [0]
        assert planar_polygon_area(points, counts).tolist() == [0.0]
        degenerate = np.zeros((4, 2)) + 0.5
        assert clipped_area(degenerate, UNIT_SQUARE) == 0.0

    def test_a_subject_inside_is_kept(self) -> None:
        small = 0.25 + 0.5 * UNIT_SQUARE
        assert clipped_area(small, UNIT_SQUARE) == 0.25
        assert clipped_area(UNIT_SQUARE, small) == 0.25

    def test_a_clockwise_subject_gives_a_negative_area(self) -> None:
        assert clipped_area((UNIT_SQUARE + 0.5)[::-1], UNIT_SQUARE) == -0.25


class TestCellOverlaps:
    def test_rejects_a_concave_target(self) -> None:
        dart = np.array([[0.0, 0.0], [2.0, 1.0], [0.0, 2.0], [0.5, 1.0]])
        with pytest.raises(GeometryError, match="'target' cells must be convex"):
            cell_overlaps(UNIT_SQUARE[None], dart[None], sphere=False)
