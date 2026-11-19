"""The host-side polygon clipper of conservative remapping (#41). The slow tier:
sweeps, oracle and property checks, anything over 0.3 s on CI (mostly einx and XLA
compiling new shapes). The clipper runs in NumPy on the host, so it has no gradient
or trace checks.
"""

from __future__ import annotations

import hypothesis.strategies as st
import jax.numpy as jnp
import numpy as np
from _harness import assert_matches_oracle
from hypothesis import given, settings

from functax._src.interp._clip import (
    cell_overlaps,
    clip_polygons,
    planar_polygon_area,
)
from functax.geometry import lonlat_to_unit, spherical_polygon_area


UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def clipped_area(subject: np.ndarray, clip: np.ndarray) -> float:
    points, counts = clip_polygons(subject[None], np.array([len(subject)]), clip[None])
    return float(planar_polygon_area(points, counts)[0])


def random_convex_quad(rng: np.random.Generator) -> np.ndarray:
    """A counter-clockwise convex 4-gon: points at increasing angles on an ellipse."""
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, 4))
    centre = rng.uniform(-0.5, 0.5, 2)
    radii = rng.uniform(0.3, 1.0, 2)
    return centre + radii * np.column_stack([np.cos(angles), np.sin(angles)])


class TestProperties:
    @settings(max_examples=20, deadline=None)
    @given(seed=st.integers(0, 2**31 - 1))
    def test_the_overlap_is_symmetric_and_bounded(self, seed: int) -> None:
        # |A ∩ B| = |B ∩ A| and 0 ≤ |A ∩ B| ≤ min(|A|, |B|) for convex A and B;
        # one 4-gon against one 4-gon, the shape the other clips here compile
        rng = np.random.default_rng(seed)
        first, second = random_convex_quad(rng), random_convex_quad(rng)
        overlap = clipped_area(first, second)
        # the shoelace area of each 4-gon
        areas = [
            0.5
            * np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])
            for q in (first, second)
        ]
        assert_matches_oracle(
            clipped_area(second, first), overlap, rtol=1e-12, atol=1e-15
        )
        assert -1e-15 <= overlap <= float(np.min(areas)) * (1.0 + 1e-12)


class TestCellOverlaps:
    def test_planar_pairs_partition_each_source_cell(self) -> None:
        # two pairs through the whole clip: shapes no fast test compiles
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
