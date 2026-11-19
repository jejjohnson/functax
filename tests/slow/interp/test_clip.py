"""The host-side polygon clipper of conservative remapping (#41). The slow tier:
sweeps, oracle and dot-product checks, gradient and trace checks, anything over
0.3 s on CI (mostly einx and XLA compiling new shapes).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from _harness import assert_matches_oracle

from functax._src.interp._clip import (
    cell_overlaps,
)
from functax.geometry import lonlat_to_unit, spherical_polygon_area


UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


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
