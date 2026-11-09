"""functax's bilinear plan against geoops interp_at, pinned at geotoolz 1e7308b as
data: values, validity and the half-pixel border.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from _harness import (
    assert_matches_oracle,
)

from functax.geometry import AffineGrid, Points
from functax.interp import Linear, Policies


GEOOPS = np.load(Path(__file__).parent / "data" / "geoops_interp_at_bilinear.npz")


class TestInterpAt:
    def test_bilinear_equals_interp_at(self) -> None:
        grid = AffineGrid(GEOOPS["transform"], GEOOPS["data"].shape)
        policies = Policies(extent="cells", boundary="edge")
        plan = Linear().plan(grid, Points(jnp.asarray(GEOOPS["xy"])), policies=policies)
        values = plan(jnp.asarray(GEOOPS["data"]).ravel())
        np.testing.assert_array_equal(plan.valid, GEOOPS["bilinear_valid"])
        assert_matches_oracle(values, GEOOPS["bilinear"], rtol=1e-12, atol=1e-14)

    def test_the_half_pixel_border_reads_the_edge(self) -> None:
        border = GEOOPS["kind"] == 1
        grid = AffineGrid(GEOOPS["transform"], GEOOPS["data"].shape)
        policies = Policies(extent="cells", boundary="edge")
        plan = Linear().plan(
            grid, Points(jnp.asarray(GEOOPS["xy"][border])), policies=policies
        )
        values = plan(jnp.asarray(GEOOPS["data"]).ravel())
        assert bool(jnp.all(plan.valid))
        assert_matches_oracle(
            values, GEOOPS["bilinear"][border], rtol=1e-12, atol=1e-14
        )
