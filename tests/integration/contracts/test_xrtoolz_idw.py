"""`InverseDistance` (#35) against xrtoolz's ``idw_to_grid`` at b3855d7, pinned as data.

The fixture ``xrtoolz_idw_to_grid.npz`` holds 200 random sources in lon/lat, a 33 × 33
grid reaching past them, and ``idw_to_grid(..., metric="euclidean", eps=0.0)`` for two
settings: k = 8, p = 2 with no radius, and k = 5, p = 1.5 with ``max_distance=1.0``,
which leaves the cells far from every source NaN. xrtoolz weighs by (d + eps)⁻ᵖ and
functax by max(d, ε)⁻ᵖ, which agree at eps = 0 wherever d > ε (no target here
coincides with a source). Its generator, ``gen_xrtoolz_idw.py``, runs knn.py from a
``git archive`` of xr_toolz b3855d7 in xr_toolz's own environment (xarray,
scikit-learn).
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import InverseDistance


DATA = Path(__file__).parent / "data"
XRTOOLZ = np.load(DATA / "xrtoolz_idw_to_grid.npz")


@pytest.mark.parametrize(
    ("name", "scheme"),
    [
        ("idw_k8_p2", InverseDistance(k=8, power=2.0)),
        ("idw_k5_p1.5_r1", InverseDistance(k=5, power=1.5, radius=1.0)),
    ],
)
def test_idw_to_grid_equals_xrtoolz(name: str, scheme: InverseDistance) -> None:
    # xrtoolz's grid is (lat, lon), so the sources are (lat, lon) too
    sources = Points(np.column_stack([XRTOOLZ["lat"], XRTOOLZ["lon"]]))
    grid = RectilinearGrid(
        (Axis(jnp.asarray(XRTOOLZ["grid_lat"])), Axis(jnp.asarray(XRTOOLZ["grid_lon"])))
    )
    plan = scheme.plan(sources, grid)
    gridded = plan(jnp.asarray(XRTOOLZ["values"]))
    expected = XRTOOLZ[name]
    assert_matches_oracle(gridded, expected.ravel(), rtol=1e-12)
    # the radius leaves cells with no source NaN, as xrtoolz does
    assert np.isnan(expected).any() == (scheme.radius is not None)
