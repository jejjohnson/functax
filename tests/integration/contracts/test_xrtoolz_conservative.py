"""functax.interp.Conservative (#40) against xrtoolz b3855d7, pinned as data.

The fixture ``xrtoolz_regrid_conservative.npz`` holds xrtoolz's
``overlap_weights_1d`` and ``regrid_conservative`` (from
``packages/xrtoolz-ops/src/xrops/interpolate/_src/grid_to_grid.py`` at
b3855d7) on random rectilinear grids: a planar regrid of a field with 15 %
NaNs under ``normalize="fracarea"`` (skipna), the same grids without NaNs under
``normalize="destarea"``, and a spherical regrid of a 0–360° source onto a
target offset by 7.5° on −180–180°, with sin-latitude weights. The target
sticks out of the planar source on every side, so its edge cells are partly
covered. The generator, ``gen_xrtoolz_conservative.py``, ran xrtoolz's module
from a ``git archive`` export with xrtoolz's own environment (xarray, SciPy).

The mapping: ``fracarea`` with skipna is ``missing="renormalise"`` (the area
outside the source counts as missing), ``destarea`` is ``outside="extrapolate"``,
and the spherical geometry is a latitude axis in sin φ (values and edges) with
a periodic longitude axis of period 360.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from _harness import assert_matches_oracle

from functax.geometry import Axis, RectilinearGrid
from functax.interp import Conservative, Policies, overlap_weights_1d


DATA = Path(__file__).parent / "data"
XRTOOLZ = np.load(DATA / "xrtoolz_regrid_conservative.npz")


def _axis(edges: np.ndarray, *, period: float | None = None) -> Axis:
    centres = 0.5 * (edges[:-1] + edges[1:])
    return Axis(jnp.asarray(centres), edges=jnp.asarray(edges), period=period)


def _grid(*edges: np.ndarray) -> RectilinearGrid:
    return RectilinearGrid(tuple(_axis(e) for e in edges))


def test_overlap_weights_equal_xrtoolz() -> None:
    dense = overlap_weights_1d(XRTOOLZ["overlap_source"], XRTOOLZ["overlap_target"])
    assert_matches_oracle(dense, XRTOOLZ["overlap"], rtol=1e-12, atol=1e-15)


def test_fracarea_with_nans_equals_renormalise() -> None:
    source = _grid(XRTOOLZ["y_edges"], XRTOOLZ["x_edges"])
    target = _grid(XRTOOLZ["ty_edges"], XRTOOLZ["tx_edges"])
    plan = Conservative().plan(source, target, policies=Policies(missing="renormalise"))
    field = XRTOOLZ["field_nan"]
    remapped = plan(jnp.asarray(field.reshape(field.shape[0], -1)))
    expected = XRTOOLZ["fracarea_nan"].reshape(field.shape[0], -1)
    assert_matches_oracle(remapped, expected, rtol=1e-12)


def test_destarea_equals_extrapolate() -> None:
    source = _grid(XRTOOLZ["y_edges"], XRTOOLZ["x_edges"])
    target = _grid(XRTOOLZ["ty_edges"], XRTOOLZ["tx_edges"])
    plan = Conservative().plan(source, target, policies=Policies(outside="extrapolate"))
    field = XRTOOLZ["field"]
    remapped = plan(jnp.asarray(field.reshape(field.shape[0], -1)))
    expected = XRTOOLZ["destarea"].reshape(field.shape[0], -1)
    assert_matches_oracle(remapped, expected, rtol=1e-12)


def test_spherical_lonlat_cells_equal_xrtoolz() -> None:
    # |cell| = Δλ (sin φ₊ − sin φ₋): an axis in sin φ makes the latitude
    # overlaps xrtoolz's, and the period makes the longitudes wrap
    sin_latitude = np.sin(np.deg2rad(XRTOOLZ["lat_edges"]))
    target_sin_latitude = np.sin(np.deg2rad(XRTOOLZ["tlat_edges"]))
    source = RectilinearGrid(
        (_axis(sin_latitude), _axis(XRTOOLZ["lon_edges"], period=360.0))
    )
    target = RectilinearGrid((_axis(target_sin_latitude), _axis(XRTOOLZ["tlon_edges"])))
    remapped = Conservative().plan(source, target)(
        jnp.asarray(XRTOOLZ["sphere_field"].ravel())
    )
    assert_matches_oracle(remapped, XRTOOLZ["sphere"].ravel(), rtol=1e-12)
