"""Chips of a warp: every window of a grid target equals the matching rows of the full
plan, bit for bit, also under missing-sample renormalisation.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from functax.geometry import AffineGrid
from functax.interp import (
    Linear,
    Plan,
    Policies,
    WindowedPlan,
)


SOURCE = AffineGrid([1.0, 0.0, 0.0, 0.0, -1.0, 10.0], (10, 12))


DESTINATION = AffineGrid([1.4, 0.0, 1.3, 0.0, -1.3, 9.6], (8, 9))


CHIPS = [
    (slice(0, 3), slice(0, 4)),
    (slice(3, 8), slice(4, 9)),
    (slice(None), slice(2, 3)),
    (slice(1, 7, 2), slice(None, None, 3)),
]


def _warp(policies: Policies | None = None) -> Plan:
    return Linear().plan(SOURCE, DESTINATION, policies=policies)


def _as_raster(values: jax.Array) -> np.ndarray:
    """The flat destination rows as an (…, 8, 9) raster."""
    return np.asarray(einx.id("... (row col) -> ... row col", values, row=8))


class TestChips:
    @pytest.mark.parametrize("chip", CHIPS, ids=lambda w: str(w))
    def test_chips_equal_the_full_warp_bit_for_bit(self, chip: tuple) -> None:
        warp = _warp()
        assert not bool(jnp.all(warp.valid))
        fields = jax.random.normal(jax.random.key(0), (3, 120))
        windowed = warp[chip]
        assert isinstance(windowed, WindowedPlan)
        assert windowed.window == chip
        full = _as_raster(warp(fields))[(slice(None), *chip)]
        assert windowed.grid_shape == full.shape[1:]
        chip_values = einx.id(
            "... (row col) -> ... row col", windowed(fields), row=full.shape[1]
        )
        np.testing.assert_array_equal(chip_values, full)
        np.testing.assert_array_equal(
            einx.id("(row col) -> row col", windowed.valid, row=full.shape[1]),
            _as_raster(warp.valid)[chip],
        )

    def test_chips_under_renormalise_are_bit_for_bit(self) -> None:
        warp = _warp(Policies(missing="renormalise"))
        field = jax.random.normal(jax.random.key(1), (120,)).at[::7].set(jnp.nan)
        full = _as_raster(warp(field))
        for chip in CHIPS:
            np.testing.assert_array_equal(
                einx.id(
                    "(row col) -> row col", warp[chip](field), row=full[chip].shape[0]
                ),
                full[chip],
            )
