"""functax's lon/lat lift against geonnax's lonlat_to_cartesian3d, pinned at geonnax
be5537d as data.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax.geometry import (
    lonlat_to_unit,
)


GEONNAX = np.load(Path(__file__).parent / "data" / "geonnax_lonlat_to_cartesian3d.npz")


class TestLonlatToUnit:
    @pytest.mark.parametrize(
        ("inputs", "outputs", "unit"),
        [
            ("lonlat_degrees", "unit_from_degrees", "degrees"),
            ("lonlat_radians", "unit_from_radians", "radians"),
        ],
    )
    def test_equals_geonnax(self, inputs: str, outputs: str, unit: str) -> None:
        lifted = lonlat_to_unit(jnp.asarray(GEONNAX[inputs]), unit=unit)
        assert_matches_oracle(lifted, GEONNAX[outputs], rtol=0.0, atol=1e-15)
