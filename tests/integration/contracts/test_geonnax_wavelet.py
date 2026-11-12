"""functax.transform (#111) against geonnax, pinned as data: the spherical wavelet
transform of geonnax 6e32651, its windows, its analysis of fields that are not
band-limited, its synthesis of random band maps, and two truncated scales.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax.transform import SphericalHarmonicTransform, SphericalWaveletTransform


GEONNAX = np.load(Path(__file__).parent / "data" / "geonnax_wavelet.npz")


@pytest.mark.parametrize("grid", [(9, 17, 7), (16, 32, 12)])
@pytest.mark.parametrize("n_scales", [None, 2])
def test_wavelet_transform_equals_geonnax(grid, n_scales):
    n_lat, n_lon, l_max = grid
    key = f"{n_lat}_{n_lon}_{l_max}_{'full' if n_scales is None else n_scales}"
    sht = SphericalHarmonicTransform.init(n_lat, n_lon, l_max)
    swt = SphericalWaveletTransform.init(sht, n_scales=n_scales)
    assert_matches_oracle(swt.filters, GEONNAX[f"filters_{key}"], rtol=0, atol=0)
    forward = swt.forward(jnp.asarray(GEONNAX[f"field_{key}"]))
    assert_matches_oracle(forward, GEONNAX[f"forward_{key}"], rtol=1e-12, atol=1e-13)
    inverse = swt.inverse(jnp.asarray(GEONNAX[f"bands_{key}"]))
    assert_matches_oracle(inverse, GEONNAX[f"inverse_{key}"], rtol=1e-12, atol=1e-13)
