"""functax's `legendre_series` (#47) against kernellib's, pinned at kernellib
c2d1fce as data. The generator exports ``src/kernellib/functional/_geo.py`` with
``git archive``, loads it with a stub for its unused ``geonnax.geo`` import, and
evaluates the series on 405 points of [−1, 1] (with both ends, 0 and 1 + 1e-12,
which is clipped) and on a (3, 7) grid, for L = 0, 1, 2, 5, 50 and 200 standard
normal coefficients.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax.basis.polynomial import legendre_series


KERNELLIB = np.load(Path(__file__).parent / "data" / "kernellib_legendre_series.npz")


@pytest.mark.parametrize("degree", [0, 1, 2, 5, 50, 200])
def test_legendre_series_equals_kernellib(degree: int) -> None:
    coeffs = jnp.asarray(KERNELLIB[f"coeffs_{degree}"])
    for points, key in (("x", "series"), ("x_grid", "series_grid")):
        values = legendre_series(jnp.asarray(KERNELLIB[points]), coeffs)
        expected = KERNELLIB[f"{key}_{degree}"]
        assert_matches_oracle(values, expected, rtol=1e-13, atol=1e-13)
