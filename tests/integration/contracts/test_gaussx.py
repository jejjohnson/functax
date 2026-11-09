"""functax's Keys cubic stencil against gaussx cubic_interpolation_weights, pinned at
gaussx 1adbf07 as data.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import (
    assert_matches_oracle,
)

from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import (
    Cubic,
)


GAUSSX = np.load(
    Path(__file__).parent / "data" / "gaussx_cubic_interpolation_weights.npz"
)


class TestCubicInterpolationWeights:
    @pytest.mark.parametrize(("name", "sizes"), [("1d", (10,)), ("2d", (6, 5))])
    def test_equals_gaussx_weights(self, name: str, sizes: tuple[int, ...]) -> None:
        grid = RectilinearGrid(tuple(Axis(jnp.linspace(0.0, 1.0, n)) for n in sizes))
        plan = Cubic().plan(grid, Points(jnp.asarray(GAUSSX[f"x_{name}"])))
        np.testing.assert_array_equal(plan.read.indices, GAUSSX[f"indices_{name}"])
        assert_matches_oracle(
            plan.read.weights, GAUSSX[f"weights_{name}"], rtol=1e-12, atol=1e-14
        )
