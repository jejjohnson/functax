"""`fill_gaps` (#42) against xrtoolz ``fillnan_laplacian``, pinned as data.

The fixture holds the test images of xrtoolz's
``packages/xrtoolz-ops/tests/interpolate/test_gap_fill.py`` at ``b3855d7`` with
``boundary="reflect"`` (the corner, the two harmonic holes, the two leading
slices and the image with an infinite cell; the ``"wrap"`` image is left out,
since `fill_gaps` has mirror edges only) and one random image with 30 % holes.
Each was filled by xrtoolz's own ``_fillnan_laplacian_slice``, extracted from
``packages/xrtoolz-ops/src/xrops/interpolate/_src/gap_fill.py`` (the rest of the
module needs xarray), with SOR at ω = 1.8 and ``tol=1e-15``, so the sweeps ran
to the discrete solution (the residual of the neighbour average at the holes is
below 7e-16). Its red/black Gauss–Seidel fixed point, with the edge pixel as its
own ghost neighbour, is the mirror-edge five-point Laplace equation that
`fill_gaps` solves directly, so the two agree to round-off. With its default
``tol=1e-4`` xrtoolz stops early and differs from both by up to that tolerance.
The generator is ``gen_xrtoolz_gap_fill.py`` in the Wave 3 scratch directory.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax.project import fill_gaps


DATA = np.load(Path(__file__).parent / "data" / "xrtoolz_gap_fill.npz")
IMAGES = sorted({key.removesuffix("_image") for key in DATA if key.endswith("_image")})


@pytest.mark.parametrize("name", IMAGES)
def test_laplace_fill_equals_xrtoolz_fillnan_laplacian(name: str) -> None:
    image = jnp.asarray(DATA[f"{name}_image"])
    expected = DATA[f"{name}_filled"]
    mask = jnp.isfinite(image)
    scale = float(np.max(np.abs(expected[np.isfinite(DATA[f"{name}_image"])])))
    # a tight CG solve agrees to round-off
    tight = fill_gaps(image, mask=mask, solver=lx.CG(rtol=1e-13, atol=1e-13))
    assert_matches_oracle(tight, expected, rtol=0, atol=1e-12 * scale)
    # the default CG at √ε of float64 meets the issue's 1e-8
    default = fill_gaps(image, mask=mask)
    assert_matches_oracle(default, expected, rtol=0, atol=1e-8 * scale)
