"""`fill_gaps` (#42) against xrtoolz ``fillnan_laplacian``, pinned as data.

The fixture holds the test images of xrtoolz's
``packages/xrtoolz-ops/tests/interpolate/test_gap_fill.py`` at ``b3855d7`` with
``boundary="reflect"`` (the corner, the two harmonic holes, the two leading
slices and the image with an infinite cell) and one random image with 30 % holes.
Each was filled by xrtoolz's own ``_fillnan_laplacian_slice``, extracted from
``packages/xrtoolz-ops/src/xrops/interpolate/_src/gap_fill.py`` (the rest of the
module needs xarray), with SOR at ω = 1.8 and ``tol=1e-15``, so the sweeps ran
to the discrete solution (the residual of the neighbour average at the holes is
below 7e-16). Its red/black Gauss–Seidel fixed point, with the edge pixel as its
own ghost neighbour, is the mirror-edge five-point Laplace equation that
`fill_gaps` solves directly, so the two agree to round-off. With its default
``tol=1e-4`` xrtoolz stops early and differs from both by up to that tolerance.
The generator is ``gen_xrtoolz_gap_fill.py`` in the Wave 3 scratch directory.

``xrtoolz_gap_fill_wrap.npz`` holds the same extraction run with
``boundary="wrap"`` (longitude, the columns, wraps; latitude reflects), on
xrtoolz's own seam test image, a periodic field with a hole across the seam
and a random image with 30 % holes, again to a residual below 2e-15 at the
holes. It checks ``fill_gaps(..., periodic=(False, True))`` (#119). Its
generator is ``gen_xrtoolz_gap_fill_wrap.py`` in the Wave 5 scratch directory.
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


WRAP = np.load(Path(__file__).parent / "data" / "xrtoolz_gap_fill_wrap.npz")
WRAP_IMAGES = sorted(
    {key.removesuffix("_image") for key in WRAP if key.endswith("_image")}
)


@pytest.mark.parametrize("name", WRAP_IMAGES)
def test_periodic_fill_equals_xrtoolz_wrap_boundary(name: str) -> None:
    image = jnp.asarray(WRAP[f"{name}_image"])
    expected = WRAP[f"{name}_filled"]
    mask = jnp.isfinite(image)
    scale = float(np.max(np.abs(expected[np.isfinite(WRAP[f"{name}_image"])])))
    tight = fill_gaps(
        image,
        mask=mask,
        periodic=(False, True),
        solver=lx.CG(rtol=1e-13, atol=1e-13),
    )
    assert_matches_oracle(tight, expected, rtol=0, atol=1e-12 * scale)
    # the seam image's hole reads 6, the mean of 4, 8, 2 and 10 across the seam
    if name == "seam":
        assert abs(float(tight[1, 0]) - 6.0) < 1e-12
