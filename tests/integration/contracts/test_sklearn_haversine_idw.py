"""`InverseDistance` under the geodesic metric (#120) against scikit-learn's haversine.

The fixture ``sklearn_haversine_idw.npz`` holds 400 sources in (λ, φ) degrees, 300
uniform on the sphere and 100 in clusters at both poles and either side of the
dateline, with 240 targets drawn the same way. For each target it holds the IDW
of the k nearest by ``BallTree(metric="haversine")`` (scikit-learn 1.9.0), weighted
by θ⁻ᵖ and normalised, for k = 8, p = 2 with no radius and for k = 5, p = 1.5 with
an arc radius of 0.1 rad (whose targets with no source that close read NaN). Its
generator, ``gen_haversine_idw.py`` in the Wave 5 scratch directory, runs in
xr_toolz's environment, which has scikit-learn.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax.geometry import Points
from functax.interp import InverseDistance


DATA = np.load(Path(__file__).parent / "data" / "sklearn_haversine_idw.npz")


@pytest.mark.parametrize(
    ("name", "scheme"),
    [
        ("k8_p2", InverseDistance(k=8, power=2.0)),
        ("k5_p1.5_r0.1", InverseDistance(k=5, power=1.5, radius=0.1)),
    ],
)
def test_equals_scikit_learn_haversine_idw(name: str, scheme: InverseDistance) -> None:
    sources = Points(jnp.asarray(DATA["sources"]), metric="geodesic")
    targets = Points(jnp.asarray(DATA["targets"]), metric="geodesic")
    plan = scheme.plan(sources, targets, unit="degrees")
    expected = DATA[name]
    assert_matches_oracle(plan(jnp.asarray(DATA["values"])), expected, rtol=1e-12)
    # the arc radius leaves the targets with no source within 0.1 rad NaN
    assert np.isnan(expected).any() == (scheme.radius is not None)
