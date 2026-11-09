"""Plans in turn: a warp then a sample, and three plans, folded into one stencil and
equal to applying them one by one.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from _harness import (
    assert_matches_oracle,
)

from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import (
    Linear,
    Plan,
)


def _spectral_and_srf() -> tuple[Plan, Plan]:
    """Spectral interpolation onto 12 band centres, then a 7-channel read-out."""
    rng = np.random.default_rng(20)
    fine = Axis(jnp.asarray(np.cumsum(rng.uniform(0.5, 1.5, size=20))))
    centres = np.sort(rng.uniform(float(fine.values[0]), float(fine.values[-1]), 12))
    spectral = Linear().plan(fine, Points(jnp.asarray(centres)[:, None]))
    channels = np.sort(rng.uniform(centres[0], centres[-1], 7))
    srf = Linear().plan(
        Axis(jnp.asarray(centres)), Points(jnp.asarray(channels)[:, None])
    )
    return srf, spectral


class TestWarpThenSample:
    def test_a_warp_then_a_sample(self) -> None:
        source = RectilinearGrid((Axis(jnp.arange(6.0)), Axis(jnp.arange(7.0))))
        destination = RectilinearGrid(
            (Axis(jnp.linspace(0.2, 4.8, 4)), Axis(jnp.linspace(0.3, 5.7, 5)))
        )
        warp = Linear().plan(source, destination)
        sites = Points(jax.random.uniform(jax.random.key(24), (8, 2), maxval=4.5))
        sample = Linear().plan(destination, sites)
        field = jax.random.normal(jax.random.key(25), (42,))
        observed = (sample @ warp)(field)
        assert_matches_oracle(observed, sample(warp(field)), rtol=1e-12, atol=1e-14)

    def test_three_plans(self) -> None:
        srf, spectral = _spectral_and_srf()
        binning = Linear().plan(
            Axis(jnp.arange(7.0)), Points(jnp.array([[1.5], [4.5]]))
        )
        radiance = jax.random.normal(jax.random.key(27), (20,))
        chained = (binning @ srf) @ spectral
        assert_matches_oracle(
            chained(radiance),
            binning(srf(spectral(radiance))),
            rtol=1e-12,
            atol=1e-14,
        )
