"""functax.transform (#23) against geonnax and spectraldiffx, pinned as data: the
spherical-harmonic transform and the non-uniform FFTs against geonnax be5537d,
and the analysis against spectraldiffx d68f8f6's complex SHT on the same
Gauss–Legendre grid.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from _harness import assert_matches_oracle

from functax.quad import gauss_legendre
from functax.transform import (
    SphericalHarmonicTransform,
    nonuniform_irfft,
    nonuniform_rfft,
)


DATA = Path(__file__).parent / "data"
GEONNAX = np.load(DATA / "geonnax_transform.npz")
SPECTRALDIFFX = np.load(DATA / "spectraldiffx_sht.npz")


def test_sht_equals_geonnax() -> None:
    sht = SphericalHarmonicTransform.init(9, 17, 7)
    forward = sht.forward(jnp.asarray(GEONNAX["field"]))
    assert_matches_oracle(forward, GEONNAX["forward"], rtol=1e-12, atol=1e-13)
    inverse = sht.inverse(jnp.asarray(GEONNAX["coeffs"]))
    assert_matches_oracle(inverse, GEONNAX["inverse"], rtol=1e-12, atol=1e-13)


def test_nonuniform_ffts_equal_geonnax() -> None:
    xi, u, w = (jnp.asarray(GEONNAX[k]) for k in ("xi", "u", "w"))
    blocks = nonuniform_rfft(u, xi, w, (3, 4))
    for index, block in enumerate(blocks):
        assert_matches_oracle(block, GEONNAX[f"rfft_{index}"], rtol=1e-12, atol=1e-14)
    values = nonuniform_irfft(blocks, jnp.asarray(GEONNAX["query"]), (3, 4))
    assert_matches_oracle(values, GEONNAX["irfft"], rtol=1e-12, atol=1e-13)
    (block,) = nonuniform_rfft(u[:, :40], jnp.asarray(GEONNAX["xi1"]), w[:40], (5,))
    assert_matches_oracle(block, GEONNAX["rfft_1d"], rtol=1e-12, atol=1e-14)


def test_analysis_equals_spectraldiffx_through_the_documented_mapping() -> None:
    # gate of #23: spectraldiffx's û(l, m) is an FFT in longitude and orthonormal
    # P̃_l^|m| (with the Condon–Shortley phase) in latitude, rows north to south,
    # on Gauss–Legendre nodes in cos θ and longitudes 2πk/n_lon from 0: the grid
    # of this transform with the rows reversed. Then c_l0 = √(2π)/n_lon Re û(l, 0),
    # and for m > 0, c_lm = 2√π/n_lon Re û(l, m) and c_l,−m = −2√π/n_lon Im û(l, m).
    n_lat, n_lon = SPECTRALDIFFX["field"].shape
    l_max = n_lat - 1
    sht = SphericalHarmonicTransform.init(n_lat, n_lon, l_max)
    # the same grid: its nodes in cos θ, reversed, are this transform's sines of
    # latitude, and its longitudes start at 0
    nodes, _ = gauss_legendre(n_lat)
    assert_matches_oracle(SPECTRALDIFFX["cos_theta"][::-1], nodes, rtol=0, atol=1e-14)
    longitudes = 2 * np.pi * np.arange(n_lon) / n_lon
    assert_matches_oracle(SPECTRALDIFFX["phi"], longitudes, rtol=0, atol=1e-14)
    field = jnp.asarray(SPECTRALDIFFX["field"][::-1])[None]
    coefficients = np.asarray(sht.forward(field)[0])
    u_hat = SPECTRALDIFFX["coefficients"]
    expected = np.zeros_like(coefficients)
    for ell in range(l_max + 1):
        expected[ell * ell + ell] = math.sqrt(2 * math.pi) / n_lon * u_hat[ell, 0].real
        for m in range(1, ell + 1):
            expected[ell * ell + ell + m] = (
                2 * math.sqrt(math.pi) / n_lon * u_hat[ell, m].real
            )
            expected[ell * ell + ell - m] = (
                -2 * math.sqrt(math.pi) / n_lon * u_hat[ell, m].imag
            )
    assert_matches_oracle(
        coefficients, expected, rtol=0, atol=1e-10 * np.abs(expected).max()
    )
