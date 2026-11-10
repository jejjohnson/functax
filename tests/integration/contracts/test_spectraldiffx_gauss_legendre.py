"""functax's sphere Gauss–Legendre rule (#45) against spectraldiffx's
``_gauss_legendre_nodes_weights``, pinned at spectraldiffx d68f8f6 as data.

The fixture was written by ``gen_spectraldiffx_gauss_legendre.py`` (in the Wave 4
scratch directory), which loads ``spectraldiffx/_src/spherical/grid.py`` from a
``git archive`` of d68f8f6 by path and saves the nodes μ = cos θ and weights for
each n. spectraldiffx orders μ from north to south (+1 down to −1); functax's
latitudes ascend south to north, so the reference is read reversed. On the
sphere, μ = cos θ = sin φ is the z of functax's nodes, and spectraldiffx's
weights (summing to 2) are functax's per-ring weights times n_lon / 2π.

The nodes agree to 2.3e-16. The weights agree to 4.2e-15 absolute but not to
1e-14 relative: NumPy's ``leggauss`` and SciPy's ``roots_legendre`` both miss
the smallest end weights by about 1e-12 relative at n = 64 (checked against
mpmath at 40 digits), in opposite directions, so the weights are compared
absolutely, as they are O(1/n) and sum to 2.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax.quad import gauss_legendre, sphere_gauss_legendre


SPECTRALDIFFX = np.load(
    Path(__file__).parent / "data" / "spectraldiffx_gauss_legendre.npz"
)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 8, 16, 32, 33, 64])
def test_sphere_nodes_and_weights_equal_spectraldiffx(n: int) -> None:
    n_lon = 2 * n
    rule = sphere_gauss_legendre(n, n_lon)
    # one entry per latitude ring: latitude-major, so every n_lon-th node
    ring_sines = np.asarray(rule.nodes[::n_lon, 2])
    ring_weights = np.asarray(rule.weights[::n_lon]) * n_lon / (2.0 * math.pi)
    expected_nodes = SPECTRALDIFFX[f"nodes_{n}"][::-1]
    expected_weights = SPECTRALDIFFX[f"weights_{n}"][::-1]
    assert_matches_oracle(ring_sines, expected_nodes, rtol=0.0, atol=1e-14)
    assert_matches_oracle(ring_weights, expected_weights, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 8, 16, 32, 33, 64])
def test_interval_rule_equals_spectraldiffx(n: int) -> None:
    nodes, weights = gauss_legendre(n)
    assert_matches_oracle(
        nodes, SPECTRALDIFFX[f"nodes_{n}"][::-1], rtol=0.0, atol=1e-14
    )
    assert_matches_oracle(
        weights, SPECTRALDIFFX[f"weights_{n}"][::-1], rtol=0.0, atol=1e-14
    )
