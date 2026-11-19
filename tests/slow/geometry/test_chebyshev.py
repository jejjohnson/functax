"""Tests for `chebyshev_nodes`. The slow tier: on any interval and for degrees up to
64, the nodes are the zeros (Gauss) and extrema (Lobatto) of T_N.
"""

from __future__ import annotations

import hypothesis.strategies as st
import numpy as np
from _harness import assert_matches_oracle
from hypothesis import given, settings

from functax.geometry import chebyshev_nodes


@settings(max_examples=20, deadline=None)
@given(
    degree=st.integers(1, 64),
    lower=st.floats(-10.0, 10.0),
    width=st.floats(0.1, 10.0),
)
def test_nodes_are_the_zeros_and_extrema_of_t_n(
    degree: int, lower: float, width: float
) -> None:
    # back on [−1, 1]: T_N(ξ) = 0 at the Gauss nodes, T_N′(ξ) = 0 at the
    # interior Lobatto nodes and T_N(ξ) = (−1)ʲ at all of them
    interval = (lower, lower + width)
    t_n = np.zeros(degree + 1)
    t_n[-1] = 1.0
    derivative = np.polynomial.chebyshev.chebder(t_n)

    def reference(nodes: object) -> np.ndarray:
        return (2.0 * np.asarray(nodes) - (2.0 * lower + width)) / width

    gauss = reference(chebyshev_nodes(degree, kind="gauss", interval=interval))
    lobatto = reference(chebyshev_nodes(degree, interval=interval))
    # mapping back costs a few ulps of the interval, amplified by |T_N′| ≤ N²
    atol = 1e-13 * degree**2 * max(1.0, abs(lower) + width) / width
    assert_matches_oracle(
        np.polynomial.chebyshev.chebval(gauss, t_n), np.zeros(degree), rtol=0, atol=atol
    )
    assert_matches_oracle(
        np.polynomial.chebyshev.chebval(lobatto, t_n),
        (-1.0) ** np.arange(degree + 1),
        rtol=0,
        atol=atol,
    )
    assert_matches_oracle(
        np.polynomial.chebyshev.chebval(lobatto[1:-1], derivative),
        np.zeros(degree - 1),
        rtol=0,
        atol=atol * degree,
    )
