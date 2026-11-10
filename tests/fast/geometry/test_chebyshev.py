"""Tests for `chebyshev_nodes`. The fast tier: NumPy's Chebyshev points, the order,
the interval map, the pairing with `clenshaw_curtis`, and validation.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax import GeometryError, ValidationError
from functax.geometry import chebyshev_nodes
from functax.quad import clenshaw_curtis


@pytest.mark.parametrize("degree", [1, 2, 5, 16, 33])
def test_nodes_are_numpy_chebyshev_points_descending(degree: int) -> None:
    # chebpts2 (second kind, the extrema) and chebpts1 (first kind, the zeros)
    # ascend; spectraldiffx's order, kept here, descends
    lobatto = chebyshev_nodes(degree)
    expected = np.polynomial.chebyshev.chebpts2(degree + 1)[::-1]
    assert_matches_oracle(lobatto, expected, rtol=0.0, atol=1e-15)
    gauss = chebyshev_nodes(degree, kind="gauss")
    expected = np.polynomial.chebyshev.chebpts1(degree)[::-1]
    assert_matches_oracle(gauss, expected, rtol=0.0, atol=1e-15)


def test_lobatto_nodes_hit_both_ends_and_gauss_nodes_avoid_them() -> None:
    lobatto = chebyshev_nodes(7, interval=(-3.0, 4.0))
    assert lobatto.shape == (8,)
    assert lobatto[0] == 4.0
    assert lobatto[-1] == -3.0
    assert bool(jnp.all(jnp.diff(lobatto) < 0))
    gauss = chebyshev_nodes(7, kind="gauss", interval=(-3.0, 4.0))
    assert gauss.shape == (7,)
    assert bool(jnp.all((gauss > -3.0) & (gauss < 4.0)))
    assert bool(jnp.all(jnp.diff(gauss) < 0))


@pytest.mark.parametrize("kind", ["lobatto", "gauss"])
def test_nodes_on_an_interval_are_the_affine_map(kind: str) -> None:
    reference = chebyshev_nodes(9, kind=kind)
    mapped = chebyshev_nodes(9, kind=kind, interval=(2.0, 6.0))
    assert_matches_oracle(mapped, 4.0 + 2.0 * reference, rtol=1e-15)


def test_clenshaw_curtis_pairs_with_the_reversed_nodes() -> None:
    # clenshaw_curtis ascends; its weights integrate f at the reversed nodes,
    # exactly for degree ≤ N + 1 (N even): ∫₀² x⁵ dx = 32/3
    nodes = chebyshev_nodes(6, interval=(0.0, 2.0))
    rule = clenshaw_curtis(7, interval=(0.0, 2.0))
    assert_matches_oracle(rule.nodes, nodes[::-1], rtol=0.0, atol=1e-15)
    assert abs(float(rule.weights @ nodes[::-1] ** 5) - 32.0 / 3.0) < 1e-13


@pytest.mark.parametrize("n", [0, -1, 2.0, True])
def test_n_must_be_a_positive_integer(n: object) -> None:
    with pytest.raises(GeometryError, match="'n' must be an integer of at least 1"):
        inspect.unwrap(chebyshev_nodes)(n)


def test_unknown_kind_and_bad_intervals_are_rejected() -> None:
    with pytest.raises(GeometryError, match="'kind' must be one of"):
        chebyshev_nodes(4, kind="gauss-lobatto")
    with pytest.raises(GeometryError, match="'interval' must have a < b"):
        chebyshev_nodes(4, interval=(1.0, -1.0))
    with pytest.raises(ValidationError, match="'interval' must be two finite"):
        chebyshev_nodes(4, interval=(0.0, float("inf")))
