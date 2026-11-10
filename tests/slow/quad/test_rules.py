"""Tests for `functax.quad`'s trapezoid and Simpson rules (#44). The slow tier:
gradients in the nodes and jit/vmap invariance, which compile per shape.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from hypothesis import given, settings, strategies as st

from functax.quad import clenshaw_curtis, gauss_legendre, simpson, trapezoid


def _monomial_integral(degree: int, lower: float, upper: float) -> float:
    """∫ₐᵇ xᵈ dx = (bᵈ⁺¹ − aᵈ⁺¹)/(d + 1)."""
    return (upper ** (degree + 1) - lower ** (degree + 1)) / (degree + 1)


def _random_nodes(count: int, seed: int) -> np.ndarray:
    """Sorted, non-uniform nodes on [0, 3]."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0.0, 3.0, count))


def _nodes() -> jax.Array:
    rng = np.random.default_rng(0)
    return jnp.asarray(np.sort(rng.uniform(0.0, 3.0, 7)))


def test_trapezoid_gradient_in_the_nodes() -> None:
    values = jnp.cos(_nodes())
    assert_grad_matches_fd(lambda x: trapezoid(x).weights @ values, _nodes())


def test_simpson_gradient_in_the_nodes() -> None:
    values = jnp.cos(_nodes())
    assert_grad_matches_fd(lambda x: simpson(x).weights @ values, _nodes())


def test_jit_and_vmap_agree_with_a_loop() -> None:
    rng = np.random.default_rng(1)
    batch = jnp.asarray(np.sort(rng.uniform(0.0, 3.0, (3, 6)), axis=1))
    assert_traces(lambda x: trapezoid(x).weights, batch)
    assert_traces(lambda x: simpson(x).weights, batch)


class TestExactness:
    """Gate G4-exact: each rule integrates the polynomials its degree promises."""

    @pytest.mark.parametrize("n", [2, 3, 5, 9, 17])
    def test_clenshaw_curtis_is_exact_to_its_degree(self, n: int) -> None:
        rule = clenshaw_curtis(n, interval=(0.0, 2.0))
        degree_of_nodes = n - 1
        # N + 1 when N is even, by symmetry
        top = degree_of_nodes + 1 if degree_of_nodes % 2 == 0 else degree_of_nodes
        for degree in range(top + 1):
            estimate = np.sum(rule.weights * rule.nodes**degree)
            exact = _monomial_integral(degree, 0.0, 2.0)
            assert_matches_oracle(estimate, exact, rtol=1e-13, atol=1e-15)

    def test_simpson_is_exact_for_quadratics_on_uneven_nodes(self) -> None:
        nodes = jnp.asarray(_random_nodes(9, seed=4))
        rule = simpson(nodes)
        exact = _monomial_integral(2, float(nodes[0]), float(nodes[-1]))
        assert_matches_oracle(rule.weights @ rule.nodes**2, exact, rtol=1e-13)


@settings(max_examples=25, deadline=None)
@given(
    n=st.integers(min_value=2, max_value=12),
    lower=st.floats(-5.0, 5.0),
    width=st.floats(0.1, 5.0),
)
def test_weights_sum_to_the_interval_length(n: int, lower: float, width: float) -> None:
    upper = lower + width
    for rule in (
        gauss_legendre(n, interval=(lower, upper)),
        clenshaw_curtis(n, interval=(lower, upper)),
    ):
        assert_matches_oracle(np.sum(rule.weights), width, rtol=1e-13)
    nodes = jnp.linspace(lower, upper, n)
    for rule in (trapezoid(nodes), simpson(nodes)):
        assert_matches_oracle(jnp.sum(rule.weights), width, rtol=1e-13)
