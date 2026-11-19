"""Tests for `functax.quad`'s rules (#44). The slow tier: gradients in the nodes,
jit/vmap invariance, Simpson's SciPy weights on longer node vectors and the
weights' sum on random intervals, which compile per shape.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from hypothesis import given, settings, strategies as st
from scipy.integrate import simpson as scipy_simpson

from functax.quad import clenshaw_curtis, gauss_legendre, simpson, trapezoid


def _monomial_integral(degree: int, lower: float, upper: float) -> float:
    """∫ₐᵇ xᵈ dx = (bᵈ⁺¹ − aᵈ⁺¹)/(d + 1)."""
    return (upper ** (degree + 1) - lower ** (degree + 1)) / (degree + 1)


def _random_nodes(count: int, seed: int) -> np.ndarray:
    """Sorted, non-uniform nodes on [0, 3]."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0.0, 3.0, count))


def _scipy_weights(integrator, nodes: np.ndarray) -> np.ndarray:
    """The weights SciPy uses on these nodes: its integral of each unit vector."""
    return np.array([integrator(unit, x=nodes) for unit in np.eye(nodes.size)])


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


_TRAPEZOID_AND_SIMPSON = jax.jit(
    lambda nodes: (trapezoid(nodes).weights, simpson(nodes).weights)
)


class TestExactness:
    """Gate G4-exact: each rule integrates the polynomials its degree promises."""

    @settings(max_examples=20, deadline=None)
    @given(
        n=st.integers(min_value=2, max_value=24),
        lower=st.floats(-5.0, 5.0),
        width=st.floats(0.1, 5.0),
    )
    def test_gauss_and_clenshaw_curtis_are_exact_on_any_interval(
        self, n: int, lower: float, width: float
    ) -> None:
        # ∫ₐᵇ tᵏ dx = h (1 − (−1)ᵏ⁺¹)/(k + 1) with t = (x − c)/h, for every k up
        # to 2n − 1 (Gauss) and n − 1, or n when n is odd (Clenshaw–Curtis)
        centre, half = lower + width / 2, width / 2
        top_cc = n if n % 2 == 1 else n - 1
        for rule, top in (
            (gauss_legendre(n, interval=(lower, lower + width)), 2 * n - 1),
            (clenshaw_curtis(n, interval=(lower, lower + width)), top_cc),
        ):
            scaled = (np.asarray(rule.nodes) - centre) / half
            powers = np.arange(top + 1)
            estimates = np.asarray(rule.weights) @ scaled[:, None] ** powers
            exact = half * (1 - (-1.0) ** (powers + 1)) / (powers + 1)
            assert_matches_oracle(estimates, exact, rtol=0, atol=1e-13 * width)

    @settings(max_examples=20, deadline=None)
    @given(seed=st.integers(0, 2**31 - 1))
    def test_trapezoid_and_simpson_are_exact_on_uneven_nodes(self, seed: int) -> None:
        # trapezoid for lines, Simpson for quadratics, on 9 random sorted nodes
        nodes = _random_nodes(9, seed)
        coefficients = np.random.default_rng(seed).normal(size=3)
        line = coefficients[0] + coefficients[1] * nodes
        quadratic = line + coefficients[2] * nodes**2
        a, b = nodes[0], nodes[-1]
        exact_line = coefficients[0] * (b - a) + coefficients[1] * (b**2 - a**2) / 2
        exact_quadratic = exact_line + coefficients[2] * (b**3 - a**3) / 3
        trapezoid_weights, simpson_weights = _TRAPEZOID_AND_SIMPSON(jnp.asarray(nodes))
        assert_matches_oracle(
            trapezoid_weights @ line, exact_line, rtol=1e-12, atol=1e-13
        )
        assert_matches_oracle(
            simpson_weights @ quadratic, exact_quadratic, rtol=1e-12, atol=1e-13
        )

    def test_simpson_is_exact_for_quadratics_on_uneven_nodes(self) -> None:
        nodes = jnp.asarray(_random_nodes(9, seed=4))
        rule = simpson(nodes)
        exact = _monomial_integral(2, float(nodes[0]), float(nodes[-1]))
        assert_matches_oracle(rule.weights @ rule.nodes**2, exact, rtol=1e-13)


class TestMatchesScipy:
    @pytest.mark.parametrize("count", [8, 11])
    @pytest.mark.parametrize("descending", [False, True])
    def test_simpson_weights_over_several_pairs(
        self, count: int, descending: bool
    ) -> None:
        nodes = _random_nodes(count, seed=10 + count)
        nodes = nodes[::-1] if descending else nodes
        rule = simpson(jnp.asarray(nodes))
        expected = _scipy_weights(scipy_simpson, nodes)
        assert_matches_oracle(rule.weights, expected, rtol=1e-14, atol=1e-14)


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
