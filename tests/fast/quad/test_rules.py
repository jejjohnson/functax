"""Tests for `functax.quad`'s `Rule` and its constructors (#44). The fast tier:
gate G4-exact for Gauss–Legendre and Clenshaw–Curtis, SciPy for the trapezoid
and Simpson rules on random non-uniform nodes, the interval map, and validation.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle
from jaxtyping import TypeCheckError
from scipy.integrate import simpson as scipy_simpson, trapezoid as scipy_trapezoid

from functax import ValidationError
from functax.quad import Rule, clenshaw_curtis, gauss_legendre, simpson, trapezoid


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


class TestExactness:
    """Gate G4-exact: each rule integrates the polynomials its degree promises."""

    @pytest.mark.parametrize("n", [1, 3, 8])
    @pytest.mark.parametrize("interval", [(-1.0, 1.0), (0.5, 2.0)])
    def test_gauss_legendre_is_exact_to_degree_2n_minus_1(
        self, n: int, interval: tuple[float, float]
    ) -> None:
        rule = gauss_legendre(n, interval=interval)
        for degree in range(2 * n):
            estimate = np.sum(rule.weights * rule.nodes**degree)
            exact = _monomial_integral(degree, *interval)
            assert_matches_oracle(estimate, exact, rtol=1e-13, atol=1e-15)

    def test_trapezoid_is_exact_for_lines(self) -> None:
        nodes = jnp.asarray(_random_nodes(7, seed=5))
        rule = trapezoid(nodes)
        exact = _monomial_integral(1, float(nodes[0]), float(nodes[-1]))
        assert_matches_oracle(rule.weights @ rule.nodes, exact, rtol=1e-14)


class TestMatchesScipy:
    @pytest.mark.parametrize("count", [1, 2, 3, 4, 7, 10])
    @pytest.mark.parametrize("descending", [False, True])
    def test_trapezoid_weights(self, count: int, descending: bool) -> None:
        nodes = _random_nodes(count, seed=count)
        nodes = nodes[::-1] if descending else nodes
        rule = trapezoid(jnp.asarray(nodes))
        expected = _scipy_weights(scipy_trapezoid, nodes)
        assert_matches_oracle(rule.weights, expected, rtol=1e-14, atol=1e-15)

    @pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 8, 11])
    @pytest.mark.parametrize("descending", [False, True])
    def test_simpson_weights(self, count: int, descending: bool) -> None:
        nodes = _random_nodes(count, seed=10 + count)
        nodes = nodes[::-1] if descending else nodes
        rule = simpson(jnp.asarray(nodes))
        expected = _scipy_weights(scipy_simpson, nodes)
        assert_matches_oracle(rule.weights, expected, rtol=1e-14, atol=1e-14)


class TestRule:
    def test_unpacks_into_nodes_and_weights(self) -> None:
        nodes, weights = gauss_legendre(4)
        reference_nodes, reference_weights = np.polynomial.legendre.leggauss(4)
        np.testing.assert_array_equal(nodes, reference_nodes)
        np.testing.assert_array_equal(weights, reference_weights)

    def test_gauss_nodes_are_host_float64(self) -> None:
        rule = gauss_legendre(5, interval=(2.0, 3.0))
        assert isinstance(rule.nodes, np.ndarray)
        assert rule.nodes.dtype == np.float64
        assert np.all(np.diff(rule.nodes) > 0)

    def test_a_reversed_interval_negates_the_rule(self) -> None:
        forward = clenshaw_curtis(6, interval=(0.0, 2.0))
        backward = clenshaw_curtis(6, interval=(2.0, 0.0))
        np.testing.assert_allclose(backward.weights, -forward.weights, rtol=1e-15)

    def test_float32_nodes_keep_float32(self) -> None:
        nodes = jnp.linspace(0.0, 1.0, 5, dtype=jnp.float32)
        assert trapezoid(nodes).weights.dtype == jnp.float32
        assert simpson(nodes).weights.dtype == jnp.float32

    def test_integer_nodes_become_floating(self) -> None:
        rule = trapezoid(jnp.arange(4))
        assert jnp.issubdtype(rule.weights.dtype, jnp.floating)

    def test_rejects_weights_that_are_not_1d(self) -> None:
        # the import hook in conftest.py checks the annotation before the body
        with pytest.raises((ValidationError, TypeCheckError), match="weights"):
            Rule(jnp.zeros(2), jnp.zeros((2, 2)))

    def test_rejects_nodes_without_one_entry_per_weight(self) -> None:
        with pytest.raises((ValidationError, TypeCheckError), match=r"weights|nodes"):
            Rule(jnp.zeros(3), jnp.zeros(2))


class TestValidation:
    @pytest.mark.parametrize("n", [0, -2, True])
    def test_gauss_legendre_rejects_a_bad_count(self, n: int) -> None:
        with pytest.raises(
            ValidationError, match="'n' must be an integer of at least 1"
        ):
            gauss_legendre(n)

    @pytest.mark.parametrize("n", [1, 0])
    def test_clenshaw_curtis_needs_two_nodes(self, n: int) -> None:
        with pytest.raises(
            ValidationError, match="'n' must be an integer of at least 2"
        ):
            clenshaw_curtis(n)

    def test_rejects_a_non_integer_count(self) -> None:
        with pytest.raises(ValidationError, match="'n' must be an integer"):
            inspect.unwrap(clenshaw_curtis)(2.5)

    @pytest.mark.parametrize("interval", [(0.0, np.inf), (np.nan, 1.0), ("a", 1.0)])
    def test_rejects_a_bad_interval(self, interval) -> None:
        with pytest.raises(ValidationError, match="'interval' must be two finite"):
            inspect.unwrap(gauss_legendre)(3, interval=interval)

    @pytest.mark.parametrize("nodes", [jnp.zeros((2, 2)), jnp.zeros(0)])
    def test_rejects_nodes_that_are_not_a_vector(self, nodes) -> None:
        with pytest.raises(ValidationError, match="'nodes' must be a non-empty 1-D"):
            inspect.unwrap(trapezoid)(nodes)
