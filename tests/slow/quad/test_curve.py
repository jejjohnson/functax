"""Tests for `functax.quad.polyline`. The slow tier: the rule traces (eager, `jit`
and `vmap` agree) and differentiates in the vertices (`jacrev` against central
differences), so a transect can move under an optimiser.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_traces

from functax.quad import polyline


ZIGZAG = jnp.array([[0.0, 0.0], [3.0, 1.0], [2.5, -2.0], [-1.0, 0.7], [0.0, 5.0]])


def _length(vertices: jnp.ndarray) -> float:
    edges = np.diff(np.asarray(vertices), axis=0)
    return float(np.sum(np.linalg.norm(edges, axis=1)))


def _flux(vertices: jnp.ndarray, rule: str) -> jnp.ndarray:
    """Φ = ∑ₖ Δsₖ c(xₖ) (u(xₖ) · nₖ) for a smooth c and a rotating wind u."""
    transect = polyline(vertices, 3, rule=rule)
    x, y = transect.nodes[:, 0], transect.nodes[:, 1]
    concentration = jnp.exp(-(x**2) - 0.5 * y**2)
    wind = jnp.stack([1.0 + 0.3 * y, -0.2 * x], axis=-1)
    normal_wind = jnp.sum(wind * transect.normals, axis=-1)
    return jnp.sum(transect.weights * concentration * normal_wind)


@pytest.mark.parametrize("rule", ["midpoint", "gauss_legendre"])
def test_flux_gradient_in_the_vertices_matches_finite_differences(rule: str) -> None:
    vertices = jnp.array([[-1.0, -0.5], [0.2, 0.1], [0.8, 1.2]])
    assert_grad_matches_fd(lambda v: _flux(v, rule), vertices)


@pytest.mark.parametrize("rule", ["midpoint", "gauss_legendre"])
def test_rule_traces_under_jit_and_vmap(rule: str) -> None:
    rng = np.random.default_rng(0)
    batch = jnp.asarray(rng.normal(size=(4, 5, 2)))

    def fields(vertices: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        transect = polyline(vertices, 3, rule=rule)
        return transect.nodes, transect.weights, transect.normals

    assert_traces(fields, batch)


@pytest.mark.parametrize("rule", ["midpoint", "gauss_legendre"])
@pytest.mark.parametrize("n", [1, 3, 8])
def test_weights_sum_to_the_length(rule: str, n: int) -> None:
    transect = polyline(ZIGZAG, n, rule=rule)
    length = _length(ZIGZAG)
    assert transect.nodes.shape == (n * 4, 2)
    assert abs(float(transect.weights.sum()) - length) < 1e-14 * length


def test_weights_sum_to_the_length_in_three_dimensions() -> None:
    rng = np.random.default_rng(0)
    vertices = jnp.asarray(rng.normal(size=(6, 3)))
    transect = polyline(vertices, 5, rule="gauss_legendre")
    length = _length(vertices)
    assert transect.normals is None
    assert abs(float(transect.weights.sum()) - length) < 1e-14 * length


@pytest.mark.parametrize("rule", ["midpoint", "gauss_legendre"])
def test_normals_flip_with_the_vertex_order(rule: str) -> None:
    n = 4
    forward = polyline(ZIGZAG, n, rule=rule)
    backward = polyline(ZIGZAG[::-1], n, rule=rule)
    # the reversed rule visits the same nodes in reverse order
    np.testing.assert_allclose(backward.nodes[::-1], forward.nodes, atol=1e-14)
    np.testing.assert_allclose(backward.weights[::-1], forward.weights, rtol=1e-14)
    np.testing.assert_allclose(backward.normals[::-1], -forward.normals, atol=1e-15)


def test_uniform_flux_across_a_straight_transect() -> None:
    # Φ = ∫ u · n dℓ: a uniform wind (0, −2) across 5 m heading +x gives −10
    transect = polyline(jnp.array([[0.0, 0.0], [5.0, 0.0]]), 4)
    wind = jnp.array([0.0, -2.0])
    flux = jnp.sum(transect.weights * (transect.normals @ wind))
    assert abs(float(flux) + 10.0) < 1e-14


def test_nodes_are_spread_by_arc_length_on_each_segment() -> None:
    transect = polyline(jnp.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0]]), 2)
    assert transect.nodes.tolist() == [[1.0, 0.0], [3.0, 0.0], [4.0, 0.5], [4.0, 1.5]]
    assert transect.weights.tolist() == [2.0, 2.0, 1.0, 1.0]


def test_keeps_float32_and_promotes_integers() -> None:
    single = polyline(ZIGZAG.astype(jnp.float32), 2, rule="gauss_legendre")
    assert single.nodes.dtype == jnp.float32
    assert single.weights.dtype == jnp.float32
    assert single.normals.dtype == jnp.float32
    integers = polyline(jnp.array([[0, 0], [3, 4]]), 1)
    assert jnp.issubdtype(integers.weights.dtype, jnp.floating)
    assert math.isclose(float(integers.weights[0]), 5.0)


def test_midpoint_is_exact_for_linear_fields() -> None:
    transect = polyline(ZIGZAG, 3)
    nodes = np.asarray(transect.nodes)
    # c(x, y) = 2x − y + 1 integrates segment by segment as ℓⱼ c(midpoint)
    vertices = np.asarray(ZIGZAG)
    midpoints = 0.5 * (vertices[1:] + vertices[:-1])
    lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    exact = float(np.dot(lengths, 2 * midpoints[:, 0] - midpoints[:, 1] + 1))
    approx = float(np.dot(transect.weights, 2 * nodes[:, 0] - nodes[:, 1] + 1))
    assert abs(approx - exact) < 1e-13 * abs(exact)


def test_zero_length_segment_has_zero_weight_and_normal() -> None:
    transect = polyline(jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]), 2)
    assert transect.weights.tolist() == [0.0, 0.0, 0.5, 0.5]
    assert np.all(np.isfinite(transect.normals))
    assert transect.normals[:2].tolist() == [[0.0, 0.0], [0.0, 0.0]]
