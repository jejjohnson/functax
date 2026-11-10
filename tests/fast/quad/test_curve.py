"""Tests for `functax.quad.polyline` and `CurveRule`. The fast tier: the weights sum
to the length, Gauss–Legendre exactness along a straight segment, the normals'
length, orientation and sign, the node layout, and validation.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from functax import ValidationError
from functax.quad import CurveRule, Rule, polyline


ZIGZAG = jnp.array([[0.0, 0.0], [3.0, 1.0], [2.5, -2.0], [-1.0, 0.7], [0.0, 5.0]])


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_gauss_legendre_is_exact_along_a_segment_to_degree_2n_minus_1(n: int) -> None:
    start, end = np.array([0.5, -1.0]), np.array([2.0, 3.0])
    transect = polyline(jnp.asarray(np.stack([start, end])), n, rule="gauss_legendre")
    length = float(np.linalg.norm(end - start))
    nodes = np.asarray(transect.nodes)
    # arc length of each node from the start
    arc = np.linalg.norm(nodes - start, axis=1)
    for degree in range(2 * n):
        # ∫₀^ℓ sᵈ ds = ℓᵈ⁺¹/(d + 1)
        exact = length ** (degree + 1) / (degree + 1)
        approx = float(np.dot(transect.weights, arc**degree))
        assert abs(approx - exact) < 1e-13 * exact
    # one degree past exactness, the rule misses
    degree = 2 * n
    exact = length ** (degree + 1) / (degree + 1)
    assert abs(float(np.dot(transect.weights, arc**degree)) - exact) > 1e-8 * exact


def test_gauss_legendre_is_exact_for_a_polynomial_in_the_coordinates() -> None:
    # c(x, y) = x²y − 3xy² + y³ is cubic along any line, so n = 2 is exact
    start, end = np.array([1.0, 2.0]), np.array([-2.0, 0.5])
    transect = polyline(jnp.asarray(np.stack([start, end])), 2, rule="gauss_legendre")
    x, y = np.asarray(transect.nodes).T
    approx = float(np.dot(transect.weights, x**2 * y - 3 * x * y**2 + y**3))
    # the same integral with 20 Gauss–Legendre nodes as the reference
    reference = polyline(jnp.asarray(np.stack([start, end])), 20, rule="gauss_legendre")
    rx, ry = np.asarray(reference.nodes).T
    exact = float(np.dot(reference.weights, rx**2 * ry - 3 * rx * ry**2 + ry**3))
    assert abs(approx - exact) < 1e-13 * abs(exact)


@pytest.mark.parametrize("rule", ["midpoint", "gauss_legendre"])
def test_normals_are_unit_and_orthogonal_to_the_tangent(rule: str) -> None:
    n = 3
    transect = polyline(ZIGZAG, n, rule=rule)
    normals = np.asarray(transect.normals)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-15)
    edges = np.diff(np.asarray(ZIGZAG), axis=0)
    tangents = np.repeat(edges / np.linalg.norm(edges, axis=1)[:, None], n, axis=0)
    np.testing.assert_allclose(np.sum(normals * tangents, axis=1), 0.0, atol=1e-15)
    # +90°: the cross product t × n is +1, the normal on the left
    cross = tangents[:, 0] * normals[:, 1] - tangents[:, 1] * normals[:, 0]
    np.testing.assert_allclose(cross, 1.0, atol=1e-15)


def test_is_a_rule_that_unpacks() -> None:
    transect = polyline(ZIGZAG, 2)
    assert isinstance(transect, Rule)
    nodes, weights = transect
    assert nodes.shape == (8, 2)
    assert weights.shape == (8,)


def test_one_dimensional_polyline() -> None:
    transect = polyline(jnp.array([[0.0], [2.0], [-1.0]]), 1)
    assert transect.normals is None
    assert transect.weights.tolist() == [2.0, 3.0]


@pytest.mark.parametrize("n", [0, -1, True])
def test_rejects_a_bad_count(n: int) -> None:
    with pytest.raises(ValidationError, match="'n' must be an integer"):
        polyline(ZIGZAG, n)


def test_rejects_an_unknown_rule() -> None:
    with pytest.raises(ValidationError, match="'rule' must be one of"):
        polyline(ZIGZAG, 2, rule="simpson")


@pytest.mark.parametrize("shape", [(1, 2), (4,), (3, 0), (2, 2, 2)])
def test_rejects_bad_vertices(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValidationError, match="'vertices' must be"):
        inspect.unwrap(polyline)(jnp.zeros(shape), 2)


def test_curve_rule_rejects_misshapen_normals() -> None:
    # the import hook in conftest.py checks the annotation before the body
    with pytest.raises((ValidationError, TypeCheckError), match="normals"):
        CurveRule(jnp.zeros((3, 2)), jnp.ones(3), jnp.zeros((2, 2)))
    with pytest.raises(ValidationError, match="'normals' are defined for 2-D"):
        CurveRule(jnp.zeros((3, 3)), jnp.ones(3), jnp.zeros((3, 3)))
