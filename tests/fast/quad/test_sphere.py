"""Tests for the sphere rules in `functax.quad`. The fast tier: the stated degree,
the weights' sum on a sphere of radius a, the wrapped geometry grid and
validation. The harmonic exactness sweep is in the slow tier.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from functax import GeometryError, ValidationError
from functax.geometry import gauss_legendre_grid
from functax.quad import (
    Rule,
    SphereRule,
    sphere_fibonacci,
    sphere_gauss_legendre,
    sphere_healpix,
    sphere_latlon,
)


RULES = [
    pytest.param(lambda: sphere_gauss_legendre(1), id="gauss_legendre-1"),
    pytest.param(lambda: sphere_gauss_legendre(4), id="gauss_legendre-4"),
    pytest.param(lambda: sphere_gauss_legendre(8, 9), id="gauss_legendre-8x9"),
    pytest.param(lambda: sphere_gauss_legendre(5, 3), id="gauss_legendre-5x3"),
    pytest.param(lambda: sphere_gauss_legendre(16, 32), id="gauss_legendre-16"),
    pytest.param(lambda: sphere_latlon(18, 36), id="latlon-18x36"),
    pytest.param(lambda: sphere_latlon(3, 1), id="latlon-3x1"),
    pytest.param(lambda: sphere_healpix(1), id="healpix-1"),
    pytest.param(lambda: sphere_healpix(4, nest=False), id="healpix-4-ring"),
    pytest.param(lambda: sphere_fibonacci(100), id="fibonacci-100"),
]


@pytest.mark.parametrize(
    ("n_lat", "n_lon", "degree"),
    [(1, None, 1), (4, None, 7), (8, 9, 8), (5, 3, 2), (6, 40, 11)],
)
def test_gauss_legendre_degree_is_min_of_both_limits(
    n_lat: int, n_lon: int | None, degree: int
) -> None:
    assert sphere_gauss_legendre(n_lat, n_lon).degree == degree


@pytest.mark.parametrize("make", RULES)
def test_unit_weights_sum_to_four_pi(make) -> None:
    assert abs(float(make().weights.sum()) - 4.0 * math.pi) < 1e-13


@pytest.mark.parametrize(
    "build",
    [
        lambda radius: sphere_gauss_legendre(6, radius=radius),
        lambda radius: sphere_latlon(4, 8, radius=radius),
        lambda radius: sphere_healpix(2, radius=radius),
        lambda radius: sphere_fibonacci(30, radius=radius),
    ],
)
def test_radius_scales_the_weights_by_its_square(build) -> None:
    radius = 6371.0
    unit, scaled = build(1.0), build(radius)
    assert scaled.radius == radius
    np.testing.assert_allclose(scaled.weights, unit.weights * radius**2, rtol=1e-15)
    np.testing.assert_array_equal(scaled.nodes, unit.nodes)
    expected = 4.0 * math.pi * radius**2
    assert abs(float(scaled.weights.sum()) - expected) < 1e-13 * expected


def test_is_a_rule_that_unpacks_to_unit_vector_nodes() -> None:
    rule = sphere_gauss_legendre(3)
    assert isinstance(rule, Rule)
    nodes, weights = rule
    assert nodes.shape == (18, 3)
    assert weights.shape == (18,)
    np.testing.assert_allclose(jnp.linalg.norm(nodes, axis=-1), 1.0, atol=1e-15)


def test_gauss_legendre_wraps_the_geometry_grid() -> None:
    # latitudes ascend south to north and longitudes start at −π
    rule = sphere_gauss_legendre(4, 6)
    points, weights = gauss_legendre_grid(4, 6)
    np.testing.assert_array_equal(rule.nodes, points)
    np.testing.assert_array_equal(rule.weights, weights)
    z = np.asarray(rule.nodes[::6, 2])
    assert np.all(np.diff(z) > 0)
    first_longitude = math.atan2(float(rule.nodes[0, 1]), float(rule.nodes[0, 0]))
    assert abs(abs(first_longitude) - math.pi) < 1e-15


@pytest.mark.parametrize("radius", [0.0, -1.0, math.inf, math.nan])
def test_rejects_a_bad_radius(radius: float) -> None:
    with pytest.raises(GeometryError, match="'radius' must be positive and finite"):
        sphere_gauss_legendre(2, radius=radius)


def test_rejects_a_bad_count() -> None:
    with pytest.raises(GeometryError, match="'n_lat' must be an integer"):
        sphere_gauss_legendre(0)
    with pytest.raises(GeometryError, match="'nside' must be a power of two"):
        sphere_healpix(3)


def test_rejects_nodes_that_are_not_three_dimensional() -> None:
    with pytest.raises(ValidationError, match=r"'nodes' must be \(Q, 3\)"):
        SphereRule(jnp.zeros((4, 2)), jnp.ones(4), radius=1.0, degree=0)


@pytest.mark.parametrize("degree", [-1, True])
def test_rejects_a_bad_degree(degree) -> None:
    with pytest.raises(ValidationError, match="'degree' must be an integer"):
        SphereRule(jnp.zeros((4, 3)), jnp.ones(4), radius=1.0, degree=degree)
