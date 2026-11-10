"""Tests for the sphere maths and the pairwise distances in `functax.geometry`. The
slow tier: oracle sweeps, dot-product tests across policies, gradient and trace
checks, each a second or more (mostly einx compiling new shapes).
"""

from __future__ import annotations

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from hypothesis import given, settings
from scipy.spatial.distance import cdist

from functax.geometry import (
    Points,
    chord_distance,
    geodesic_distance,
    icosphere,
    lonlat_to_unit,
    pairwise_distance,
    spherical_polygon_area,
    unit_to_lonlat,
)


def _random_units(count: int, seed: int) -> jax.Array:
    """Unit vectors uniform on the sphere (normalised Gaussians)."""
    gaussian = jax.random.normal(jax.random.key(seed), (count, 3))
    norms = jnp.sqrt(jnp.sum(gaussian**2, axis=-1, keepdims=True))
    return gaussian / norms


class TestLift:
    def test_gradient_finite_at_the_poles(self) -> None:
        poles = jnp.array([[0.3, jnp.pi / 2], [-1.0, -jnp.pi / 2]])
        jacobian = jax.jacrev(lonlat_to_unit)(poles)
        assert bool(jnp.all(jnp.isfinite(jacobian)))
        assert_grad_matches_fd(lonlat_to_unit, poles)

    def test_unit_length_and_batch_dims(self) -> None:
        lonlat = jax.random.uniform(
            jax.random.key(0), (3, 4, 2), minval=-3.0, maxval=3.0
        )
        lifted = lonlat_to_unit(lonlat)
        assert lifted.shape == (3, 4, 3)
        assert_matches_oracle(jnp.sum(lifted**2, axis=-1), jnp.ones((3, 4)), rtol=1e-15)

    @pytest.mark.parametrize("unit", ["degrees", "radians"])
    def test_round_trip_away_from_the_poles(self, unit: str) -> None:
        rng = np.random.default_rng(1)
        scale = 180.0 if unit == "degrees" else np.pi
        # λ ∈ (−π, π], φ within 89.9° of the equator
        lonlat = np.stack(
            [
                rng.uniform(-0.999, 1.0, 500) * scale,
                rng.uniform(-0.999, 0.999, 500) * scale / 2,
            ],
            axis=-1,
        )
        back = unit_to_lonlat(lonlat_to_unit(jnp.asarray(lonlat), unit=unit), unit=unit)
        assert_matches_oracle(back, lonlat, rtol=1e-12, atol=1e-12 * scale)


class TestDistances:
    @settings(max_examples=20, deadline=None)
    @given(seed=st.integers(0, 2**31 - 1), radius=st.floats(1e-3, 1e4))
    def test_chord_is_twice_the_sine_of_half_the_angle(
        self, seed: int, radius: float
    ) -> None:
        # d_chord = 2a sin(d_geo / 2a), on 10,000 random pairs per example
        u = _random_units(10_000, seed)
        v = _random_units(10_000, seed + 1)
        chord = jax.vmap(
            lambda a, b: chord_distance(a[None], b[None], radius=radius)[0, 0]
        )(u, v)
        geodesic = jax.vmap(
            lambda a, b: geodesic_distance(a[None], b[None], radius=radius)[0, 0]
        )(u, v)
        expected = 2.0 * radius * jnp.sin(geodesic / (2.0 * radius))
        assert_matches_oracle(chord, expected, rtol=1e-12, atol=1e-14 * radius)

    def test_antipodes(self) -> None:
        u = _random_units(50, seed=2)
        geodesic = jnp.diagonal(geodesic_distance(u, -u, radius=6371.0))
        chord = jnp.diagonal(chord_distance(u, -u, radius=6371.0))
        assert_matches_oracle(geodesic, jnp.full(50, jnp.pi * 6371.0), rtol=1e-12)
        assert_matches_oracle(chord, jnp.full(50, 2.0 * 6371.0), rtol=1e-12)

    def test_zero_on_the_diagonal_and_symmetric(self) -> None:
        u = _random_units(30, seed=3)
        for distance in (chord_distance, geodesic_distance):
            matrix = distance(u, u)
            assert_matches_oracle(
                jnp.diagonal(matrix), jnp.zeros(30), rtol=0.0, atol=1e-15
            )
            assert_matches_oracle(matrix, matrix.T, rtol=1e-12, atol=1e-15)

    def test_a_traced_radius_is_differentiable(self) -> None:
        u = _random_units(3, seed=7)
        v = _random_units(2, seed=8)
        # ∂/∂a of Σ a‖u − v‖ is Σ ‖u − v‖
        slope = jax.jit(jax.grad(lambda a: jnp.sum(chord_distance(u, v, radius=a))))(
            2.0
        )
        assert_matches_oracle(slope, jnp.sum(chord_distance(u, v)), rtol=1e-12)

    @pytest.mark.parametrize("distance", [chord_distance, geodesic_distance])
    def test_gradient_finite_at_zero_distance(self, distance) -> None:  # type: ignore[no-untyped-def]
        u = _random_units(4, seed=4)
        gradient = jax.grad(lambda points: jnp.sum(distance(points, u)))(u)
        assert bool(jnp.all(jnp.isfinite(gradient)))

    def test_geodesic_gradient_through_the_lift_at_the_poles(self) -> None:
        poles = jnp.array([[0.0, jnp.pi / 2], [2.0, -jnp.pi / 2], [1.0, 0.3]])

        def total(lonlat: jax.Array) -> jax.Array:
            units = lonlat_to_unit(lonlat)
            return jnp.sum(geodesic_distance(units, units))

        assert bool(jnp.all(jnp.isfinite(jax.grad(total)(poles))))

    @pytest.mark.parametrize("distance", [chord_distance, geodesic_distance])
    def test_gradient_matches_finite_differences(self, distance) -> None:  # type: ignore[no-untyped-def]
        u = _random_units(3, seed=5)
        v = _random_units(4, seed=6)
        assert_grad_matches_fd(lambda points: distance(points, v, radius=2.0), u)

    @pytest.mark.parametrize("distance", [chord_distance, geodesic_distance])
    def test_traces(self, distance) -> None:  # type: ignore[no-untyped-def]
        u = jnp.stack([_random_units(3, seed=s) for s in range(4)])
        v = jnp.stack([_random_units(5, seed=10 + s) for s in range(4)])
        assert_traces(lambda a, b: distance(a, b, radius=3.0), u, v)


class TestPairwiseDistance:
    def test_traces_and_differentiates(self) -> None:
        targets = Points(jnp.asarray([[0.0, 10.0], [50.0, -20.0]]), metric="geodesic")

        def distances(xy: jax.Array) -> jax.Array:
            sources = Points(xy, metric="geodesic")
            return pairwise_distance(sources, targets, unit="degrees")

        xy = jnp.asarray([[[1.0, 2.0], [3.0, 4.0]], [[-5.0, 6.0], [7.0, -8.0]]])
        assert_traces(distances, xy)
        assert_grad_matches_fd(distances, xy[0])

    def test_euclidean_equals_scipy(self) -> None:
        rng = np.random.default_rng(7)
        sources = rng.normal(size=(20, 3))
        targets = rng.normal(size=(30, 3))
        distances = pairwise_distance(
            Points(jnp.asarray(sources)), Points(jnp.asarray(targets))
        )
        assert_matches_oracle(
            distances, cdist(sources, targets), rtol=1e-12, atol=1e-15
        )

    @pytest.mark.parametrize(
        ("metric", "distance"),
        [("chord", chord_distance), ("geodesic", geodesic_distance)],
    )
    def test_sphere_metrics_lift_lonlat(self, metric: str, distance) -> None:  # type: ignore[no-untyped-def]
        rng = np.random.default_rng(8)
        lonlat = np.stack(
            [rng.uniform(-180, 180, 12), rng.uniform(-90, 90, 12)], axis=-1
        )
        points = Points(jnp.asarray(lonlat), metric=metric)
        units = lonlat_to_unit(jnp.asarray(lonlat), unit="degrees")
        assert_matches_oracle(
            pairwise_distance(points, points, radius=6371.0, unit="degrees"),
            distance(units, units, radius=6371.0),
            rtol=1e-12,
            atol=1e-9,
        )

    @pytest.mark.parametrize("metric", ["euclidean", "chord", "geodesic"])
    def test_chunked_equals_unchunked(self, metric: str) -> None:
        rng = np.random.default_rng(9)
        sources = Points(jnp.asarray(rng.uniform(-1, 1, (7, 2))), metric=metric)
        targets = Points(jnp.asarray(rng.uniform(-1, 1, (23, 2))), metric=metric)
        whole = pairwise_distance(sources, targets)
        # 7 · 23 = 161 pairs; chunks of 16 // 7 = 2 targets, the last one padded
        chunked = pairwise_distance(sources, targets, max_pairs=16)
        assert chunked.shape == (7, 23)
        assert_matches_oracle(chunked, whole, rtol=1e-15)

    def test_padding_is_infinitely_far(self) -> None:
        sources = Points(jnp.zeros((2, 2)), valid=jnp.array([True, False]))
        targets = Points(jnp.ones((3, 2)), valid=jnp.array([True, True, False]))
        distances = pairwise_distance(sources, targets)
        expected = np.array(
            [[np.sqrt(2.0), np.sqrt(2.0), np.inf], [np.inf, np.inf, np.inf]]
        )
        assert_matches_oracle(distances, expected, rtol=1e-15)


class TestSphericalPolygonArea:
    @pytest.mark.parametrize("level", [0, 1, 2, 3])
    def test_icosphere_faces_sum_to_four_pi(self, level: int) -> None:
        vertices, triangles = icosphere(level)
        areas = spherical_polygon_area(vertices[triangles])
        assert bool(jnp.all(areas > 0))
        assert_matches_oracle(areas.sum(), 4 * np.pi, rtol=1e-12)

    def test_closed_forms(self) -> None:
        # the octant is an eighth of the sphere
        octant = jnp.eye(3)
        theta = 0.3
        cap_quad = lonlat_to_unit(
            jnp.array([[0.0, 0.0], [theta, 0.0], [theta, 1.0], [0.0, 1.0]])
        )
        assert_matches_oracle(spherical_polygon_area(octant), np.pi / 2, rtol=1e-14)
        # a quad from the equator to latitude 1 rad: its great-circle top edge
        # bulges poleward of the latitude circle, so it holds more than the band
        band = theta * np.sin(1.0)
        area = float(spherical_polygon_area(cap_quad))
        assert band < area < 1.01 * band
        assert_matches_oracle(
            spherical_polygon_area(octant, radius=2.0), 2 * np.pi, rtol=1e-14
        )

    def test_clockwise_is_negative_and_padding_adds_nothing(self) -> None:
        octant = jnp.eye(3)
        assert_matches_oracle(
            spherical_polygon_area(octant[::-1]), -np.pi / 2, rtol=1e-14
        )
        padded = jnp.concatenate([octant, octant[-1:], octant[-1:]])
        assert_matches_oracle(spherical_polygon_area(padded), np.pi / 2, rtol=1e-14)

    def test_small_cells_keep_their_digits(self) -> None:
        # a 1e-4 rad square at the equator: area ≈ h², where ∑α − 2π loses ~8 digits
        h = 1e-4
        square = lonlat_to_unit(jnp.array([[0.0, 0.0], [h, 0.0], [h, h], [0.0, h]]))
        exact = h * np.sin(h)
        # the great-circle top edge differs from the latitude circle by O(h⁴)
        assert_matches_oracle(spherical_polygon_area(square), exact, rtol=1e-7)
