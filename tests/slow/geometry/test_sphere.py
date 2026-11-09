"""Tests for the sphere maths and the pairwise distances in `functax.geometry`. The
slow tier: oracle sweeps, dot-product tests across policies, gradient and trace
checks, each a second or more (mostly einx compiling new shapes).
"""

from __future__ import annotations

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from hypothesis import given, settings

from functax.geometry import (
    Points,
    chord_distance,
    geodesic_distance,
    lonlat_to_unit,
    pairwise_distance,
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
