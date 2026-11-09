"""Tests for the sphere maths and the pairwise distances in `functax.geometry`."""

from __future__ import annotations

import inspect
from pathlib import Path

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from hypothesis import given, settings
from scipy.spatial.distance import cdist

from functax import GeometryError
from functax.geometry import (
    Points,
    chord_distance,
    geodesic_distance,
    lonlat_to_unit,
    pairwise_distance,
    unit_to_lonlat,
)


# geonnax's lonlat_to_cartesian3d at be5537d (src/geonnax/geo.py), evaluated on
# 200 random points in degrees and 200 in radians and pinned as data, so the
# test does not import geonnax.
GEONNAX = np.load(Path(__file__).parent / "data" / "geonnax_lonlat_to_cartesian3d.npz")

# The functions behind the jaxtyping hook in conftest.py, which would reject a
# wrong shape or dtype before the module's own GeometryError check runs.
raw_lonlat_to_unit = inspect.unwrap(lonlat_to_unit)
raw_unit_to_lonlat = inspect.unwrap(unit_to_lonlat)


def _random_units(count: int, seed: int) -> jax.Array:
    """Unit vectors uniform on the sphere (normalised Gaussians)."""
    gaussian = jax.random.normal(jax.random.key(seed), (count, 3))
    norms = jnp.sqrt(jnp.sum(gaussian**2, axis=-1, keepdims=True))
    return gaussian / norms


class TestLift:
    @pytest.mark.parametrize(
        ("inputs", "outputs", "unit"),
        [
            ("lonlat_degrees", "unit_from_degrees", "degrees"),
            ("lonlat_radians", "unit_from_radians", "radians"),
        ],
    )
    def test_equals_geonnax(self, inputs: str, outputs: str, unit: str) -> None:
        lifted = lonlat_to_unit(jnp.asarray(GEONNAX[inputs]), unit=unit)
        assert_matches_oracle(lifted, GEONNAX[outputs], rtol=0.0, atol=1e-15)

    def test_unit_length_and_batch_dims(self) -> None:
        lonlat = jax.random.uniform(
            jax.random.key(0), (3, 4, 2), minval=-3.0, maxval=3.0
        )
        lifted = lonlat_to_unit(lonlat)
        assert lifted.shape == (3, 4, 3)
        assert_matches_oracle(jnp.sum(lifted**2, axis=-1), jnp.ones((3, 4)), rtol=1e-15)

    def test_keeps_float32(self) -> None:
        lifted = lonlat_to_unit(jnp.zeros((2, 2), dtype=jnp.float32))
        assert lifted.dtype == jnp.float32

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

    def test_unit_to_lonlat_ignores_the_length(self) -> None:
        lonlat = unit_to_lonlat(jnp.array([[0.0, 5.0, 5.0]]), unit="degrees")
        assert_matches_oracle(lonlat, np.array([[90.0, 45.0]]), rtol=1e-15)

    @pytest.mark.slow
    def test_gradient_finite_at_the_poles(self) -> None:
        poles = jnp.array([[0.3, jnp.pi / 2], [-1.0, -jnp.pi / 2]])
        jacobian = jax.jacrev(lonlat_to_unit)(poles)
        assert bool(jnp.all(jnp.isfinite(jacobian)))
        assert_grad_matches_fd(lonlat_to_unit, poles)

    @pytest.mark.parametrize(
        ("call", "match"),
        [
            (
                lambda: raw_lonlat_to_unit(jnp.zeros((2, 3))),
                "'lonlat' must have a last axis of length 2",
            ),
            (
                lambda: raw_lonlat_to_unit(jnp.zeros((2, 2), dtype=int)),
                "'lonlat' must have a floating",
            ),
            (
                lambda: lonlat_to_unit(jnp.zeros((2, 2)), unit="gradians"),
                "'unit' must be one of",
            ),
            (
                lambda: raw_unit_to_lonlat(jnp.zeros((2, 2))),
                "'xyz' must have a last axis of length 3",
            ),
            (
                lambda: unit_to_lonlat(jnp.zeros((2, 3)), unit="turns"),
                "'unit' must be one of",
            ),
        ],
    )
    def test_rejects_bad_inputs(self, call, match: str) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(GeometryError, match=match):
            call()


class TestDistances:
    @pytest.mark.slow
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

    @pytest.mark.slow
    def test_antipodes(self) -> None:
        u = _random_units(50, seed=2)
        geodesic = jnp.diagonal(geodesic_distance(u, -u, radius=6371.0))
        chord = jnp.diagonal(chord_distance(u, -u, radius=6371.0))
        assert_matches_oracle(geodesic, jnp.full(50, jnp.pi * 6371.0), rtol=1e-12)
        assert_matches_oracle(chord, jnp.full(50, 2.0 * 6371.0), rtol=1e-12)

    @pytest.mark.slow
    def test_zero_on_the_diagonal_and_symmetric(self) -> None:
        u = _random_units(30, seed=3)
        for distance in (chord_distance, geodesic_distance):
            matrix = distance(u, u)
            assert_matches_oracle(
                jnp.diagonal(matrix), jnp.zeros(30), rtol=0.0, atol=1e-15
            )
            assert_matches_oracle(matrix, matrix.T, rtol=1e-12, atol=1e-15)

    @pytest.mark.slow
    def test_a_traced_radius_is_differentiable(self) -> None:
        u = _random_units(3, seed=7)
        v = _random_units(2, seed=8)
        # ∂/∂a of Σ a‖u − v‖ is Σ ‖u − v‖
        slope = jax.jit(jax.grad(lambda a: jnp.sum(chord_distance(u, v, radius=a))))(
            2.0
        )
        assert_matches_oracle(slope, jnp.sum(chord_distance(u, v)), rtol=1e-12)

    def test_geodesic_is_accurate_at_small_angles(self) -> None:
        # 1e-9 rad apart: arccos(u · v) would return 0 or ~2e-8 here
        angle = 1e-9
        u = jnp.array([[1.0, 0.0, 0.0]])
        v = jnp.array([[jnp.cos(angle), jnp.sin(angle), 0.0]])
        assert_matches_oracle(geodesic_distance(u, v), jnp.array([[angle]]), rtol=1e-12)

    @pytest.mark.slow
    @pytest.mark.parametrize("distance", [chord_distance, geodesic_distance])
    def test_gradient_finite_at_zero_distance(self, distance) -> None:  # type: ignore[no-untyped-def]
        u = _random_units(4, seed=4)
        gradient = jax.grad(lambda points: jnp.sum(distance(points, u)))(u)
        assert bool(jnp.all(jnp.isfinite(gradient)))

    @pytest.mark.slow
    def test_geodesic_gradient_through_the_lift_at_the_poles(self) -> None:
        poles = jnp.array([[0.0, jnp.pi / 2], [2.0, -jnp.pi / 2], [1.0, 0.3]])

        def total(lonlat: jax.Array) -> jax.Array:
            units = lonlat_to_unit(lonlat)
            return jnp.sum(geodesic_distance(units, units))

        assert bool(jnp.all(jnp.isfinite(jax.grad(total)(poles))))

    @pytest.mark.slow
    @pytest.mark.parametrize("distance", [chord_distance, geodesic_distance])
    def test_gradient_matches_finite_differences(self, distance) -> None:  # type: ignore[no-untyped-def]
        u = _random_units(3, seed=5)
        v = _random_units(4, seed=6)
        assert_grad_matches_fd(lambda points: distance(points, v, radius=2.0), u)

    @pytest.mark.slow
    @pytest.mark.parametrize("distance", [chord_distance, geodesic_distance])
    def test_traces(self, distance) -> None:  # type: ignore[no-untyped-def]
        u = jnp.stack([_random_units(3, seed=s) for s in range(4)])
        v = jnp.stack([_random_units(5, seed=10 + s) for s in range(4)])
        assert_traces(lambda a, b: distance(a, b, radius=3.0), u, v)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"u": jnp.zeros(3), "v": jnp.zeros((2, 3))}, r"'u' must be 2-D"),
            (
                {"u": jnp.zeros((2, 3)), "v": jnp.zeros((2, 2))},
                "'v' must have a last axis of length 3",
            ),
            (
                {"u": jnp.zeros((2, 3)), "v": jnp.zeros((2, 3)), "radius": 0.0},
                "'radius' must be positive",
            ),
            (
                {
                    "u": jnp.zeros((2, 3)),
                    "v": jnp.zeros((2, 3)),
                    "radius": float("nan"),
                },
                "'radius' must be positive",
            ),
        ],
    )
    @pytest.mark.parametrize("distance", [chord_distance, geodesic_distance])
    def test_rejects_bad_inputs(
        self, distance, kwargs: dict[str, object], match: str
    ) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(GeometryError, match=match):
            inspect.unwrap(distance)(**kwargs)


class TestPairwiseDistance:
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

    @pytest.mark.slow
    def test_traces_and_differentiates(self) -> None:
        targets = Points(jnp.asarray([[0.0, 10.0], [50.0, -20.0]]), metric="geodesic")

        def distances(xy: jax.Array) -> jax.Array:
            sources = Points(xy, metric="geodesic")
            return pairwise_distance(sources, targets, unit="degrees")

        xy = jnp.asarray([[[1.0, 2.0], [3.0, 4.0]], [[-5.0, 6.0], [7.0, -8.0]]])
        assert_traces(distances, xy)
        assert_grad_matches_fd(distances, xy[0])

    def test_rejects_mixed_metrics(self) -> None:
        with pytest.raises(GeometryError, match="must share a metric"):
            pairwise_distance(
                Points(jnp.zeros((1, 2))), Points(jnp.zeros((1, 2)), metric="chord")
            )

    def test_sphere_metrics_need_lonlat(self) -> None:
        points = Points(jnp.zeros((2, 3)), metric="geodesic")
        with pytest.raises(GeometryError, match=r"'sources.xy' must be \(λ, φ\)"):
            pairwise_distance(points, points)

    def test_sphere_metrics_check_unit_and_radius(self) -> None:
        points = Points(jnp.zeros((2, 2)), metric="chord")
        with pytest.raises(GeometryError, match="'unit' must be one of"):
            pairwise_distance(points, points, unit="grad")
        with pytest.raises(GeometryError, match="'radius' must be positive"):
            pairwise_distance(points, points, radius=-1.0)
