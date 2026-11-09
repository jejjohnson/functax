"""Tests for the sphere maths and the pairwise distances in `functax.geometry`. The
fast tier: behaviour, validation and small closed-form checks, each well under a
second.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle
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


raw_lonlat_to_unit = inspect.unwrap(lonlat_to_unit)


raw_unit_to_lonlat = inspect.unwrap(unit_to_lonlat)


class TestLift:
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
    def test_geodesic_is_accurate_at_small_angles(self) -> None:
        # 1e-9 rad apart: arccos(u · v) would return 0 or ~2e-8 here
        angle = 1e-9
        u = jnp.array([[1.0, 0.0, 0.0]])
        v = jnp.array([[jnp.cos(angle), jnp.sin(angle), 0.0]])
        assert_matches_oracle(geodesic_distance(u, v), jnp.array([[angle]]), rtol=1e-12)

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
