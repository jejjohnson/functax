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

from functax import GeometryError
from functax.geometry import (
    Points,
    chord_distance,
    geodesic_distance,
    lonlat_to_unit,
    pairwise_distance,
    spherical_polygon_area,
    unit_to_lonlat,
)


raw_lonlat_to_unit = inspect.unwrap(lonlat_to_unit)


raw_unit_to_lonlat = inspect.unwrap(unit_to_lonlat)


class TestLift:
    def test_keeps_float32(self) -> None:
        lifted = lonlat_to_unit(jnp.zeros((2, 2), dtype=jnp.float32))
        assert lifted.dtype == jnp.float32

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


class TestSphericalPolygonArea:
    @pytest.mark.parametrize(
        ("vertices", "match"),
        [
            (jnp.zeros((2, 3)), "at least three vertices"),
            (jnp.zeros((3, 2)), "last axis of length 3"),
            (jnp.zeros((3, 3), dtype=jnp.int32), "floating dtype"),
        ],
    )
    def test_rejects_bad_vertices(self, vertices: jax.Array, match: str) -> None:
        with pytest.raises(GeometryError, match=match):
            inspect.unwrap(spherical_polygon_area)(vertices)
