"""Tests for the frames of `functax.geometry`: ECEF, east/north/up, tangent planes
and the rotation to the pole. The fast tier: closed forms, round trips and
validation, each well under a second.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax import GeometryError
from functax.geometry import (
    cartesian_to_tangent,
    ecef_to_lonlat,
    enu_basis,
    local_tangent_plane,
    lonlat_to_ecef,
    lonlat_to_unit,
    rotation_to_pole,
    tangent_to_cartesian,
)


# WGS84, which functax never hard-codes: the caller (or geophysix) passes it
WGS84 = {"semi_major": 6378137.0, "flattening": 1.0 / 298.257223563}


def _points(n: int = 200, seed: int = 0) -> jax.Array:
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-180.0, 180.0, n)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    return jnp.asarray(np.stack([lon, lat], axis=-1))


class TestEcef:
    def test_known_points_on_wgs84(self) -> None:
        xyz = lonlat_to_ecef(
            jnp.array([[0.0, 0.0], [0.0, 90.0]]), unit="degrees", **WGS84
        )
        semi_minor = WGS84["semi_major"] * (1 - WGS84["flattening"])
        assert_matches_oracle(
            xyz[0], [WGS84["semi_major"], 0.0, 0.0], rtol=0, atol=1e-6
        )
        assert_matches_oracle(xyz[1], [0.0, 0.0, semi_minor], rtol=0, atol=1e-6)

    @pytest.mark.parametrize(
        "flattening", [WGS84["flattening"], 0.0], ids=["wgs84", "sphere"]
    )
    def test_round_trip_to_a_micrometre(self, flattening: float) -> None:
        # #20 asks for 1e-9 m, but one float64 ulp at a = 6.4e6 m is 1.4e-9 m;
        # the worst error seen is two ulps, so the bound is 1e-8 m (seven ulps).
        # Compare in ECEF, since the longitude is undefined at the poles.
        lonlat = _points()
        height = jnp.asarray(np.random.default_rng(1).uniform(-1e4, 1e4, 200))
        kwargs = {"semi_major": 6378137.0, "flattening": flattening, "unit": "degrees"}
        xyz = lonlat_to_ecef(lonlat, height, **kwargs)
        back, height_back = ecef_to_lonlat(xyz, **kwargs)
        again = lonlat_to_ecef(back, height_back, **kwargs)
        assert_matches_oracle(again, xyz, rtol=0, atol=1e-8)
        assert_matches_oracle(height_back, height, rtol=0, atol=1e-8)

    def test_the_sphere_is_the_radius_times_the_unit_vector(self) -> None:
        lonlat = _points()
        xyz = lonlat_to_ecef(lonlat, semi_major=2.0, unit="degrees")
        assert_matches_oracle(
            xyz, 2.0 * lonlat_to_unit(lonlat, unit="degrees"), rtol=0, atol=1e-12
        )

    def test_a_traced_ellipsoid_is_accepted(self) -> None:
        lonlat = _points(5)

        def ecef(semi_major: jax.Array, flattening: jax.Array) -> jax.Array:
            return lonlat_to_ecef(
                lonlat, semi_major=semi_major, flattening=flattening, unit="degrees"
            )

        traced = jax.jit(ecef)(
            jnp.asarray(WGS84["semi_major"]), jnp.asarray(WGS84["flattening"])
        )
        assert_matches_oracle(
            traced, lonlat_to_ecef(lonlat, unit="degrees", **WGS84), rtol=1e-14
        )

    def test_integer_degrees_become_float32(self) -> None:
        import inspect

        frame = inspect.unwrap(enu_basis)(jnp.array([[0, 0], [90, 45]]), unit="degrees")
        assert frame.dtype == jnp.float32

    def test_radians_and_degrees_agree(self) -> None:
        lonlat = _points(10)
        degrees = lonlat_to_ecef(lonlat, unit="degrees", **WGS84)
        radians = lonlat_to_ecef(jnp.deg2rad(lonlat), unit="radians", **WGS84)
        assert_matches_oracle(radians, degrees, rtol=1e-14, atol=1e-8)


class TestEnu:
    def test_is_orthonormal_right_handed_and_up_is_the_unit_vector(self) -> None:
        lonlat = jnp.concatenate([_points(), jnp.array([[30.0, 90.0], [-60.0, -90.0]])])
        frame = np.asarray(enu_basis(lonlat, unit="degrees"))
        gram = np.einsum("nij,nkj->nik", frame, frame)
        assert_matches_oracle(
            gram, np.broadcast_to(np.eye(3), gram.shape), rtol=0, atol=1e-12
        )
        assert_matches_oracle(
            np.cross(frame[:, 0], frame[:, 1]), frame[:, 2], rtol=0, atol=1e-12
        )
        assert_matches_oracle(
            frame[:, 2], lonlat_to_unit(lonlat, unit="degrees"), rtol=0, atol=1e-12
        )

    def test_tangent_round_trip_and_tangency(self) -> None:
        lonlat = _points()
        uv = jnp.asarray(np.random.default_rng(2).normal(size=(200, 2)))
        vec = tangent_to_cartesian(lonlat, uv, unit="degrees")
        back = cartesian_to_tangent(lonlat, vec, unit="degrees")
        assert_matches_oracle(back, uv, rtol=0, atol=1e-12)
        radial = np.sum(
            np.asarray(vec) * np.asarray(lonlat_to_unit(lonlat, unit="degrees")), axis=1
        )
        assert_matches_oracle(radial, np.zeros(200), rtol=0, atol=1e-12)


class TestTangentPlane:
    def test_azimuthal_equidistant_keeps_the_distance_from_the_origin(self) -> None:
        origin = jnp.array([12.0, -30.0])
        lonlat = _points()
        plane = np.asarray(
            local_tangent_plane(lonlat, origin, radius=1.0, unit="degrees")
        )
        points = np.asarray(lonlat_to_unit(lonlat, unit="degrees"))
        centre = np.asarray(lonlat_to_unit(origin, unit="degrees"))
        angle = np.arctan2(
            np.linalg.norm(np.cross(points, centre), axis=1), points @ centre
        )
        assert_matches_oracle(np.linalg.norm(plane, axis=1), angle, rtol=0, atol=1e-10)

    def test_gnomonic_is_tan_of_the_angle(self) -> None:
        plane = local_tangent_plane(
            jnp.array([[10.0, 0.0]]),
            jnp.array([0.0, 0.0]),
            radius=1.0,
            projection="gnomonic",
            unit="degrees",
        )
        assert_matches_oracle(
            plane, [[np.tan(np.radians(10.0)), 0.0]], rtol=1e-12, atol=1e-15
        )

    def test_the_gradient_is_finite_at_the_origin(self) -> None:
        origin = jnp.array([0.0, 0.0])

        def total(lonlat: jax.Array) -> jax.Array:
            return jnp.sum(
                local_tangent_plane(lonlat, origin, radius=1.0, unit="degrees")
            )

        assert bool(jnp.isfinite(jax.grad(total)(jnp.array([[0.0, 0.0]]))).all())


class TestRotationToPole:
    def test_takes_the_centre_to_the_pole(self) -> None:
        centre = jnp.array([40.0, 25.0])
        rotation = np.asarray(rotation_to_pole(centre, unit="degrees"))
        unit_vector = np.asarray(lonlat_to_unit(centre, unit="degrees"))
        assert_matches_oracle(
            rotation @ unit_vector, [0.0, 0.0, 1.0], rtol=0, atol=1e-12
        )
        assert_matches_oracle(rotation @ rotation.T, np.eye(3), rtol=0, atol=1e-12)
        assert abs(np.linalg.det(rotation) - 1.0) < 1e-12


class TestValidation:
    def test_the_unit_has_no_default(self) -> None:
        with pytest.raises(TypeError, match="unit"):
            enu_basis(jnp.zeros((1, 2)))  # ty: ignore[missing-argument]

    @pytest.mark.parametrize(
        "call",
        [
            lambda: enu_basis(jnp.zeros((1, 2)), unit="gradians"),
            lambda: lonlat_to_ecef(jnp.zeros((1, 2)), semi_major=1.0, unit="turns"),
            lambda: ecef_to_lonlat(jnp.ones((1, 3)), semi_major=1.0, unit="turns"),
            lambda: rotation_to_pole(jnp.zeros(2), unit="turns"),
        ],
    )
    def test_rejects_an_unknown_unit(self, call: object) -> None:
        with pytest.raises(GeometryError, match="'unit' must be one of"):
            call()  # ty: ignore[call-non-callable]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"semi_major": 0.0}, "'semi_major' must be positive"),
            (
                {"semi_major": 1.0, "flattening": 1.0},
                r"'flattening' must lie in \[0, 1\)",
            ),
            (
                {"semi_major": 1.0, "flattening": -0.1},
                r"'flattening' must lie in \[0, 1\)",
            ),
        ],
    )
    def test_rejects_a_bad_ellipsoid(
        self, kwargs: dict[str, float], match: str
    ) -> None:
        with pytest.raises(GeometryError, match=match):
            lonlat_to_ecef(jnp.zeros((1, 2)), unit="degrees", **kwargs)
        with pytest.raises(GeometryError, match=match):
            ecef_to_lonlat(jnp.ones((1, 3)), unit="degrees", **kwargs)

    def test_rejects_bad_shapes_behind_the_type_hook(self) -> None:
        import inspect

        with pytest.raises(GeometryError, match=r"'lonlat' must have shape \(N, 2\)"):
            inspect.unwrap(enu_basis)(jnp.zeros((3, 3)), unit="degrees")
        with pytest.raises(GeometryError, match=r"'xyz' must have shape \(N, 3\)"):
            inspect.unwrap(ecef_to_lonlat)(
                jnp.zeros((3, 2)), semi_major=1.0, unit="degrees"
            )
        with pytest.raises(
            GeometryError, match=r"'lonlat_centre' must have shape \(2,\)"
        ):
            inspect.unwrap(rotation_to_pole)(jnp.zeros(3), unit="degrees")

    def test_rejects_an_unknown_projection_and_a_bad_radius(self) -> None:
        origin = jnp.zeros(2)
        with pytest.raises(GeometryError, match="'projection' must be one of"):
            local_tangent_plane(
                jnp.zeros((1, 2)),
                origin,
                radius=1.0,
                projection="mercator",
                unit="degrees",
            )
        with pytest.raises(GeometryError, match="'radius' must be positive"):
            local_tangent_plane(jnp.zeros((1, 2)), origin, radius=-1.0, unit="degrees")
