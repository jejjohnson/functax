"""Tests for the frames of `functax.geometry`. The slow tier: gradient and trace
checks.
"""

from __future__ import annotations

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import (
    assert_adjoint,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)
from hypothesis import given, settings

from functax.geometry import (
    cartesian_to_tangent,
    ecef_to_lonlat,
    enu_basis,
    local_tangent_plane,
    lonlat_to_ecef,
    lonlat_to_unit,
    rotation_to_pole,
    tangent_to_cartesian,
    unit_to_lonlat,
)


WGS84 = {"semi_major": 6378137.0, "flattening": 1.0 / 298.257223563}


def _points(n: int = 200, seed: int = 0) -> jax.Array:
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-180.0, 180.0, n)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    return jnp.asarray(np.stack([lon, lat], axis=-1))


SPHERE = {"semi_major": 2.0, "flattening": 0.1}


LONLAT = jnp.array([[0.3, 0.2], [-1.1, 0.9], [2.0, -0.4]])


ORIGIN = jnp.array([0.1, 0.3])


FRAMES = {
    "ecef_to_lonlat": lambda ll: ecef_to_lonlat(
        lonlat_to_ecef(ll, 0.1, unit="radians", **SPHERE), unit="radians", **SPHERE
    ),
    "enu_basis": lambda ll: enu_basis(ll, unit="radians"),
    "tangent_to_cartesian": lambda ll: tangent_to_cartesian(
        ll, ll[:, ::-1], unit="radians"
    ),
    "cartesian_to_tangent": lambda ll: cartesian_to_tangent(
        ll, jnp.concatenate([ll, ll[:, :1]], axis=1), unit="radians"
    ),
    "azimuthal_equidistant": lambda ll: local_tangent_plane(
        ll, ORIGIN, radius=2.0, unit="radians"
    ),
    "gnomonic": lambda ll: local_tangent_plane(
        ll * 0.3, ORIGIN, radius=2.0, projection="gnomonic", unit="radians"
    ),
}


@pytest.mark.parametrize("name", FRAMES)
def test_gradients_match_finite_differences(name: str) -> None:
    assert_grad_matches_fd(FRAMES[name], LONLAT)


@pytest.mark.parametrize("name", FRAMES)
def test_jit_and_vmap_match_a_loop(name: str) -> None:
    batches = jax.random.uniform(jax.random.key(0), (3, 4, 2), minval=-1.0, maxval=1.0)
    assert_traces(FRAMES[name], batches)


def test_rotation_to_pole_traces() -> None:
    centres = jax.random.uniform(jax.random.key(1), (4, 2), minval=-1.0, maxval=1.0)
    assert_traces(lambda centre: rotation_to_pole(centre, unit="radians"), centres)


def test_rotation_to_pole_gradient_matches_finite_differences() -> None:
    # differentiated in the centre (λ, φ), away from the poles
    centre = jnp.array([0.7, -0.4])
    assert_grad_matches_fd(lambda c: rotation_to_pole(c, unit="radians"), centre)


_ROTATION = jax.jit(lambda centre: rotation_to_pole(centre, unit="radians"))


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_rotation_to_pole_is_a_rotation_taking_the_centre_up(seed: int) -> None:
    # QᵀQ = I, det Q = 1, Q n(centre) = e_z and Q e_north(centre) = e_y
    rng = np.random.default_rng(seed)
    centre = np.array([rng.uniform(-np.pi, np.pi), np.arcsin(rng.uniform(-1, 1))])
    rotation = np.asarray(_ROTATION(jnp.asarray(centre)))
    frame = np.asarray(enu_basis(jnp.asarray(centre[None]), unit="radians"))[0]
    assert_matches_oracle(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-15)
    assert_matches_oracle(np.linalg.det(rotation), 1.0, rtol=1e-14)
    assert_matches_oracle(rotation @ frame[2], [0.0, 0.0, 1.0], rtol=0, atol=1e-15)
    assert_matches_oracle(rotation @ frame[1], [0.0, 1.0, 0.0], rtol=0, atol=1e-15)


_ECEF_ROUND_TRIP = jax.jit(
    lambda lonlat, height, semi_major, flattening: lonlat_to_ecef(
        *ecef_to_lonlat(
            lonlat_to_ecef(
                lonlat,
                height,
                semi_major=semi_major,
                flattening=flattening,
                unit="radians",
            ),
            semi_major=semi_major,
            flattening=flattening,
            unit="radians",
        ),
        semi_major=semi_major,
        flattening=flattening,
        unit="radians",
    )
)


@settings(max_examples=20, deadline=None)
@given(
    seed=st.integers(0, 2**31 - 1),
    semi_major=st.floats(1.0, 1e7),
    flattening=st.floats(0.0, 0.05),
)
def test_ecef_round_trips_on_any_ellipsoid(
    seed: int, semi_major: float, flattening: float
) -> None:
    # xyz → (λ, φ, h) → xyz, compared in ECEF since λ is undefined at the poles
    rng = np.random.default_rng(seed)
    lonlat = np.deg2rad(np.asarray(_points(50, seed % 1000)))
    height = rng.uniform(-0.01, 0.01, 50) * semi_major
    xyz = lonlat_to_ecef(
        jnp.asarray(lonlat),
        jnp.asarray(height),
        semi_major=semi_major,
        flattening=flattening,
        unit="radians",
    )
    again = _ECEF_ROUND_TRIP(
        jnp.asarray(lonlat),
        jnp.asarray(height),
        jnp.asarray(semi_major),
        jnp.asarray(flattening),
    )
    # a few ulps of the semi-major axis, as in TestEcef's round trip
    assert_matches_oracle(again, xyz, rtol=0, atol=1e-14 * semi_major)


_TANGENT_TO_CARTESIAN = jax.jit(
    lambda lonlat, uv: tangent_to_cartesian(lonlat, uv, unit="radians")
)
_CARTESIAN_TO_TANGENT = jax.jit(
    lambda lonlat, vec: cartesian_to_tangent(lonlat, vec, unit="radians")
)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_cartesian_to_tangent_is_the_adjoint_of_tangent_to_cartesian(
    seed: int,
) -> None:
    # ⟨T uv, v⟩ = ⟨uv, Tᵀ v⟩ with T the (3, 2) east/north columns at each point
    lonlat = np.deg2rad(np.asarray(_points(30, seed % 1000)))
    lonlat = jnp.asarray(lonlat)
    assert_adjoint(
        lambda uv: _TANGENT_TO_CARTESIAN(lonlat, uv.reshape(30, 2)).ravel(),
        lambda vec: _CARTESIAN_TO_TANGENT(lonlat, vec.reshape(30, 3)).ravel(),
        60,
        90,
        key=jax.random.key(seed),
    )


_ENU = jax.jit(lambda lonlat: enu_basis(lonlat, unit="radians"))


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), shift=st.floats(-np.pi, np.pi))
def test_enu_basis_turns_with_the_longitude(seed: int, shift: float) -> None:
    # the frame at (λ + α, φ) is the frame at (λ, φ) turned by R_z(α)
    lonlat = np.deg2rad(np.asarray(_points(30, seed % 1000)))
    turned = lonlat + np.array([shift, 0.0])
    cos, sin = np.cos(shift), np.sin(shift)
    rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    expected = np.asarray(_ENU(jnp.asarray(lonlat))) @ rotation.T
    assert_matches_oracle(_ENU(jnp.asarray(turned)), expected, rtol=0, atol=4e-15)


_PLANES = jax.jit(
    lambda lonlat, origin: (
        local_tangent_plane(lonlat, origin, radius=2.0, unit="radians"),
        local_tangent_plane(
            lonlat, origin, radius=2.0, projection="gnomonic", unit="radians"
        ),
    )
)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_tangent_planes_keep_distances_and_straighten_great_circles(
    seed: int,
) -> None:
    # azimuthal equidistant: ‖p‖ = R ψ; gnomonic: a great circle maps to a line
    rng = np.random.default_rng(seed)
    origin = np.array([rng.uniform(-np.pi, np.pi), np.arcsin(rng.uniform(-1, 1))])
    centre = np.asarray(lonlat_to_unit(jnp.asarray(origin)))
    # a great-circle arc through a point within 0.4 rad of the origin
    start = centre + 0.4 * rng.uniform(-1, 1, 3)
    start /= np.linalg.norm(start)
    heading = rng.normal(size=3)
    heading -= (heading @ start) * start
    heading /= np.linalg.norm(heading)
    steps = np.linspace(-0.5, 0.5, 7)[:, None]
    arc = np.cos(steps) * start + np.sin(steps) * heading
    lonlat = unit_to_lonlat(jnp.asarray(arc))
    equidistant, gnomonic = _PLANES(lonlat, jnp.asarray(origin))
    angle = np.arctan2(np.linalg.norm(np.cross(arc, centre), axis=1), arc @ centre)
    assert_matches_oracle(
        np.linalg.norm(equidistant, axis=1), 2.0 * angle, rtol=1e-12, atol=1e-15
    )
    # (p_k − p_0) × (p_last − p_0) = 0 for every k
    chords = np.asarray(gnomonic) - np.asarray(gnomonic)[0]
    cross = chords[:, 0] * chords[-1, 1] - chords[:, 1] * chords[-1, 0]
    scale = float(np.max(np.linalg.norm(chords, axis=1)) ** 2)
    assert_matches_oracle(cross, np.zeros(7), rtol=0, atol=1e-13 * scale)


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

    def test_integer_degrees_become_float32(self) -> None:
        import inspect

        frame = inspect.unwrap(enu_basis)(jnp.array([[0, 0], [90, 45]]), unit="degrees")
        assert frame.dtype == jnp.float32

    def test_radians_and_degrees_agree(self) -> None:
        lonlat = _points(10)
        degrees = lonlat_to_ecef(lonlat, unit="degrees", **WGS84)
        radians = lonlat_to_ecef(jnp.deg2rad(lonlat), unit="radians", **WGS84)
        assert_matches_oracle(radians, degrees, rtol=1e-14, atol=1e-8)

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

    def test_the_gradient_is_finite_at_the_origin(self) -> None:
        origin = jnp.array([0.0, 0.0])

        def total(lonlat: jax.Array) -> jax.Array:
            return jnp.sum(
                local_tangent_plane(lonlat, origin, radius=1.0, unit="degrees")
            )

        assert bool(jnp.isfinite(jax.grad(total)(jnp.array([[0.0, 0.0]]))).all())

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
