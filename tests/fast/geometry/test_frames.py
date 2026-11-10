"""Tests for the frames of `functax.geometry`: ECEF, east/north/up, tangent planes
and the rotation to the pole. The fast tier: closed forms, round trips and
validation, each well under a second.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax import GeometryError
from functax.geometry import (
    ecef_to_lonlat,
    enu_basis,
    local_tangent_plane,
    lonlat_to_ecef,
    lonlat_to_unit,
    rotation_to_pole,
)


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
