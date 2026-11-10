"""Tests for the frames of `functax.geometry`. The slow tier: gradient and trace
checks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from _harness import assert_grad_matches_fd, assert_traces

from functax.geometry import (
    cartesian_to_tangent,
    ecef_to_lonlat,
    enu_basis,
    local_tangent_plane,
    lonlat_to_ecef,
    rotation_to_pole,
    tangent_to_cartesian,
)


# a sphere of radius 2 keeps the finite differences well scaled
SPHERE = {"semi_major": 2.0, "flattening": 0.1}
LONLAT = jnp.array([[0.3, 0.2], [-1.1, 0.9], [2.0, -0.4]])
ORIGIN = jnp.array([0.1, 0.3])


FRAMES = {
    "lonlat_to_ecef": lambda ll: lonlat_to_ecef(ll, unit="radians", **SPHERE),
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
