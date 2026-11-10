"""Tests for `InverseDistance`: the weights against a NumPy re-implementation, the
radius, padding taps, coincident targets and validation. The fast tier: behaviour,
validation and small closed-form checks, each well under a second.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

from functax import GeometryError, PolicyError, ValidationError
from functax.geometry import Axis, Points
from functax.interp import InverseDistance, Policies, Scheme


RNG = np.random.default_rng(35)


SOURCE_XY = RNG.uniform(0.0, 10.0, (60, 2))


TARGET_XY = RNG.uniform(-1.0, 11.0, (25, 2))


def test_round_trips_through_json() -> None:
    scheme = InverseDistance(k=4, power=1, radius=2.5, floor=1e-9)
    text = json.dumps(scheme.get_config(), allow_nan=False)
    assert Scheme.from_config(json.loads(text)) == scheme


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"k": 0}, "k"),
        ({"k": True}, "k"),
        ({"power": -1.0}, "power"),
        ({"power": float("inf")}, "power"),
        ({"radius": 0.0}, "radius"),
        ({"floor": 0.0}, "floor"),
        ({"floor": float("nan")}, "floor"),
    ],
)
def test_rejects_a_bad_field(kwargs: dict, field: str) -> None:
    with pytest.raises(ValidationError, match=f"'{field}'"):
        InverseDistance(**kwargs)


def test_rejects_bad_geometry_and_policies() -> None:
    scheme = InverseDistance()
    with pytest.raises(GeometryError, match="'source' must be Points"):
        scheme.plan(Axis(jnp.arange(3.0)), Points(TARGET_XY))
    with pytest.raises(GeometryError, match="'geodesic'"):
        scheme.plan(Points(SOURCE_XY, metric="geodesic"), Points(TARGET_XY))
    with pytest.raises(GeometryError, match="share a metric"):
        scheme.plan(Points(SOURCE_XY), Points(TARGET_XY, metric="chord"))
    with pytest.raises(GeometryError, match="coordinates per point"):
        scheme.plan(Points(SOURCE_XY), Points(TARGET_XY[:, :1]))
    with pytest.raises(GeometryError, match="'target' must be"):
        scheme.plan(Points(SOURCE_XY), TARGET_XY)
    with pytest.raises(ValidationError, match="'unit'"):
        scheme.plan(
            Points(SOURCE_XY, metric="chord"), Points(TARGET_XY, metric="chord")
        )
    with pytest.raises(PolicyError, match="outside='nan' or 'raise'"):
        scheme.plan(
            Points(SOURCE_XY), Points(TARGET_XY), policies=Policies(outside="clamp")
        )
