"""Tests for `InverseDistance`: the weights against a NumPy re-implementation, the
radius, padding taps, coincident targets and validation. The fast tier: behaviour,
validation and small closed-form checks, each well under a second.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax import GeometryError, OutsideDomainError, PolicyError, ValidationError
from functax.geometry import Axis, Points
from functax.interp import InverseDistance, Policies, Scheme


RNG = np.random.default_rng(35)
SOURCE_XY = RNG.uniform(0.0, 10.0, (60, 2))
TARGET_XY = RNG.uniform(-1.0, 11.0, (25, 2))
VALUES = RNG.normal(size=60)


def _idw_reference(
    source: np.ndarray,
    target: np.ndarray,
    values: np.ndarray,
    *,
    k: int,
    power: float,
    radius: float | None = None,
    floor: float = 1e-12,
) -> np.ndarray:
    """The issue's formula by brute force: wᵢⱼ ∝ max(dᵢⱼ, ε)⁻ᵖ, NaN if empty."""
    distances = np.linalg.norm(target[:, None, :] - source[None, :, :], axis=-1)
    order = np.argsort(distances, axis=1)[:, :k]
    nearest = np.take_along_axis(distances, order, axis=1)
    found = nearest < (np.inf if radius is None else radius)
    weights = np.where(found, np.maximum(nearest, floor) ** -power, 0.0)
    total = weights.sum(axis=1)
    with np.errstate(invalid="ignore"):
        estimate = (weights * values[order]).sum(axis=1) / total
    return np.where(total > 0, estimate, np.nan)


def test_equals_the_formula_in_numpy() -> None:
    plan = InverseDistance().plan(Points(SOURCE_XY), Points(TARGET_XY))
    expected = _idw_reference(SOURCE_XY, TARGET_XY, VALUES, k=8, power=2.0)
    assert_matches_oracle(plan(jnp.asarray(VALUES)), expected, rtol=1e-12)


def test_a_radius_with_no_neighbours_is_invalid() -> None:
    far = np.array([[50.0, 50.0], SOURCE_XY[0] + 0.1])
    plan = InverseDistance(k=3, radius=1.0).plan(Points(SOURCE_XY), Points(far))
    assert plan.valid.tolist() == [False, True]
    values = plan(jnp.asarray(VALUES))
    assert np.isnan(values[0]) and np.isfinite(values[1])
    # the empty row reads nothing
    assert plan.read.weights[0].tolist() == [0.0, 0.0, 0.0]
    assert plan.read.indices[0].tolist() == [0, 0, 0]
    with pytest.raises(OutsideDomainError, match="1 of 2 targets"):
        InverseDistance(k=3, radius=1.0).plan(
            Points(SOURCE_XY), Points(far), policies=Policies(outside="raise")
        )


def test_missing_taps_repeat_the_nearest_index() -> None:
    plan = InverseDistance(k=8, radius=1.5).plan(Points(SOURCE_XY), Points(TARGET_XY))
    indices = np.asarray(plan.read.indices)
    weights = np.asarray(plan.read.weights)
    rows = np.asarray(plan.valid)
    padding = (weights == 0) & rows[:, None]
    assert padding.any()
    assert (
        indices[padding] == np.broadcast_to(indices[:, :1], indices.shape)[padding]
    ).all()


def test_more_neighbours_than_sources_reads_them_all() -> None:
    plan = InverseDistance(k=10, power=0.0).plan(
        Points(SOURCE_XY[:3]), Points(TARGET_XY[:2])
    )
    assert plan.read.n_taps == 3
    assert_matches_oracle(
        plan(jnp.asarray(VALUES[:3])), np.full(2, VALUES[:3].mean()), rtol=1e-12
    )


def test_a_coincident_target_takes_the_source_value() -> None:
    plan = InverseDistance(k=4).plan(Points(SOURCE_XY), Points(SOURCE_XY[[3, 17]]))
    assert_matches_oracle(plan(jnp.asarray(VALUES)), VALUES[[3, 17]], rtol=1e-12)


def test_a_high_power_does_not_overflow_in_float32() -> None:
    source = Points(jnp.asarray(SOURCE_XY, dtype=jnp.float32))
    target = Points(jnp.asarray(SOURCE_XY[:2], dtype=jnp.float32))
    plan = InverseDistance(k=4, power=6.0).plan(source, target)
    assert plan.read.weights.dtype == jnp.float32
    assert np.isfinite(np.asarray(plan.read.weights)).all()
    values = plan(jnp.asarray(VALUES, dtype=jnp.float32))
    np.testing.assert_allclose(values, VALUES[:2], rtol=1e-6)


def test_renormalise_drops_a_missing_source() -> None:
    values = VALUES.copy()
    plan = InverseDistance(k=4).plan(
        Points(SOURCE_XY), Points(TARGET_XY), policies=Policies(missing="renormalise")
    )
    nearest = int(plan.read.indices[0, 0])
    values[nearest] = np.nan
    result = plan(jnp.asarray(values))
    assert np.isfinite(result).all()


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
