"""Tests for `Binning`: empty cells, samples off the grid, padding, the count's missing
linear map, the JSON round trip and validation. The fast tier: behaviour, validation
and small closed-form checks, each well under a second.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import pytest

from functax import (
    GeometryError,
    NonlinearSchemeError,
    PolicyError,
    ValidationError,
)
from functax.geometry import Axis, Points
from functax.interp import Binning, Policies, Scheme


CELLS = Axis(jnp.arange(4.0))


SAMPLES = Points(jnp.array([[0.1], [0.4], [2.2], [9.0], [-0.6]]))


VALUES = jnp.array([1.0, 3.0, 5.0, 7.0, 11.0])


def test_a_count_has_no_linear_map() -> None:
    plan = Binning("count").plan(SAMPLES, CELLS)
    for method in (plan.operator, lambda: plan.adjoint(jnp.ones(4)), plan.linearize):
        with pytest.raises(NonlinearSchemeError, match="does not depend on the data"):
            method()


def test_padding_counts_in_no_cell() -> None:
    padded = Points(SAMPLES.xy, valid=jnp.array([True, False, True, False, False]))
    plan = Binning("sum").plan(padded, CELLS, policies=Policies(outside="raise"))
    assert plan(VALUES).tolist() == [1.0, 0.0, 5.0, 0.0]


def test_round_trips_through_json() -> None:
    scheme = Binning("sum", weights="linear")
    text = json.dumps(scheme.get_config(), allow_nan=False)
    assert Scheme.from_config(json.loads(text)) == scheme


def test_rejects_bad_fields_geometry_and_policies() -> None:
    with pytest.raises(ValidationError, match="'statistic'"):
        Binning("median")
    with pytest.raises(ValidationError, match="'weights'"):
        Binning(weights="cubic")
    with pytest.raises(GeometryError, match="'source' must be Points"):
        Binning().plan(CELLS, CELLS)
    with pytest.raises(GeometryError, match="'source' must be Points"):
        Binning().plan(Points(jnp.zeros((2, 2)), metric="chord"), CELLS)
    with pytest.raises(GeometryError, match="'target' must be a grid"):
        Binning().plan(SAMPLES, SAMPLES)
    with pytest.raises(PolicyError, match="outside='nan' or 'raise'"):
        Binning().plan(SAMPLES, CELLS, policies=Policies(outside="clamp"))
    with pytest.raises(PolicyError, match="renormalise"):
        Binning().plan(SAMPLES, CELLS, policies=Policies(missing="renormalise"))
