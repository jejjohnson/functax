"""Tests for `Binning`: empty cells, samples off the grid, padding, the count's missing
linear map, the JSON round trip and validation. The fast tier: behaviour, validation
and small closed-form checks, each well under a second.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

from functax import (
    GeometryError,
    NonlinearSchemeError,
    OutsideDomainError,
    PolicyError,
    ValidationError,
)
from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import Binning, BinningPlan, Policies, Scheme, TransposedStencil


CELLS = Axis(jnp.arange(4.0))
SAMPLES = Points(jnp.array([[0.1], [0.4], [2.2], [9.0], [-0.6]]))
VALUES = jnp.array([1.0, 3.0, 5.0, 7.0, 11.0])


def test_empty_cells_give_nan_means_and_zero_sums() -> None:
    means = Binning("mean").plan(SAMPLES, CELLS)
    assert means(VALUES).tolist()[:1] == [2.0]
    assert np.isnan(np.asarray(means(VALUES))[[1, 3]]).all()
    assert means.valid.tolist() == [True, False, True, False]
    sums = Binning("sum").plan(SAMPLES, CELLS)
    assert sums(VALUES).tolist() == [4.0, 0.0, 5.0, 0.0]
    assert bool(sums.valid.all())


def test_samples_off_the_grid_are_dropped_not_clamped() -> None:
    counts = Binning("count").plan(SAMPLES, CELLS)
    assert counts(VALUES).tolist() == [2.0, 0.0, 1.0, 0.0]
    assert counts.counts.tolist() == [2.0, 0.0, 1.0, 0.0]
    with pytest.raises(OutsideDomainError, match="2 of 5 targets"):
        Binning().plan(SAMPLES, CELLS, policies=Policies(outside="raise"))


def test_the_fill_value_marks_empty_means() -> None:
    plan = Binning("mean").plan(SAMPLES, CELLS, policies=Policies(fill_value=-1.0))
    assert plan(VALUES).tolist() == [2.0, -1.0, 5.0, -1.0]
    # the linear map has a zero row there, and the adjoint ignores it
    assert plan.operator().mv(VALUES).tolist() == [2.0, 0.0, 5.0, 0.0]
    assert plan.adjoint(jnp.ones(4)).tolist() == [0.5, 0.5, 1.0, 0.0, 0.0]


def test_a_count_has_no_linear_map() -> None:
    plan = Binning("count").plan(SAMPLES, CELLS)
    for method in (plan.operator, lambda: plan.adjoint(jnp.ones(4)), plan.linearize):
        with pytest.raises(NonlinearSchemeError, match="does not depend on the data"):
            method()


def test_padding_counts_in_no_cell() -> None:
    padded = Points(SAMPLES.xy, valid=jnp.array([True, False, True, False, False]))
    plan = Binning("sum").plan(padded, CELLS, policies=Policies(outside="raise"))
    assert plan(VALUES).tolist() == [1.0, 0.0, 5.0, 0.0]


def test_batches_and_grid_shape() -> None:
    grid = RectilinearGrid((Axis(jnp.arange(2.0)), Axis(jnp.arange(3.0))))
    samples = Points(jnp.array([[0.1, 0.2], [0.9, 1.8], [1.2, 2.1]]))
    plan = Binning("sum", weights="linear").plan(samples, grid)
    assert isinstance(plan, BinningPlan)
    assert isinstance(plan.read, TransposedStencil)
    assert plan.grid_shape == (2, 3)
    assert plan(jnp.ones((4, 3))).shape == (4, 6)


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
