"""Tests for `Delaunay`: the outside rule, given simplices, padding sources, the JSON
round trip and validation. The fast tier: behaviour, validation and small closed-form
checks, each well under a second.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from functax import GeometryError, OutsideDomainError, PolicyError, ValidationError
from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import Delaunay, Policies, Scheme


SQUARE = Points(jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
TARGETS = Points(jnp.array([[0.25, 0.5], [0.9, 0.8], [2.0, 2.0], [-0.1, 0.5]]))


def test_outside_the_hull_is_invalid_with_taps_at_vertex_zero() -> None:
    plan = Delaunay().plan(SQUARE, TARGETS)
    assert plan.valid.tolist() == [True, True, False, False]
    assert plan.read.indices[2:].tolist() == [[0, 0, 0], [0, 0, 0]]
    assert plan.read.weights[2:].tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    values = plan(jnp.array([1.0, 3.0, 4.0, 6.0]))
    assert np.isnan(values[2:]).all()
    with pytest.raises(OutsideDomainError, match="2 of 4 targets"):
        Delaunay().plan(SQUARE, TARGETS, policies=Policies(outside="raise"))


def test_weights_sum_to_one_inside() -> None:
    plan = Delaunay().plan(SQUARE, TARGETS)
    totals = np.asarray(plan.read.weights).sum(axis=1)
    np.testing.assert_allclose(totals[:2], 1.0, rtol=1e-14)


def test_a_given_mesh_equals_its_fresh_triangulation() -> None:
    fresh = Delaunay().plan(SQUARE, TARGETS)
    mesh = Delaunay(simplices=[[0, 1, 3], [0, 3, 2]]).plan(SQUARE, TARGETS)
    np.testing.assert_allclose(mesh.read.to_dense(), fresh.read.to_dense(), atol=1e-15)
    assert mesh.valid.tolist() == fresh.valid.tolist()


def test_padding_sources_are_left_out() -> None:
    padded = Points(
        jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]),
        valid=jnp.array([True, True, True, True, False]),
    )
    plan = Delaunay().plan(padded, TARGETS)
    reads = np.asarray(plan.read.indices)[np.asarray(plan.read.weights) != 0]
    assert 4 not in reads
    mesh = Delaunay(simplices=[[0, 1, 4], [0, 1, 3], [0, 3, 2]]).plan(padded, TARGETS)
    assert 4 not in np.asarray(mesh.read.indices)


def test_float32_coordinates_keep_float32_weights() -> None:
    source = Points(jnp.asarray(SQUARE.xy, dtype=jnp.float32))
    target = Points(jnp.asarray(TARGETS.xy, dtype=jnp.float32))
    assert Delaunay().plan(source, target).read.weights.dtype == jnp.float32


def test_a_grid_target_keeps_its_shape() -> None:
    grid = RectilinearGrid(
        (Axis(jnp.linspace(0.0, 1.0, 3)), Axis(jnp.linspace(0.0, 1.0, 4)))
    )
    plan = Delaunay().plan(SQUARE, grid)
    assert plan.grid_shape == (3, 4)
    assert bool(plan.valid.all())


def test_round_trips_through_json() -> None:
    for scheme in (Delaunay(), Delaunay(simplices=np.array([[0, 1, 3], [0, 3, 2]]))):
        text = json.dumps(scheme.get_config(), allow_nan=False)
        assert Scheme.from_config(json.loads(text)) == scheme


@pytest.mark.parametrize(
    ("simplices", "match"),
    [
        ([[0, 1]], "at least 3 vertices"),
        (np.zeros((0, 3), dtype=int), "non-empty"),
        ([[0.0, 1.0, 2.0]], "integer"),
        ([[0, -1, 2]], "non-negative"),
    ],
)
def test_rejects_bad_simplices(simplices: object, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        Delaunay(simplices=simplices)


def test_rejects_bad_geometry_and_policies() -> None:
    with pytest.raises(GeometryError, match="'source' must be Points"):
        Delaunay().plan(Axis(jnp.arange(3.0)), TARGETS)
    with pytest.raises(GeometryError, match="'euclidean'"):
        Delaunay().plan(Points(SQUARE.xy, metric="chord"), TARGETS)
    with pytest.raises(GeometryError, match="at least 2 coordinates"):
        Delaunay().plan(Points(SQUARE.xy[:, :1]), Points(TARGETS.xy[:, :1]))
    with pytest.raises(GeometryError, match="coordinates per point"):
        Delaunay().plan(SQUARE, Points(TARGETS.xy[:, :1]))
    with pytest.raises(GeometryError, match="4 vertices per simplex"):
        Delaunay(simplices=[[0, 1, 2]]).plan(
            Points(jnp.zeros((4, 3))), Points(jnp.zeros((1, 3)))
        )
    with pytest.raises(ValidationError, match="index the 4 sources"):
        Delaunay(simplices=[[0, 1, 4]]).plan(SQUARE, TARGETS)
    with pytest.raises(GeometryError, match="degenerate"):
        line = Points(jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
        Delaunay(simplices=[[0, 1, 2]]).plan(line, TARGETS)
    with pytest.raises(PolicyError, match="outside='nan' or 'raise'"):
        Delaunay().plan(SQUARE, TARGETS, policies=Policies(outside="extrapolate"))
    with pytest.raises(ValidationError, match="concrete"):
        jax.jit(lambda xy: Delaunay().plan(SQUARE, Points(xy)).read)(TARGETS.xy)
