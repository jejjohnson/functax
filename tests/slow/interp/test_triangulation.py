"""Tests for `Delaunay`: SciPy's LinearNDInterpolator in 2-D and 3-D, affine
reproduction, convex weights on random clouds, the dot-product test, gradient and
trace checks, and a given icosphere mesh on a tangent plane. The slow tier: each
compiles new shapes eagerly.
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
    assert_transpose_matches,
)
from hypothesis import given, settings
from scipy.interpolate import LinearNDInterpolator

from functax.geometry import (
    Axis,
    Points,
    RectilinearGrid,
    icosphere,
    local_tangent_plane,
    lonlat_to_unit,
    unit_to_lonlat,
)
from functax.interp import Delaunay


SQUARE = Points(jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))


TARGETS = Points(jnp.array([[0.25, 0.5], [0.9, 0.8], [2.0, 2.0], [-0.1, 0.5]]))


RNG = np.random.default_rng(38)


@pytest.mark.parametrize("n_dims", [2, 3])
def test_equals_linear_nd_interpolator(n_dims: int) -> None:
    sources = RNG.uniform(0.0, 1.0, (80, n_dims))
    targets = RNG.uniform(-0.1, 1.1, (300, n_dims))
    values = RNG.normal(size=80)
    plan = Delaunay().plan(Points(sources), Points(targets))
    expected = LinearNDInterpolator(sources, values)(targets)
    # NaN outside the hull in both
    assert_matches_oracle(plan(jnp.asarray(values)), expected, rtol=1e-12, atol=1e-14)
    assert np.isnan(expected).any()
    np.testing.assert_array_equal(np.asarray(plan.valid), np.isfinite(expected))


def test_reproduces_affine_functions() -> None:
    sources = RNG.uniform(-3.0, 5.0, (60, 2))
    targets = RNG.uniform(-3.0, 5.0, (200, 2))
    # f = 2 − 0.5 x + 3 y
    affine = 2.0 - 0.5 * sources[:, 0] + 3.0 * sources[:, 1]
    plan = Delaunay().plan(Points(sources), Points(targets))
    expected = 2.0 - 0.5 * targets[:, 0] + 3.0 * targets[:, 1]
    inside = np.asarray(plan.valid)
    assert inside.sum() > 100
    assert_matches_oracle(
        plan(jnp.asarray(affine)), expected, rtol=1e-12, atol=1e-13, where=inside
    )


def test_passes_the_dot_product_test() -> None:
    sources = RNG.uniform(0.0, 1.0, (40, 2))
    targets = RNG.uniform(-0.2, 1.2, (70, 2))
    plan = Delaunay().plan(Points(sources), Points(targets))
    assert_adjoint(plan.operator().mv, plan.adjoint, 40, 70, key=jax.random.key(1))
    assert_transpose_matches(plan.operator().mv, plan.adjoint, jnp.ones(40))


def test_a_given_icosphere_mesh_on_a_tangent_plane_equals_a_fresh_triangulation() -> (
    None
):
    vertices, triangles = icosphere(3)
    vertices = np.asarray(vertices)
    triangles = np.asarray(triangles)
    origin = np.array([0.3, 0.4])
    pole = np.asarray(lonlat_to_unit(jnp.asarray(origin), unit="radians"))
    # the vertices within 0.4 rad of the tangent point, and the triangles among them
    near = vertices @ pole > np.cos(0.4)
    kept = np.flatnonzero(near)
    renumber = np.full(vertices.shape[0], -1)
    renumber[kept] = np.arange(kept.size)
    cap_triangles = renumber[triangles[near[triangles].all(axis=1)]]
    lonlat = unit_to_lonlat(jnp.asarray(vertices[kept]), unit="radians")
    # the gnomonic projection maps the flat triangles to straight-edged ones
    plane = local_tangent_plane(
        lonlat, jnp.asarray(origin), radius=1.0, projection="gnomonic", unit="radians"
    )
    radius = RNG.uniform(0.0, 0.2, 200)
    angle = RNG.uniform(0.0, 2.0 * np.pi, 200)
    targets = Points(
        jnp.asarray(np.column_stack([radius * np.cos(angle), radius * np.sin(angle)]))
    )
    mesh = Delaunay(simplices=cap_triangles).plan(Points(plane), targets)
    fresh = Delaunay().plan(Points(plane), targets)
    assert bool(mesh.valid.all()) and bool(fresh.valid.all())
    assert_matches_oracle(
        mesh.read.to_dense(), fresh.read.to_dense(), rtol=1e-12, atol=1e-14
    )


def test_outside_the_hull_is_invalid_with_taps_at_vertex_zero() -> None:
    plan = Delaunay().plan(SQUARE, TARGETS)
    assert plan.valid.tolist() == [True, True, False, False]
    assert plan.read.indices[2:].tolist() == [[0, 0, 0], [0, 0, 0]]
    assert plan.read.weights[2:].tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    values = plan(jnp.array([1.0, 3.0, 4.0, 6.0]))
    assert np.isnan(values[2:]).all()


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


def test_a_given_mesh_equals_its_fresh_triangulation() -> None:
    fresh = Delaunay().plan(SQUARE, TARGETS)
    mesh = Delaunay(simplices=[[0, 1, 3], [0, 3, 2]]).plan(SQUARE, TARGETS)
    assert_matches_oracle(
        mesh.read.to_dense(), fresh.read.to_dense(), rtol=1e-14, atol=1e-15
    )
    assert mesh.valid.tolist() == fresh.valid.tolist()


class TestProperties:
    @settings(max_examples=12, deadline=None)
    @given(seed=st.integers(0, 2**31 - 1))
    def test_weights_are_convex_and_reproduce_affine_fields(self, seed: int) -> None:
        # inside the hull bₖ ≥ 0 and ∑ₖ bₖ = 1, so f̂ = a + bᵀx_* for f = a + bᵀx
        rng = np.random.default_rng(seed)
        sources = rng.uniform(0.0, 1.0, (30, 2))
        targets = rng.uniform(-0.1, 1.1, (40, 2))
        offset, slope = rng.normal(), rng.normal(size=2)
        plan = Delaunay().plan(Points(sources), Points(targets))
        weights = np.asarray(plan.read.weights)
        inside = np.asarray(plan.valid)
        # a target on an edge can round to bₖ ≈ −1e-16
        assert bool(np.all(weights[inside] >= -1e-14))
        assert_matches_oracle(
            weights.sum(axis=1), np.ones(40), rtol=1e-13, where=inside
        )
        values = plan(jnp.asarray(offset + sources @ slope))
        expected = offset + targets @ slope
        assert_matches_oracle(values, expected, rtol=1e-12, atol=1e-13, where=inside)
        assert bool(np.all(np.isnan(np.asarray(values)[~inside])))


class TestTransforms:
    # the plan is built on the host and rejects traced coordinates, so only the
    # data y is traced and differentiated
    rng = np.random.default_rng(39)
    sources = rng.uniform(0.0, 1.0, (25, 2))
    targets = rng.uniform(0.05, 0.95, (12, 2))
    fields = rng.normal(size=(3, 25))

    def test_jit_and_vmap_agree_with_a_loop(self) -> None:
        plan = Delaunay().plan(Points(self.sources), Points(self.targets))
        assert_traces(lambda values: plan(values), jnp.asarray(self.fields))

    def test_gradient_in_the_data_matches_finite_differences(self) -> None:
        # ∂f̂ / ∂y, the barycentric weights scattered to their vertices
        plan = Delaunay().plan(Points(self.sources), Points(self.targets))
        assert_grad_matches_fd(plan, jnp.asarray(self.fields[0]))
