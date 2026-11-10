"""Tests for `functax.quad.cell_area`. The fast tier: lon/lat band areas, affine
areas equal |ae − bd| exactly, rectilinear areas sum to the domain, the node
order matches a field's flattened values, and validation. The global grids and
the float32 path are in the slow tier.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from functax import GeometryError, ValidationError
from functax.geometry import AffineGrid, Axis, RectilinearGrid, pixel_centres
from functax.quad import Rule, cell_area


def test_lonlat_cell_has_the_band_area() -> None:
    # one cell 10° wide between 30° and 40° north on a sphere of radius 2
    grid = AffineGrid([10.0, 0.0, 0.0, 0.0, -10.0, 40.0], (1, 1))
    rule = cell_area(grid, sphere_radius=2.0, unit="degrees")
    # a² Δλ (sin φ_top − sin φ_bottom)
    expected = 4.0 * math.radians(10.0) * (math.sin(math.radians(40.0)) - 0.5)
    assert abs(float(rule.weights[0]) - expected) < 1e-15 * expected


def test_latitude_edges_past_the_poles_are_clipped() -> None:
    # nodes at ±60° with midpoint edges at ±120°: the bands stop at the poles
    grid = RectilinearGrid(
        (Axis(jnp.array([0.0, 180.0]), period=360.0), Axis(jnp.array([-60.0, 60.0])))
    )
    rule = cell_area(grid, sphere_radius=1.0, unit="degrees")
    assert abs(float(rule.weights.sum()) - 4.0 * math.pi) < 1e-14


@pytest.mark.parametrize(
    "transform",
    [
        [20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0],
        [0.3, 0.1, -5.0, -0.2, 0.7, 2.0],
        [-1.5, 2.0, 0.0, 0.5, 0.25, 1.0],
    ],
)
def test_affine_areas_equal_the_determinant_exactly(transform: list[float]) -> None:
    a, b, _, d, e, _ = transform
    rule = cell_area(AffineGrid(transform, (3, 4)))
    assert rule.weights.dtype == jnp.float64
    assert np.all(np.asarray(rule.weights) == abs(a * e - b * d))


def test_affine_nodes_are_the_pixel_centres_row_major() -> None:
    grid = AffineGrid([0.3, 0.1, -5.0, -0.2, 0.7, 2.0], (3, 4))
    rule = cell_area(grid)
    x, y = pixel_centres(grid)
    np.testing.assert_array_equal(rule.nodes[:, 0], np.asarray(x).ravel())
    np.testing.assert_array_equal(rule.nodes[:, 1], np.asarray(y).ravel())


def test_rectilinear_midpoint_edges_and_a_descending_axis() -> None:
    # without edges, the cells end half a spacing past the end nodes
    axes = (Axis(jnp.array([3.0, 2.0, 0.0])), Axis(jnp.array([0.0, 1.0])))
    rule = cell_area(RectilinearGrid(axes))
    np.testing.assert_allclose(rule.weights, [1.0, 1.0, 1.5, 1.5, 2.0, 2.0])
    # the domain is [−1, 3.5] × [−0.5, 1.5]
    assert abs(float(rule.weights.sum()) - 4.5 * 2.0) < 1e-14


def test_rectilinear_nodes_follow_the_axes() -> None:
    axes = (Axis(jnp.array([0.0, 1.0])), Axis(jnp.array([5.0, 6.0, 7.0])))
    rule = cell_area(RectilinearGrid(axes))
    assert isinstance(rule, Rule)
    assert rule.nodes.tolist()[:4] == [[0.0, 5.0], [0.0, 6.0], [0.0, 7.0], [1.0, 5.0]]


def test_unit_goes_with_the_radius() -> None:
    grid = AffineGrid([1.0, 0.0, 0.0, 0.0, -1.0, 0.0], (2, 2))
    with pytest.raises(ValidationError, match="'unit' must be None without"):
        cell_area(grid, unit="degrees")
    with pytest.raises(ValidationError, match="'unit' must be 'degrees' or 'radians'"):
        cell_area(grid, sphere_radius=1.0)
    with pytest.raises(GeometryError, match="'unit' must be one of"):
        cell_area(grid, sphere_radius=1.0, unit="gradians")


@pytest.mark.parametrize("radius", [0.0, -1.0, math.inf, math.nan])
def test_rejects_a_bad_radius(radius: float) -> None:
    grid = AffineGrid([1.0, 0.0, 0.0, 0.0, -1.0, 0.0], (2, 2))
    with pytest.raises(ValidationError, match="'sphere_radius' must be positive"):
        cell_area(grid, sphere_radius=radius, unit="degrees")


def test_rejects_a_rotated_affine_grid_on_the_sphere() -> None:
    grid = AffineGrid([1.0, 0.1, 0.0, 0.0, -1.0, 0.0], (2, 2))
    with pytest.raises(GeometryError, match="north-up"):
        cell_area(grid, sphere_radius=1.0, unit="degrees")


def test_rejects_a_rectilinear_grid_without_two_axes_on_the_sphere() -> None:
    grid = RectilinearGrid((Axis(jnp.array([0.0, 1.0])),))
    with pytest.raises(GeometryError, match="two axes"):
        cell_area(grid, sphere_radius=1.0, unit="degrees")


def test_rejects_another_geometry() -> None:
    import inspect

    with pytest.raises(GeometryError, match="'grid' must be an AffineGrid"):
        inspect.unwrap(cell_area)(Axis(jnp.array([0.0, 1.0])))
