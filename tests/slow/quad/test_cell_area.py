"""Tests for `functax.quad.cell_area`. The slow tier (each new grid shape compiles
its einx calls): a global 1° lon/lat grid sums to 4πa² to 1e-12 relative, as an
affine and as a rectilinear grid, the two agree cell by cell, and float32 axes
keep float32.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from functax.geometry import AffineGrid, Axis, RectilinearGrid
from functax.quad import cell_area


EARTH_LIKE_RADIUS = 6.371e6


def _global_rectilinear(unit: str, step: float) -> RectilinearGrid:
    """A global (λ, φ) grid of square cells, centres at half steps."""
    full = 360.0 if unit == "degrees" else 2.0 * math.pi
    half = full / 4.0
    longitude = Axis(jnp.arange(-full / 2 + step / 2, full / 2, step), period=full)
    latitude = Axis(jnp.arange(-half + step / 2, half, step))
    return RectilinearGrid((longitude, latitude))


@pytest.mark.parametrize("radius", [1.0, EARTH_LIKE_RADIUS])
def test_global_one_degree_affine_grid_sums_to_the_sphere_area(radius: float) -> None:
    globe = AffineGrid([1.0, 0.0, -180.0, 0.0, -1.0, 90.0], (180, 360))
    rule = cell_area(globe, sphere_radius=radius, unit="degrees")
    expected = 4.0 * math.pi * radius**2
    assert abs(float(rule.weights.sum()) - expected) < 1e-12 * expected


@pytest.mark.parametrize(
    ("unit", "step"), [("degrees", 1.0), ("radians", math.pi / 180.0)]
)
def test_global_one_degree_rectilinear_grid_sums_to_the_sphere_area(
    unit: str, step: float
) -> None:
    rule = cell_area(
        _global_rectilinear(unit, step), sphere_radius=EARTH_LIKE_RADIUS, unit=unit
    )
    expected = 4.0 * math.pi * EARTH_LIKE_RADIUS**2
    assert rule.weights.shape == (360 * 180,)
    assert abs(float(rule.weights.sum()) - expected) < 1e-12 * expected


def test_affine_and_rectilinear_lonlat_rules_agree() -> None:
    # rows are latitude in the affine grid, the second axis in the rectilinear
    affine = AffineGrid([2.0, 0.0, 10.0, 0.0, -3.0, 60.0], (4, 5))
    rectilinear = RectilinearGrid(
        (
            Axis(jnp.arange(11.0, 20.0, 2.0)),
            Axis(jnp.arange(58.5, 47.0, -3.0)),
        )
    )
    from_affine = cell_area(affine, sphere_radius=3.0, unit="degrees")
    from_rectilinear = cell_area(rectilinear, sphere_radius=3.0, unit="degrees")
    # (row col) against (lon lat): the same cells, transposed
    transposed = np.asarray(from_rectilinear.weights).reshape(5, 4).T.ravel()
    np.testing.assert_allclose(from_affine.weights, transposed, rtol=1e-14)


def test_rectilinear_keeps_float32() -> None:
    axes = (Axis(jnp.arange(3, dtype=jnp.float32)),) * 2
    rule = cell_area(RectilinearGrid(axes))
    assert rule.weights.dtype == jnp.float32
    globe = _global_rectilinear("degrees", 10.0)
    globe32 = RectilinearGrid(
        tuple(
            Axis(axis.values.astype(jnp.float32), period=axis.period)
            for axis in globe.axes
        )
    )
    on_sphere = cell_area(globe32, sphere_radius=1.0, unit="degrees")
    assert on_sphere.weights.dtype == jnp.float32


def test_rectilinear_areas_sum_to_the_domain_area() -> None:
    rng = np.random.default_rng(0)
    x_edges = np.cumsum(rng.uniform(0.1, 1.0, 8)) - 2.0
    y_edges = np.cumsum(rng.uniform(0.1, 1.0, 6)) + 3.0
    axes = tuple(
        Axis(jnp.asarray(0.5 * (edges[:-1] + edges[1:])), edges=jnp.asarray(edges))
        for edges in (x_edges, y_edges)
    )
    rule = cell_area(RectilinearGrid(axes))
    domain = (x_edges[-1] - x_edges[0]) * (y_edges[-1] - y_edges[0])
    assert abs(float(rule.weights.sum()) - domain) < 1e-14 * domain
    # each cell is the product of its two widths, the first axis outer
    expected = np.outer(np.diff(x_edges), np.diff(y_edges)).ravel()
    np.testing.assert_allclose(rule.weights, expected, rtol=1e-15)


@pytest.mark.parametrize("count", [1, 3])
def test_rectilinear_grid_of_any_dimension(count: int) -> None:
    axes = tuple(Axis(jnp.array([0.5, 1.5, 2.5])) for _ in range(count))
    rule = cell_area(RectilinearGrid(axes))
    assert rule.nodes.shape == (3**count, count)
    assert abs(float(rule.weights.sum()) - 3.0**count) < 1e-13
