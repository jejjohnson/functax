"""Tests for `functax.quad.cell_area`. The slow tier (each new grid shape compiles
its einx calls): a global 1° lon/lat grid sums to 4πa² to 1e-12 relative, as an
affine and as a rectilinear grid, the two agree cell by cell, and float32 axes
keep float32.
"""

from __future__ import annotations

import math

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from hypothesis import given, settings

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


def _random_edges(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Uneven λ edges spanning under 2π and φ edges inside (−π/2, π/2), radians."""
    rng = np.random.default_rng(seed)
    lon_edges = rng.uniform(-np.pi, np.pi) + np.cumsum(rng.uniform(0.05, 0.7, 9))
    lat_edges = np.sort(rng.uniform(-1.5, 1.5, 7))
    return lon_edges, lat_edges


def _lonlat_areas(
    lon_edges: jax.Array, lat_edges: jax.Array, radius: float = 2.5
) -> jax.Array:
    """The cell areas of a (λ, φ) grid with traced edges, in radians.

    The radius stays a Python number: ``sphere_radius`` is typed ``float | int``.
    """
    # the nodes stay concrete; traced edges skip the Axis value checks
    axes = tuple(
        Axis(jnp.arange(edges.shape[0] - 1, dtype=edges.dtype), edges=edges)
        for edges in (lon_edges, lat_edges)
    )
    rule = cell_area(RectilinearGrid(axes), sphere_radius=radius, unit="radians")
    return rule.weights


def _plane_areas(lon_edges: jax.Array, lat_edges: jax.Array) -> jax.Array:
    """The same cells in the plane: |Δλ| |Δφ|."""
    axes = tuple(
        Axis(jnp.arange(edges.shape[0] - 1, dtype=edges.dtype), edges=edges)
        for edges in (lon_edges, lat_edges)
    )
    return cell_area(RectilinearGrid(axes)).weights


_AREAS = jax.jit(
    lambda lon, lat, radius: (_lonlat_areas(lon, lat, radius), _plane_areas(lon, lat)),
    static_argnums=2,
)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), radius=st.sampled_from([1.0, EARTH_LIKE_RADIUS]))
def test_uneven_cells_add_up_to_the_domain(seed: int, radius: float) -> None:
    # ∑|C| = R² (λ_end − λ_start)(sin φ_end − sin φ_start) on the sphere and
    # (λ_end − λ_start)(φ_end − φ_start) in the plane, whatever the inner edges
    lon_edges, lat_edges = _random_edges(seed)
    sphere, plane = _AREAS(jnp.asarray(lon_edges), jnp.asarray(lat_edges), radius)
    lon_span = lon_edges[-1] - lon_edges[0]
    sine_span = np.sin(lat_edges[-1]) - np.sin(lat_edges[0])
    assert sphere.shape == plane.shape == (8, 6)
    assert_matches_oracle(sphere.sum(), radius**2 * lon_span * sine_span, rtol=1e-13)
    assert_matches_oracle(
        plane.sum(), lon_span * (lat_edges[-1] - lat_edges[0]), rtol=1e-13
    )


def test_gradient_in_the_edges_matches_finite_differences() -> None:
    # differentiated in the λ and φ edges, with no edge at a pole
    lon_edges, lat_edges = (jnp.asarray(edges[:4]) for edges in _random_edges(1))
    assert_grad_matches_fd(lambda edges: _lonlat_areas(*edges), (lon_edges, lat_edges))


def test_gradient_in_the_sphere_radius_is_twice_the_area_over_r() -> None:
    # A = R² Δλ Δsin φ summed, so ∂A/∂R = 2A/R; the radius arrives traced
    globe = AffineGrid([10.0, 0.0, -180.0, 0.0, -10.0, 90.0], (18, 36))

    def total_area(radius):  # type: ignore[no-untyped-def]
        return cell_area(globe, sphere_radius=radius, unit="degrees").weights.sum()

    radius = jnp.asarray(2.0)
    assert_grad_matches_fd(total_area, radius)
    expected = 2.0 * total_area(radius) / radius
    assert_matches_oracle(jax.grad(total_area)(radius), expected, rtol=1e-12)


def test_jit_and_vmap_agree_with_a_loop() -> None:
    pairs = [_random_edges(seed) for seed in range(3)]
    lon_edges = jnp.asarray(np.stack([lon for lon, _ in pairs]))
    lat_edges = jnp.asarray(np.stack([lat for _, lat in pairs]))
    assert_traces(_lonlat_areas, lon_edges, lat_edges)


@pytest.mark.parametrize("radius", [EARTH_LIKE_RADIUS])
def test_global_one_degree_affine_grid_sums_to_the_sphere_area(radius: float) -> None:
    globe = AffineGrid([1.0, 0.0, -180.0, 0.0, -1.0, 90.0], (180, 360))
    rule = cell_area(globe, sphere_radius=radius, unit="degrees")
    expected = 4.0 * math.pi * radius**2
    assert_matches_oracle(rule.weights.sum(), expected, rtol=1e-12)


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
    assert rule.weights.shape == (360, 180)
    assert_matches_oracle(rule.weights.sum(), expected, rtol=1e-12)


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
    transposed = np.asarray(from_rectilinear.weights).T
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
    assert_matches_oracle(rule.weights.sum(), domain, rtol=1e-14)
    # each cell is the product of its two widths, one axis per grid axis
    expected = np.outer(np.diff(x_edges), np.diff(y_edges))
    np.testing.assert_allclose(rule.weights, expected, rtol=1e-15)


@pytest.mark.parametrize("count", [1, 3])
def test_rectilinear_grid_of_any_dimension(count: int) -> None:
    axes = tuple(Axis(jnp.array([0.5, 1.5, 2.5])) for _ in range(count))
    rule = cell_area(RectilinearGrid(axes))
    assert rule.weights.shape == (3,) * count
    assert rule.nodes.shape == (*(3,) * count, count)
    assert_matches_oracle(rule.weights.sum(), 3.0**count, rtol=0, atol=1e-13)
