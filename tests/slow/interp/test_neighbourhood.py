"""Tests for `InverseDistance`: the weights against a NumPy re-implementation, the
oracle sweep, the tree and brute-force searches, the chord metric, grid targets, the
dot-product test and gradients. The slow tier: each compiles new shapes eagerly.
"""

from __future__ import annotations

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

import functax._src.interp._neighbourhood as neighbourhood
from functax.geometry import AffineGrid, Axis, Points, RectilinearGrid
from functax.interp import InverseDistance


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


def _unit(lonlat_degrees: np.ndarray) -> np.ndarray:
    lon, lat = np.deg2rad(lonlat_degrees).T
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )


@pytest.mark.parametrize(
    ("k", "power", "radius"), [(8, 2.0, None), (5, 1.5, 1.5), (1, 3.0, None)]
)
def test_equals_the_formula_in_numpy(
    k: int, power: float, radius: float | None
) -> None:
    plan = InverseDistance(k=k, power=power, radius=radius).plan(
        Points(SOURCE_XY), Points(TARGET_XY)
    )
    expected = _idw_reference(
        SOURCE_XY, TARGET_XY, VALUES, k=k, power=power, radius=radius
    )
    assert_matches_oracle(plan(jnp.asarray(VALUES)), expected, rtol=1e-12)


def test_tree_and_brute_force_give_identical_stencils() -> None:
    scheme = InverseDistance(k=6, radius=2.0)
    valid = np.arange(60) % 7 != 0
    source = Points(SOURCE_XY, valid=valid)
    tree = scheme.plan(source, Points(TARGET_XY)).read
    # traced targets take the brute-force search
    brute = jax.jit(lambda xy: scheme.plan(source, Points(xy)).read)(TARGET_XY)
    np.testing.assert_array_equal(brute.indices, tree.indices)
    np.testing.assert_array_equal(brute.valid, tree.valid)
    assert_matches_oracle(brute.weights, tree.weights, rtol=1e-12, atol=1e-15)
    # padding sources are never read
    assert not np.isin(
        np.asarray(tree.indices)[np.asarray(tree.weights) > 0], np.flatnonzero(~valid)
    ).any()


def test_brute_force_in_chunks_equals_one_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheme = InverseDistance(k=4)
    source = Points(SOURCE_XY)
    whole = jax.jit(lambda xy: scheme.plan(source, Points(xy)).read)(TARGET_XY)
    # 60 · 7 pairs per chunk: four chunks of 7 targets, the last one padded
    monkeypatch.setattr(neighbourhood, "MAX_PAIRS", 60 * 7)
    chunked = jax.jit(lambda xy: scheme.plan(source, Points(xy)).read)(TARGET_XY)
    np.testing.assert_array_equal(chunked.indices, whole.indices)
    assert_matches_oracle(chunked.weights, whole.weights, rtol=1e-12)


def test_the_chord_metric_searches_across_the_dateline() -> None:
    lonlat = np.array([[179.0, 10.0], [-179.5, 10.0], [170.0, 10.0], [-170.0, 10.0]])
    target = np.array([[180.0, 10.0], [-179.9, 9.0]])
    plan = InverseDistance(k=2).plan(
        Points(lonlat, metric="chord"), Points(target, metric="chord"), unit="degrees"
    )
    # the two sources either side of the dateline, which euclidean lon/lat misses
    assert [sorted(row) for row in plan.read.indices.tolist()] == [[0, 1], [0, 1]]
    planar = InverseDistance(k=2).plan(Points(lonlat), Points(target))
    assert sorted(planar.read.indices[0].tolist()) != [0, 1]
    values = np.array([1.0, 2.0, 3.0, 4.0])
    chords = np.linalg.norm(_unit(target)[:, None] - _unit(lonlat)[None, :2], axis=-1)
    expected = (chords**-2 @ values[:2]) / (chords**-2).sum(axis=1)
    assert_matches_oracle(plan(jnp.asarray(values)), expected, rtol=1e-12)
    radians = InverseDistance(k=2).plan(
        Points(np.deg2rad(lonlat), metric="chord"),
        Points(np.deg2rad(target), metric="chord"),
        unit="radians",
    )
    assert_matches_oracle(radians(jnp.asarray(values)), expected, rtol=1e-12)


def test_onto_a_grid_keeps_its_shape() -> None:
    grid = RectilinearGrid(
        (Axis(jnp.linspace(0.0, 10.0, 4)), Axis(jnp.linspace(0.0, 10.0, 5)))
    )
    plan = InverseDistance().plan(Points(SOURCE_XY), grid)
    assert plan.grid_shape == (4, 5)
    lat, lon = np.meshgrid(
        np.linspace(0.0, 10.0, 4), np.linspace(0.0, 10.0, 5), indexing="ij"
    )
    nodes = np.column_stack([lat.ravel(), lon.ravel()])
    expected = _idw_reference(SOURCE_XY, nodes, VALUES, k=8, power=2.0)
    assert_matches_oracle(plan(jnp.asarray(VALUES)), expected, rtol=1e-12)
    affine = AffineGrid([1.0, 0.0, 0.0, 0.0, -1.0, 10.0], (10, 10))
    assert InverseDistance().plan(Points(SOURCE_XY), affine).grid_shape == (10, 10)
    line = InverseDistance().plan(Points(SOURCE_XY[:, :1]), Axis(jnp.arange(3.0)))
    assert line.grid_shape == (3,)


def test_passes_the_dot_product_test() -> None:
    plan = InverseDistance(k=5, radius=2.0).plan(Points(SOURCE_XY), Points(TARGET_XY))
    key = jax.random.key(0)
    assert_adjoint(plan.operator().mv, plan.adjoint, 60, 25, key=key)
    assert_transpose_matches(plan.operator().mv, plan.adjoint, jnp.asarray(VALUES))


def test_gradients_reach_the_targets_through_the_weights() -> None:
    scheme = InverseDistance(k=5, power=2.0)
    source = Points(SOURCE_XY)
    values = jnp.asarray(VALUES)

    def read(xy: jax.Array) -> jax.Array:
        return scheme.plan(source, Points(xy))(values)

    # interior targets, away from ties in the neighbour order
    assert_grad_matches_fd(read, jnp.asarray(TARGET_XY[3:6]), rtol=1e-6)


def test_jit_and_vmap_agree_with_a_loop() -> None:
    scheme = InverseDistance(k=4, radius=3.0)
    source = Points(SOURCE_XY)
    values = jnp.asarray(VALUES)

    def read(xy: jax.Array) -> jax.Array:
        return scheme.plan(source, Points(xy))(values)

    batched = jnp.asarray(TARGET_XY[:24]).reshape(3, 8, 2)
    assert_traces(read, batched)
