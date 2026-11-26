"""Tests for `Binning`: SciPy's binned_statistic_dd for the mean, sum and count,
cloud-in-cell conservation, totals and constants kept on random samples, the gradient
in the positions, and the dot-product test. The slow tier: each compiles new shapes
eagerly.
"""

from __future__ import annotations

import functools
import itertools

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
from scipy.stats import binned_statistic_dd

from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import Binning, BinningPlan, Policies, TransposedStencil


CELLS = Axis(jnp.arange(4.0))


SAMPLES = Points(jnp.array([[0.1], [0.4], [2.2], [9.0], [-0.6]]))


VALUES = jnp.array([1.0, 3.0, 5.0, 7.0, 11.0])


RNG = np.random.default_rng(39)


NODES = (np.sort(RNG.uniform(0.0, 10.0, 7)), np.linspace(-2.0, 2.0, 5))


GRID = RectilinearGrid(tuple(Axis(jnp.asarray(nodes)) for nodes in NODES))


N_SAMPLES = 400


def _cell_edges(nodes: np.ndarray) -> np.ndarray:
    """Half-way between nodes, and half a cell past the end nodes."""
    middle = 0.5 * (nodes[1:] + nodes[:-1])
    first = nodes[0] - 0.5 * (nodes[1] - nodes[0])
    last = nodes[-1] + 0.5 * (nodes[-1] - nodes[-2])
    return np.concatenate([[first], middle, [last]])


def _samples() -> tuple[np.ndarray, np.ndarray]:
    # some samples fall past the grid on either axis, and are dropped
    positions = np.column_stack(
        [RNG.uniform(-2.0, 12.0, N_SAMPLES), RNG.uniform(-3.0, 3.0, N_SAMPLES)]
    )
    return positions, RNG.normal(size=N_SAMPLES)


@pytest.mark.parametrize("statistic", ["mean", "sum", "count"])
def test_equals_binned_statistic_dd(statistic: str) -> None:
    positions, values = _samples()
    plan = Binning(statistic).plan(Points(positions), GRID)
    edges = [_cell_edges(nodes) for nodes in NODES]
    expected = binned_statistic_dd(positions, values, statistic, bins=edges).statistic
    # the mean is NaN in an empty cell, the sum and the count 0, as in SciPy
    assert_matches_oracle(plan(jnp.asarray(values)), expected.ravel(), rtol=1e-12)
    if statistic == "mean":
        assert np.isnan(expected).any()


def test_cloud_in_cell_conserves_the_total() -> None:
    positions, _ = _samples()
    masses = RNG.uniform(0.5, 2.0, N_SAMPLES)
    plan = Binning("sum", weights="linear").plan(Points(positions), GRID)
    deposited = plan(jnp.asarray(masses))
    # the samples inside the hull of the nodes
    inside = np.ones(N_SAMPLES, dtype=bool)
    for column, nodes in enumerate(NODES):
        inside &= (positions[:, column] >= nodes[0]) & (
            positions[:, column] <= nodes[-1]
        )
    assert 0 < inside.sum() < N_SAMPLES
    assert_matches_oracle(jnp.sum(deposited), masses[inside].sum(), rtol=1e-12)


def test_the_gradient_of_a_cloud_in_cell_loss_reaches_the_positions() -> None:
    # samples well inside the grid, so ±ε stays in the same cells
    positions = jnp.asarray(
        np.column_stack([RNG.uniform(1.0, 9.0, 30), RNG.uniform(-1.5, 1.5, 30)])
    )
    masses = jnp.asarray(RNG.uniform(0.5, 2.0, 30))
    target = jnp.asarray(RNG.normal(size=35))

    def loss(xy: jax.Array) -> jax.Array:
        concentration = Binning("sum", weights="linear").plan(Points(xy), GRID)(masses)
        return jnp.sum((concentration - target) ** 2)

    gradient = jax.grad(loss)(positions)
    assert float(jnp.max(jnp.abs(gradient))) > 0.0
    assert_grad_matches_fd(loss, positions, eps=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("statistic", "weights"),
    [("mean", "nearest"), ("sum", "linear")],
)
def test_passes_the_dot_product_test(statistic: str, weights: str) -> None:
    positions, _ = _samples()
    plan = Binning(statistic, weights=weights).plan(Points(positions), GRID)
    apply = plan.operator().mv
    assert_adjoint(apply, plan.adjoint, N_SAMPLES, 35, key=jax.random.key(2))
    assert_transpose_matches(apply, plan.adjoint, jnp.ones(N_SAMPLES))


def test_jit_and_vmap_agree_with_a_loop() -> None:
    masses = jnp.asarray(RNG.uniform(0.5, 2.0, 20))

    def deposit(xy: jax.Array) -> jax.Array:
        return Binning("sum", weights="linear").plan(Points(xy), GRID)(masses)

    batched = jnp.asarray(
        np.stack(
            [
                np.column_stack(
                    [RNG.uniform(0.0, 10.0, 20), RNG.uniform(-2.0, 2.0, 20)]
                )
                for _ in range(3)
            ]
        )
    )
    assert_traces(deposit, batched)


@functools.partial(jax.jit, static_argnames="weights")
def _bin(
    xy: jax.Array, values: jax.Array, level: jax.Array, weights: str
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The sums of ``values`` and the means of the constant ``level`` on GRID."""
    sums = Binning("sum", weights=weights).plan(Points(xy), GRID)
    means = Binning("mean", weights=weights).plan(Points(xy), GRID)
    flat = jnp.full(xy.shape[0], level)
    return sums(values), means(flat), means.valid


@settings(max_examples=20, deadline=None)
@given(
    seed=st.integers(0, 2**31 - 1),
    weights=st.sampled_from(["nearest", "linear"]),
    level=st.floats(-1e3, 1e3).filter(lambda c: c == 0.0 or abs(c) > 1e-100),
)
def test_sums_keep_the_total_and_means_keep_a_constant(
    seed: int, weights: str, level: float
) -> None:
    # ∑ⱼ (Hᵀ y)ⱼ = ∑ᵢ yᵢ over the samples the grid holds, since each row of H
    # sums to one there (and is zero off the grid), and the mean of a constant
    # c is c in every non-empty cell
    rng = np.random.default_rng(seed)
    xy = np.column_stack([rng.uniform(-2.0, 12.0, 60), rng.uniform(-3.0, 3.0, 60)])
    values = rng.normal(size=60)
    sums, means, valid = _bin(
        jnp.asarray(xy), jnp.asarray(values), jnp.asarray(level), weights
    )
    # nearest holds the samples in the cells' edges, linear those in the nodes' hull
    held = np.ones(60, dtype=bool)
    for column, nodes in enumerate(NODES):
        bounds = _cell_edges(nodes) if weights == "nearest" else nodes
        held &= (xy[:, column] >= bounds[0]) & (xy[:, column] <= bounds[-1])
    assert_matches_oracle(jnp.sum(sums), values[held].sum(), rtol=1e-12, atol=1e-13)
    assert_matches_oracle(
        means, np.full(35, level), rtol=1e-14, where=np.asarray(valid)
    )


def test_empty_cells_give_nan_means_and_zero_sums() -> None:
    means = Binning("mean").plan(SAMPLES, CELLS)
    assert means(VALUES).tolist()[:1] == [2.0]
    assert np.isnan(np.asarray(means(VALUES))[[1, 3]]).all()
    assert means.valid.tolist() == [True, False, True, False]
    sums = Binning("sum").plan(SAMPLES, CELLS)
    assert sums(VALUES).tolist() == [4.0, 0.0, 5.0, 0.0]
    assert bool(sums.valid.all())


def test_the_fill_value_marks_empty_means() -> None:
    plan = Binning("mean").plan(SAMPLES, CELLS, policies=Policies(fill_value=-1.0))
    assert plan(VALUES).tolist() == [2.0, -1.0, 5.0, -1.0]
    # the linear map has a zero row there, and the adjoint ignores it
    assert plan.operator().mv(VALUES).tolist() == [2.0, 0.0, 5.0, 0.0]
    assert plan.adjoint(jnp.ones(4)).tolist() == [0.5, 0.5, 1.0, 0.0, 0.0]


def test_batches_and_grid_shape() -> None:
    grid = RectilinearGrid((Axis(jnp.arange(2.0)), Axis(jnp.arange(3.0))))
    samples = Points(jnp.array([[0.1, 0.2], [0.9, 1.8], [1.2, 2.1]]))
    plan = Binning("sum", weights="linear").plan(samples, grid)
    assert isinstance(plan, BinningPlan)
    assert isinstance(plan.read, TransposedStencil)
    assert plan.grid_shape == (2, 3)
    assert plan(jnp.ones((4, 3))).shape == (4, 6)


# --- explicit edges (#168) -----------------------------------------------------

STRETCHED = ((np.arange(11) / 10.0) ** 2, np.array([-3.0, -1.0, 0.5, 2.5, 3.0]))


def _with_edges(edges: np.ndarray, *, descending: bool = False) -> Axis:
    centres = 0.5 * (edges[1:] + edges[:-1])
    if descending:
        centres, edges = centres[::-1], edges[::-1]
    return Axis(jnp.asarray(centres), edges=jnp.asarray(edges))


@pytest.mark.parametrize("statistic", ["mean", "sum", "count"])
def test_nearest_bins_between_explicit_edges(statistic: str) -> None:
    rng = np.random.default_rng(7)
    # past the first and the last edge on both axes too
    positions = np.column_stack(
        [rng.uniform(-0.1, 1.1, 500), rng.uniform(-3.5, 3.5, 500)]
    )
    values = rng.normal(size=500)
    grid = RectilinearGrid([_with_edges(edges) for edges in STRETCHED])
    plan = Binning(statistic).plan(Points(jnp.asarray(positions)), grid)
    expected = binned_statistic_dd(
        positions, values, statistic, bins=list(STRETCHED)
    ).statistic
    assert_matches_oracle(plan(jnp.asarray(values)), expected.ravel(), rtol=1e-12)


def test_a_descending_axis_bins_between_its_edges() -> None:
    edges = np.array([0.0, 1.0, 4.0, 10.0])
    x = jnp.array([[0.5], [1.2], [3.9], [4.5], [9.9], [10.5]])
    rising = Binning("count").plan(Points(x), _with_edges(edges))(jnp.ones(6))
    falling = Binning("count").plan(Points(x), _with_edges(edges, descending=True))(
        jnp.ones(6)
    )
    assert rising.tolist() == [1.0, 2.0, 2.0]
    assert falling.tolist() == [2.0, 2.0, 1.0]


def test_an_axis_without_edges_bins_as_before() -> None:
    """Edges on one axis change nothing on the other."""
    positions, values = _samples()
    edges = _cell_edges(NODES[0])
    with_edges = Axis(jnp.asarray(NODES[0]), edges=jnp.asarray(edges))
    mixed = RectilinearGrid([with_edges, Axis(jnp.asarray(NODES[1]))])
    found = Binning("sum").plan(Points(positions), mixed)(jnp.asarray(values))
    before = Binning("sum").plan(Points(positions), GRID)(jnp.asarray(values))
    assert_matches_oracle(found, before, rtol=1e-12)


def test_nearest_with_edges_passes_the_dot_product_test() -> None:
    rng = np.random.default_rng(8)
    positions = np.column_stack([rng.uniform(0.0, 1.0, 40), rng.uniform(-3, 3, 40)])
    grid = RectilinearGrid([_with_edges(edges) for edges in STRETCHED])
    plan = Binning("sum").plan(Points(jnp.asarray(positions)), grid)
    apply = plan.operator().mv
    assert_adjoint(apply, plan.adjoint, 40, 40, key=jax.random.key(3))


def test_nearest_with_edges_traces() -> None:
    grid = _with_edges(np.array([0.0, 1.0, 4.0, 10.0]))
    values = jnp.ones(4)

    def counts(x: jax.Array) -> jax.Array:
        return Binning("count").plan(Points(x), grid)(values)

    batch = jnp.asarray(np.random.default_rng(9).uniform(-1.0, 11.0, (3, 4, 1)))
    assert_traces(counts, batch)


# --- linear weights over the cells (#169) --------------------------------------

OVER_CELLS = Policies(extent="cells")


def _grid(dim: int, *, stretched: bool) -> RectilinearGrid:
    edges = (np.arange(9) / 8.0) ** 2 * 10.0 if stretched else np.linspace(0, 10, 9)
    return RectilinearGrid([_with_edges(edges)] * dim)


def _ghost_and_fold(grid: RectilinearGrid, xy: np.ndarray, m: np.ndarray) -> np.ndarray:
    """A ghost node per end, reflected in the end edge, deposited and folded back.

    Samples outside the edges are dropped first, as the caller of this
    construction drops them.
    """
    inside = np.ones(len(m), dtype=bool)
    for column, axis in enumerate(grid.axes):
        edges = np.asarray(axis.edges)
        inside &= (xy[:, column] >= edges[0]) & (xy[:, column] < edges[-1])
    m = np.where(inside, m, 0.0)
    ghosts = []
    for axis in grid.axes:
        edges, nodes = np.asarray(axis.edges), np.asarray(axis.values)
        ghosts.append(
            Axis(
                jnp.asarray(
                    [2 * edges[0] - nodes[0], *nodes, 2 * edges[-1] - nodes[-1]]
                )
            )
        )
    plan = Binning("sum", weights="linear").plan(
        Points(jnp.asarray(xy)), RectilinearGrid(ghosts)
    )
    padded = np.asarray(plan(jnp.asarray(m))).reshape([n + 2 for n in grid.shape])
    for axis in range(len(grid.shape)):
        inner = np.take(padded, range(1, padded.shape[axis] - 1), axis=axis)
        first = [slice(None)] * inner.ndim
        first[axis] = slice(0, 1)
        last = [slice(None)] * inner.ndim
        last[axis] = slice(-1, None)
        inner[tuple(first)] += np.take(padded, [0], axis=axis)
        inner[tuple(last)] += np.take(padded, [-1], axis=axis)
        padded = inner
    return padded.ravel()


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("stretched", [False, True], ids=["uniform", "stretched"])
def test_linear_over_the_cells_keeps_every_mass_inside_the_edges(
    dim: int, stretched: bool
) -> None:
    rng = np.random.default_rng(10 + dim)
    # some past the edges on each axis, which are dropped
    xy = rng.uniform(-0.5, 10.5, (800, dim))
    m = rng.uniform(0.5, 2.0, 800)
    grid = _grid(dim, stretched=stretched)
    plan = Binning("sum", weights="linear").plan(
        Points(jnp.asarray(xy)), grid, policies=OVER_CELLS
    )
    deposited = plan(jnp.asarray(m))
    inside = ((xy >= 0.0) & (xy < 10.0)).all(axis=1)
    assert 0 < inside.sum() < 800
    assert_matches_oracle(jnp.sum(deposited), m[inside].sum(), rtol=1e-12)
    assert_matches_oracle(
        deposited, _ghost_and_fold(grid, xy, m), rtol=1e-12, atol=1e-12
    )


def test_extent_nodes_is_unchanged_bit_for_bit() -> None:
    positions, _ = _samples()
    masses = jnp.asarray(RNG.uniform(0.5, 2.0, N_SAMPLES))
    default = Binning("sum", weights="linear").plan(Points(positions), GRID)(masses)
    nodes = Binning("sum", weights="linear").plan(
        Points(positions), GRID, policies=Policies(extent="nodes")
    )(masses)
    assert jnp.array_equal(default, nodes)


def test_linear_over_the_cells_passes_the_dot_product_test() -> None:
    rng = np.random.default_rng(20)
    xy = jnp.asarray(rng.uniform(0.0, 10.0, (50, 2)))
    plan = Binning("sum", weights="linear").plan(
        Points(xy), _grid(2, stretched=True), policies=OVER_CELLS
    )
    assert_adjoint(plan.operator().mv, plan.adjoint, 50, 64, key=jax.random.key(4))


def test_linear_over_the_cells_has_the_gradient_of_its_weights() -> None:
    """Away from the nodes; in a half cell at a wall the weights are constant."""
    grid = _grid(1, stretched=True)
    nodes = np.asarray(grid.axes[0].values)
    # one sample in each interior gap, and one in the half cell below the first node
    xy = jnp.asarray(
        [
            [0.6 * nodes[0]],
            *[[0.5 * (a + b)] for a, b in itertools.pairwise(nodes)],
        ]
    )
    masses = jnp.ones(xy.shape[0])
    target = jnp.asarray(np.random.default_rng(21).normal(size=8))

    def loss(points: jax.Array) -> jax.Array:
        plan = Binning("sum", weights="linear").plan(
            Points(points), grid, policies=OVER_CELLS
        )
        return jnp.sum((plan(masses) - target) ** 2)

    assert_grad_matches_fd(loss, xy, eps=1e-6, rtol=1e-6)
    assert float(jax.grad(loss)(xy)[0, 0]) == 0.0


def test_linear_over_the_cells_needs_a_rectilinear_target() -> None:
    from functax import PolicyError
    from functax.geometry import AffineGrid

    grid = AffineGrid.from_affine((1.0, 0.0, 0.0, 0.0, -1.0, 4.0), (4, 4))
    with pytest.raises(PolicyError, match="extent='cells'"):
        Binning("sum", weights="linear").plan(
            Points(jnp.ones((3, 2))), grid, policies=OVER_CELLS
        )
