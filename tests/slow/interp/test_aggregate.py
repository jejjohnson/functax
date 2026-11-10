"""Tests for `Binning`: SciPy's binned_statistic_dd for the mean, sum and count,
cloud-in-cell conservation, the gradient in the positions, and the dot-product test.
The slow tier: each compiles new shapes eagerly.
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
from scipy.stats import binned_statistic_dd

from functax import (
    OutsideDomainError,
)
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
    [("mean", "nearest"), ("sum", "linear"), ("mean", "linear")],
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


def test_samples_off_the_grid_are_dropped_not_clamped() -> None:
    counts = Binning("count").plan(SAMPLES, CELLS)
    assert counts(VALUES).tolist() == [2.0, 0.0, 1.0, 0.0]
    assert counts.counts.tolist() == [2.0, 0.0, 1.0, 0.0]
    with pytest.raises(OutsideDomainError, match="2 of 5 targets"):
        Binning().plan(SAMPLES, CELLS, policies=Policies(outside="raise"))
