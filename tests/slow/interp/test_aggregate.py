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

from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import Binning


RNG = np.random.default_rng(39)
# an irregular and a regular axis
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
