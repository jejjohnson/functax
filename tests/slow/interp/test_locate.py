"""Tests for `locate`, `index_positions`, `remap_taps` and `Policies`. The slow tier:
oracle sweeps, dot-product tests across policies, gradient and trace checks, each a
second or more (mostly einx compiling new shapes).
"""

from __future__ import annotations

import equinox as eqx
import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
from _harness import assert_grad_matches_fd, assert_matches_oracle
from hypothesis import given, settings

from functax.geometry import Axis
from functax.interp import (
    index_positions,
    locate,
)


def _reference_positions(nodes: np.ndarray, x: np.ndarray) -> np.ndarray:
    """u from NumPy's searchsorted on increasing nodes, linear past the ends."""
    cell = np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, nodes.size - 2)
    return cell + (x - nodes[cell]) / (nodes[cell + 1] - nodes[cell])


class TestIndexPositions:
    def test_equals_searchsorted_on_10000_random_axes(self) -> None:
        rng = np.random.default_rng(0)
        n_axes, n_nodes, n_targets = 10_000, 12, 8
        steps = rng.uniform(0.1, 2.0, size=(n_axes, n_nodes - 1))
        nodes = np.cumsum(
            np.concatenate([rng.normal(size=(n_axes, 1)), steps], axis=1), axis=1
        )
        descending = rng.random(n_axes) < 0.5
        values = np.where(descending[:, None], nodes[:, ::-1], nodes)
        span = nodes[:, -1:] - nodes[:, :1]
        targets = nodes[:, :1] + rng.uniform(-0.2, 1.2, size=(n_axes, n_targets)) * span
        # u on the increasing nodes, then reversed for a decreasing axis
        ascending = np.stack(
            [_reference_positions(n, x) for n, x in zip(nodes, targets, strict=True)]
        )
        expected = np.where(descending[:, None], (n_nodes - 1) - ascending, ascending)

        # one template axis, its values swapped for the whole stack
        template = Axis(jnp.asarray(values[0]))
        axes = eqx.tree_at(lambda axis: axis.values, template, jnp.asarray(values))
        located = jax.vmap(index_positions)(axes, jnp.asarray(targets))
        assert_matches_oracle(located, expected, rtol=1e-12, atol=1e-12)

    @settings(max_examples=50, deadline=None)
    @given(
        n_nodes=st.integers(2, 20),
        seed=st.integers(0, 2**31 - 1),
        descending=st.booleans(),
    )
    def test_equals_searchsorted_on_random_axes(
        self, n_nodes: int, seed: int, descending: bool
    ) -> None:
        rng = np.random.default_rng(seed)
        nodes = np.cumsum(rng.uniform(0.1, 2.0, size=n_nodes))
        values = nodes[::-1] if descending else nodes
        targets = rng.uniform(nodes[0] - 1.0, nodes[-1] + 1.0, size=16)
        ascending = _reference_positions(nodes, targets)
        expected = (n_nodes - 1) - ascending if descending else ascending
        located = index_positions(Axis(jnp.asarray(values)), jnp.asarray(targets))
        assert_matches_oracle(located, expected, rtol=1e-12, atol=1e-12)

    def test_gradient_is_one_over_the_spacing(self) -> None:
        axis = Axis(jnp.array([0.0, 2.0, 3.0, 7.0]))
        targets = jnp.array([1.0, 2.5, 5.0])
        slopes = jax.vmap(jax.grad(lambda x: index_positions(axis, x)))(targets)
        assert_matches_oracle(slopes, jnp.array([0.5, 1.0, 0.25]), rtol=1e-15)
        assert_grad_matches_fd(lambda x: index_positions(axis, x), targets)

    def test_regular_arithmetic_equals_the_search(self) -> None:
        axis = Axis(jnp.linspace(-90.0, 90.0, 37))
        assert axis.is_regular
        targets = jnp.linspace(-100.0, 100.0, 41)
        searched = jax.jit(index_positions)(axis, targets)
        assert_matches_oracle(
            index_positions(axis, targets), searched, rtol=1e-13, atol=1e-13
        )


class TestPeriodic:
    def test_a_period_apart_locates_identically(self) -> None:
        longitude = Axis(jnp.arange(0.0, 360.0, 2.5), period=360.0)
        targets = jnp.linspace(-170.0, 170.0, 23)
        for shift in (-720.0, -360.0, 360.0, 1080.0):
            assert_matches_oracle(
                index_positions(longitude, targets + shift),
                index_positions(longitude, targets),
                rtol=1e-12,
                atol=1e-12,
            )

    def test_the_wrap_cell(self) -> None:
        longitude = Axis(jnp.arange(0.0, 360.0, 90.0), period=360.0)
        # 315° lies halfway between the last node (270°) and 360° ≡ 0°
        assert index_positions(longitude, jnp.array([315.0, -45.0])).tolist() == [
            3.5,
            3.5,
        ]
        cell, offset, inside = locate(longitude, jnp.array([315.0, 1e6]))
        assert cell.tolist()[0] == 3
        assert offset.tolist()[0] == 0.5
        assert inside.tolist() == [True, True]

    def test_an_irregular_periodic_axis(self) -> None:
        longitude = Axis(jnp.array([0.0, 100.0, 300.0]), period=360.0)
        # the wrap cell runs from 300° to 360°
        assert index_positions(longitude, jnp.array([330.0])).tolist() == [2.5]


class TestLocate:
    def test_descending_cells_and_offsets(self) -> None:
        # wavelength from wavenumber: decreasing nodes
        axis = Axis(jnp.array([2.5, 2.0, 1.0]))
        cell, offset, inside = locate(axis, jnp.array([2.25, 1.5]))
        assert cell.tolist() == [0, 1]
        assert offset.tolist() == [0.5, 0.5]
        assert inside.tolist() == [True, True]
