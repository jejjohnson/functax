"""Tests for `KernelSolve`, the saddle-point fit [c; b] = A⁻¹ [y; 0] of a kernel and a
tail. The slow tier: the dense-solve oracle and the block-equation property over
random kernels, each a second or more (mostly XLA compiling the factorisations).
"""

from __future__ import annotations

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle
from hypothesis import given, settings

from functax.interp import KernelSolve


N_SOURCE = 9


def _kernel_and_tail(
    seed: int, n_terms: int, *, nugget: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """A Gaussian-kernel K̃ = K_XX + βI on random 2-D sources, and the tail [1, x, y].

    The Gaussian kernel is positive definite, so C = Q₂ᵀ K̃ Q₂ is too for any tail.
    """
    rng = np.random.default_rng(seed)
    sources = rng.uniform(0.0, 3.0, size=(N_SOURCE, 2))
    squared = np.sum((sources[:, None] - sources[None]) ** 2, axis=-1)
    gram = np.exp(-squared) + nugget * np.eye(N_SOURCE)
    terms = np.column_stack([np.ones(N_SOURCE), sources])[:, :n_terms]
    return gram, terms


def _dense_saddle_point(gram: np.ndarray, terms: np.ndarray) -> np.ndarray:
    """A = [[K̃, P], [Pᵀ, 0]], assembled for NumPy's dense solve."""
    n_terms = terms.shape[1]
    return np.block([[gram, terms], [terms.T, np.zeros((n_terms, n_terms))]])


@jax.jit
def _block_residuals(
    gram: jax.Array, terms: jax.Array, rhs: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """(K̃ c + P b − g, Pᵀ c − h) for [c; b] = A⁻¹ [g; h], compiled once."""
    n_terms = terms.shape[1]
    solution = KernelSolve(gram, terms).solve(rhs)
    weights, tail = solution[:-n_terms], solution[-n_terms:]
    values, constraints = rhs[:-n_terms], rhs[-n_terms:]
    return gram @ weights + terms @ tail - values, terms.T @ weights - constraints


class TestKernelSolve:
    @pytest.mark.parametrize(
        "n_terms", [0, 1, 3], ids=["no-tail", "constant", "affine"]
    )
    def test_equals_the_dense_saddle_point_solve(self, n_terms: int) -> None:
        gram, terms = _kernel_and_tail(n_terms, n_terms)
        data = np.random.default_rng(10 + n_terms).normal(size=(2, N_SOURCE))
        fit = KernelSolve(jnp.asarray(gram), jnp.asarray(terms) if n_terms else None)
        # [c; b] = A⁻¹ [y; 0], by NumPy's LU on the assembled A
        rhs = np.concatenate([data, np.zeros((2, n_terms))], axis=-1)
        expected = np.linalg.solve(_dense_saddle_point(gram, terms), rhs.T).T
        # cond(A) is 1.3e2 to 7.4e2 for these three, so 1e-12 leaves headroom
        assert_matches_oracle(
            fit.apply(jnp.asarray(data)), expected, rtol=1e-12, atol=1e-12
        )

    @settings(max_examples=20, deadline=None)
    @given(seed=st.integers(0, 2**31 - 1))
    def test_the_solve_satisfies_both_block_equations(self, seed: int) -> None:
        # K̃ c + P b = g and Pᵀ c = h, for any right-hand side [g; h]
        gram, terms = _kernel_and_tail(seed, 3, nugget=0.1)
        rhs = np.random.default_rng(seed + 1).normal(size=N_SOURCE + 3)
        values, constraints = _block_residuals(
            jnp.asarray(gram), jnp.asarray(terms), jnp.asarray(rhs)
        )
        # with the nugget 0.1, cond(C) stays under 20 (200 seeds checked)
        assert_matches_oracle(values, np.zeros(N_SOURCE), rtol=0.0, atol=1e-12)
        assert_matches_oracle(constraints, np.zeros(3), rtol=0.0, atol=1e-12)
