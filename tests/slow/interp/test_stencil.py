"""Tests for `Stencil`, the sparse read-out H with its hand-written adjoint. The slow
tier: oracle sweeps, dot-product tests across policies, gradient and trace checks,
each a second or more (mostly einx compiling new shapes).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from _harness import (
    assert_adjoint,
    assert_matches_oracle,
    assert_traces,
)

from functax.interp import Stencil


N_SOURCE = 11


N_TARGET = 7


N_TAPS = 4


def _random_stencil(seed: int, *, dtype: jnp.dtype = jnp.float64) -> Stencil:
    """A random stencil whose rows repeat indices, so the scatter must add."""
    key_indices, key_weights = jax.random.split(jax.random.key(seed))
    indices = jax.random.randint(key_indices, (N_TARGET, N_TAPS), 0, 3)
    # three distinct indices per row at most: repeats are certain
    indices = indices + jnp.arange(N_TARGET)[:, None]
    weights = jax.random.normal(key_weights, (N_TARGET, N_TAPS), dtype=dtype)
    return Stencil(indices, weights, n_source=N_SOURCE)


class TestAdjoint:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_dot_product(self, seed: int) -> None:
        stencil = _random_stencil(seed)
        assert jnp.any(stencil.indices[:, 0] == stencil.indices[:, 1:].T)
        assert_adjoint(
            stencil.apply,
            stencil.apply_adjoint,
            N_SOURCE,
            N_TARGET,
            key=jax.random.key(seed),
        )


class TestDense:
    def test_apply_equals_the_dense_product(self) -> None:
        stencil = _random_stencil(8)
        coefficients = jax.random.normal(jax.random.key(9), (2, 3, N_SOURCE))
        dense = stencil.to_dense()
        assert_matches_oracle(
            stencil.apply(coefficients), coefficients @ dense.T, rtol=1e-12, atol=1e-14
        )


class TestTraces:
    def test_jit_and_vmap_over_a_batch_of_fields(self) -> None:
        stencil = _random_stencil(14)
        batch = jax.random.normal(jax.random.key(15), (4, N_SOURCE))
        # lambdas, since jax.jit hashes its function and a bound method of a
        # module hashes the module's arrays
        assert_traces(lambda c: stencil.apply(c), batch)
        residuals = jax.random.normal(jax.random.key(16), (4, N_TARGET))
        assert_traces(lambda r: stencil.apply_adjoint(r), residuals)

    def test_vmap_over_a_batch_of_stencils(self) -> None:
        stencils = [_random_stencil(seed) for seed in (17, 18, 19)]
        batch = jax.tree.map(lambda *leaves: jnp.stack(leaves), *stencils)
        coefficients = jax.random.normal(jax.random.key(20), (3, N_SOURCE))
        assert_traces(lambda stencil, c: stencil.apply(c), batch, coefficients)
