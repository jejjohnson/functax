"""Tests for `Stencil`, the sparse read-out H with its hand-written adjoint. The slow
tier: oracle sweeps, dot-product tests across policies, gradient and trace checks,
each a second or more (mostly einx compiling new shapes).
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

    def test_equals_linear_transpose(self) -> None:
        stencil = _random_stencil(3)
        assert_transpose_matches(
            stencil.apply, stencil.apply_adjoint, jnp.zeros(N_SOURCE)
        )

    def test_lineax_operator_and_its_transpose(self) -> None:
        stencil = _random_stencil(4)
        operator = stencil.operator()
        assert operator.in_size() == N_SOURCE
        assert operator.out_size() == N_TARGET
        coefficients = jax.random.normal(jax.random.key(5), (N_SOURCE,))
        residual = jax.random.normal(jax.random.key(6), (N_TARGET,))
        assert_matches_oracle(
            operator.mv(coefficients), stencil.apply(coefficients), rtol=1e-15
        )
        assert_matches_oracle(
            operator.transpose().mv(residual),
            stencil.apply_adjoint(residual),
            rtol=1e-12,
            atol=1e-15,
        )


class TestDense:
    def test_apply_equals_the_dense_product(self) -> None:
        stencil = _random_stencil(8)
        coefficients = jax.random.normal(jax.random.key(9), (2, 3, N_SOURCE))
        dense = stencil.to_dense()
        assert_matches_oracle(
            stencil.apply(coefficients), coefficients @ dense.T, rtol=1e-12, atol=1e-14
        )

    def test_round_trip(self) -> None:
        rng = np.random.default_rng(7)
        dense = rng.normal(size=(9, 13))
        # at most three non-zeros per row
        for row in dense:
            row[rng.permutation(13)[rng.integers(0, 4) :]] = 0.0
        stencil = Stencil.from_dense(dense, max_taps=3)
        assert stencil.n_taps == 3
        np.testing.assert_array_equal(stencil.to_dense(), dense)


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

    def test_grad_with_respect_to_the_weights(self) -> None:
        stencil = _random_stencil(21)
        # one weight exactly zero: its gradient is still c at that tap
        weights = stencil.weights.at[0, 0].set(0.0)
        coefficients = jax.random.normal(jax.random.key(22), (N_SOURCE,))

        def readout(w: jax.Array) -> jax.Array:
            return Stencil(stencil.indices, w, n_source=N_SOURCE).apply(coefficients)

        assert_grad_matches_fd(readout, weights)


class TestApply:
    def test_batched_input(self) -> None:
        stencil = _random_stencil(10)
        coefficients = jnp.ones((5, 3, N_SOURCE))
        assert stencil.apply(coefficients).shape == (5, 3, N_TARGET)
        assert stencil.apply_adjoint(jnp.ones((5, 3, N_TARGET))).shape == (
            5,
            3,
            N_SOURCE,
        )

    def test_float32_stays_float32(self) -> None:
        stencil = _random_stencil(11, dtype=jnp.float32)
        coefficients = jnp.ones(N_SOURCE, dtype=jnp.float32)
        assert stencil.apply(coefficients).dtype == jnp.float32
        assert (
            stencil.apply_adjoint(jnp.ones(N_TARGET, dtype=jnp.float32)).dtype
            == jnp.float32
        )
