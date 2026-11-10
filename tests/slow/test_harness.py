"""Self-tests for the oracle harness in `tests/_harness.py`. The slow tier:
sweeps, oracle and dot-product checks, gradient and trace checks, anything over
0.3 s on CI (mostly einx and XLA compiling new shapes).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from _harness import (
    assert_adjoint,
    assert_traces,
)


N_SOURCE = 7


N_TARGET = 5


PERTURBATION = 1e-6


@pytest.fixture
def dense_operator() -> jax.Array:
    """A random dense H ∈ ℝ^(M × N)."""
    return jax.random.normal(jax.random.key(1), (N_TARGET, N_SOURCE))


@pytest.fixture
def perturbed_adjoint(dense_operator: jax.Array) -> jax.Array:
    """G = Hᵀ + ε E, with E a random matrix of Hᵀ's shape."""
    noise = jax.random.normal(jax.random.key(2), (N_SOURCE, N_TARGET))
    return jnp.transpose(dense_operator) + PERTURBATION * noise


class TestAdjoint:
    def test_passes_for_the_transpose(self, dense_operator: jax.Array) -> None:
        assert_adjoint(
            lambda x: dense_operator @ x,
            lambda r: jnp.transpose(dense_operator) @ r,
            N_SOURCE,
            N_TARGET,
            key=jax.random.key(0),
        )

    def test_fails_for_a_perturbed_adjoint(
        self, dense_operator: jax.Array, perturbed_adjoint: jax.Array
    ) -> None:
        with pytest.raises(AssertionError, match="dot-product test failed"):
            assert_adjoint(
                lambda x: dense_operator @ x,
                lambda r: perturbed_adjoint @ r,
                N_SOURCE,
                N_TARGET,
                key=jax.random.key(0),
            )


class TestTraces:
    def test_passes_for_a_pure_function(self) -> None:
        batch = jax.random.normal(jax.random.key(3), (4, 3))
        assert_traces(lambda x: jnp.sin(x) * jnp.sum(x), batch)
