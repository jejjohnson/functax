"""Self-tests for the oracle harness in `tests/_harness.py`. The fast tier: behaviour,
validation and small closed-form checks, each well under a second.
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


def _differs_under_tracing(x: jax.Array) -> jax.Array:
    """Returns 2x eagerly and 3x when traced: a stand-in for a tracing bug."""
    factor = 3.0 if isinstance(x, jax.core.Tracer) else 2.0
    return factor * x


class TestMatchesOracle:
    def test_passes_within_tolerance(self) -> None:
        expected = np.array([1.0, -2.0, 3.0])
        assert_matches_oracle(expected * (1.0 + 1e-14), expected, rtol=1e-12)

    def test_names_the_worst_index(self) -> None:
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        actual = expected.copy()
        actual[1, 0] += 1e-3
        actual[0, 1] += 1e-9
        with pytest.raises(AssertionError, match=r"2 of 4 .* worst at index \(1, 0\)"):
            assert_matches_oracle(actual, expected, rtol=1e-12)

    def test_where_skips_masked_entries(self) -> None:
        expected = np.array([1.0, 2.0, 3.0])
        actual = np.array([1.0, 99.0, 3.0])
        assert_matches_oracle(
            actual, expected, rtol=1e-12, where=np.array([True, False, True])
        )

    def test_nan_matches_only_nan(self) -> None:
        expected = np.array([np.nan, 1.0])
        assert_matches_oracle(np.array([np.nan, 1.0]), expected, rtol=1e-12)
        with pytest.raises(AssertionError, match=r"worst at index \(0,\)"):
            assert_matches_oracle(np.array([0.0, 1.0]), expected, rtol=1e-12)

    def test_booleans_and_integers_match_exactly(self) -> None:
        assert_matches_oracle(
            np.array([True, False]), np.array([True, False]), rtol=1.0
        )
        with pytest.raises(AssertionError, match=r"worst at index \(1,\)"):
            assert_matches_oracle(np.array([3, 4]), np.array([3, 5]), rtol=1.0)

    def test_shape_mismatch_fails(self) -> None:
        with pytest.raises(AssertionError, match="shape mismatch"):
            assert_matches_oracle(np.zeros(3), np.zeros(4), rtol=1e-12)


class TestAdjoint:
    def test_fails_on_a_wrong_output_shape(self, dense_operator: jax.Array) -> None:
        with pytest.raises(AssertionError, match="expected \\(7,\\)"):
            assert_adjoint(
                lambda x: dense_operator @ x,
                lambda r: r,
                N_SOURCE,
                N_TARGET,
                key=jax.random.key(0),
            )

    def test_zero_operator_passes(self) -> None:
        assert_adjoint(
            lambda x: jnp.zeros(N_TARGET),
            lambda r: jnp.zeros(N_SOURCE),
            N_SOURCE,
            N_TARGET,
            key=jax.random.key(0),
        )


class TestTransposeMatches:
    def test_passes_for_the_transpose(self, dense_operator: jax.Array) -> None:
        assert_transpose_matches(
            lambda x: dense_operator @ x,
            lambda r: jnp.transpose(dense_operator) @ r,
            jnp.zeros(N_SOURCE),
        )

    def test_fails_for_a_perturbed_adjoint(
        self, dense_operator: jax.Array, perturbed_adjoint: jax.Array
    ) -> None:
        with pytest.raises(AssertionError, match="worst at index"):
            assert_transpose_matches(
                lambda x: dense_operator @ x,
                lambda r: perturbed_adjoint @ r,
                jnp.zeros(N_SOURCE),
            )


class TestTraces:
    def test_passes_for_pytree_arguments_and_outputs(self) -> None:
        inputs = {"a": jnp.arange(6.0).reshape(2, 3), "b": jnp.ones(2)}
        assert_traces(lambda tree: (tree["a"] * tree["b"], tree["b"]), inputs)

    def test_fails_when_tracing_changes_the_result(self) -> None:
        with pytest.raises(AssertionError, match=r"jit\(fn\)"):
            assert_traces(_differs_under_tracing, jnp.ones((3, 2)))

    def test_fails_on_mismatched_batch_axes(self) -> None:
        with pytest.raises(AssertionError, match="same leading batch axis"):
            assert_traces(lambda x, y: x + y[0], jnp.ones(3), jnp.ones(4))


class TestGradMatchesFd:
    def test_passes_for_a_smooth_function(self) -> None:
        point = jnp.array([0.3, -1.2, 2.0])
        assert_grad_matches_fd(lambda x: jnp.sin(x) * jnp.exp(x[0]), point)

    def test_fails_for_a_wrong_derivative(self) -> None:
        @jax.custom_jvp
        def wrong_sin(x: jax.Array) -> jax.Array:
            return jnp.sin(x)

        @wrong_sin.defjvp
        def _wrong_sin_jvp(primals, tangents):  # type: ignore[no-untyped-def]
            (x,), (dx,) = primals, tangents
            # d sin x / dx is cos x; 1.01 cos x is wrong by 1 %
            return jnp.sin(x), 1.01 * jnp.cos(x) * dx

        with pytest.raises(AssertionError, match="worst at index"):
            assert_grad_matches_fd(wrong_sin, jnp.array([0.2, 0.9]))


def test_helpers_refuse_float32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jax.config, "read", lambda name: False)
    with pytest.raises(AssertionError, match="jax_enable_x64"):
        assert_traces(jnp.sin, jnp.ones(2))
