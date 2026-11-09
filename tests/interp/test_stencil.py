"""Tests for `Stencil`, the sparse read-out H with its hand-written adjoint."""

from __future__ import annotations

import inspect

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

from functax import ValidationError
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
    def test_round_trip(self) -> None:
        rng = np.random.default_rng(7)
        dense = rng.normal(size=(9, 13))
        # at most three non-zeros per row
        for row in dense:
            row[rng.permutation(13)[rng.integers(0, 4) :]] = 0.0
        stencil = Stencil.from_dense(dense, max_taps=3)
        assert stencil.n_taps == 3
        np.testing.assert_array_equal(stencil.to_dense(), dense)

    def test_apply_equals_the_dense_product(self) -> None:
        stencil = _random_stencil(8)
        coefficients = jax.random.normal(jax.random.key(9), (2, 3, N_SOURCE))
        dense = stencil.to_dense()
        assert_matches_oracle(
            stencil.apply(coefficients), coefficients @ dense.T, rtol=1e-12, atol=1e-14
        )

    def test_from_dense_rejects_full_rows_and_bad_input(self) -> None:
        with pytest.raises(ValidationError, match="row 0 has 3 non-zeros"):
            Stencil.from_dense(np.ones((2, 3)), max_taps=2)
        with pytest.raises(ValidationError, match="'matrix' must be 2-D"):
            Stencil.from_dense(np.ones(3), max_taps=2)
        with pytest.raises(ValidationError, match="'matrix' must have a floating"):
            Stencil.from_dense(np.ones((2, 2), dtype=int), max_taps=2)


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

    def test_padding_repeats_a_read_index(self) -> None:
        stencil = Stencil.from_dense(
            np.array([[0.0, 0.0, 2.0], [1.0, 3.0, 0.0]]), max_taps=3
        )
        assert stencil.indices.tolist() == [[2, 2, 2], [0, 1, 0]]
        # a NaN the rows never read stays out of them
        values = stencil.apply(jnp.array([1.0, 1.0, 1.0]).at[2].set(jnp.nan))
        assert bool(jnp.isnan(values[0]))
        assert values[1] == 4.0

    def test_nan_at_a_weighted_tap_propagates(self) -> None:
        stencil = Stencil(jnp.array([[0, 1]]), jnp.array([[0.5, 0.5]]), n_source=2)
        assert bool(jnp.isnan(stencil.apply(jnp.array([jnp.nan, 2.0]))[0]))

    def test_valid_defaults_to_all_rows(self) -> None:
        assert bool(jnp.all(_random_stencil(12).valid))

    def test_rejects_a_wrong_last_axis(self) -> None:
        stencil = _random_stencil(13)
        with pytest.raises(
            ValidationError, match="'coefficients' must have a last axis of length 11"
        ):
            stencil.apply(jnp.ones(N_SOURCE + 1))
        with pytest.raises(
            ValidationError, match="'residual' must have a last axis of length 7"
        ):
            stencil.apply_adjoint(jnp.ones(N_TARGET + 1))


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


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            (
                {"indices": jnp.zeros(3, dtype=int), "weights": jnp.zeros(3)},
                "'indices' must be 2-D",
            ),
            (
                {"indices": jnp.zeros((2, 2), dtype=int), "weights": jnp.zeros((2, 3))},
                "'weights' must have the shape",
            ),
            (
                {"indices": jnp.zeros((2, 2)), "weights": jnp.zeros((2, 2))},
                "'indices' must have an integer",
            ),
            (
                {
                    "indices": jnp.zeros((2, 2), dtype=int),
                    "weights": jnp.zeros((2, 2), dtype=int),
                },
                "'weights' must have a floating",
            ),
            (
                {
                    "indices": jnp.zeros((2, 2), dtype=int),
                    "weights": jnp.zeros((2, 2)),
                    "n_source": 0,
                },
                "'n_source' must be positive",
            ),
            (
                {
                    "indices": jnp.zeros((2, 2), dtype=int),
                    "weights": jnp.zeros((2, 2)),
                    "n_source": 2.0,
                },
                "'n_source' must be an integer",
            ),
            (
                {"indices": jnp.array([[0, 3]]), "weights": jnp.zeros((1, 2))},
                r"'indices' must lie in \[0, 3\)",
            ),
            (
                {"indices": jnp.array([[-1, 0]]), "weights": jnp.zeros((1, 2))},
                r"'indices' must lie in \[0, 3\)",
            ),
            (
                {
                    "indices": jnp.zeros((2, 2), dtype=int),
                    "weights": jnp.zeros((2, 2)),
                    "valid": jnp.ones(3, dtype=bool),
                },
                "'valid' must be boolean",
            ),
        ],
    )
    def test_rejects_bad_fields(self, kwargs: dict[str, object], match: str) -> None:
        kwargs = {"n_source": 3, **kwargs}
        with pytest.raises(ValidationError, match=match):
            inspect.unwrap(Stencil.__init__)(object.__new__(Stencil), **kwargs)

    def test_checks_indices_under_jit(self) -> None:
        build = jax.jit(lambda i: Stencil(i, jnp.ones((1, 2)), n_source=3).indices)
        with pytest.raises(Exception, match=r"'indices' must lie in \[0, 3\)"):
            build(jnp.array([[0, 5]])).block_until_ready()
