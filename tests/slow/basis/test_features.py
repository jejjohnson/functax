"""Tests for the feature maps of `functax.basis.features`. The slow tier: closed forms
and layouts (each eager call compiles its einx kernels, a second or so), the
random-feature kernel estimate at 10⁴ features, and gradient, trace and dtype checks.
"""

from __future__ import annotations

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from hypothesis import given, settings

from functax.basis.features import (
    fourier_features,
    gaussian_window_features,
    rff_cosine_forward,
    rff_forward,
    seasonal_features,
)


W = jnp.array([[1.0, -0.5, 2.0], [0.3, 1.0, 0.5]])


PHASES = jnp.array([0.1, 2.0, 4.0])


POINT = jnp.array([0.4, -1.1])


class TestRandomFourierFeatures:
    def test_rff_forward_is_cosines_then_sines(self) -> None:
        projection = np.asarray(POINT) @ np.asarray(W) / 0.8
        expected = np.sqrt(1.0 / 3.0) * np.concatenate(
            [np.cos(projection), np.sin(projection)]
        )
        assert_matches_oracle(rff_forward(W, 0.8, 3, POINT), expected, rtol=1e-12)

    def test_rff_forward_has_unit_norm(self) -> None:
        # cos² + sin² = 1 for each of the F pairs, times 1/F
        features = rff_forward(W, 0.8, 3, POINT)
        assert_matches_oracle(features @ features, 1.0, rtol=1e-12)

    def test_rff_cosine_forward_matches_its_closed_form(self) -> None:
        phases = jnp.array([0.1, 2.0, 4.0])
        projection = np.asarray(POINT) @ np.asarray(W) / 0.8 + np.asarray(phases)
        expected = np.sqrt(2.0 / 3.0) * np.cos(projection)
        assert_matches_oracle(
            rff_cosine_forward(W, phases, 0.8, 3, POINT), expected, rtol=1e-12
        )

    def test_a_traced_lengthscale_is_accepted(self) -> None:
        features = jax.jit(lambda scale: rff_forward(W, scale, 3, POINT))(
            jnp.asarray(0.8)
        )
        assert_matches_oracle(features, rff_forward(W, 0.8, 3, POINT), rtol=1e-12)


class TestFourierFeatures:
    def test_layout_is_cosines_then_sines_at_dyadic_frequencies(self) -> None:
        x = jnp.array([0.1, 0.35])
        phase = 2 * np.pi * np.outer(np.asarray(x), [1.0, 2.0, 4.0])
        expected = np.concatenate([np.cos(phase), np.sin(phase)], axis=1)
        assert_matches_oracle(fourier_features(x, 3), expected, rtol=1e-12, atol=1e-15)

    def test_rescale_divides_each_pair_by_its_degree(self) -> None:
        x = jnp.array([0.1, 0.35])
        ratio = fourier_features(x, 3, rescale=True) / fourier_features(x, 3)
        assert_matches_oracle(
            ratio, np.tile([1.0, 1 / 2, 1 / 3] * 2, (2, 1)), rtol=1e-12
        )


class TestSeasonalFeatures:
    def test_layout_is_cosines_then_sines(self) -> None:
        t = jnp.array([0.0, 1.5, 3.0])
        phase = 2 * np.pi * np.outer(np.asarray(t), [1 / 7.0, 2 / 7.0])
        expected = np.concatenate([np.cos(phase), np.sin(phase)], axis=1)
        assert_matches_oracle(
            seasonal_features(t, [7.0], [2]), expected, rtol=1e-12, atol=1e-15
        )

    def test_rescale_divides_by_the_harmonic_within_each_period(self) -> None:
        t = jnp.array([0.4, 1.5])
        ratio = seasonal_features(t, [7.0, 30.0], [2, 2], rescale=True) / (
            seasonal_features(t, [7.0, 30.0], [2, 2])
        )
        assert_matches_oracle(ratio, np.tile([1.0, 0.5] * 4, (2, 1)), rtol=1e-12)

    def test_traces_with_python_periods(self) -> None:
        t = jnp.linspace(0.0, 10.0, 4)
        traced = jax.jit(lambda u: seasonal_features(u, (7.0,), (2,)))(t)
        assert_matches_oracle(traced, seasonal_features(t, (7.0,), (2,)), rtol=1e-12)

    def test_float64_time_keeps_a_float64_phase(self) -> None:
        # seconds since the epoch with a daily period: a float32 1/86400 puts the
        # phase off by a visible fraction of a cycle at t ≈ 1.7e9 s
        t = jnp.array([1.7e9, 1.7e9 + 3600.0])
        phase = 2 * np.pi * np.asarray(t) / 86_400.0
        expected = np.stack([np.cos(phase), np.sin(phase)], axis=1)
        # a float64 phase of about 1.2e5 rad rounds off near 1e-11 rad
        assert_matches_oracle(
            seasonal_features(t, (86_400.0,), (1,)), expected, rtol=0.0, atol=1e-9
        )


class TestGaussianWindowFeatures:
    def test_matches_its_closed_form(self) -> None:
        t = jnp.linspace(0.0, 10.0, 11)
        windows = gaussian_window_features(t, jnp.array([5.0]), jnp.array([2.0]))
        expected = np.exp(-0.5 * ((np.asarray(t) - 5.0) / 2.0) ** 2)
        assert_matches_oracle(windows[:, 0], expected, rtol=1e-12)
        assert float(windows[5, 0]) == 1.0


class TestKernelEstimate:
    def test_rff_inner_products_approach_the_squared_exponential(self) -> None:
        n_features = 10_000
        lengthscale = 0.9
        frequencies = jax.random.normal(jax.random.key(0), (3, n_features))
        points = jax.random.normal(jax.random.key(1), (6, 3))
        features = jax.vmap(
            lambda point: rff_forward(frequencies, lengthscale, n_features, point)
        )(points)
        estimate = features @ features.T
        # k(x, x′) = exp(−‖x − x′‖² / (2ℓ²))
        difference = np.asarray(points)[:, None, :] - np.asarray(points)[None, :, :]
        kernel = np.exp(-(difference**2).sum(-1) / (2 * lengthscale**2))
        # each entry is a mean of F terms cos(·), each of variance ≤ 1/2, so its
        # standard error is ≤ √(1/(2F)) ≈ 0.0071; 0.05 is seven standard errors
        np.testing.assert_allclose(np.asarray(estimate), kernel, atol=0.05)


class TestGradients:
    def test_rff_forward(self) -> None:
        assert_grad_matches_fd(
            lambda x: rff_forward(W, 0.8, 3, x), jnp.array([0.4, -1.1])
        )

    def test_rff_cosine_forward(self) -> None:
        assert_grad_matches_fd(
            lambda x: rff_cosine_forward(W, PHASES, 0.8, 3, x), jnp.array([0.4, -1.1])
        )

    def test_fourier_features(self) -> None:
        assert_grad_matches_fd(
            lambda x: fourier_features(x, 3, rescale=True), jnp.array([0.1, 0.35])
        )

    def test_seasonal_features(self) -> None:
        assert_grad_matches_fd(
            lambda t: seasonal_features(t, (7.0, 30.0), (2, 1)), jnp.array([0.4, 3.5])
        )

    def test_gaussian_window_features_in_the_centres(self) -> None:
        t = jnp.linspace(0.0, 10.0, 5)
        assert_grad_matches_fd(
            lambda centres: gaussian_window_features(t, centres, jnp.array([2.0, 3.0])),
            jnp.array([2.5, 7.0]),
        )


class TestTraces:
    @pytest.mark.parametrize(
        "fn",
        [
            lambda x: rff_forward(W, 0.8, 3, x),
            lambda x: rff_cosine_forward(W, PHASES, 0.8, 3, x),
        ],
        ids=["rff_forward", "rff_cosine_forward"],
    )
    def test_random_features_over_points(self, fn: object) -> None:
        points = jax.random.normal(jax.random.key(2), (4, 2))
        assert_traces(fn, points)

    @pytest.mark.parametrize(
        "fn",
        [
            lambda x: fourier_features(x, 3, rescale=True),
            lambda x: seasonal_features(x, (7.0, 30.0), (2, 1), rescale=True),
            lambda x: gaussian_window_features(
                x, jnp.array([2.0, 8.0]), jnp.array([1.0, 1.5])
            ),
        ],
        ids=["fourier_features", "seasonal_features", "gaussian_window_features"],
    )
    def test_coordinate_features_over_batches(self, fn: object) -> None:
        batches = jax.random.uniform(jax.random.key(3), (3, 5), maxval=10.0)
        assert_traces(fn, batches)


# compiled once per module, so each hypothesis example reuses the kernels
@jax.jit
def _rff_gram(frequencies: jax.Array, points: jax.Array) -> jax.Array:
    """⟨φ(xᵢ), φ(xⱼ)⟩ for 16 random features of lengthscale 0.7."""
    features = jax.vmap(lambda x: rff_forward(frequencies, 0.7, 16, x))(points)
    return features @ features.T


_FOURIER = jax.jit(lambda x: fourier_features(x, 4, rescale=True))


_SEASONAL = jax.jit(lambda x: seasonal_features(x, (7.0, 28.0), (3, 2)))


_WINDOWS = jax.jit(gaussian_window_features)


class TestProperties:
    @settings(max_examples=20, deadline=None)
    @given(seed=st.integers(0, 2**31 - 1))
    def test_rff_gram_is_translation_invariant(self, seed: int) -> None:
        # ⟨φ(x + s), φ(y + s)⟩ = (1/F) ∑_f cos(w_fᵀ(x − y)/ℓ), whatever s
        keys = jax.random.split(jax.random.key(seed), 3)
        frequencies = jax.random.normal(keys[0], (3, 16))
        points = jax.random.normal(keys[1], (5, 3))
        shift = 3.0 * jax.random.normal(keys[2], (3,))
        assert_matches_oracle(
            _rff_gram(frequencies, points + shift),
            _rff_gram(frequencies, points),
            rtol=0.0,
            atol=1e-12,
        )

    @settings(max_examples=20, deadline=None)
    @given(seed=st.integers(0, 2**31 - 1), cycles=st.integers(-3, 3))
    def test_features_are_periodic(self, seed: int, cycles: int) -> None:
        # the dyadic frequencies 2ᵈ repeat over a unit shift, and the harmonics
        # of 7 and 28 over 28
        x = jax.random.uniform(jax.random.key(seed), (6,), minval=-5.0, maxval=5.0)
        assert_matches_oracle(_FOURIER(x + cycles), _FOURIER(x), rtol=0.0, atol=1e-12)
        assert_matches_oracle(
            _SEASONAL(x + 28.0 * cycles), _SEASONAL(x), rtol=0.0, atol=1e-12
        )

    @settings(max_examples=20, deadline=None)
    @given(seed=st.integers(0, 2**31 - 1), shift=st.floats(-50.0, 50.0))
    def test_windows_move_with_their_centres(self, seed: int, shift: float) -> None:
        # χ(t + s; t_a + s, T_a) = χ(t; t_a, T_a): only t − t_a enters
        keys = jax.random.split(jax.random.key(seed), 3)
        t = jax.random.uniform(keys[0], (7,), maxval=10.0)
        centres = jax.random.uniform(keys[1], (3,), maxval=10.0)
        widths = jax.random.uniform(keys[2], (3,), minval=0.5, maxval=3.0)
        assert_matches_oracle(
            _WINDOWS(t + shift, centres + shift, widths),
            _WINDOWS(t, centres, widths),
            rtol=0.0,
            atol=1e-12,
        )


class TestDtypes:
    @staticmethod
    def _fourier(x: jax.Array) -> jax.Array:
        return fourier_features(x, 4, rescale=True)

    @staticmethod
    def _seasonal(x: jax.Array) -> jax.Array:
        return seasonal_features(x, (7.0, 365.25), (2, 3), rescale=True)

    @pytest.mark.parametrize("name", ["_fourier", "_seasonal"])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_floating_inputs_keep_their_dtype(self, name: str, dtype: type) -> None:
        x = jnp.linspace(0.0, 3.0, 5, dtype=dtype)
        assert getattr(self, name)(x).dtype == dtype

    @pytest.mark.parametrize("name", ["_fourier", "_seasonal"])
    def test_integers_become_float32(self, name: str) -> None:
        fn = getattr(self, name)
        features = fn(jnp.arange(5))
        assert features.dtype == jnp.float32
        expected = fn(jnp.arange(5, dtype=jnp.float32))
        assert_matches_oracle(features, expected, rtol=0.0, atol=1e-6)
