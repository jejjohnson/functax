"""Tests for the feature maps of `functax.basis.features`. The fast tier: closed forms,
layouts, dtypes and validation, each well under a second.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax import ValidationError
from functax.basis.features import (
    fourier_features,
    gaussian_window_features,
    rff_cosine_forward,
    rff_forward,
    seasonal_features,
    seasonal_frequencies,
)


W = jnp.array([[1.0, -0.5, 2.0], [0.3, 1.0, 0.5]])


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

    def test_zero_degrees_gives_zero_columns(self) -> None:
        assert fourier_features(jnp.ones(4), 0).shape == (4, 0)

    def test_rejects_a_negative_degree(self) -> None:
        with pytest.raises(ValidationError, match="'max_degree' must be non-negative"):
            fourier_features(jnp.ones(4), -1)


class TestSeasonalFeatures:
    def test_frequencies_are_harmonics_over_periods(self) -> None:
        index, frequencies = seasonal_frequencies([7.0, 365.25], [2, 1])
        assert index == [0, 0, 1]
        assert frequencies == [1 / 7.0, 2 / 7.0, 1 / 365.25]

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

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValidationError, match="the same length, got 2 and 1"):
            seasonal_frequencies([7.0, 30.0], [1])


class TestGaussianWindowFeatures:
    def test_matches_its_closed_form(self) -> None:
        t = jnp.linspace(0.0, 10.0, 11)
        windows = gaussian_window_features(t, jnp.array([5.0]), jnp.array([2.0]))
        expected = np.exp(-0.5 * ((np.asarray(t) - 5.0) / 2.0) ** 2)
        assert_matches_oracle(windows[:, 0], expected, rtol=1e-12)
        assert float(windows[5, 0]) == 1.0
