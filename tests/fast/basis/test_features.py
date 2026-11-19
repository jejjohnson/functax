"""Tests for the feature maps of `functax.basis.features`. The fast tier: the
frequency layout, empty features and validation, none of which compiles more than a
trivial kernel.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from functax import ValidationError
from functax.basis.features import fourier_features, seasonal_frequencies


class TestFourierFeatures:
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

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValidationError, match="the same length, got 2 and 1"):
            seasonal_frequencies([7.0, 30.0], [1])
