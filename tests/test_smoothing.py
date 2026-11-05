"""Tests for ``mypackage.smoothing``."""

from __future__ import annotations

import math

import pytest

from mypackage import (
    EmptySeriesError,
    Padding,
    ValidationError,
    exponential_moving_average,
    median_filter,
    moving_average,
    summarize,
)
from mypackage.smoothing import _reflect_index


SMOOTHERS = [
    pytest.param(lambda s: moving_average(s, window=3), id="moving_average"),
    pytest.param(lambda s: median_filter(s, window=3), id="median_filter"),
    pytest.param(lambda s: exponential_moving_average(s, alpha=0.5), id="exponential"),
]


class TestPadding:
    def test_is_a_string_enum(self) -> None:
        assert Padding.EDGE == "edge"
        assert Padding("reflect") is Padding.REFLECT

    def test_string_aliases_are_accepted(self, simple_series: list[float]) -> None:
        assert moving_average(simple_series, 3, padding="edge") == moving_average(
            simple_series, 3, padding=Padding.EDGE
        )

    def test_unknown_padding_raises(self, simple_series: list[float]) -> None:
        with pytest.raises(ValidationError, match="'padding' must be one of"):
            moving_average(simple_series, 3, padding="wrap")

    @pytest.mark.parametrize("padding", list(Padding))
    def test_every_strategy_runs(
        self, padding: Padding, simple_series: list[float]
    ) -> None:
        assert moving_average(simple_series, 3, padding=padding)


class TestReflectIndex:
    """The reflection helper is exercised directly: the ``length == 1`` guard
    is unreachable through the public API, because a one-sample series only
    admits a window of one, which needs no padding at all."""

    def test_single_element_series_collapses_to_index_zero(self) -> None:
        assert all(_reflect_index(index, 1) == 0 for index in (-3, 0, 5))

    @pytest.mark.parametrize(
        ("index", "expected"),
        [(-1, 1), (-2, 2), (0, 0), (2, 2), (3, 1), (4, 0), (5, 1)],
    )
    def test_mirrors_without_repeating_the_endpoints(
        self, index: int, expected: int
    ) -> None:
        assert _reflect_index(index, 3) == expected


class TestMovingAverage:
    def test_window_of_one_is_the_identity(self, simple_series: list[float]) -> None:
        assert moving_average(simple_series, 1) == simple_series

    def test_interior_is_the_plain_mean(self, simple_series: list[float]) -> None:
        smoothed = moving_average(simple_series, 3)
        assert smoothed[1:4] == pytest.approx([2.0, 3.0, 4.0])

    def test_edge_padding_preserves_length(self, simple_series: list[float]) -> None:
        assert len(moving_average(simple_series, 3)) == len(simple_series)

    def test_no_padding_shortens_the_output(self, simple_series: list[float]) -> None:
        smoothed = moving_average(simple_series, 3, padding=Padding.NONE)
        assert smoothed == pytest.approx([2.0, 3.0, 4.0])

    def test_zero_padding_pulls_ends_toward_zero(self) -> None:
        assert moving_average(
            [3.0, 3.0, 3.0], 3, padding=Padding.ZERO
        ) == pytest.approx([2.0, 3.0, 2.0])

    def test_reflect_padding_mirrors_without_repeating(self) -> None:
        # Reflecting [1, 2, 3] gives [2, 1, 2, 3, 2] -> means 5/3, 2, 7/3.
        assert moving_average([1.0, 2.0, 3.0], 3, padding=Padding.REFLECT) == (
            pytest.approx([5.0 / 3.0, 2.0, 7.0 / 3.0])
        )

    def test_reflect_handles_single_element(self) -> None:
        assert moving_average([4.0], 1, padding=Padding.REFLECT) == [4.0]

    def test_preserves_a_constant_series(self) -> None:
        constant = [7.0] * 10
        assert moving_average(constant, 5) == pytest.approx(constant)

    def test_reduces_noise_variance(self, sine_series: list[float]) -> None:
        noise = [0.4 * (-1.0) ** index for index in range(len(sine_series))]
        noisy = [
            clean + wobble for clean, wobble in zip(sine_series, noise, strict=True)
        ]
        residual_before = summarize(
            [n - c for n, c in zip(noisy, sine_series, strict=True)]
        ).variance
        smoothed = moving_average(noisy, 3)
        residual_after = summarize(
            [s - c for s, c in zip(smoothed, sine_series, strict=True)]
        ).variance
        assert residual_after < residual_before

    @pytest.mark.parametrize("window", [0, -1, 100])
    def test_invalid_window_raises(
        self, window: int, simple_series: list[float]
    ) -> None:
        with pytest.raises(ValidationError):
            moving_average(simple_series, window)

    def test_boolean_window_is_rejected(self, simple_series: list[float]) -> None:
        with pytest.raises(ValidationError, match="positive integer"):
            moving_average(simple_series, True)  # type: ignore[arg-type]

    def test_empty_series_raises(self) -> None:
        with pytest.raises(EmptySeriesError):
            moving_average([], 3)


class TestExponentialMovingAverage:
    def test_first_value_is_the_seed(self, simple_series: list[float]) -> None:
        assert exponential_moving_average(simple_series, 0.3)[0] == simple_series[0]

    def test_alpha_one_is_the_identity(self, simple_series: list[float]) -> None:
        assert exponential_moving_average(simple_series, 1.0) == simple_series

    def test_recurrence(self) -> None:
        assert exponential_moving_average([1.0, 2.0, 3.0], 0.5) == pytest.approx(
            [1.0, 1.5, 2.25]
        )

    def test_converges_to_a_step(self) -> None:
        step = [0.0] + [1.0] * 200
        assert exponential_moving_average(step, 0.1)[-1] == pytest.approx(1.0, abs=1e-6)

    def test_preserves_length(self, simple_series: list[float]) -> None:
        assert len(exponential_moving_average(simple_series, 0.2)) == len(simple_series)

    @pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5, math.nan])
    def test_invalid_alpha_raises(
        self, alpha: float, simple_series: list[float]
    ) -> None:
        with pytest.raises(ValidationError, match=r"\(0, 1\]"):
            exponential_moving_average(simple_series, alpha)


class TestMedianFilter:
    def test_removes_an_isolated_spike(self, spiky_series: list[float]) -> None:
        assert median_filter(spiky_series, 3) == pytest.approx([1.0] * 7)

    def test_mean_filter_smears_the_same_spike(self, spiky_series: list[float]) -> None:
        smeared = moving_average(spiky_series, 3)
        assert sum(value > 1.5 for value in smeared) == 3

    def test_window_of_one_is_the_identity(self, simple_series: list[float]) -> None:
        assert median_filter(simple_series, 1) == simple_series

    def test_preserves_a_step_edge(self) -> None:
        """A mean filter blurs this edge; the median keeps it sharp."""
        step = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        assert median_filter(step, 3) == pytest.approx(step)

    def test_no_padding_shortens_the_output(self, spiky_series: list[float]) -> None:
        filtered = median_filter(spiky_series, 3, padding=Padding.NONE)
        assert len(filtered) == len(spiky_series) - 2

    def test_even_window_is_supported(self, simple_series: list[float]) -> None:
        assert len(median_filter(simple_series, 2)) == len(simple_series)

    def test_empty_series_raises(self) -> None:
        with pytest.raises(EmptySeriesError):
            median_filter([], 3)


@pytest.mark.parametrize("smoother", SMOOTHERS)
def test_smoothers_do_not_mutate_input(smoother, simple_series: list[float]) -> None:
    original = list(simple_series)
    smoother(simple_series)
    assert simple_series == original


@pytest.mark.parametrize("smoother", SMOOTHERS)
def test_smoothers_return_a_new_list(smoother, simple_series: list[float]) -> None:
    result = smoother(simple_series)
    assert isinstance(result, list)
    assert result is not simple_series


@pytest.mark.parametrize("smoother", SMOOTHERS)
def test_smoothers_are_idempotent_on_constants(smoother) -> None:
    constant = [2.5] * 8
    assert smoother(constant) == pytest.approx(constant)
