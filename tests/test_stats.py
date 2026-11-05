"""Tests for ``mypackage.stats``."""

from __future__ import annotations

import math

import pytest

from mypackage import (
    EmptySeriesError,
    RunningStats,
    Summary,
    ValidationError,
    quantile,
    summarize,
    zscores,
)


class TestSummarize:
    def test_known_values(self, wikipedia_series: list[float]) -> None:
        summary = summarize(wikipedia_series, ddof=0)
        assert summary.count == 8
        assert summary.mean == pytest.approx(5.0)
        assert summary.variance == pytest.approx(4.0)
        assert summary.std == pytest.approx(2.0)
        assert summary.minimum == 2.0
        assert summary.maximum == 9.0
        assert summary.median == pytest.approx(4.5)
        assert summary.spread == 7.0

    def test_sample_variance_is_the_default(self, simple_series: list[float]) -> None:
        assert summarize(simple_series).ddof == 1
        assert summarize(simple_series).variance == pytest.approx(2.5)
        assert summarize(simple_series, ddof=0).variance == pytest.approx(2.0)

    def test_summary_is_immutable(self, simple_series: list[float]) -> None:
        summary = summarize(simple_series)
        with pytest.raises(AttributeError):
            summary.mean = 0.0  # type: ignore[misc]

    def test_summary_is_hashable_and_comparable(
        self, simple_series: list[float]
    ) -> None:
        assert summarize(simple_series) == summarize(list(simple_series))
        assert len({summarize(simple_series), summarize(simple_series)}) == 1

    def test_to_dict_includes_derived_fields(self, simple_series: list[float]) -> None:
        data = summarize(simple_series).to_dict()
        assert set(data) == {
            "count",
            "ddof",
            "maximum",
            "mean",
            "median",
            "minimum",
            "spread",
            "std",
            "variance",
        }
        assert data["std"] == pytest.approx(math.sqrt(data["variance"]))

    def test_does_not_mutate_input(self, simple_series: list[float]) -> None:
        original = list(simple_series)
        summarize(simple_series)
        assert simple_series == original

    def test_accepts_any_sequence(self) -> None:
        assert summarize(range(5), ddof=0).mean == pytest.approx(2.0)
        assert summarize((1, 2, 3)).count == 3

    def test_is_numerically_stable_for_large_offsets(self) -> None:
        """The naive sum-of-squares formula loses all precision here."""
        offset = 1e9
        values = [offset + delta for delta in (1.0, 2.0, 3.0, 4.0)]
        assert summarize(values).variance == pytest.approx(5.0 / 3.0)

    def test_mean_survives_an_overflowing_sum(self) -> None:
        summary = summarize([1e308, 1e308], ddof=0)
        assert summary.mean == 1e308
        assert summary.variance == 0.0

    def test_overflow_path_matches_mixed_magnitudes(self) -> None:
        values = [1.7e308, 1.7e308, 1.0]
        # Divide before multiplying: the literal 3.4e308 is already inf.
        assert summarize(values).mean == pytest.approx(1.7e308 / 3 * 2)

    def test_unrepresentable_variance_is_inf_not_an_error(self) -> None:
        summary = summarize([1e308, -1e308], ddof=0)
        assert summary.mean == 0.0
        assert math.isinf(summary.variance)

    def test_empty_series_raises(self) -> None:
        with pytest.raises(EmptySeriesError, match="at least one value"):
            summarize([])

    @pytest.mark.parametrize("ddof", [-1, 5, 6])
    def test_invalid_ddof_raises(self, ddof: int) -> None:
        with pytest.raises(ValidationError):
            summarize([1.0, 2.0, 3.0], ddof=ddof)

    def test_non_numeric_raises(self) -> None:
        with pytest.raises(ValidationError, match="only numbers"):
            summarize([1.0, "two"])


class TestQuantile:
    @pytest.mark.parametrize(
        ("q", "expected"),
        [(0.0, 1.0), (0.25, 1.75), (0.5, 2.5), (0.75, 3.25), (1.0, 4.0)],
    )
    def test_linear_interpolation(self, q: float, expected: float) -> None:
        assert quantile([1.0, 2.0, 3.0, 4.0], q) == pytest.approx(expected)

    def test_is_order_invariant(self) -> None:
        assert quantile([4.0, 1.0, 3.0, 2.0], 0.5) == pytest.approx(2.5)

    def test_single_observation(self) -> None:
        assert quantile([7.5], 0.0) == 7.5
        assert quantile([7.5], 1.0) == 7.5

    def test_median_matches_summarize(self, wikipedia_series: list[float]) -> None:
        assert quantile(wikipedia_series, 0.5) == summarize(wikipedia_series).median

    @pytest.mark.parametrize("q", [-0.01, 1.01, 2.0])
    def test_out_of_range_raises(self, q: float) -> None:
        with pytest.raises(ValidationError, match=r"\[0, 1\]"):
            quantile([1.0, 2.0], q)

    def test_empty_series_raises(self) -> None:
        with pytest.raises(EmptySeriesError):
            quantile([], 0.5)


class TestZscores:
    def test_zero_mean_unit_variance(self, wikipedia_series: list[float]) -> None:
        scores = zscores(wikipedia_series)
        assert math.fsum(scores) == pytest.approx(0.0, abs=1e-12)
        assert summarize(scores).std == pytest.approx(1.0)

    def test_symmetric_series(self) -> None:
        assert zscores([1.0, 2.0, 3.0]) == pytest.approx([-1.0, 0.0, 1.0])

    def test_constant_series_raises(self) -> None:
        with pytest.raises(ValidationError, match="constant series"):
            zscores([2.0, 2.0, 2.0])

    def test_empty_series_raises(self) -> None:
        with pytest.raises(EmptySeriesError):
            zscores([])


class TestRunningStats:
    def test_matches_batch_summary(self, wikipedia_series: list[float]) -> None:
        stats = RunningStats().update_many(wikipedia_series)
        batch = summarize(wikipedia_series)
        assert stats.count == batch.count
        assert stats.mean == pytest.approx(batch.mean)
        assert stats.variance == pytest.approx(batch.variance)
        assert stats.std == pytest.approx(batch.std)
        assert stats.minimum == batch.minimum
        assert stats.maximum == batch.maximum

    def test_updates_are_chainable(self) -> None:
        stats = RunningStats().update(1.0).update(2.0).update(3.0)
        assert stats.count == 3
        assert stats.mean == pytest.approx(2.0)

    def test_incremental_equals_batch_at_every_step(
        self, simple_series: list[float]
    ) -> None:
        stats = RunningStats()
        for index, value in enumerate(simple_series, start=1):
            stats.update(value)
            if index >= 2:
                expected = summarize(simple_series[:index])
                assert stats.mean == pytest.approx(expected.mean)
                assert stats.variance == pytest.approx(expected.variance)

    def test_fresh_instance_has_zero_statistics(self) -> None:
        stats = RunningStats()
        assert (stats.count, stats.mean, stats.variance, stats.std) == (
            0,
            0.0,
            0.0,
            0.0,
        )

    def test_variance_needs_more_than_ddof_observations(self) -> None:
        assert RunningStats(ddof=1).update(5.0).variance == 0.0
        assert RunningStats(ddof=0).update(5.0).variance == 0.0

    def test_population_ddof(self, wikipedia_series: list[float]) -> None:
        stats = RunningStats(ddof=0).update_many(wikipedia_series)
        assert stats.variance == pytest.approx(4.0)

    def test_extrema_before_any_update_raise(self) -> None:
        stats = RunningStats()
        with pytest.raises(EmptySeriesError):
            _ = stats.minimum
        with pytest.raises(EmptySeriesError):
            _ = stats.maximum

    @pytest.mark.parametrize("ddof", [-1, -5])
    def test_negative_ddof_is_rejected(self, ddof: int) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            RunningStats(ddof=ddof)

    @pytest.mark.parametrize("ddof", [1.5, "1", True])
    def test_non_integer_ddof_is_rejected(self, ddof: object) -> None:
        with pytest.raises(ValidationError, match="must be an integer"):
            RunningStats(ddof=ddof)  # type: ignore[arg-type]

    def test_non_numeric_update_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be a number"):
            RunningStats().update("nope")  # type: ignore[arg-type]

    def test_is_stable_for_large_offsets(self) -> None:
        offset = 1e9
        values = [offset + delta for delta in (1.0, 2.0, 3.0, 4.0)]
        assert RunningStats().update_many(values).variance == pytest.approx(5.0 / 3.0)

    def test_repr_hides_private_state(self) -> None:
        assert repr(RunningStats()) == "RunningStats(ddof=1)"


def test_summary_can_be_constructed_directly() -> None:
    summary = Summary(
        count=2, mean=1.5, variance=0.5, minimum=1.0, maximum=2.0, median=1.5
    )
    assert summary.ddof == 1
    assert summary.std == pytest.approx(math.sqrt(0.5))
