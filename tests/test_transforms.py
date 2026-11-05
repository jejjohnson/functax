"""Tests for ``mypackage.transforms``."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from mypackage import (
    Clip,
    EmptySeriesError,
    Fittable,
    MinMaxScale,
    MovingAverage,
    NotFittedError,
    Pipeline,
    Series,
    Standardize,
    Transform,
    ValidationError,
    chain,
    summarize,
)


@dataclass(frozen=True, slots=True)
class Square:
    """A user-defined transform that never inherits from anything."""

    def apply(self, series: Series) -> list[float]:
        return [value**2 for value in series]


class TestStandardize:
    def test_fit_apply_standardises(self, wikipedia_series: list[float]) -> None:
        scaled = Standardize().fit_apply(wikipedia_series)
        summary = summarize(scaled)
        assert summary.mean == pytest.approx(0.0, abs=1e-12)
        assert summary.std == pytest.approx(1.0)

    def test_learned_statistics_transfer_to_new_data(self) -> None:
        scaler = Standardize().fit([10.0, 12.0, 14.0])
        assert scaler.mean_ == pytest.approx(12.0)
        assert scaler.apply([16.0])[0] == pytest.approx(2.0)

    def test_inverse_round_trips(self, simple_series: list[float]) -> None:
        scaler = Standardize().fit(simple_series)
        assert scaler.inverse(scaler.apply(simple_series)) == pytest.approx(
            simple_series
        )

    def test_apply_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError, match=r"call \.fit\(series\) first"):
            Standardize().apply([1.0, 2.0])

    def test_inverse_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            Standardize().inverse([1.0, 2.0])

    def test_constant_series_raises(self) -> None:
        with pytest.raises(ValidationError, match="constant series"):
            Standardize().fit([3.0, 3.0, 3.0])

    def test_population_ddof(self, wikipedia_series: list[float]) -> None:
        scaler = Standardize(ddof=0).fit(wikipedia_series)
        assert scaler.std_ == pytest.approx(2.0)

    def test_empty_series_raises(self) -> None:
        with pytest.raises(EmptySeriesError):
            Standardize().fit([])


class TestMinMaxScale:
    def test_default_range_is_unit_interval(self) -> None:
        assert MinMaxScale().fit_apply([0.0, 5.0, 10.0]) == pytest.approx(
            [0.0, 0.5, 1.0]
        )

    def test_custom_range(self) -> None:
        scaler = MinMaxScale(feature_range=(-1.0, 1.0))
        assert scaler.fit_apply([0.0, 5.0, 10.0]) == pytest.approx([-1.0, 0.0, 1.0])

    def test_extrapolates_beyond_the_fitted_range(self) -> None:
        scaler = MinMaxScale().fit([0.0, 10.0])
        assert scaler.apply([20.0])[0] == pytest.approx(2.0)

    def test_apply_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError, match="MinMaxScale is not fitted"):
            MinMaxScale().apply([1.0])

    def test_constant_series_raises(self) -> None:
        with pytest.raises(ValidationError, match="constant series"):
            MinMaxScale().fit([4.0, 4.0])

    @pytest.mark.parametrize("bounds", [(1.0, 0.0), (0.0, 0.0)])
    def test_inverted_range_raises(self, bounds: tuple[float, float]) -> None:
        with pytest.raises(ValidationError, match="low < high"):
            MinMaxScale(feature_range=bounds)


class TestClip:
    def test_clamps_both_sides(self) -> None:
        assert Clip(lower=0.0, upper=1.0).apply([-2.0, 0.5, 3.0]) == [0.0, 0.5, 1.0]

    def test_one_sided_bounds(self) -> None:
        assert Clip(lower=0.0).apply([-2.0, 3.0]) == [0.0, 3.0]
        assert Clip(upper=1.0).apply([-2.0, 3.0]) == [-2.0, 1.0]

    def test_unbounded_is_the_identity(self, simple_series: list[float]) -> None:
        assert Clip().apply(simple_series) == simple_series

    def test_is_stateless(self, simple_series: list[float]) -> None:
        clip = Clip(lower=2.0)
        assert clip.fit(simple_series) is clip

    def test_fit_apply_is_a_plain_apply(self, simple_series: list[float]) -> None:
        clip = Clip(upper=3.0)
        assert clip.fit_apply(simple_series) == clip.apply(simple_series)

    def test_is_frozen(self) -> None:
        with pytest.raises(AttributeError):
            Clip().lower = 1.0  # type: ignore[misc]

    def test_inverted_bounds_raise(self) -> None:
        with pytest.raises(ValidationError, match="must not exceed"):
            Clip(lower=2.0, upper=1.0)


class TestMovingAverageTransform:
    def test_matches_the_function(self, simple_series: list[float]) -> None:
        from mypackage import moving_average

        assert MovingAverage(window=3).apply(simple_series) == moving_average(
            simple_series, 3
        )

    def test_is_stateless(self, simple_series: list[float]) -> None:
        smoother = MovingAverage(window=3)
        assert smoother.fit(simple_series) is smoother

    def test_fit_apply_is_a_plain_apply(self, simple_series: list[float]) -> None:
        smoother = MovingAverage(window=3)
        assert smoother.fit_apply(simple_series) == smoother.apply(simple_series)

    def test_invalid_window_raises_at_construction(self) -> None:
        with pytest.raises(ValidationError, match="positive integer"):
            MovingAverage(window=0)


class TestProtocols:
    @pytest.mark.parametrize(
        "transform",
        [Standardize(), MinMaxScale(), Clip(), MovingAverage(window=2), Square()],
    )
    def test_transform_protocol(self, transform: Transform) -> None:
        assert isinstance(transform, Transform)

    @pytest.mark.parametrize("transform", [Standardize(), MinMaxScale()])
    def test_fittable_protocol(self, transform: Fittable) -> None:
        assert isinstance(transform, Fittable)

    def test_non_transform_is_rejected(self) -> None:
        assert not isinstance("definitely not a transform", Transform)


class TestPipeline:
    def test_applies_steps_in_order(self) -> None:
        pipeline = Pipeline([Clip(lower=0.0), Square()])
        assert pipeline.fit_apply([-3.0, 2.0]) == [0.0, 4.0]

    def test_order_matters(self) -> None:
        reversed_pipeline = Pipeline([Square(), Clip(lower=0.0)])
        assert reversed_pipeline.fit_apply([-3.0, 2.0]) == [9.0, 4.0]

    def test_empty_pipeline_is_the_identity(self, simple_series: list[float]) -> None:
        assert Pipeline().fit_apply(simple_series) == simple_series

    def test_composes_with_or(self) -> None:
        combined = Pipeline([Clip(lower=0.0)]) | Square()
        assert len(combined) == 2
        assert combined.fit_apply([-3.0, 2.0]) == [0.0, 4.0]

    def test_composes_pipelines(self) -> None:
        combined = Pipeline([Clip(lower=0.0)]) | Pipeline([Square(), Clip(upper=5.0)])
        assert len(combined) == 3

    def test_or_does_not_mutate_the_operands(self) -> None:
        left = Pipeline([Clip(lower=0.0)])
        _ = left | Square()
        assert len(left) == 1

    def test_is_iterable_and_indexable(self) -> None:
        pipeline = Pipeline([Clip(), Square()])
        assert len(list(pipeline)) == 2
        assert isinstance(pipeline[1], Square)

    def test_steps_are_fitted_on_upstream_output(self) -> None:
        """The scaler must see the *clipped* data, not the raw data."""
        pipeline = Pipeline([Clip(upper=10.0), Standardize()])
        pipeline.fit([0.0, 10.0, 1000.0])
        scaler = pipeline[1]
        assert isinstance(scaler, Standardize)
        assert scaler.mean_ == pytest.approx(summarize([0.0, 10.0, 10.0]).mean)

    def test_apply_before_fit_propagates_not_fitted(self) -> None:
        with pytest.raises(NotFittedError):
            Pipeline([Standardize()]).apply([1.0, 2.0])

    def test_rejects_a_non_callable_apply(self) -> None:
        """`runtime_checkable` alone would accept this and fail at apply time."""

        class NotCallable:
            apply = 7

        assert isinstance(NotCallable(), Transform)  # the protocol's blind spot
        with pytest.raises(ValidationError, match="step 0 does not implement"):
            Pipeline([NotCallable()])  # type: ignore[list-item]

    def test_composition_revalidates_steps(self) -> None:
        class NotCallable:
            apply = 7

        with pytest.raises(ValidationError, match="step 1 does not implement"):
            _ = Pipeline([Clip()]) | NotCallable()  # type: ignore[operator]

    def test_rejects_non_transform_steps(self) -> None:
        with pytest.raises(ValidationError, match="does not implement"):
            Pipeline([Clip(), "not a transform"])  # type: ignore[list-item]

    def test_does_not_mutate_input(self, simple_series: list[float]) -> None:
        original = list(simple_series)
        Pipeline([Standardize()]).fit_apply(simple_series)
        assert simple_series == original


class TestChain:
    def test_builds_a_pipeline(self) -> None:
        pipeline = chain(MinMaxScale(), MovingAverage(window=3))
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline) == 2

    def test_matches_manual_construction(self, simple_series: list[float]) -> None:
        steps = [Standardize(), Clip(lower=-1.0)]
        assert chain(*steps).fit_apply(simple_series) == Pipeline(
            list(steps)
        ).fit_apply(simple_series)

    def test_empty_chain(self, simple_series: list[float]) -> None:
        assert chain().fit_apply(simple_series) == simple_series
