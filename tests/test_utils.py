"""Tests for ``mypackage.utils``."""

from __future__ import annotations

import pytest

from mypackage import EmptySeriesError, ValidationError, Window, as_floats, timer
from mypackage.utils import Elapsed, require_non_empty, require_positive


class TestAsFloats:
    @pytest.mark.parametrize(
        ("series", "expected"),
        [
            ([1, 2, 3], [1.0, 2.0, 3.0]),
            ((1.5, 2.5), [1.5, 2.5]),
            (range(3), [0.0, 1.0, 2.0]),
            ([], []),
            (["1.5", "2"], [1.5, 2.0]),
        ],
    )
    def test_coerces_sequences(self, series, expected: list[float]) -> None:
        assert as_floats(series) == expected

    def test_returns_a_new_list(self) -> None:
        values = [1.0, 2.0]
        assert as_floats(values) is not values

    def test_reports_the_offending_index(self) -> None:
        with pytest.raises(ValidationError, match="at index 2"):
            as_floats([1.0, 2.0, "three"])

    def test_uses_the_supplied_name(self) -> None:
        with pytest.raises(ValidationError, match="'weights'"):
            as_floats([None], name="weights")

    def test_chains_the_original_error(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            as_floats(["nan-ish"])
        assert isinstance(excinfo.value.__cause__, ValueError)


class TestRequireNonEmpty:
    def test_passes_through_values(self) -> None:
        assert require_non_empty((1, 2)) == [1.0, 2.0]

    def test_rejects_empty(self) -> None:
        with pytest.raises(EmptySeriesError, match="at least one value"):
            require_non_empty([])

    def test_empty_series_error_is_a_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one value"):
            require_non_empty([])


class TestRequirePositive:
    @pytest.mark.parametrize("value", [1, 2, 1000])
    def test_accepts_positive_integers(self, value: int) -> None:
        assert require_positive(value) == value

    @pytest.mark.parametrize("value", [0, -1, 1.5, "3", None, True, False])
    def test_rejects_everything_else(self, value: object) -> None:
        with pytest.raises(ValidationError, match="positive integer"):
            require_positive(value)  # type: ignore[arg-type]


class TestWindow:
    def test_evicts_the_oldest_element(self) -> None:
        window: Window[int] = Window(capacity=3)
        for value in (1, 2, 3, 4):
            window.push(value)
        assert list(window) == [2, 3, 4]

    def test_push_is_chainable(self) -> None:
        assert list(Window[str](capacity=2).push("a").push("b").push("c")) == ["b", "c"]

    def test_is_full_tracks_capacity(self) -> None:
        window: Window[int] = Window(capacity=2)
        assert not window.is_full
        window.push(1)
        assert not window.is_full
        window.push(2)
        assert window.is_full

    def test_len(self) -> None:
        window: Window[int] = Window(capacity=5)
        window.push(1).push(2)
        assert len(window) == 2

    def test_clear(self) -> None:
        window: Window[int] = Window(capacity=2)
        window.push(1).push(2).clear()
        assert len(window) == 0
        assert not window.is_full

    def test_holds_arbitrary_element_types(self) -> None:
        window: Window[str] = Window(capacity=2)
        window.push("a").push("b")
        assert list(window) == ["a", "b"]

    def test_repr_is_informative(self) -> None:
        window: Window[int] = Window(capacity=2)
        window.push(7)
        assert repr(window) == "Window(capacity=2, items=[7])"

    @pytest.mark.parametrize("capacity", [0, -3, 2.0])
    def test_invalid_capacity_raises(self, capacity: object) -> None:
        with pytest.raises(ValidationError, match="'capacity'"):
            Window(capacity)  # type: ignore[arg-type]


class TestTimer:
    def test_records_a_non_negative_duration(self) -> None:
        with timer() as elapsed:
            sum(range(1000))
        assert elapsed.seconds >= 0.0

    def test_milliseconds_is_consistent(self) -> None:
        with timer() as elapsed:
            pass
        assert elapsed.milliseconds == pytest.approx(elapsed.seconds * 1e3)

    def test_is_zero_before_the_block_exits(self) -> None:
        with timer() as elapsed:
            assert elapsed.seconds == 0.0

    def test_records_even_when_the_block_raises(self) -> None:
        elapsed_ref: Elapsed | None = None
        with pytest.raises(RuntimeError), timer() as elapsed:
            elapsed_ref = elapsed
            raise RuntimeError("boom")
        assert elapsed_ref is not None
        assert elapsed_ref.seconds >= 0.0
