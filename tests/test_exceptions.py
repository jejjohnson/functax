"""Tests for the exception hierarchy.

The hierarchy is part of the public contract: downstream code catches these,
so a change in the inheritance graph is a breaking change.
"""

from __future__ import annotations

import pytest

from mypackage import (
    EmptySeriesError,
    MypackageError,
    NotFittedError,
    Standardize,
    ValidationError,
    summarize,
)


@pytest.mark.parametrize(
    "error",
    [ValidationError, EmptySeriesError, NotFittedError],
)
def test_every_error_derives_from_the_base(error: type[Exception]) -> None:
    assert issubclass(error, MypackageError)


@pytest.mark.parametrize(
    ("error", "stdlib_base"),
    [
        (ValidationError, ValueError),
        (EmptySeriesError, ValueError),
        (NotFittedError, RuntimeError),
    ],
)
def test_errors_alias_the_closest_stdlib_exception(
    error: type[Exception], stdlib_base: type[Exception]
) -> None:
    assert issubclass(error, stdlib_base)


def test_empty_series_error_specialises_validation_error() -> None:
    assert issubclass(EmptySeriesError, ValidationError)


def test_a_single_except_clause_catches_the_package() -> None:
    caught: list[MypackageError] = []
    for call in (lambda: summarize([]), lambda: Standardize().apply([1.0])):
        try:
            call()
        except MypackageError as err:
            caught.append(err)
    assert len(caught) == 2
    assert isinstance(caught[0], EmptySeriesError)
    assert isinstance(caught[1], NotFittedError)


def test_messages_name_the_offending_argument() -> None:
    with pytest.raises(EmptySeriesError) as excinfo:
        summarize([])
    assert "'series'" in str(excinfo.value)
