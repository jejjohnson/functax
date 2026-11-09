"""Tests for the exception hierarchy.

The hierarchy is part of the public contract: downstream code catches these,
so a change in the inheritance graph is a breaking change.
"""

from __future__ import annotations

import pytest

from functax import (
    FunctaxError,
    GeometryError,
    MissingExtraError,
    OutsideDomainError,
    PolicyError,
    ValidationError,
)


@pytest.mark.parametrize(
    "error",
    [
        ValidationError,
        GeometryError,
        PolicyError,
        OutsideDomainError,
        MissingExtraError,
    ],
)
def test_every_error_derives_from_the_base(error: type[Exception]) -> None:
    assert issubclass(error, FunctaxError)


@pytest.mark.parametrize(
    ("error", "stdlib_base"),
    [
        (ValidationError, ValueError),
        (GeometryError, ValueError),
        (PolicyError, ValueError),
        (OutsideDomainError, ValueError),
        (MissingExtraError, ImportError),
    ],
)
def test_errors_alias_the_closest_stdlib_exception(
    error: type[Exception], stdlib_base: type[Exception]
) -> None:
    assert issubclass(error, stdlib_base)


def test_geometry_error_specialises_validation_error() -> None:
    assert issubclass(GeometryError, ValidationError)
