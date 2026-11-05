"""Shared pytest fixtures.

Fixtures live here rather than in individual test modules so that every test
file sees the same reference data — when an expected value changes, it changes
in exactly one place.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def simple_series() -> list[float]:
    """A short ascending series with easy closed-form statistics."""
    return [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.fixture
def wikipedia_series() -> list[float]:
    """The canonical variance example: mean 5, population variance 4."""
    return [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]


@pytest.fixture
def spiky_series() -> list[float]:
    """A flat series with a single large outlier in the middle."""
    return [1.0, 1.0, 1.0, 99.0, 1.0, 1.0, 1.0]


@pytest.fixture
def sine_series() -> list[float]:
    """A noiseless sine wave, sampled over one period."""
    return [math.sin(2.0 * math.pi * index / 32.0) for index in range(32)]


@pytest.fixture
def series_file(tmp_path: Path, simple_series: list[float]) -> Iterator[Path]:
    """A temporary file holding ``simple_series`` as whitespace-separated text."""
    path = tmp_path / "series.txt"
    path.write_text(" ".join(str(value) for value in simple_series), encoding="utf-8")
    yield path
