"""Smoothing filters for one-dimensional series.

Three filters, all pure functions that return a new list and never touch their
input:

| Filter | Kind | Good at | Bad at |
|---|---|---|---|
| `moving_average` | linear, finite | white noise | preserving edges |
| `exponential_moving_average` | linear, infinite | streaming | symmetric response |
| `median_filter` | non-linear, finite | isolated spikes | smooth gradients |

Finite-support filters need values beyond the ends of the series. The
[`Padding`][mypackage.Padding] strategy decides what those values are.
"""

from __future__ import annotations

import math
import statistics
from enum import StrEnum

from mypackage._typing import Series
from mypackage.exceptions import ValidationError
from mypackage.utils import Window, require_non_empty, require_positive


__all__ = [
    "Padding",
    "exponential_moving_average",
    "median_filter",
    "moving_average",
]


class Padding(StrEnum):
    """How a finite-support filter extends a series past its boundaries.

    Being a `StrEnum`, every member compares equal to its own name, so callers
    can pass the plain string ``"edge"`` instead of importing the enum.

    Attributes:
        EDGE: Repeat the first and last value. Keeps the output length equal to
            the input length and does not pull the ends toward zero.
        ZERO: Pad with ``0.0``. Biases the ends toward zero, which is only
            sensible for mean-zero signals.
        REFLECT: Mirror the series about its endpoints without repeating them.
            Preserves local slope better than ``EDGE``.
        NONE: Do not pad. The output is shorter than the input by
            ``window - 1`` samples.

    Examples:
        >>> Padding("edge") is Padding.EDGE
        True
        >>> moving_average([0.0, 0.0, 9.0], window=3, padding="zero")
        [0.0, 3.0, 3.0]
    """

    EDGE = "edge"
    ZERO = "zero"
    REFLECT = "reflect"
    NONE = "none"


def _reflect_index(index: int, length: int) -> int:
    """Map any integer index into ``[0, length)`` by mirroring at the ends."""
    if length == 1:
        return 0
    period = 2 * (length - 1)
    index = abs(index) % period
    return index if index < length else period - index


def _pad(values: list[float], left: int, right: int, padding: Padding) -> list[float]:
    """Extend ``values`` by ``left``/``right`` samples under ``padding``."""
    if padding is Padding.NONE:
        return list(values)
    if padding is Padding.ZERO:
        return [0.0] * left + values + [0.0] * right
    if padding is Padding.EDGE:
        return [values[0]] * left + values + [values[-1]] * right

    length = len(values)
    head = [values[_reflect_index(-offset, length)] for offset in range(left, 0, -1)]
    tail = [
        values[_reflect_index(length - 1 + offset, length)]
        for offset in range(1, right + 1)
    ]
    return head + values + tail


def _validate_window(values: list[float], window: int) -> int:
    """Validate ``window`` against the length of ``values``."""
    window = require_positive(window, name="window")
    if window > len(values):
        msg = (
            f"'window' must not exceed the series length, "
            f"got window={window} for {len(values)} value(s)"
        )
        raise ValidationError(msg)
    return window


def moving_average(
    series: Series,
    window: int,
    *,
    padding: Padding | str = Padding.EDGE,
) -> list[float]:
    """Smooth ``series`` with a centred, equally weighted moving average.

    With w the window length and l = ⌊w/2⌋,

    yᵢ = (1/w) ∑ₖ xᵢ₊ₖ,  for k = −l, …, w − 1 − l.

    Args:
        series: The observations to smooth.
        window: Number of samples averaged per output point. Must be positive
            and no larger than the series.
        padding: Boundary strategy, see [`Padding`][mypackage.Padding]. With
            `Padding.NONE` the output is ``window - 1`` samples shorter.

    Returns:
        A new list of smoothed values.

    Raises:
        EmptySeriesError: If ``series`` is empty.
        ValidationError: If ``window`` is not positive or exceeds the series
            length, or if ``padding`` is not a valid strategy.

    Examples:
        >>> moving_average([1.0, 2.0, 3.0, 4.0, 5.0], window=3)
        [1.3333333333333333, 2.0, 3.0, 4.0, 4.666666666666667]

        A window of 1 is the identity:

        >>> moving_average([1.0, 5.0, 2.0], window=1)
        [1.0, 5.0, 2.0]

        Dropping the padded ends shortens the output instead:

        >>> moving_average([1.0, 2.0, 3.0, 4.0, 5.0], window=3, padding="none")
        [2.0, 3.0, 4.0]
    """
    values = require_non_empty(series)
    window = _validate_window(values, window)
    strategy = _as_padding(padding)

    left = window // 2
    right = window - 1 - left
    padded = _pad(values, left, right, strategy)
    return [
        math.fsum(padded[start : start + window]) / window
        for start in range(len(padded) - window + 1)
    ]


def exponential_moving_average(series: Series, alpha: float) -> list[float]:
    """Smooth ``series`` with a causal exponentially weighted moving average.

    y₀ = x₀,  yᵢ = α xᵢ + (1 − α) yᵢ₋₁,

    with α the smoothing factor ``alpha``.

    The filter is *causal*: ``y[i]`` depends only on samples up to ``i``, so it
    can run on a live stream. The cost is a phase lag of roughly
    (1 − α)/α samples.

    Args:
        series: The observations to smooth.
        alpha: Smoothing factor in ``(0, 1]``. ``1.0`` returns the input
            unchanged; values near zero smooth aggressively.

    Returns:
        A new list of smoothed values, the same length as ``series``.

    Raises:
        EmptySeriesError: If ``series`` is empty.
        ValidationError: If ``alpha`` is outside ``(0, 1]``.

    Examples:
        >>> exponential_moving_average([1.0, 2.0, 3.0], alpha=0.5)
        [1.0, 1.5, 2.25]
        >>> exponential_moving_average([1.0, 2.0, 3.0], alpha=1.0)
        [1.0, 2.0, 3.0]
    """
    values = require_non_empty(series)
    if not 0.0 < alpha <= 1.0:
        msg = f"'alpha' must lie in (0, 1], got {alpha!r}"
        raise ValidationError(msg)

    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(alpha * value + (1.0 - alpha) * smoothed[-1])
    return smoothed


def median_filter(
    series: Series,
    window: int,
    *,
    padding: Padding | str = Padding.EDGE,
) -> list[float]:
    """Smooth ``series`` with a centred running median.

    Unlike the moving average, the median is a *non-linear* statistic with a
    breakdown point of 50%: up to half the samples in a window can be
    arbitrarily corrupted without moving the output. That is what makes it the
    right tool for isolated spikes, which a mean filter merely smears.

    Uses a [`Window`][mypackage.Window] internally so the streaming and batch
    code paths share one buffer implementation.

    Args:
        series: The observations to smooth.
        window: Number of samples per running median.
        padding: Boundary strategy, see [`Padding`][mypackage.Padding].

    Returns:
        A new list of filtered values.

    Raises:
        EmptySeriesError: If ``series`` is empty.
        ValidationError: If ``window`` is invalid or ``padding`` is unknown.

    Examples:
        A single spike is removed outright, not smeared across neighbours:

        >>> median_filter([1.0, 1.0, 99.0, 1.0, 1.0], window=3)
        [1.0, 1.0, 1.0, 1.0, 1.0]

        The moving average, by contrast, spreads the spike over three samples:

        >>> [round(v, 2) for v in moving_average([1.0, 1.0, 99.0, 1.0, 1.0], 3)]
        [1.0, 33.67, 33.67, 33.67, 1.0]
    """
    values = require_non_empty(series)
    window = _validate_window(values, window)
    strategy = _as_padding(padding)

    left = window // 2
    right = window - 1 - left
    padded = _pad(values, left, right, strategy)

    buffer: Window[float] = Window(capacity=window)
    filtered: list[float] = []
    for value in padded:
        buffer.push(value)
        if buffer.is_full:
            filtered.append(statistics.median(buffer))
    return filtered


def _as_padding(padding: Padding | str) -> Padding:
    """Coerce ``padding`` to a [`Padding`][mypackage.Padding] member."""
    if isinstance(padding, Padding):
        return padding
    try:
        return Padding(padding)
    except ValueError as err:
        valid = ", ".join(repr(member.value) for member in Padding)
        msg = f"'padding' must be one of {valid}, got {padding!r}"
        raise ValidationError(msg) from err
