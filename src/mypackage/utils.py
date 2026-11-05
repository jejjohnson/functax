"""Small, dependency-free helpers shared across the package.

Three things live here:

- **Validators** ([`as_floats`][mypackage.as_floats],
  [`require_non_empty`][mypackage.require_non_empty],
  [`require_positive`][mypackage.require_positive]) that coerce and check
  inputs once, at the public API boundary, and raise
  [`ValidationError`][mypackage.ValidationError] with a uniform message.
- [`Window`][mypackage.Window], a generic fixed-capacity sliding window used
  by the streaming filters in [`mypackage.smoothing`][].
- [`timer`][mypackage.timer], a context manager for wall-clock measurements
  in examples and benchmarks.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from mypackage._typing import Series
from mypackage.exceptions import EmptySeriesError, ValidationError


__all__ = [
    "Elapsed",
    "Window",
    "as_floats",
    "require_non_empty",
    "require_positive",
    "timer",
]


def as_floats(series: Series | Sequence[str], *, name: str = "series") -> list[float]:
    """Coerce a sequence of numbers to a fresh `list[float]`.

    Strings that parse as numbers are accepted too, which is what lets
    [`mypackage.cli`][] hand raw tokens straight to this function instead of
    duplicating the validation logic.

    Args:
        series: The sequence to coerce. Never mutated.
        name: Argument name used in the error message.

    Returns:
        A new list containing every element of ``series`` as a `float`.

    Raises:
        ValidationError: If any element cannot be converted to `float`, or if
            ``series`` is not a sequence.

    Examples:
        >>> as_floats([1, 2, 3])
        [1.0, 2.0, 3.0]
        >>> as_floats(range(3))
        [0.0, 1.0, 2.0]
        >>> as_floats(["1.5", "2"])
        [1.5, 2.0]
        >>> try:
        ...     as_floats([1.0, "oops"])
        ... except ValidationError as err:
        ...     print(err)
        'series' must contain only numbers, got 'oops' at index 1
    """
    try:
        values = list(series)
    except TypeError as err:  # pragma: no cover - defensive
        msg = f"{name!r} must be a sequence of numbers, got {type(series).__name__}"
        raise ValidationError(msg) from err

    out: list[float] = []
    for index, value in enumerate(values):
        try:
            out.append(float(value))
        except (TypeError, ValueError) as err:
            msg = f"{name!r} must contain only numbers, got {value!r} at index {index}"
            raise ValidationError(msg) from err
    return out


def require_non_empty(series: Series, *, name: str = "series") -> list[float]:
    """Coerce ``series`` to floats and require at least one element.

    Args:
        series: The sequence to validate.
        name: Argument name used in the error message.

    Returns:
        A new `list[float]` with the validated contents.

    Raises:
        EmptySeriesError: If ``series`` has no elements.
        ValidationError: If any element is not a number.

    Examples:
        >>> require_non_empty((1, 2))
        [1.0, 2.0]
        >>> try:
        ...     require_non_empty([])
        ... except EmptySeriesError as err:
        ...     print(err)
        'series' must contain at least one value, got 0
    """
    values = as_floats(series, name=name)
    if not values:
        msg = f"{name!r} must contain at least one value, got 0"
        raise EmptySeriesError(msg)
    return values


def require_positive(value: int, *, name: str = "window") -> int:
    """Require ``value`` to be a strictly positive integer.

    Args:
        value: The integer to validate. `bool` is rejected explicitly, since
            ``True`` would otherwise silently pass as ``1``.
        name: Argument name used in the error message.

    Returns:
        The validated integer.

    Raises:
        ValidationError: If ``value`` is not a positive `int`.

    Examples:
        >>> require_positive(3, name="window")
        3
        >>> try:
        ...     require_positive(0, name="window")
        ... except ValidationError as err:
        ...     print(err)
        'window' must be a positive integer, got 0
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{name!r} must be a positive integer, got {value!r}"
        raise ValidationError(msg)
    return value


class Window[T]:
    """A fixed-capacity sliding window over a stream of values.

    Pushing past ``capacity`` evicts the oldest element, so the window always
    holds the most recent ``capacity`` items. The element type is generic: the
    window itself never inspects the values it stores.

    Args:
        capacity: Maximum number of retained elements. Must be positive.

    Raises:
        ValidationError: If ``capacity`` is not a positive integer.

    Examples:
        >>> window: Window[int] = Window(capacity=3)
        >>> for value in (1, 2, 3, 4):
        ...     _ = window.push(value)
        >>> list(window)
        [2, 3, 4]
        >>> len(window), window.is_full
        (3, True)

        Pushing returns ``self``, so calls chain:

        >>> list(Window[str](capacity=2).push("a").push("b").push("c"))
        ['b', 'c']
    """

    __slots__ = ("_items", "capacity")

    def __init__(self, capacity: int) -> None:
        self.capacity = require_positive(capacity, name="capacity")
        self._items: deque[T] = deque(maxlen=self.capacity)

    def push(self, value: T) -> Window[T]:
        """Append ``value``, evicting the oldest element if the window is full.

        Args:
            value: The item to append.

        Returns:
            ``self``, so that pushes can be chained.
        """
        self._items.append(value)
        return self

    def clear(self) -> None:
        """Drop every retained element."""
        self._items.clear()

    @property
    def is_full(self) -> bool:
        """Whether the window holds exactly ``capacity`` elements."""
        return len(self._items) == self.capacity

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Window(capacity={self.capacity}, items={list(self._items)!r})"


@dataclass(slots=True)
class Elapsed:
    """Wall-clock duration recorded by [`timer`][mypackage.timer].

    Attributes:
        seconds: Elapsed wall-clock time. Zero until the ``with`` block exits.
    """

    seconds: float = 0.0

    @property
    def milliseconds(self) -> float:
        """Elapsed time in milliseconds."""
        return self.seconds * 1e3


@contextmanager
def timer() -> Iterator[Elapsed]:
    """Measure the wall-clock time spent inside a ``with`` block.

    The yielded [`Elapsed`][mypackage.Elapsed] is populated when the block
    exits — including when it exits via an exception.

    Yields:
        An `Elapsed` record that is filled in on exit.

    Examples:
        >>> with timer() as elapsed:
        ...     total = sum(range(1_000))
        >>> elapsed.seconds >= 0.0
        True
        >>> elapsed.milliseconds == elapsed.seconds * 1e3
        True
    """
    elapsed = Elapsed()
    start = time.perf_counter()
    try:
        yield elapsed
    finally:
        elapsed.seconds = time.perf_counter() - start
