"""Exception hierarchy for `mypackage`.

Every error raised by this package derives from
[`MypackageError`][mypackage.MypackageError], so a caller can trap the whole
package with a single ``except`` clause while still being able to discriminate
between failure modes when it matters.

The concrete errors also inherit from the closest standard-library exception
(`ValueError`, `RuntimeError`), so code written against the stdlib keeps
working without importing anything from this package.
"""

from __future__ import annotations


__all__ = [
    "EmptySeriesError",
    "MypackageError",
    "NotFittedError",
    "ValidationError",
]


class MypackageError(Exception):
    """Base class for every exception raised by ``mypackage``.

    Examples:
        >>> from mypackage import EmptySeriesError, MypackageError
        >>> issubclass(EmptySeriesError, MypackageError)
        True
    """


class ValidationError(MypackageError, ValueError):
    """Raised when an argument fails validation.

    Also a `ValueError`, so ``except ValueError`` still catches it.

    Examples:
        >>> from mypackage import moving_average
        >>> try:
        ...     moving_average([1.0, 2.0, 3.0], window=0)
        ... except ValueError as err:
        ...     print(err)
        'window' must be a positive integer, got 0
    """


class EmptySeriesError(ValidationError):
    """Raised when an operation requires at least one observation.

    Examples:
        >>> from mypackage import EmptySeriesError, summarize
        >>> try:
        ...     summarize([])
        ... except EmptySeriesError as err:
        ...     print(err)
        'series' must contain at least one value, got 0
    """


class NotFittedError(MypackageError, RuntimeError):
    """Raised when a stateful transform is used before it has been fitted.

    Examples:
        >>> from mypackage import NotFittedError, Standardize
        >>> try:
        ...     Standardize().apply([1.0, 2.0])
        ... except NotFittedError as err:
        ...     print(err)
        Standardize is not fitted; call .fit(series) first
    """
