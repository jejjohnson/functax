"""Exception hierarchy for `functax`.

Every error raised by this package derives from
[`FunctaxError`][functax.FunctaxError], so a caller can trap the whole
package with a single ``except`` clause. Each concrete error also inherits from
the closest standard-library exception, so code written against the stdlib
keeps working without importing anything from this package.
"""

from __future__ import annotations


__all__ = [
    "FunctaxError",
    "GeometryError",
    "MissingExtraError",
    "ValidationError",
]


class FunctaxError(Exception):
    """Base class for every exception raised by ``functax``.

    Examples:
        >>> from functax import FunctaxError, ValidationError
        >>> issubclass(ValidationError, FunctaxError)
        True
    """


class ValidationError(FunctaxError, ValueError):
    """Raised when an argument fails validation.

    Also a `ValueError`, so ``except ValueError`` still catches it.

    Examples:
        >>> from functax import ValidationError
        >>> issubclass(ValidationError, ValueError)
        True
    """


class GeometryError(ValidationError):
    """Raised when an axis, grid or point set is malformed.

    A `ValidationError`, so also a `ValueError`. The message names the field
    and the value that failed.

    Examples:
        >>> from functax import GeometryError, ValidationError
        >>> issubclass(GeometryError, ValidationError)
        True
        >>> issubclass(GeometryError, ValueError)
        True
    """


class MissingExtraError(FunctaxError, ImportError):
    """Raised when a module needs the ``functax[jax]`` extra and it is missing.

    The base install depends on NumPy alone, so every module that needs JAX
    checks for it on import and raises this error, whose message names the
    extra to install. It is also an `ImportError`, so ``except ImportError``
    still catches it.

    Examples:
        >>> from functax import MissingExtraError
        >>> issubclass(MissingExtraError, ImportError)
        True
    """
