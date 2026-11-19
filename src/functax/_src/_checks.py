"""Value checks that raise on concrete arrays and defer to run time under ``jit``."""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Shaped

from functax.exceptions import FunctaxError, ValidationError


__all__ = [
    "check_values",
    "require_floating",
]


def check_values(
    array: Shaped[Array, "*shape"],
    bad: Bool[Array, ""],
    message: str,
    *,
    error: type[FunctaxError] = ValidationError,
) -> Shaped[Array, "*shape"]:
    """Raise if ``bad`` holds; under tracing, defer the check to run time.

    On concrete arrays the error is ``error`` and its message ends with the
    offending array. Inside ``jit`` the values are unknown, so the same message
    goes to ``eqx.error_if``, which raises when the compiled code runs.

    Args:
        array: The array the check is about, returned (wrapped under a trace).
        bad: A boolean scalar, true when the check fails.
        message: What must hold, naming the field (``"'values' must be …"``).
        error: The `FunctaxError` subclass to raise on concrete arrays.

    Returns:
        ``array``, threaded through ``eqx.error_if`` when traced.

    Raises:
        FunctaxError: ``error``, if ``bad`` holds on concrete arrays.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax._src._checks import check_values
        >>> values = jnp.array([1.0, 2.0])
        >>> checked = check_values(values, jnp.any(values <= 0), "must be positive")
        >>> checked.shape
        (2,)

        A failing check raises on concrete arrays:

        >>> bad = jnp.any(values > 1)
        >>> check_values(values, bad, "'values' must be ≤ 1")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValidationError: 'values' must be ≤ 1, got array([1., 2.])
    """
    if isinstance(array, jax.core.Tracer) or isinstance(bad, jax.core.Tracer):
        return eqx.error_if(array, bad, message)
    if bool(bad):
        raise error(f"{message}, got {np.asarray(array)!r}")
    return array


def require_floating(
    array: Shaped[Array, "*shape"],
    name: str,
    *,
    error: type[FunctaxError] = ValidationError,
) -> None:
    """Raise ``error`` unless ``array`` has a floating dtype; integers are not cast.

    Args:
        array: The array to check.
        name: The field name used in the message.
        error: The `FunctaxError` subclass to raise.

    Raises:
        FunctaxError: ``error``, if ``array`` has an integer or boolean dtype.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax._src._checks import require_floating
        >>> require_floating(jnp.zeros(3), "values")
        >>> integers = jnp.arange(3)
        >>> require_floating(
        ...     integers, "values"
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValidationError: 'values' must have a floating dtype, got int64
    """
    if not jnp.issubdtype(array.dtype, jnp.floating):
        msg = f"'{name}' must have a floating dtype, got {array.dtype}"
        raise error(msg)
