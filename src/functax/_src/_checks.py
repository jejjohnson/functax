"""Value checks that raise on concrete arrays and defer to run time under ``jit``."""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import numpy as np

from functax.exceptions import FunctaxError, ValidationError


__all__ = [
    "check_values",
    "require_floating",
]


def check_values(
    array: jax.Array,
    bad: jax.Array,
    message: str,
    *,
    error: type[FunctaxError] = ValidationError,
) -> jax.Array:
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
    """
    if isinstance(array, jax.core.Tracer) or isinstance(bad, jax.core.Tracer):
        return eqx.error_if(array, bad, message)
    if bool(bad):
        raise error(f"{message}, got {np.asarray(array)!r}")
    return array


def require_floating(
    array: jax.Array, name: str, *, error: type[FunctaxError] = ValidationError
) -> None:
    """Raise ``error`` unless ``array`` has a floating dtype; integers are not cast."""
    if not jnp.issubdtype(array.dtype, jnp.floating):
        msg = f"'{name}' must have a floating dtype, got {array.dtype}"
        raise error(msg)
