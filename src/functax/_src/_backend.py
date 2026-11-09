"""Checks for the optional ``functax[jax]`` extra.

The base install depends on NumPy alone. A module that needs JAX calls
[`require_jax`][functax._src._backend.require_jax] before its first JAX import,
so a NumPy-only install fails with a message that names the extra rather than
with a bare ``ModuleNotFoundError: No module named 'jax'``.
"""

from __future__ import annotations

import importlib

from functax.exceptions import MissingExtraError


__all__ = [
    "JAX_EXTRA",
    "require_jax",
]


JAX_EXTRA: tuple[str, ...] = ("jax", "equinox", "lineax", "einx", "jaxtyping")
"""The import names of the distributions in the ``functax[jax]`` extra."""


def require_jax(module: str) -> None:
    """Raise unless every distribution of the ``functax[jax]`` extra imports.

    Args:
        module: The dotted name of the calling module, used in the message.

    Raises:
        MissingExtraError: If any name in `JAX_EXTRA` fails to import. The
            message names the module, the missing import and the extra.

    Examples:
        >>> from functax._src._backend import require_jax
        >>> require_jax("functax.interp")  # silent when the extra is installed
    """
    for name in JAX_EXTRA:
        try:
            importlib.import_module(name)
        except ImportError as err:
            msg = (
                f"{module} needs the functax[jax] extra, but {name!r} could not "
                f"be imported; install it with: pip install 'functax[jax]'"
            )
            raise MissingExtraError(msg) from err
