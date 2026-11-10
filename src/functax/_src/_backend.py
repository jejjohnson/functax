"""Checks for the optional ``functax[jax]`` and ``functax[scipy]`` extras.

The base install depends on NumPy alone. A module that needs JAX calls
[`require_jax`][functax._src._backend.require_jax] before its first JAX import,
so a NumPy-only install fails with a message that names the extra rather than
with a bare ``ModuleNotFoundError: No module named 'jax'``. A scheme whose fit
needs SciPy calls [`import_scipy`][functax._src._backend.import_scipy] when it
builds a plan, so functax itself imports without SciPy. The NumPy-only
kernels call [`namespace`][functax._src._backend.namespace] instead, which
imports JAX only when a JAX array has already arrived (the array namespace of
an input).
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any

import numpy as np

from functax.exceptions import MissingExtraError


__all__ = [
    "JAX_EXTRA",
    "import_scipy",
    "is_jax_array",
    "is_tracer",
    "namespace",
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


def import_scipy(submodule: str, *, needed_by: str) -> ModuleType:
    """``scipy.<submodule>``, imported when a plan is built, or an error naming it.

    SciPy is the ``functax[scipy]`` extra: only the host-side fits of some
    schemes (k-d trees, triangulations) need it, so it is imported at fit
    time rather than when functax is.

    Args:
        submodule: The SciPy submodule, such as ``"spatial"``.
        needed_by: What needs it, used in the message (``"InverseDistance.plan"``).

    Returns:
        The imported submodule.

    Raises:
        MissingExtraError: If SciPy cannot be imported; the message names
            ``needed_by`` and the extra.

    Examples:
        >>> from functax._src._backend import import_scipy
        >>> import_scipy("spatial", needed_by="InverseDistance.plan").__name__
        'scipy.spatial'
    """
    try:
        return importlib.import_module(f"scipy.{submodule}")
    except ImportError as err:
        msg = (
            f"{needed_by} needs the functax[scipy] extra, but 'scipy.{submodule}' "
            "could not be imported; install it with: pip install 'functax[scipy]'"
        )
        raise MissingExtraError(msg) from err


def is_jax_array(x: Any) -> bool:
    """Whether ``x`` is a JAX array or tracer, without importing JAX.

    A JAX array can only exist once JAX is imported, so when ``jax`` is not in
    ``sys.modules`` the answer is ``False`` and nothing is imported.

    Examples:
        >>> import numpy as np
        >>> from functax._src._backend import is_jax_array
        >>> is_jax_array(np.zeros(3))
        False
    """
    if "jax" not in sys.modules:
        return False
    import jax

    return isinstance(x, jax.Array)


def is_tracer(x: Any) -> bool:
    """Whether ``x`` is a JAX tracer (inside ``jit``, ``grad`` or ``vmap``).

    Examples:
        >>> import numpy as np
        >>> from functax._src._backend import is_tracer
        >>> is_tracer(np.zeros(3))
        False
    """
    if "jax" not in sys.modules:
        return False
    import jax.core

    return isinstance(x, jax.core.Tracer)


def namespace(*arrays: Any) -> ModuleType:
    """The array module of ``arrays``: ``jax.numpy`` if any is JAX, else ``numpy``.

    It dispatches on the array type, as geoarray's ``namespace`` does, so it
    needs no NumPy 2 ``__array_namespace__`` and works with ``numpy>=1.26``.
    Lists, tuples and scalars count as NumPy. JAX is imported only when one of
    ``arrays`` is already a JAX array.

    Args:
        *arrays: The inputs of a kernel.

    Returns:
        ``numpy`` or ``jax.numpy``.

    Examples:
        >>> import numpy as np
        >>> from functax._src._backend import namespace
        >>> namespace(np.zeros(3), (1.0, 2.0)).__name__
        'numpy'
        >>> import jax.numpy as jnp
        >>> namespace(np.zeros(3), jnp.zeros(3)).__name__
        'jax.numpy'
    """
    if any(is_jax_array(x) for x in arrays):
        import jax.numpy as jnp

        return jnp
    return np
