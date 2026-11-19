"""Dtype rules shared across areas: how an integer input becomes a floating one."""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import jax.numpy as jnp
from jaxtyping import Array, Shaped


__all__ = ["promote_integers"]


def promote_integers(array: Shaped[Array, "*shape"]) -> Shaped[Array, "*shape"]:
    """``array`` as float32 when it holds integers, and unchanged otherwise.

    An integer input to a feature map or a field (time steps, cell indices)
    would truncate under the trigonometric and affine maps that follow. This is
    geonnax's rule, kept so the moved code gives the same output: integers
    become float32 whatever the x64 setting, and a floating input keeps its
    dtype.

    Args:
        array: The input, of any dtype.

    Returns:
        ``array`` cast to float32 if it has an integer dtype, else ``array``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax._src._dtypes import promote_integers
        >>> promote_integers(jnp.arange(3)).dtype
        dtype('float32')
        >>> promote_integers(jnp.zeros(3, dtype=jnp.float64)).dtype
        dtype('float64')
    """
    if jnp.issubdtype(array.dtype, jnp.integer):
        return array.astype(jnp.float32)
    return array
