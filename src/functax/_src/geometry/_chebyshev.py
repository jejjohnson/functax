"""Chebyshev nodes on an interval: the Gauss–Lobatto and the Gauss points.

The nodes of spectraldiffx's ``ChebyshevGrid1D/2D/3D`` at ``d68f8f6``, moved so
that the Chebyshev transforms of `functax.transform` and the Chebyshev read-out
of `functax.interp` share one grid. They keep spectraldiffx's order, from the
upper end of the interval down to the lower one.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from functax._src.geometry._sphere_grids import _check_count
from functax._src.quad._rules import _interval
from functax.exceptions import GeometryError


__all__ = [
    "CHEBYSHEV_KINDS",
    "chebyshev_nodes",
]


CHEBYSHEV_KINDS: tuple[str, ...] = ("lobatto", "gauss")
"""The Chebyshev node kinds: Gauss–Lobatto (with the ends) and Gauss (without)."""


def chebyshev_nodes(
    n: int,
    *,
    kind: str = "lobatto",
    interval: tuple[float, float] = (-1.0, 1.0),
) -> Float[Array, " node"]:
    """The Chebyshev nodes of degree N on [a, b], from b down to a.

    With c = (a + b)/2 and h = (b − a)/2, the nodes are xⱼ = c + h ξⱼ for

        Gauss–Lobatto:  ξⱼ = cos(jπ/N),           j = 0, …, N  (N + 1 nodes),
        Gauss:          ξⱼ = cos((2j + 1)π/(2N)), j = 0, …, N − 1  (N nodes),

    the extrema of T_N with the ends ±1, and the zeros of T_N. They follow
    spectraldiffx's ``ChebyshevGrid1D.x``, decreasing from ξ₀ = 1 (it is a
    move, and the transforms of `ChebyshevTransform1D` are written in that
    order). `clenshaw_curtis` in `functax.quad` orders the same Lobatto nodes
    ascending, from a up to b, so ``clenshaw_curtis(N + 1, interval)`` pairs
    with these nodes reversed; its weights are symmetric, so they pair with
    either order up to rounding.

    Algorithm:
        1. ξⱼ on [−1, 1], on the host in float64.
        2. Map them affinely onto [a, b]: xⱼ = c + h ξⱼ.

    Args:
        n: N, the polynomial degree: N + 1 Lobatto nodes, or N Gauss nodes;
            spectraldiffx's ``N``. At least 1.
        kind: ``"lobatto"`` (spectraldiffx's ``"gauss-lobatto"``) or
            ``"gauss"``.
        interval: The interval (a, b), with a < b; spectraldiffx's [−L, L] is
            ``(-L, L)``.

    Returns:
        The nodes, decreasing from b (or just below it) to a, as a JAX array.

    Raises:
        GeometryError: If ``n`` is not an integer of at least 1, ``kind`` is
            unknown, or the interval does not have a < b.
        ValidationError: If the interval is not two finite numbers.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import chebyshev_nodes
        >>> # cos(jπ/4), j = 0, …, 4
        >>> nodes = chebyshev_nodes(4)
        >>> bool(jnp.allclose(nodes, jnp.cos(jnp.pi * jnp.arange(5) / 4)))
        True
        >>> chebyshev_nodes(8, kind="gauss", interval=(0.0, 2.0)).shape
        (8,)
        >>> # the ends of [a, b] are the first and last Lobatto nodes
        >>> chebyshev_nodes(3, interval=(-2.0, 5.0))[jnp.array([0, -1])].tolist()
        [5.0, -2.0]
    """
    degree = _check_count(n, "n")
    if kind not in CHEBYSHEV_KINDS:
        msg = f"'kind' must be one of {list(CHEBYSHEV_KINDS)}, got {kind!r}"
        raise GeometryError(msg)
    lower, upper = _interval(interval)
    if not lower < upper:
        msg = f"'interval' must have a < b, got {interval!r}"
        raise GeometryError(msg)
    if kind == "lobatto":
        # ξⱼ = cos(jπ/N), j = 0, …, N
        angles = np.pi * np.arange(degree + 1) / degree
    else:
        # ξⱼ = cos((2j + 1)π/(2N)), j = 0, …, N − 1
        angles = np.pi * (2 * np.arange(degree) + 1) / (2 * degree)
    reference = np.cos(angles)
    # c = (a + b)/2, h = (b − a)/2
    centre = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    # xⱼ = c + h ξⱼ
    return jnp.asarray(centre + half_width * reference)
