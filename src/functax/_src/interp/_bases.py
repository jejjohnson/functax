"""Local one-dimensional bases: the taps a target reads on an axis, and their weights.

A target at cell i and offset t ∈ [0, 1) reads taps i + r for the basis'
relative taps r, with weights that depend on t. Nearest reads one tap, linear
two; the cubic and B-spline bases of #27 follow the same pattern.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import abc
from typing import ClassVar

import einx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


__all__ = [
    "Basis",
    "LinearBasis",
    "NearestBasis",
]


class Basis(eqx.Module):
    """A local basis on one axis: S relative taps, and weights from the offsets.

    ``support`` is (r_min, r_max), the range of relative taps for offsets in
    [0, 1); `outside="extrapolate"` keeps the cell within
    [−r_min, N − 1 − r_max], so every tap stays on the axis.
    """

    taps: eqx.AbstractClassVar[int]
    support: eqx.AbstractClassVar[tuple[int, int]]

    @abc.abstractmethod
    def stencil(
        self, offsets: Float[Array, " target"]
    ) -> tuple[Int[Array, "target tap"], Float[Array, "target tap"]]:
        """The relative taps r and the weights w of each target, from its offset t."""


class NearestBasis(Basis):
    """Nearest neighbour: one tap, the closer of i and i + 1, with weight 1.

    A tie (t = ½) goes to i, the lower node, as in SciPy's
    ``RegularGridInterpolator(method="nearest")``; ``map_coordinates(order=0)``
    rounds ties up, and NumPy's ``round`` to even.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.interp import NearestBasis
        >>> taps, weights = NearestBasis().stencil(jnp.array([0.2, 0.5, 0.8]))
        >>> taps.tolist(), weights.tolist()
        ([[0], [0], [1]], [[1.0], [1.0], [1.0]])
    """

    taps: ClassVar[int] = 1
    support: ClassVar[tuple[int, int]] = (0, 1)

    def stencil(
        self, offsets: Float[Array, " target"]
    ) -> tuple[Int[Array, "target tap"], Float[Array, "target tap"]]:
        # r = 1 when t > ½, else 0: ties go to the lower node
        upper = (offsets > 0.5).astype(jnp.int32)
        taps = einx.id("target -> target 1", upper)
        weights = einx.id("target -> target 1", jnp.ones_like(offsets))
        return taps, weights


class LinearBasis(Basis):
    """Linear: taps i and i + 1, with weights 1 − t and t.

    Linear interpolation is the posterior mean under a Brownian-motion prior
    (its variance is #31). Gradients reach the target through t.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.interp import LinearBasis
        >>> taps, weights = LinearBasis().stencil(jnp.array([0.25]))
        >>> taps.tolist(), weights.tolist()
        ([[0, 1]], [[0.75, 0.25]])
    """

    taps: ClassVar[int] = 2
    support: ClassVar[tuple[int, int]] = (0, 1)

    def stencil(
        self, offsets: Float[Array, " target"]
    ) -> tuple[Int[Array, "target tap"], Float[Array, "target tap"]]:
        n_targets = offsets.shape[0]
        taps = einx.id(
            "tap -> target tap", jnp.arange(2, dtype=jnp.int32), target=n_targets
        )
        # w = (1 − t, t)
        weights = einx.id("target, target -> target (1 + 1)", 1.0 - offsets, offsets)
        return taps, weights
