"""Derivative read-outs: ∂ᵐH/∂x_dᵐ, with entries ∂ᵐφⱼ/∂x_dᵐ at the k-th target.

Here H is the read-out, φⱼ the basis function of coefficient j, x_d the d-th
coordinate of a target and m ≥ 1 the order. Every row of a local stencil
depends only on its own target, so the derivative weights are the
forward-mode derivative of the weight construction with respect to the target
coordinates, one coordinate x_d at a time: the chain rule through the index
position (1/Δ on a regular axis of spacing Δ, one over the cell width on a
rectilinear one, A⁻¹ on an affine grid of matrix A) and through the policies
comes with it. The taps are piecewise constant in x, so they stay those of
the plan.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import abc
import dataclasses
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from functax._src.interp._bases import Basis
from functax._src.interp._policies import Policies
from functax._src.interp._stencil import Stencil
from functax._src.interp._tensor import (
    coordinates_to_positions,
    descending_axes,
    source_layout,
    stencil_from_positions,
)
from functax.exceptions import ValidationError


__all__ = [
    "LocalGeometry",
    "ReadGeometry",
    "derivative_stencil",
]


class ReadGeometry(eqx.Module):
    """What a plan's stencil was built from, so `Plan.derivative` can rebuild it.

    The stencil weights are a function of the target coordinates, and their
    forward-mode derivative in one coordinate is the derivative read-out.
    Each subclass holds the (M, D) target coordinates as ``coordinates``
    (declared there rather than here, which keeps each subclass's field
    order, and so its positional constructor).

    Examples:
        A plan from a local scheme keeps its geometry, and rebuilds its own
        weights from it:

        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points
        >>> from functax.interp import Linear, ReadGeometry
        >>> plan = Linear().plan(Axis(jnp.arange(4.0)), Points(jnp.array([[1.25]])))
        >>> isinstance(plan.local, ReadGeometry)
        True
        >>> weights = plan.local.weights(plan.local.coordinates, plan.policies)
        >>> bool(jnp.allclose(weights, plan.read.weights))
        True
    """

    @abc.abstractmethod
    def weights(
        self, coordinates: Float[Array, "target dim"], policies: Policies
    ) -> Float[Array, "target tap"]:
        """The stencil weights at ``coordinates``, with the plan's taps."""


class LocalGeometry(ReadGeometry):
    """What a local plan was built from: the source, the target coordinates, the bases.

    Kept on the `Plan`, so `Plan.derivative` can rebuild the weights.

    Args:
        source: The source geometry (an `Axis` or a grid).
        coordinates: X_*, the (M, D) target coordinates in the source's frame.
        bases: One `Basis` per source axis.

    Examples:
        The weights of a linear read-out are (1 − t, t):

        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis
        >>> from functax.interp import LinearBasis, LocalGeometry, Policies
        >>> coordinates = jnp.array([[0.25], [2.5]])
        >>> geometry = LocalGeometry(
        ...     Axis(jnp.arange(4.0)), coordinates, (LinearBasis(),)
        ... )
        >>> geometry.weights(coordinates, Policies()).tolist()
        [[0.75, 0.25], [0.5, 0.5]]

        Their derivative in the target coordinate is (−1, 1) on unit cells:

        >>> import jax
        >>> def first_weights(x):
        ...     return geometry.weights(x, Policies())
        >>> _, slope = jax.jvp(first_weights, (coordinates,), (jnp.ones((2, 1)),))
        >>> slope.tolist()
        [[-1.0, 1.0], [-1.0, 1.0]]
    """

    source: Any
    coordinates: Float[Array, "target dim"]
    bases: tuple[Basis, ...]

    def weights(
        self, coordinates: Float[Array, "target dim"], policies: Policies
    ) -> Float[Array, "target tap"]:
        """The tensor stencil's weights at ``coordinates``."""
        shape, periodic = source_layout(self.source)
        extrapolate = policies.outside == "extrapolate"
        positions = coordinates_to_positions(
            self.source, coordinates, extrapolate=extrapolate
        )
        rebuilt = stencil_from_positions(
            positions,
            shape,
            periodic,
            self.bases,
            policies=policies,
            descending=descending_axes(self.source),
        )
        return rebuilt.weights


def derivative_stencil(
    geometry: ReadGeometry,
    stencil: Stencil,
    policies: Policies,
    *,
    axis: int,
    order: int,
) -> Stencil:
    """The stencil of ∂ᵐ/∂x_dᵐ, with d = ``axis`` and m = ``order``.

    Algorithm:
        1. Check ``axis`` and ``order``; relax ``outside="raise"`` to
           ``"nan"``, since the plan already checked its targets.
        2. Take the weights w(X_*) as a function of the target coordinates.
        3. Apply forward mode m times along the tangent e_d of every target.
        4. Keep the plan's taps and validity, with the derivative weights.

    Args:
        geometry: What the plan's stencil was built from.
        stencil: The plan's own stencil, whose taps and validity are kept.
        policies: The plan's policies.
        axis: d, the target coordinate to differentiate along (a column of
            ``Points.xy``; negative counts from the end).
        order: m ≥ 1.

    Returns:
        The derivative `Stencil`.

    Raises:
        ValidationError: If ``axis`` is out of range or ``order`` is not a
            positive integer.
    """
    # every subclass declares the field; ReadGeometry itself does not (see it)
    coordinates: Float[Array, "target dim"]
    coordinates = geometry.coordinates  # ty: ignore[unresolved-attribute]
    n_coordinates = coordinates.shape[1]
    if isinstance(axis, bool) or not isinstance(axis, int):
        msg = f"'axis' must be an integer, got {axis!r}"
        raise ValidationError(msg)
    if not -n_coordinates <= axis < n_coordinates:
        msg = f"'axis' must lie in [{-n_coordinates}, {n_coordinates}), got {axis}"
        raise ValidationError(msg)
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        msg = f"'order' must be a positive integer, got {order!r}"
        raise ValidationError(msg)
    column = axis % n_coordinates
    # the plan already enforced outside="raise"; the derivative re-traces
    # its targets, which "raise" would refuse
    if policies.outside == "raise":
        policies = dataclasses.replace(policies, outside="nan")

    def weights(coordinates: Float[Array, "target dim"]) -> Float[Array, "target tap"]:
        return geometry.weights(coordinates, policies)

    # v = e_d for every target: one coordinate moves, the others stay
    zeros = jnp.zeros_like(coordinates)
    tangent = zeros.at[:, column].set(1.0)
    derivative: Callable[[jax.Array], jax.Array] = weights
    # ∂ᵐw/∂x_dᵐ, one forward-mode pass per order
    for _ in range(order):
        derivative = _directional(derivative, tangent)
    derivative_weights = derivative(coordinates)
    return Stencil(
        stencil.indices,
        derivative_weights,
        n_source=stencil.n_source,
        valid=stencil.valid,
        check=False,
    )


def _directional(
    function: Callable[[jax.Array], jax.Array], tangent: Float[Array, "target dim"]
) -> Callable[[jax.Array], jax.Array]:
    """x ↦ J_f(x) v, the derivative of f along the tangent v, by forward mode."""

    def derivative(
        coordinates: Float[Array, "target dim"],
    ) -> Float[Array, "target tap"]:
        # J_f(x) v
        _, slope = jax.jvp(function, (coordinates,), (tangent,))
        return slope

    return derivative
