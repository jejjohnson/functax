"""Derivative read-outs: H′ with Hₖⱼ′ = ∂ᵐφⱼ/∂x_cᵐ at the k-th target.

Every row of a local stencil depends only on its own target, so the
derivative weights are the forward-mode derivative of the weight
construction with respect to the target coordinates, one column c at a time:
the chain rule through the index position (1/Δ on a regular axis, the cell
width on a rectilinear one, A⁻¹ on an affine grid) and through the policies
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
    """

    @abc.abstractmethod
    def weights(
        self, coordinates: Float[Array, "target dim"], policies: Policies
    ) -> Float[Array, "target tap"]:
        """The stencil weights at ``coordinates``, with the plan's taps."""


class LocalGeometry(ReadGeometry):
    """What a local plan was built from: the source, the target coordinates, the bases.

    Kept on the `Plan`, so `Plan.derivative` can rebuild the weights.
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
            positions, shape, periodic, self.bases, policies=policies
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
    """The stencil of ∂ᵐ/∂x_cᵐ, with c = ``axis`` and m = ``order``.

    Args:
        geometry: What the plan's stencil was built from.
        stencil: The plan's own stencil, whose taps and validity are kept.
        policies: The plan's policies.
        axis: c, the target coordinate to differentiate along (a column of
            ``Points.xy``; negative counts from the end).
        order: m ≥ 1.

    Returns:
        The derivative `Stencil`.

    Raises:
        ValidationError: If ``axis`` is out of range or ``order`` is not a
            positive integer.
    """
    # every subclass declares the field; ReadGeometry itself does not (see it)
    coordinates: jax.Array = geometry.coordinates  # ty: ignore[unresolved-attribute]
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

    def weights(coordinates: jax.Array) -> jax.Array:
        return geometry.weights(coordinates, policies)

    # ẋ = e_c for every target: one coordinate moves, the others stay
    tangent = jnp.zeros_like(coordinates).at[:, column].set(1.0)
    derivative: Callable[[jax.Array], jax.Array] = weights
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
    function: Callable[[jax.Array], jax.Array], tangent: jax.Array
) -> Callable[[jax.Array], jax.Array]:
    """x ↦ ∂f(x)/∂ε at ε = 0 along x + ε ẋ, by forward mode."""

    def derivative(coordinates: jax.Array) -> jax.Array:
        _, slope = jax.jvp(function, (coordinates,), (tangent,))
        return slope

    return derivative
