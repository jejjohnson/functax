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
    "derivative_stencil",
]


class LocalGeometry(eqx.Module):
    """What a local plan was built from: the source, the target coordinates, the bases.

    Kept on the `Plan`, so `Plan.derivative` can rebuild the weights.
    """

    source: Any
    coordinates: Float[Array, "target dim"]
    bases: tuple[Basis, ...]


def derivative_stencil(
    geometry: LocalGeometry,
    stencil: Stencil,
    policies: Policies,
    *,
    axis: int,
    order: int,
) -> Stencil:
    """The stencil of ∂ᵐ/∂x_cᵐ, with c = ``axis`` and m = ``order``.

    Args:
        geometry: The plan's source, target coordinates and bases.
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
    n_coordinates = geometry.coordinates.shape[1]
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
    shape, periodic = source_layout(geometry.source)
    # the plan already enforced outside="raise"; the derivative re-traces
    # its targets, which "raise" would refuse
    if policies.outside == "raise":
        policies = dataclasses.replace(policies, outside="nan")

    extrapolate = policies.outside == "extrapolate"

    def weights(coordinates: jax.Array) -> jax.Array:
        positions = coordinates_to_positions(
            geometry.source, coordinates, extrapolate=extrapolate
        )
        rebuilt = stencil_from_positions(
            positions, shape, periodic, geometry.bases, policies=policies
        )
        return rebuilt.weights

    # ẋ = e_c for every target: one coordinate moves, the others stay
    tangent = jnp.zeros_like(geometry.coordinates).at[:, column].set(1.0)
    derivative: Callable[[jax.Array], jax.Array] = weights
    for _ in range(order):
        derivative = _directional(derivative, tangent)
    derivative_weights = derivative(geometry.coordinates)
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
