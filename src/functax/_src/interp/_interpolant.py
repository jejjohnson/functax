"""Interpolants: a fitted field f̂ that is read at targets chosen inside the trace.

A `Plan` freezes its targets when it is built; an `Interpolant` keeps the
coefficients c = 𝒮(y) and the source, and locates, weighs and gathers at every
call, so particles that move inside ``lax.scan`` read the field without a
rebuild on the host. Both paths share the locate-and-weigh code, so they give
identical stencils.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import dataclasses
import math
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from functax._src.geometry._types import Points
from functax._src.interp._derivative import LocalGeometry
from functax._src.interp._fit import Identity
from functax._src.interp._plan import Plan
from functax._src.interp._policies import Policies
from functax._src.interp._prefilter import prefilter_for
from functax._src.interp._tensor import (
    check_local_source,
    coordinates_to_positions,
    source_layout,
    stencil_from_positions,
    target_coordinates,
)
from functax.exceptions import PolicyError, ValidationError


__all__ = [
    "Interpolant",
]


class Interpolant(eqx.Module):
    """A fitted field f̂(x) = ∑ₛ wₛ(t(x)) c_{iₛ(x)}, read at any targets.

    Made by ``scheme.fit(source, data)``. Calling it with target coordinates
    locates, weighs and gathers inside the trace, so it runs under ``jit``,
    ``vmap``, ``grad`` and ``lax.scan`` without host work; it equals
    ``scheme.plan(source, targets)(data)`` for every local scheme.

    Args:
        coefficients: c = 𝒮(y) over the flat source, with any leading batch
            dims (the data itself for schemes without a prefilter).
        source: The `Axis`, `RectilinearGrid` or `AffineGrid` the data lives on.
        scheme: The local scheme that fitted it.
        policies: The plan policies; ``missing="fill"`` was applied at the fit.
        derivative_axis: For a derivative interpolant, the target coordinate it
            differentiates along; ``None`` for the field itself.
        derivative_order: The derivative order, when ``derivative_axis`` is set.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis
        >>> from functax.interp import Linear
        >>> field = Linear().fit(Axis(jnp.arange(5.0)), jnp.arange(5.0) ** 2)
        >>> field(jnp.array([[1.5], [3.25]])).tolist()
        [2.5, 10.75]
        >>> jax.jit(field.__call__)(jnp.array([[2.0]])).tolist()
        [4.0]
    """

    coefficients: Float[Array, "*batch source"]
    source: Any
    # Any: the scheme module imports this one, so LocalScheme cannot be named
    scheme: Any
    policies: Policies
    derivative_axis: int | None = eqx.field(static=True, default=None)
    derivative_order: int = eqx.field(static=True, default=1)

    def _read_plan(self, coordinates: Float[Array, "target dim"]) -> Plan:
        """The plan that reads the coefficients (not the data) at the coordinates."""
        shape, periodic = source_layout(self.source)
        bases = self.scheme.bases(len(shape))
        check_local_source(self.source, bases, self.policies)
        extrapolate = self.policies.outside == "extrapolate"
        positions = coordinates_to_positions(
            self.source, coordinates, extrapolate=extrapolate
        )
        stencil = stencil_from_positions(
            positions, shape, periodic, bases, policies=self.policies
        )
        # the fill already ran at the fit, before the prefilter
        policies = self.policies
        if policies.missing == "fill":
            policies = dataclasses.replace(policies, missing="propagate")
        local = LocalGeometry(self.source, coordinates, bases)
        plan = Plan(Identity(), stencil, policies, local)
        if self.derivative_axis is None:
            return plan
        return plan.derivative(self.derivative_axis, self.derivative_order)

    def __call__(
        self, targets: Float[Array, "target dim"] | Points
    ) -> Float[Array, "*batch target"]:
        """f̂ at the targets, ``fill_value`` where they are invalid.

        Args:
            targets: The (M, D) target coordinates (one per source axis, or
                (x, y) on an `AffineGrid`), or `Points`.

        Returns:
            f̂ with the batch dims of the coefficients and the targets last.

        Raises:
            GeometryError: If the targets have the wrong number of coordinates.
        """
        coordinates = (
            target_coordinates(self.source, targets)
            if isinstance(targets, Points)
            else jnp.asarray(targets)
        )
        return self._read_plan(coordinates)(self.coefficients)

    def plan(self, target: Any) -> Plan:
        """The `Plan` that ``scheme.plan(source, target)`` builds, which reads data.

        Use it to freeze the current targets, for example to keep the adjoint.

        Args:
            target: `Points`, or a grid read at its nodes.

        Returns:
            The `Plan` (or `DerivativePlan` for a derivative interpolant).
        """
        plan = self.scheme.plan(self.source, target, policies=self.policies)
        if self.derivative_axis is None:
            return plan
        return plan.derivative(self.derivative_axis, self.derivative_order)

    def derivative(self, axis: int, order: int = 1) -> Interpolant:
        """∂ᵐf̂/∂x_cᵐ as an interpolant, with c = ``axis`` and m = ``order``.

        Raises:
            ValidationError: If the interpolant is already a derivative.
        """
        if self.derivative_axis is not None:
            msg = "derivative() of a derivative interpolant: raise 'order' instead"
            raise ValidationError(msg)
        return dataclasses.replace(self, derivative_axis=axis, derivative_order=order)


def fit_interpolant(
    scheme: Any, source: Any, data: Float[Array, "*batch source"], policies: Policies
) -> Interpolant:
    """c = 𝒮(y) for ``scheme`` on ``source``, as an `Interpolant`.

    Raises:
        ValidationError: If the last axis of ``data`` is not the source size.
        PolicyError: Under ``missing="renormalise"`` with a prefilter.
    """
    shape, periodic = source_layout(source)
    bases = scheme.bases(len(shape))
    data = jnp.asarray(data)
    n_source = math.prod(shape)
    if data.ndim < 1 or data.shape[-1] != n_source:
        msg = (
            f"'data' must have a last axis of length {n_source}, got shape {data.shape}"
        )
        raise ValidationError(msg)
    orders = tuple(basis.prefilter_order for basis in bases)
    prefilter = prefilter_for(orders, shape, periodic, policies.boundary)
    if policies.missing == "fill":
        fill = jnp.asarray(policies.fill_value, dtype=data.dtype)
        data = jnp.where(jnp.isfinite(data), data, fill)
    if prefilter is None:
        return Interpolant(data, source, scheme, policies)
    if policies.missing == "renormalise":
        msg = (
            "missing='renormalise' needs a local plan (identity fit, stencil read-out)"
        )
        raise PolicyError(msg)
    return Interpolant(prefilter.apply(data), source, scheme, policies)
