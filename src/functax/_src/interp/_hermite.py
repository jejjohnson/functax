"""Cubic Hermite read-out on one axis, from values and slopes at the knots.

On cell [xᵢ, xᵢ₊₁] of width hᵢ, at offset t, with values yᵢ, yᵢ₊₁ and slopes
mᵢ, mᵢ₊₁,

    f̂ = h₀₀(t) yᵢ + h₁₀(t) hᵢ mᵢ + h₀₁(t) yᵢ₊₁ + h₁₁(t) hᵢ mᵢ₊₁,

with h₀₀ = 2t³ − 3t² + 1, h₁₀ = t³ − 2t² + t, h₀₁ = −2t³ + 3t² and
h₁₁ = t³ − t². The cubic spline, PCHIP, Akima and Steffen differ only in how
the slopes m come from y; the fit returns [y; m], 2N coefficients, and the
read-out is a 4-tap stencil on them.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

from collections.abc import Callable

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from functax._src._backend import is_tracer
from functax._src.geometry._types import Axis
from functax._src.interp._derivative import ReadGeometry
from functax._src.interp._fit import Fit
from functax._src.interp._locate import (
    check_outside,
    clamp_positions,
    domain_bounds,
    index_positions,
)
from functax._src.interp._policies import Policies
from functax._src.interp._stencil import Stencil
from functax.exceptions import NonlinearSchemeError, ValidationError


__all__ = [
    "HermiteFit",
    "HermiteGeometry",
    "ascending_axis",
    "hermite_stencil",
]


type Slopes = Callable[[jax.Array, jax.Array], jax.Array]
"""m = slopes(x, y) for increasing knots x (N,) and values y (N,)."""


def ascending_axis(axis: Axis) -> tuple[Axis, bool]:
    """The axis with increasing knots, and whether it had to be reversed.

    Raises:
        ValidationError: If the axis is periodic (the 1-D series schemes take
            the periodic condition from the spline, not from the axis).
    """
    if axis.period is not None:
        msg = "the 1-D series schemes need a non-periodic Axis; use bc='periodic'"
        raise ValidationError(msg)
    values = axis.values
    # a concrete axis decides on the host; a traced one is taken as increasing
    if is_tracer(values):
        return axis, False
    ends = np.asarray(values)[[0, -1]]
    if bool(ends[0] > ends[1]):
        return Axis(values[::-1], repeats=axis.repeats), True
    return axis, False


class HermiteFit(Fit):
    """[y; m] from y: the values in increasing-knot order and their slopes.

    Args:
        nodes: The increasing knots x (N,).
        slopes: m = slopes(x, y) for one vector y; batch dims are vectorised.
        reversed: Whether the source axis decreases, so y is reversed first.
        linear: Whether the slopes are linear in y (the cubic spline), which
            makes `apply_adjoint` the linear transpose.
    """

    nodes: Float[Array, " node"]
    slopes: Slopes = eqx.field(static=True)
    reversed: bool = eqx.field(static=True, default=False)
    linear: bool = eqx.field(static=True, default=True)

    def apply(self, data: Float[Array, "*batch node"]) -> Float[Array, "*batch source"]:
        """c = [y; m] with the values in increasing-knot order."""
        values = data[..., ::-1] if self.reversed else data
        vectorised = jnp.vectorize(self.slopes, signature="(n),(n)->(n)")
        slopes = vectorised(self.nodes, values)
        return einx.id("... n, ... n -> ... (n + n)", values, slopes)

    def apply_adjoint(
        self, coefficients: Float[Array, "*batch source"]
    ) -> Float[Array, "*batch node"]:
        """𝒮ᵀ g, the linear transpose of the slope fit (linear fits only)."""
        if not self.linear:
            msg = "a nonlinear slope fit has no adjoint; use plan.linearize(y)"
            raise NonlinearSchemeError(msg)
        n_data = self.data_size(coefficients.shape[-1])
        structure = jax.ShapeDtypeStruct(
            (*coefficients.shape[:-1], n_data), coefficients.dtype
        )
        (transposed,) = jax.linear_transpose(self.apply, structure)(coefficients)
        return transposed

    def data_size(self, n_coefficients: int) -> int:
        return n_coefficients // 2


class HermiteGeometry(ReadGeometry):
    """What a 1-D series plan's Hermite read-out was built from.

    Kept on the plan, so `Plan.derivative` rebuilds the read-out weights at
    the targets and differentiates them: on cell i, d/dx = (1/hᵢ) d/dt, so
    order 1 reads h₀₀′/hᵢ, h₀₁′/hᵢ on the values and h₁₀′, h₁₁′ on the
    slopes, f̂′(xₖ) = mₖ at a knot, and orders 4 and above are zero.

    Args:
        axis: The increasing source axis.
        coordinates: x, the (M, 1) target coordinates, before any wrap.
        periodic: Whether the spline is periodic, so the targets wrap into
            one period P = x_{N−1} − x₀.
    """

    axis: Axis
    coordinates: Float[Array, "target 1"]
    periodic: bool = eqx.field(static=True, default=False)

    def weights(
        self, coordinates: Float[Array, "target 1"], policies: Policies
    ) -> Float[Array, "target tap"]:
        """The four Hermite weights of each target, on [y; m]."""
        positions = coordinates[:, 0]
        if self.periodic:
            # x ↦ x₀ + ((x − x₀) mod P), as the plan wraps its targets
            first = self.axis.values[0]
            period = self.axis.values[-1] - first
            positions = first + jnp.mod(positions - first, period)
        return hermite_stencil(self.axis, positions, policies).weights


def hermite_stencil(
    axis: Axis,
    coordinates: Float[Array, " target"],
    policies: Policies,
) -> Stencil:
    """The 4-tap stencil of the Hermite read-out on [y; m], at the targets.

    Args:
        axis: The increasing source axis.
        coordinates: x, the M target coordinates.
        policies: The outside rules; ``boundary`` does not apply (every tap is
            a real knot of the cell), and outside the knots the end cubic is
            continued under ``extrapolate``. On an axis with repeats, a
            repeat at an end leaves a piece of one knot there, which is
            continued as a constant.

    Returns:
        The `Stencil` on 2N coefficients: taps i, i + 1 (values) and N + i,
        N + i + 1 (slopes).
    """
    nodes = axis.values
    n_nodes = nodes.shape[0]
    lower, upper = domain_bounds(n_nodes, policies.extent)
    position = index_positions(axis, coordinates)
    finite = jnp.isfinite(position)
    inside = finite & (position >= lower) & (position <= upper)
    check_outside(inside, policies)
    position = jnp.where(finite, position, 0.0)
    if policies.outside == "clamp":
        position = clamp_positions(position, lower, upper)
    # a real cell, so both knots and both slopes exist
    cell = jnp.clip(jnp.floor(position), 0, n_nodes - 2)
    offset = position - cell
    cell = cell.astype(jnp.int32)
    # hᵢ = xᵢ₊₁ − xᵢ
    width = nodes[cell + 1] - nodes[cell]
    if axis.repeats == "allow":
        # a cell of zero width is only reached past a repeated end: hold its knot
        offset = jnp.where(width > 0, offset, jnp.clip(offset, 0.0, 1.0))
    t2 = offset * offset
    t3 = t2 * offset
    # h₀₀, h₀₁ on the values; h₁₀ hᵢ, h₁₁ hᵢ on the slopes
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h01 = -2.0 * t3 + 3.0 * t2
    h10 = (t3 - 2.0 * t2 + offset) * width
    h11 = (t3 - t2) * width
    indices = einx.id(
        "target, target, target, target -> target (1 + 1 + 1 + 1)",
        cell,
        cell + 1,
        cell + n_nodes,
        cell + 1 + n_nodes,
    )
    weights = einx.id(
        "target, target, target, target -> target (1 + 1 + 1 + 1)", h00, h01, h10, h11
    )
    row_ok = finite if policies.outside in ("clamp", "extrapolate") else inside
    zero = jnp.zeros((), weights.dtype)
    weights = einx.where("target, target tap, -> target tap", row_ok, weights, zero)
    return Stencil(indices, weights, n_source=2 * n_nodes, valid=row_ok, check=False)
