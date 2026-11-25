"""Cubic Hermite read-out on one axis, from values and slopes at the knots.

On cell [xᵢ, xᵢ₊₁] of width hᵢ, at offset t, with values yᵢ, yᵢ₊₁ and slopes
dᵢ, dᵢ₊₁,

    f̂ = φ₀₀(t) yᵢ + φ₁₀(t) hᵢ dᵢ + φ₀₁(t) yᵢ₊₁ + φ₁₁(t) hᵢ dᵢ₊₁,

with the cubic Hermite basis φ₀₀ = 2t³ − 3t² + 1, φ₁₀ = t³ − 2t² + t,
φ₀₁ = −2t³ + 3t² and φ₁₁ = t³ − t². The cubic spline, PCHIP, Akima and
Steffen differ only in how the slopes d ∈ ℝᴺ come from the N samples y; the
fit returns [y; d], 2N coefficients, and the read-out is a 4-tap stencil on
them.
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

    Args:
        axis: The source axis, increasing or decreasing, not periodic.

    Returns:
        ``(ascending, reversed)``: the axis with its knots increasing, and
        whether they were reversed to get there.

    Raises:
        ValidationError: If the axis is periodic (the 1-D series schemes take
            the periodic condition from the spline, not from the axis).

    Examples:
        A decreasing axis is reversed:

        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis
        >>> from functax._src.interp._hermite import ascending_axis
        >>> ascending, flipped = ascending_axis(Axis(jnp.array([2.0, 1.0, 0.0])))
        >>> ascending.values.tolist(), flipped
        ([0.0, 1.0, 2.0], True)

        An increasing one comes back as it is:

        >>> axis = Axis(jnp.arange(3.0))
        >>> same, flipped = ascending_axis(axis)
        >>> same is axis, flipped
        (True, False)

        A periodic axis is rejected:

        >>> ascending_axis(
        ...     Axis(jnp.arange(3.0), period=3.0)
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        functax.exceptions.ValidationError: the 1-D series schemes need a ...
    """
    if axis.period is not None and not axis.closed:
        msg = (
            "the 1-D series schemes need a non-periodic Axis, and this one is open "
            "(it has a period its nodes do not cover); drop the period and shift "
            "the targets into the range of the nodes"
        )
        raise ValidationError(msg)
    if axis.closed:
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
    """[y; d] from y: the values in increasing-knot order and their slopes.

    c = [y; 𝒮_d(y)], with d = 𝒮_d(y) the scheme's slopes at the knots.

    Args:
        nodes: x, the N increasing knots.
        slopes: d = slopes(x, y) for one vector y; batch dims are vectorised.
        reversed: Whether the source axis decreases, so y is reversed first.
        linear: Whether the slopes are linear in y (the cubic spline), which
            makes `apply_adjoint` the linear transpose.

    Raises:
        NonlinearSchemeError: From `apply_adjoint`, when the slopes are
            nonlinear in y.

    Examples:
        Values and their PCHIP slopes, 2N coefficients:

        >>> import jax.numpy as jnp
        >>> from functax._src.interp._monotone import pchip_slopes
        >>> from functax.interp import HermiteFit
        >>> fit = HermiteFit(jnp.arange(4.0), pchip_slopes, linear=False)
        >>> coefficients = fit.apply(2.0 * jnp.arange(4.0))
        >>> coefficients.shape
        (8,)
        >>> bool(jnp.allclose(coefficients[4:], 2.0))
        True

        A linear (cubic spline) fit has the adjoint of the dot-product test:

        >>> from functax._src.interp._spline1d import SPLINE_SLOPES
        >>> spline = HermiteFit(jnp.arange(4.0), SPLINE_SLOPES["natural"])
        >>> data = jnp.array([1.0, -2.0, 0.5, 3.0])
        >>> residual = jnp.arange(8.0)
        >>> forward = jnp.dot(spline.apply(data), residual)
        >>> backward = jnp.dot(data, spline.apply_adjoint(residual))
        >>> bool(jnp.allclose(forward, backward))
        True

        A nonlinear fit has no adjoint:

        >>> fit.apply_adjoint(jnp.ones(8))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        functax.exceptions.NonlinearSchemeError: a nonlinear slope fit has ...
    """

    nodes: Float[Array, " node"]
    slopes: Slopes = eqx.field(static=True)
    reversed: bool = eqx.field(static=True, default=False)
    linear: bool = eqx.field(static=True, default=True)

    def apply(self, data: Float[Array, "*batch node"]) -> Float[Array, "*batch source"]:
        """c = [y; d] with the values in increasing-knot order."""
        values = data[..., ::-1] if self.reversed else data
        vectorised = jnp.vectorize(self.slopes, signature="(n),(n)->(n)")
        # d = 𝒮_d(y)
        slopes = vectorised(self.nodes, values)
        # c = [y; d]
        return einx.id("... node, ... node -> ... (node + node)", values, slopes)

    def apply_adjoint(
        self, coefficients: Float[Array, "*batch source"]
    ) -> Float[Array, "*batch node"]:
        """𝒮ᵀ g, the linear transpose of the slope fit (linear fits only).

        Raises:
            NonlinearSchemeError: If the slopes are nonlinear in y.
        """
        if not self.linear:
            msg = "a nonlinear slope fit has no adjoint; use plan.linearize(y)"
            raise NonlinearSchemeError(msg)
        n_data = self.data_size(coefficients.shape[-1])
        structure = jax.ShapeDtypeStruct(
            (*coefficients.shape[:-1], n_data), coefficients.dtype
        )
        # 𝒮ᵀ g, by transposing the linear map y ↦ [y; d]
        transpose = jax.linear_transpose(self.apply, structure)
        (transposed,) = transpose(coefficients)
        return transposed

    def data_size(self, n_coefficients: int) -> int:
        """N, half the 2N coefficients [y; d]."""
        return n_coefficients // 2


class HermiteGeometry(ReadGeometry):
    """What a 1-D series plan's Hermite read-out was built from.

    Kept on the plan, so `Plan.derivative` rebuilds the read-out weights at
    the targets and differentiates them: on cell i, ∂/∂x = (1/hᵢ) ∂/∂t, so
    order 1 reads (∂φ₀₀/∂t)/hᵢ, (∂φ₀₁/∂t)/hᵢ on the values and ∂φ₁₀/∂t,
    ∂φ₁₁/∂t on the slopes, ∂f̂/∂x = dᵢ at knot xᵢ, and orders 4 and above are
    zero.

    Args:
        axis: The increasing source axis.
        coordinates: x, the (M, 1) target coordinates, before any wrap.
        periodic: Whether the spline is periodic, so the targets wrap into
            one period P = x_{N−1} − x₀.

    Examples:
        The weights at a cell's midpoint, on [y; d]:

        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis
        >>> from functax.interp import HermiteGeometry, Policies
        >>> targets = jnp.array([[0.5], [1.25]])
        >>> geometry = HermiteGeometry(Axis(jnp.arange(3.0)), targets)
        >>> weights = geometry.weights(targets, Policies())
        >>> weights.shape
        (2, 4)
        >>> bool(jnp.allclose(weights[0], jnp.array([0.5, 0.5, 0.125, -0.125])))
        True

        A periodic geometry wraps its targets into one period first:

        >>> periodic = HermiteGeometry(
        ...     Axis(jnp.arange(3.0)), targets, periodic=True
        ... )
        >>> shifted = periodic.weights(targets + 2.0, Policies())
        >>> bool(jnp.allclose(shifted, weights))
        True
    """

    axis: Axis
    coordinates: Float[Array, "target 1"]
    periodic: bool = eqx.field(static=True, default=False)

    def weights(
        self, coordinates: Float[Array, "target 1"], policies: Policies
    ) -> Float[Array, "target tap"]:
        """The four Hermite weights of each target, on [y; d]."""
        positions = einx.id("target 1 -> target", coordinates)
        if self.periodic:
            first = self.axis.values[0]
            # P = x_{N−1} − x₀
            period = self.axis.values[-1] - first
            # x ↦ x₀ + ((x − x₀) mod P), as the plan wraps its targets
            wrapped = jnp.mod(positions - first, period)
            positions = first + wrapped
        stencil = hermite_stencil(self.axis, positions, policies)
        return stencil.weights


def hermite_stencil(
    axis: Axis,
    coordinates: Float[Array, " target"],
    policies: Policies,
) -> Stencil:
    """The 4-tap stencil of the Hermite read-out on [y; d], at the targets.

    f̂ = φ₀₀(t) yᵢ + φ₁₀(t) hᵢ dᵢ + φ₀₁(t) yᵢ₊₁ + φ₁₁(t) hᵢ dᵢ₊₁ on cell i of
    width hᵢ, at offset t.

    Algorithm:
        1. Locate each target: its index position, after the outside rule.
        2. Split it into a cell i (a real cell, with both knots) and an
           offset t.
        3. Weight the two values by φ₀₀, φ₀₁ and the two slopes by φ₁₀ hᵢ,
           φ₁₁ hᵢ, zeroing the rows that are not read.

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

    Raises:
        OutsideDomainError: Under ``outside="raise"`` when a target is
            outside.

    Examples:
        At a cell's midpoint the values weigh ½ each:

        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis
        >>> from functax.interp import Policies
        >>> from functax._src.interp._hermite import hermite_stencil
        >>> axis = Axis(jnp.arange(4.0))
        >>> stencil = hermite_stencil(axis, jnp.array([1.5, 3.0]), Policies())
        >>> stencil.indices.tolist()
        [[1, 2, 5, 6], [2, 3, 6, 7]]
        >>> bool(jnp.allclose(stencil.weights[0, :2], 0.5))
        True

        A target outside the knots reads nothing under the default rule:

        >>> outside = hermite_stencil(axis, jnp.array([5.0]), Policies())
        >>> outside.valid.tolist()
        [False]
    """
    # --- locate ---
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

    # --- cell and offset ---
    # i = ⌊u⌋, a real cell, so both knots and both slopes exist
    floor = jnp.floor(position)
    cell = jnp.clip(floor, 0, n_nodes - 2)
    # t = u − i
    offset = position - cell
    cell = cell.astype(jnp.int32)
    next_cell = cell + 1
    # hᵢ = xᵢ₊₁ − xᵢ
    width = nodes[next_cell] - nodes[cell]
    if axis.repeats == "allow":
        # a cell of zero width is only reached past a repeated end: hold its knot
        held = jnp.clip(offset, 0.0, 1.0)
        offset = jnp.where(width > 0, offset, held)

    # --- weights ---
    # t² and t³
    offset_squared = offset * offset
    offset_cubed = offset_squared * offset
    # φ₀₀ = 2t³ − 3t² + 1
    left_value = 2.0 * offset_cubed - 3.0 * offset_squared + 1.0
    # φ₀₁ = −2t³ + 3t²
    right_value = -2.0 * offset_cubed + 3.0 * offset_squared
    # φ₁₀ hᵢ = (t³ − 2t² + t) hᵢ
    left_slope = (offset_cubed - 2.0 * offset_squared + offset) * width
    # φ₁₁ hᵢ = (t³ − t²) hᵢ
    right_slope = (offset_cubed - offset_squared) * width
    indices = einx.id(
        "target, target, target, target -> target (1 + 1 + 1 + 1)",
        cell,
        next_cell,
        cell + n_nodes,
        next_cell + n_nodes,
    )
    weights = einx.id(
        "target, target, target, target -> target (1 + 1 + 1 + 1)",
        left_value,
        right_value,
        left_slope,
        right_slope,
    )
    row_ok = finite if policies.outside in ("clamp", "extrapolate") else inside
    zero = jnp.zeros((), weights.dtype)
    weights = einx.where("target, target tap, -> target tap", row_ok, weights, zero)
    return Stencil(indices, weights, n_source=2 * n_nodes, valid=row_ok, check=False)
