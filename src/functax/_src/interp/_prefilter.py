"""The B-spline prefilter: coefficients c with y = B c, by recursive filtering per axis.

A B-spline of order k ≥ 2 does not pass through its samples, so the
coefficients come from inverting the banded matrix B of B-spline values at the
knots. B⁻¹ factors into a gain λ = ∏ₚ (1 − zₚ)(1 − 1/zₚ) and, for each pole
zₚ, a causal and an anticausal first-order recursion,

    c⁺ᵢ = yᵢ + z c⁺ᵢ₋₁  and  c⁻ᵢ = z (c⁻ᵢ₊₁ − c⁺ᵢ),

run left to right and then right to left in ``lax.scan``. The start values
come from the boundary: mirror (whole-sample symmetric, as
``scipy.ndimage``'s ``"mirror"``) or periodic (``"grid-wrap"``). Both passes
are linear in y, so 𝒮ᵀ is their linear transpose.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import math
from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from functax._src.interp._fit import Fit
from functax.exceptions import PolicyError, ValidationError


__all__ = [
    "BandedPrefilter",
    "spline_poles",
]


def spline_poles(order: int) -> tuple[float, ...]:
    """The poles zₚ of the order-k B-spline prefilter, as in ``scipy.ndimage``.

    Order 2 has z = √8 − 3, order 3 z = √3 − 2, and orders 4 and 5 two poles
    each; orders 0 and 1 have none.

    References:
        Unser, M. (1999). Splines: a perfect fit for signal and image
        processing. IEEE Signal Processing Magazine 16(6), 22-38.

    Args:
        order: k, the B-spline order, from 0 to 5.

    Returns:
        The poles, each in (−1, 0).

    Raises:
        ValidationError: If ``order`` is not an integer in [0, 5].

    Examples:
        The cubic B-spline has the one pole √3 − 2:

        >>> import math
        >>> from functax.interp import spline_poles
        >>> (pole,) = spline_poles(3)
        >>> math.isclose(pole, math.sqrt(3.0) - 2.0)
        True

        Orders 4 and 5 have two poles, inside the unit circle; 0 and 1 none:

        >>> [len(spline_poles(k)) for k in range(6)]
        [0, 0, 1, 1, 2, 2]
        >>> all(-1.0 < z < 0.0 for z in spline_poles(5))
        True
    """
    if order in (0, 1):
        return ()
    if order == 2:
        return (math.sqrt(8.0) - 3.0,)
    if order == 3:
        return (math.sqrt(3.0) - 2.0,)
    if order == 4:
        return (
            math.sqrt(664.0 - math.sqrt(438976.0)) + math.sqrt(304.0) - 19.0,
            math.sqrt(664.0 + math.sqrt(438976.0)) - math.sqrt(304.0) - 19.0,
        )
    if order == 5:
        return (
            math.sqrt(67.5 - math.sqrt(4436.25)) + math.sqrt(26.25) - 6.5,
            math.sqrt(67.5 + math.sqrt(4436.25)) - math.sqrt(26.25) - 6.5,
        )
    msg = f"'order' must be an integer in [0, 5], got {order!r}"
    raise ValidationError(msg)


def _causal_start_mirror(
    samples: Float[Array, "*batch node"], pole: float
) -> Float[Array, "*batch"]:
    """c⁺₀ for a whole-sample-symmetric signal of period 2(N − 1)."""
    n_nodes = samples.shape[-1]
    # zᴺ⁻¹
    end_power = pole ** (n_nodes - 1)
    # y₀ + zᴺ⁻¹ y_{N−1}
    total = samples[..., 0] + end_power * samples[..., -1]
    if n_nodes > 2:
        # zⁱ for i = 1, …, N − 2
        exponents = jnp.arange(1, n_nodes - 1)
        powers = jnp.asarray(pole**exponents, dtype=samples.dtype)
        # yᵢ + zᴺ⁻¹ y_{N−1−i}
        interior = samples[..., 1 : n_nodes - 1]
        mirrored = samples[..., n_nodes - 2 : 0 : -1]
        middle = interior + end_power * mirrored
        # ∑ᵢ₌₁^{N−2} zⁱ (yᵢ + zᴺ⁻¹ y_{N−1−i})
        middle_sum = einx.dot("... node, node -> ...", middle, powers)
        total = total + middle_sum
    # ÷ (1 − z^{2(N−1)})
    return total / (1.0 - end_power * end_power)


def _anticausal_start_mirror(
    causal: Float[Array, "*batch node"], pole: float
) -> Float[Array, "*batch"]:
    """c⁻_{N−1} = (z c⁺_{N−2} + c⁺_{N−1}) z / (z² − 1)."""
    # c⁻_{N−1} = (z c⁺_{N−2} + c⁺_{N−1}) z / (z² − 1)
    return (pole * causal[..., -2] + causal[..., -1]) * pole / (pole * pole - 1.0)


def _causal_start_wrap(
    samples: Float[Array, "*batch node"], pole: float
) -> Float[Array, "*batch"]:
    """c⁺₀ = ∑ₖ₌₀^{N−1} zᵏ y_{(−k) mod N} / (1 − zᴺ) for a periodic signal."""
    n_nodes = samples.shape[-1]
    # zᵏ for k = 0, …, N − 1
    exponents = jnp.arange(n_nodes)
    powers = jnp.asarray(pole**exponents, dtype=samples.dtype)
    # y₀, y_{N−1}, y_{N−2}, …, y₁: index (−k) mod N
    backwards_index = jnp.mod(-exponents, n_nodes)
    backwards = samples[..., backwards_index]
    # ∑ₖ zᵏ y_{(−k) mod N}
    total = einx.dot("... node, node -> ...", backwards, powers)
    # ÷ (1 − zᴺ)
    return total / (1.0 - pole**n_nodes)


def _anticausal_start_wrap(
    causal: Float[Array, "*batch node"], pole: float
) -> Float[Array, "*batch"]:
    """c⁻_{N−1} = −∑ₖ₌₀^{N−1} z^{k+1} c⁺_{(N−1+k) mod N} / (1 − zᴺ)."""
    n_nodes = causal.shape[-1]
    # z^{k+1} for k = 0, …, N − 1
    exponents = jnp.arange(1, n_nodes + 1)
    powers = jnp.asarray(pole**exponents, dtype=causal.dtype)
    # c⁺_{N−1}, c⁺₀, c⁺₁, …, c⁺_{N−2}: index (N − 1 + k) mod N
    shifted_index = jnp.arange(n_nodes) + n_nodes - 1
    forwards_index = jnp.mod(shifted_index, n_nodes)
    forwards = causal[..., forwards_index]
    # ∑ₖ z^{k+1} c⁺_{(N−1+k) mod N}
    total = einx.dot("... node, node -> ...", forwards, powers)
    # −… ÷ (1 − zᴺ)
    return -total / (1.0 - pole**n_nodes)


def _scan_causal(
    samples: Float[Array, "*batch node"], start: Float[Array, "*batch"], pole: float
) -> Float[Array, "*batch node"]:
    """c⁺ᵢ = yᵢ + z c⁺ᵢ₋₁ along the last axis, from c⁺₀ = ``start``."""
    leading = einx.id("... node -> node ...", samples)

    def step(
        previous: Float[Array, "*batch"], sample: Float[Array, "*batch"]
    ) -> tuple[Float[Array, "*batch"], Float[Array, "*batch"]]:
        # c⁺ᵢ = yᵢ + z c⁺ᵢ₋₁
        current = sample + pole * previous
        return current, current

    # a recursion along the axis has no einx form
    _, rest = jax.lax.scan(step, start, leading[1:])
    return einx.id("..., node ... -> ... (1 + node)", start, rest)


def _scan_anticausal(
    causal: Float[Array, "*batch node"], start: Float[Array, "*batch"], pole: float
) -> Float[Array, "*batch node"]:
    """c⁻ᵢ = z (c⁻ᵢ₊₁ − c⁺ᵢ) along the last axis, from c⁻_{N−1} = ``start``."""
    leading = einx.id("... node -> node ...", causal)

    def step(
        following: Float[Array, "*batch"], value: Float[Array, "*batch"]
    ) -> tuple[Float[Array, "*batch"], Float[Array, "*batch"]]:
        # c⁻ᵢ = z (c⁻ᵢ₊₁ − c⁺ᵢ)
        current = pole * (following - value)
        return current, current

    _, rest = jax.lax.scan(step, start, leading[:-1], reverse=True)
    return einx.id("node ..., ... -> ... (node + 1)", rest, start)


def _filter_last_axis(
    samples: Float[Array, "*batch node"], order: int, periodic: bool
) -> Float[Array, "*batch node"]:
    """B⁻¹ y along the last axis: the gain, then two passes per pole."""
    poles = spline_poles(order)
    if samples.shape[-1] == 1 or not poles:
        return samples
    # λ = ∏ₚ (1 − zₚ)(1 − 1/zₚ)
    gain = math.prod((1.0 - z) * (1.0 - 1.0 / z) for z in poles)
    # λ y
    coefficients = samples * gain
    for pole in poles:
        if periodic:
            causal_start = _causal_start_wrap(coefficients, pole)
        else:
            causal_start = _causal_start_mirror(coefficients, pole)
        causal = _scan_causal(coefficients, causal_start, pole)
        if periodic:
            anticausal_start = _anticausal_start_wrap(causal, pole)
        else:
            anticausal_start = _anticausal_start_mirror(causal, pole)
        coefficients = _scan_anticausal(causal, anticausal_start, pole)
    return coefficients


class BandedPrefilter(Fit):
    """c = B⁻¹ y per axis: the B-spline prefilter, separable on a grid.

    Algorithm:
        1. Unflatten the samples onto the grid (n₁, …, n_D).
        2. For each axis with an order k ≥ 2, move it last and run
           `B⁻¹` along it: the gain λ, then per pole a causal and an
           anticausal recursion started from the mirror or wrap boundary.
        3. Flatten the coefficients back, row-major.

    The adjoint is the linear transpose of these passes.

    Args:
        orders: The B-spline order of each axis, or ``None`` for an axis read
            without a prefilter (nearest, linear, cubic).
        shape: (n₁, …, n_D), the grid the flat source is laid out on.
        periodic: Per axis, whether the signal wraps (``"grid-wrap"``) rather
            than mirrors (``"mirror"``) at its ends.

    Raises:
        ValidationError: If the three tuples differ in length or an order is
            outside [0, 5].

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.interp import BandedPrefilter
        >>> prefilter = BandedPrefilter((3,), (5,), (False,))
        >>> coefficients = prefilter.apply(jnp.ones(5))
        >>> bool(jnp.allclose(coefficients, 1.0))
        True

        The coefficients reproduce the samples through the B-spline at the
        knots, y = B c, with B = (1/6, 2/3, 1/6) mirrored at the ends:

        >>> samples = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])
        >>> c = prefilter.apply(samples)
        >>> padded = jnp.concatenate([c[1:2], c, c[3:4]])
        >>> rebuilt = (padded[:-2] + 4.0 * padded[1:-1] + padded[2:]) / 6.0
        >>> bool(jnp.allclose(rebuilt, samples))
        True
    """

    orders: tuple[int | None, ...] = eqx.field(static=True, converter=tuple)
    shape: tuple[int, ...] = eqx.field(static=True, converter=tuple)
    periodic: tuple[bool, ...] = eqx.field(static=True, converter=tuple)

    def __check_init__(self) -> None:
        if not len(self.orders) == len(self.shape) == len(self.periodic):
            msg = (
                f"'orders', 'shape' and 'periodic' must have one entry per axis, "
                f"got {len(self.orders)}, {len(self.shape)} and {len(self.periodic)}"
            )
            raise ValidationError(msg)
        for order in self.orders:
            if order is not None:
                spline_poles(order)

    def _axis_names(self) -> list[str]:
        return [f"axis{d}" for d in range(len(self.shape))]

    def apply(
        self, data: Float[Array, "*batch source"]
    ) -> Float[Array, "*batch source"]:
        """c = B⁻¹ y, one axis at a time, over any leading batch dims."""
        names = self._axis_names()
        # Any: ty would check the ints against einx's `backend` keyword
        sizes: dict[str, Any] = dict(zip(names, self.shape, strict=True))
        grid_pattern = " ".join(names)
        coefficients = einx.id(
            f"... ({grid_pattern}) -> ... {grid_pattern}", data, **sizes
        )
        for name, order, wraps in zip(names, self.orders, self.periodic, strict=True):
            if order is None or order < 2:
                continue
            others = " ".join(other for other in names if other != name)
            moved = einx.id(f"... {grid_pattern} -> ... {others} {name}", coefficients)
            # c = B⁻¹ y along this axis
            filtered = _filter_last_axis(moved, order, wraps)
            coefficients = einx.id(
                f"... {others} {name} -> ... {grid_pattern}", filtered
            )
        return einx.id(f"... {grid_pattern} -> ... ({grid_pattern})", coefficients)

    def apply_adjoint(
        self, coefficients: Float[Array, "*batch source"]
    ) -> Float[Array, "*batch source"]:
        """𝒮ᵀ g, the linear transpose of the recursive passes."""
        structure = jax.ShapeDtypeStruct(coefficients.shape, coefficients.dtype)
        (transposed,) = jax.linear_transpose(self.apply, structure)(coefficients)
        return transposed


def prefilter_for(
    orders: tuple[int | None, ...],
    shape: tuple[int, ...],
    periodic: tuple[bool, ...],
    boundary: str,
) -> BandedPrefilter | None:
    """The prefilter a tensor plan needs, or None when no axis needs one.

    Raises:
        PolicyError: If an axis needs a prefilter and its boundary is neither
            ``"mirror"`` nor a wrap (``boundary="wrap"`` or a periodic axis).
    """
    if all(order is None for order in orders):
        return None
    wraps = []
    for order, is_periodic in zip(orders, periodic, strict=True):
        axis_wraps = is_periodic or boundary == "wrap"
        if order is not None and not axis_wraps and boundary != "mirror":
            msg = (
                "a B-spline of order ≥ 2 is prefiltered with the mirror or the "
                f"wrap boundary, got boundary={boundary!r}"
            )
            raise PolicyError(msg)
        wraps.append(axis_wraps)
    return BandedPrefilter(orders, shape, tuple(wraps))
