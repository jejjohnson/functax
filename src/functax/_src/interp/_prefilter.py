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

    Examples:
        >>> from functax.interp import spline_poles
        >>> [round(z, 12) for z in spline_poles(3)]
        [-0.267949192431]
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


def _causal_start_mirror(samples: jax.Array, pole: float) -> jax.Array:
    """c⁺₀ for a whole-sample-symmetric signal of period 2(N − 1)."""
    n = samples.shape[-1]
    # zᴺ⁻¹
    end_power = pole ** (n - 1)
    total = samples[..., 0] + end_power * samples[..., -1]
    if n > 2:
        # ∑ᵢ₌₁^{N−2} zⁱ (yᵢ + zᴺ⁻¹ y_{N−1−i})
        powers = jnp.asarray(pole ** jnp.arange(1, n - 1), dtype=samples.dtype)
        middle = samples[..., 1 : n - 1] + end_power * samples[..., n - 2 : 0 : -1]
        total = total + einx.dot("... i, i -> ...", middle, powers)
    # ÷ (1 − z^{2(N−1)})
    return total / (1.0 - end_power * end_power)


def _anticausal_start_mirror(causal: jax.Array, pole: float) -> jax.Array:
    """c⁻_{N−1} = (z c⁺_{N−2} + c⁺_{N−1}) z / (z² − 1)."""
    return (pole * causal[..., -2] + causal[..., -1]) * pole / (pole * pole - 1.0)


def _causal_start_wrap(samples: jax.Array, pole: float) -> jax.Array:
    """c⁺₀ = ∑ₖ₌₀^{N−1} zᵏ y_{(−k) mod N} / (1 − zᴺ) for a periodic signal."""
    n = samples.shape[-1]
    powers = jnp.asarray(pole ** jnp.arange(n), dtype=samples.dtype)
    # y₀, y_{N−1}, y_{N−2}, …, y₁: index (−k) mod N
    backwards = samples[..., jnp.mod(-jnp.arange(n), n)]
    total = einx.dot("... i, i -> ...", backwards, powers)
    return total / (1.0 - pole**n)


def _anticausal_start_wrap(causal: jax.Array, pole: float) -> jax.Array:
    """c⁻_{N−1} = −∑ₖ₌₀^{N−1} z^{k+1} c⁺_{(N−1+k) mod N} / (1 − zᴺ)."""
    n = causal.shape[-1]
    powers = jnp.asarray(pole ** jnp.arange(1, n + 1), dtype=causal.dtype)
    # c⁺_{N−1}, c⁺₀, c⁺₁, …, c⁺_{N−2}: index (N − 1 + k) mod N
    forwards = causal[..., jnp.mod(jnp.arange(n) + n - 1, n)]
    total = einx.dot("... i, i -> ...", forwards, powers)
    return -total / (1.0 - pole**n)


def _scan_causal(samples: jax.Array, start: jax.Array, pole: float) -> jax.Array:
    """c⁺ᵢ = yᵢ + z c⁺ᵢ₋₁ along the last axis, from c⁺₀ = ``start``."""
    leading = einx.id("... n -> n ...", samples)

    def step(previous: jax.Array, sample: jax.Array) -> tuple[jax.Array, jax.Array]:
        current = sample + pole * previous
        return current, current

    _, rest = jax.lax.scan(step, start, leading[1:])
    return einx.id("..., n ... -> ... (1 + n)", start, rest)


def _scan_anticausal(causal: jax.Array, start: jax.Array, pole: float) -> jax.Array:
    """c⁻ᵢ = z (c⁻ᵢ₊₁ − c⁺ᵢ) along the last axis, from c⁻_{N−1} = ``start``."""
    leading = einx.id("... n -> n ...", causal)

    def step(following: jax.Array, value: jax.Array) -> tuple[jax.Array, jax.Array]:
        current = pole * (following - value)
        return current, current

    _, rest = jax.lax.scan(step, start, leading[:-1], reverse=True)
    return einx.id("n ..., ... -> ... (n + 1)", rest, start)


def _filter_last_axis(samples: jax.Array, order: int, periodic: bool) -> jax.Array:
    """B⁻¹ y along the last axis: the gain, then two passes per pole."""
    poles = spline_poles(order)
    if samples.shape[-1] == 1 or not poles:
        return samples
    # λ = ∏ₚ (1 − zₚ)(1 − 1/zₚ)
    gain = math.prod((1.0 - z) * (1.0 - 1.0 / z) for z in poles)
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
        for d, (order, wraps) in enumerate(
            zip(self.orders, self.periodic, strict=True)
        ):
            if order is None or order < 2:
                continue
            others = " ".join(name for k, name in enumerate(names) if k != d)
            moved = einx.id(
                f"... {grid_pattern} -> ... {others} {names[d]}", coefficients
            )
            filtered = _filter_last_axis(moved, order, wraps)
            coefficients = einx.id(
                f"... {others} {names[d]} -> ... {grid_pattern}", filtered
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
