"""Local one-dimensional bases: the taps a target reads on an axis, and their weights.

A target at cell i and offset t ∈ [0, 1) reads taps i + r for the basis'
relative taps r, with weights that depend on t. Nearest reads one tap, linear
two; the cubic and B-spline bases of #27 follow the same pattern.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import abc
import math
from typing import ClassVar

import einx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from functax.exceptions import ValidationError


__all__ = [
    "BSplineBasis",
    "Basis",
    "CubicBasis",
    "LinearBasis",
    "NearestBasis",
    "bspline",
]


class Basis(eqx.Module):
    """A local basis on one axis: S relative taps, and weights from the offsets.

    ``support`` is (r_min, r_max), the range of relative taps for offsets in
    [0, 1); `outside="extrapolate"` keeps the cell within
    [−r_min, N − 1 − r_max], so every tap stays on the axis.

    Examples:
        Every local basis implements the protocol; linear interpolation reads
        two taps, i and i + 1:

        >>> from functax.interp import Basis, LinearBasis
        >>> basis = LinearBasis()
        >>> isinstance(basis, Basis), basis.taps, basis.support
        (True, 2, (0, 1))
        >>> basis.prefilter_order is None
        True
    """

    @property
    @abc.abstractmethod
    def taps(self) -> int:
        """S, the number of taps per target on this axis."""

    @property
    @abc.abstractmethod
    def support(self) -> tuple[int, int]:
        """(r_min, r_max), the range of relative taps for offsets in [0, 1)."""

    @property
    def prefilter_order(self) -> int | None:
        """The B-spline order whose prefilter the axis needs, or None (no fit)."""
        return None

    @abc.abstractmethod
    def stencil(
        self, offsets: Float[Array, " target"]
    ) -> tuple[Int[Array, "target tap"], Float[Array, "target tap"]]:
        """The relative taps r and the weights w of each target, from its offset t."""


class NearestBasis(Basis):
    """Nearest neighbour: one tap, the closer of i and i + 1, with weight 1.

    A tie (t = ½) goes to the node of lower coordinate, as in SciPy's
    ``RegularGridInterpolator(method="nearest")``: i on an increasing axis,
    and i + 1 on a decreasing one (``ties_up``). ``map_coordinates(order=0)``
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
        self,
        offsets: Float[Array, " target"],
        *,
        ties_up: bool | Bool[Array, ""] = False,
    ) -> tuple[Int[Array, "target tap"], Float[Array, "target tap"]]:
        """The tap r = 1 when t > ½ (or t ≥ ½ with ``ties_up``), else r = 0.

        Args:
            offsets: t, the offset of each target in its cell.
            ties_up: Whether a tie goes to i + 1, the node of lower
                coordinate on a decreasing axis; it may be traced.
        """
        # r = 1 when t > ½, else 0: ties go to i, unless they go up
        goes_up = jnp.where(ties_up, offsets >= 0.5, offsets > 0.5)
        upper = goes_up.astype(jnp.int32)
        taps = einx.id("target -> target 1", upper)
        # w = 1 on the one tap
        ones = jnp.ones_like(offsets)
        weights = einx.id("target -> target 1", ones)
        return taps, weights


class LinearBasis(Basis):
    """Linear: taps i and i + 1, with weights 1 − t and t.

    Linear interpolation is the posterior mean under a Brownian-motion prior
    (``Linear(prior_variance_rate=σ²)`` adds its variance). Gradients reach
    the target through t.

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
        # r = (0, 1) for every target
        relative = jnp.arange(2, dtype=jnp.int32)
        taps = einx.id("tap -> target tap", relative, target=n_targets)
        # w = (1 − t, t)
        lower_weight = 1.0 - offsets
        weights = einx.id("target, target -> target (1 + 1)", lower_weight, offsets)
        return taps, weights


class CubicBasis(Basis):
    """Keys cubic convolution: taps i − 1 … i + 2, weights W(1 + t) … W(2 − t).

    W(s) = (a + 2)|s|³ − (a + 3)|s|² + 1 for |s| ≤ 1, and
    W(s) = a|s|³ − 5a|s|² + 8a|s| − 4a for 1 < |s| < 2. The weights sum to 1,
    and with a = −½ the scheme reproduces quadratics. It interpolates the
    samples, so it needs no prefilter.

    Args:
        a: The kernel parameter; −½ by default, as in gaussx.

    References:
        Keys, R. (1981). Cubic convolution interpolation for digital image
        processing. IEEE Transactions on Acoustics, Speech, and Signal
        Processing 29(6), 1153-1160.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.interp import CubicBasis
        >>> taps, weights = CubicBasis().stencil(jnp.array([0.0, 0.5]))
        >>> taps.tolist()[0], weights.tolist()
        ([-1, 0, 1, 2], [[0.0, 1.0, 0.0, 0.0], [-0.0625, 0.5625, 0.5625, -0.0625]])
    """

    a: float = eqx.field(static=True, default=-0.5)
    taps: ClassVar[int] = 4
    support: ClassVar[tuple[int, int]] = (-1, 2)

    def stencil(
        self, offsets: Float[Array, " target"]
    ) -> tuple[Int[Array, "target tap"], Float[Array, "target tap"]]:
        n_targets = offsets.shape[0]
        # r = (−1, 0, 1, 2) for every target
        relative = jnp.arange(-1, 3, dtype=jnp.int32)
        taps = einx.id("tap -> target tap", relative, target=n_targets)
        a = self.a
        # s = |u − j| for the taps i + 1, i + 2 and i − 1
        near = 1.0 - offsets
        far = 2.0 - offsets
        left = 1.0 + offsets
        # W₂(s) = a s³ − 5a s² + 8a s − 4a, for the taps 1 < s < 2 away
        w_left = ((a * left - 5.0 * a) * left + 8.0 * a) * left - 4.0 * a
        w_far = ((a * far - 5.0 * a) * far + 8.0 * a) * far - 4.0 * a
        # W₁(s) = (a + 2) s³ − (a + 3) s² + 1, for the taps s ≤ 1 away
        w_here = ((a + 2.0) * offsets - (a + 3.0)) * offsets * offsets + 1.0
        w_near = ((a + 2.0) * near - (a + 3.0)) * near * near + 1.0
        weights = einx.id(
            "target, target, target, target -> target (1 + 1 + 1 + 1)",
            w_left,
            w_here,
            w_near,
            w_far,
        )
        return taps, weights


def _truncated_power(
    values: Float[Array, "*shape"], degree: int
) -> Float[Array, "*shape"]:
    """(z)₊ᵏ: zᵏ for z > 0 and 0 otherwise, with (z)₊⁰ = 1 for z ≥ 0."""
    if degree == 0:
        # (z)₊⁰ = [z ≥ 0]
        non_negative = values >= 0
        return non_negative.astype(values.dtype)
    zero = jnp.zeros((), values.dtype)
    # (z)₊ = max(z, 0)
    positive = jnp.where(values > 0, values, zero)
    # (z)₊ᵏ
    return positive**degree


def bspline(values: Float[Array, "*shape"], order: int) -> Float[Array, "*shape"]:
    """The centred B-spline βᵏ of degree k at x.

    βᵏ(x) = (1/k!) ∑ₗ₌₀^{k+1} (−1)ˡ C(k + 1, l) (x + (k + 1)/2 − l)₊ᵏ,
    supported on |x| < (k + 1)/2, with β⁰ = 1 on [−½, ½).

    Args:
        values: x, where to evaluate, any shape.
        order: k, the degree.

    Returns:
        βᵏ(x), the shape of ``values``.

    Examples:
        The cubic B-spline at the integers is (1/6, 2/3, 1/6):

        >>> import jax.numpy as jnp
        >>> from functax.interp import bspline
        >>> values = bspline(jnp.array([-1.0, 0.0, 1.0]), 3)
        >>> bool(jnp.allclose(values, jnp.array([1.0, 4.0, 1.0]) / 6.0))
        True

        It vanishes from |x| = (k + 1)/2 on:

        >>> bspline(jnp.array([-2.0, 2.0, 3.0]), 3).tolist()
        [0.0, 0.0, 0.0]

        Its integer shifts sum to one (partition of unity), here for the hat β¹:

        >>> shifts = jnp.array([-0.7, 0.3, 1.3])
        >>> bool(jnp.allclose(bspline(shifts, 1).sum(), 1.0))
        True
    """
    half_width = (order + 1) / 2.0
    total = jnp.zeros_like(values)
    for term in range(order + 2):
        # (−1)ˡ C(k + 1, l)
        sign = (-1.0) ** term * math.comb(order + 1, term)
        # x + (k + 1)/2 − l
        shifted = values + half_width - term
        # (x + (k + 1)/2 − l)₊ᵏ
        power = _truncated_power(shifted, order)
        # ∑ₗ (−1)ˡ C(k + 1, l) (x + (k + 1)/2 − l)₊ᵏ
        total = total + sign * power
    # βᵏ(x) = (1/k!) ∑ₗ …
    return total / math.factorial(order)


class BSplineBasis(Basis):
    """The centred B-spline of order k (degree k): k + 1 taps per axis.

    Odd orders read taps i − (k − 1)/2 … i + (k + 1)/2 around the cell; even
    orders centre on the nearest node, rounding a tie up as
    ``scipy.ndimage.map_coordinates`` does. Orders 0 and 1 are nearest and
    linear; from order 2 the samples go through the prefilter first.

    Args:
        order: k, from 0 to 5.

    Raises:
        ValidationError: If ``order`` is not an integer in [0, 5].

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.interp import BSplineBasis
        >>> taps, weights = BSplineBasis(3).stencil(jnp.array([0.0]))
        >>> taps.tolist()
        [[-1, 0, 1, 2]]
        >>> bool(jnp.allclose(weights, jnp.array([[1.0, 4.0, 1.0, 0.0]]) / 6.0))
        True

        Even orders centre on the nearest node, so a tie rounds up:

        >>> BSplineBasis(2).stencil(jnp.array([0.25, 0.5]))[0].tolist()
        [[-1, 0, 1], [0, 1, 2]]

        An order past 5 is refused:

        >>> BSplineBasis(6)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        functax.exceptions.ValidationError: 'order' must be an integer in [0, 5]
    """

    order: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        if (
            isinstance(self.order, bool)
            or not isinstance(self.order, int)
            or not 0 <= self.order <= 5
        ):
            msg = f"'order' must be an integer in [0, 5], got {self.order!r}"
            raise ValidationError(msg)

    @property
    def taps(self) -> int:
        return self.order + 1

    @property
    def support(self) -> tuple[int, int]:
        if self.order % 2:
            return (-(self.order - 1) // 2, (self.order + 1) // 2)
        return (-(self.order // 2), self.order // 2 + 1)

    @property
    def prefilter_order(self) -> int | None:
        return self.order if self.order >= 2 else None

    def stencil(
        self, offsets: Float[Array, " target"]
    ) -> tuple[Int[Array, "target tap"], Float[Array, "target tap"]]:
        relative = jnp.arange(self.order + 1, dtype=jnp.int32)
        if self.order % 2:
            # odd: the first tap is (k − 1)/2 left of the cell
            first = jnp.full(offsets.shape, -((self.order - 1) // 2), jnp.int32)
        else:
            # even: centred on the nearest node, a tie (t = ½) rounding up
            first = (offsets >= 0.5).astype(jnp.int32) - self.order // 2
        # r = r₀ + (0, …, k)
        taps = einx.add("target, tap -> target tap", first, relative)
        float_taps = taps.astype(offsets.dtype)
        # x = u − j = t − r for each tap r
        distance = einx.subtract(
            "target, target tap -> target tap", offsets, float_taps
        )
        # w = βᵏ(t − r)
        weights = bspline(distance, self.order)
        return taps, weights
