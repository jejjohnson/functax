"""One-dimensional quadrature rules as data: `Rule` and its four constructors."""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import math
import operator
from collections.abc import Iterator
from typing import Any

import einx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from functax._src._dtypes import promote_integers
from functax.exceptions import ValidationError


__all__ = [
    "Rule",
    "clenshaw_curtis",
    "gauss_legendre",
    "simpson",
    "trapezoid",
]


class Rule(eqx.Module):
    """A quadrature rule as data: nodes xₙ and weights wₙ, ∫ g dx ≈ ∑ₙ wₙ g(xₙ).

    A rule has no ``integrate`` method: integrix sums, scans and adapts with
    it (integrix design decision 3). Unpacking gives ``(nodes, weights)``, so
    ``nodes, weights = gauss_legendre(n)`` keeps working.

    Attributes:
        nodes: The nodes xₙ.
        weights: The weights wₙ, one per node.

    Raises:
        ValidationError: If ``weights`` is not 1-D or ``nodes`` does not have
            one entry per weight along its leading axis.

    Examples:
        >>> import jax.numpy as jnp
        >>> rule = Rule(jnp.array([0.0, 1.0]), jnp.array([0.5, 0.5]))
        >>> nodes, weights = rule
        >>> weights.shape
        (2,)
    """

    nodes: Float[Array | np.ndarray, " node *dim"]
    weights: Float[Array | np.ndarray, " node"]

    def __check_init__(self) -> None:
        if self.weights.ndim != 1:
            msg = f"'weights' must be 1-D, got shape {self.weights.shape}"
            raise ValidationError(msg)
        if self.nodes.ndim == 0 or self.nodes.shape[0] != self.weights.shape[0]:
            msg = (
                f"'nodes' must have one entry per weight along its leading axis, "
                f"got shape {self.nodes.shape} for {self.weights.shape[0]} weights"
            )
            raise ValidationError(msg)

    def __iter__(self) -> Iterator[Any]:
        # (nodes, weights): the tuple gauss_legendre returned before Rule existed
        return iter((self.nodes, self.weights))


def gauss_legendre(n: int, interval: tuple[float, float] = (-1.0, 1.0)) -> Rule:
    """The n-point Gauss–Legendre rule on an interval [a, b].

    ∫ₐᵇ p(x) dx = ∑ₙ wₙ p(xₙ) exactly for every polynomial p of degree at
    most 2n − 1, with xₙ = (a + b)/2 + (b − a)/2 ξₙ and wₙ = (b − a)/2 ωₙ
    from the rule (ξₙ, ωₙ) on [−1, 1]. The rule is a host-side constant
    (NumPy's ``leggauss``, float64), as in geonnax, so it can size static
    arrays.

    Algorithm:
        1. The nodes ξₙ and weights ωₙ on [−1, 1] from ``leggauss``.
        2. Map them affinely onto [a, b].

    Args:
        n: The number of nodes, at least 1.
        interval: The interval (a, b); a > b gives the negated rule, as
            ∫ₐᵇ = −∫ᵇₐ.

    Returns:
        A `Rule` with float64 NumPy nodes ascending from a to b and weights.

    Raises:
        ValidationError: If ``n`` is not an integer of at least 1 or the
            interval is not two finite numbers.

    Examples:
        >>> from functax.quad import gauss_legendre
        >>> nodes, weights = gauss_legendre(3)
        >>> nodes.shape, bool(abs(weights.sum() - 2.0) < 1e-14)
        ((3,), True)
        >>> # exact up to degree 5: ∫ t³ + t² dt = 2/3
        >>> bool(abs(weights @ (nodes**3 + nodes**2) - 2 / 3) < 1e-14)
        True
        >>> rule = gauss_legendre(4, interval=(0.0, 2.0))
        >>> bool(abs(rule.weights @ rule.nodes**7 - 2.0**8 / 8) < 1e-12)
        True
    """
    count = _node_count(n, smallest=1)
    lower, upper = _interval(interval)
    reference_nodes, reference_weights = np.polynomial.legendre.leggauss(count)
    return _mapped(reference_nodes, reference_weights, lower, upper)


def clenshaw_curtis(n: int, interval: tuple[float, float] = (-1.0, 1.0)) -> Rule:
    """The n-point Clenshaw–Curtis rule on [a, b], at the Chebyshev extreme points.

    With N = n − 1, the nodes are ξₖ = −cos(kπ/N), k = 0, …, N (ascending),
    and the weights (Waldvogel 2006, eq. 2.8) are

        ωₖ = (cₖ / N) [1 − ∑ⱼ₌₁^⌊(N−1)/2⌋ 2 cos(2jkπ/N) / (4j² − 1)
                         − cos(kπ) / (N² − 1) if N is even],

    with cₖ = 1 at the ends and 2 inside, mapped onto [a, b] as in
    `gauss_legendre`. Exact for polynomials of degree at most N (N + 1 when
    N is even). Built on the host in float64.

    Algorithm:
        1. ξₖ = −cos(kπ/N).
        2. The cosine series of ωₖ, with its even-N term.
        3. Map nodes and weights onto [a, b].

    Args:
        n: The number of nodes, at least 2.
        interval: The interval (a, b), as in `gauss_legendre`.

    Returns:
        A `Rule` with float64 NumPy nodes ascending from a to b and weights.

    Raises:
        ValidationError: If ``n`` is not an integer of at least 2 or the
            interval is not two finite numbers.

    Examples:
        >>> from functax.quad import clenshaw_curtis
        >>> rule = clenshaw_curtis(5)
        >>> rule.nodes.shape, bool(abs(rule.weights.sum() - 2.0) < 1e-14)
        ((5,), True)
        >>> # N = 4 is even, so degree 5 is exact: ∫₋₁¹ x⁴ dx = 2/5
        >>> bool(abs(rule.weights @ rule.nodes**4 - 0.4) < 1e-14)
        True

    References:
        Waldvogel, J. (2006). Fast construction of the Fejér and
        Clenshaw–Curtis quadrature rules. BIT 46, 195–202.
    """
    count = _node_count(n, smallest=2)
    lower, upper = _interval(interval)
    degree = count - 1
    index = np.arange(count, dtype=np.float64)
    # ξₖ = −cos(kπ/N), ascending
    reference_nodes = -np.cos(np.pi * index / degree)
    # j = 1, …, ⌊(N − 1)/2⌋, empty when N ≤ 2
    frequencies = np.arange(1, (degree - 1) // 2 + 1, dtype=np.float64)
    series = np.zeros(count)
    if frequencies.size:
        # 2 / (4j² − 1)
        coefficients = 2.0 / (4.0 * frequencies**2 - 1.0)
        # 2jkπ/N, one row per node k
        angles = einx.multiply("k, j -> k j", index, 2.0 * np.pi * frequencies / degree)
        # ∑ⱼ 2 cos(2jkπ/N) / (4j² − 1)
        series = einx.dot("k j, j -> k", np.cos(angles), coefficients)
    if degree % 2 == 0:
        # the j = N/2 term: cos(kπ) / (N² − 1)
        series = series + np.cos(np.pi * index) / (degree**2 - 1.0)
    # cₖ = 1 at the ends, 2 inside
    end_factors = np.full(count, 2.0)
    end_factors[[0, -1]] = 1.0
    # ωₖ = (cₖ / N)(1 − series)
    reference_weights = end_factors * (1.0 - series) / degree
    return _mapped(reference_nodes, reference_weights, lower, upper)


def trapezoid(nodes: ArrayLike) -> Rule:
    """The composite trapezoid rule on given nodes, uniform or not.

    w₀ = (x₁ − x₀)/2, wₙ = (xₙ₊₁ − xₙ₋₁)/2, w_N = (x_N − x_{N−1})/2, so
    ∑ₙ wₙ g(xₙ) equals ``scipy.integrate.trapezoid(g, x)``. Descending nodes
    give negated weights, as ∫ₐᵇ = −∫ᵇₐ. Traceable: the weights are
    differentiable in the nodes.

    Algorithm:
        1. The interval widths hₙ = xₙ₊₁ − xₙ.
        2. Each node takes half of each interval it bounds.

    Args:
        nodes: The nodes x₀, …, x_N, a 1-D array of at least one entry
            (a single node gets weight 0). Integers become floats.

    Returns:
        A `Rule` on the nodes, in their floating dtype.

    Raises:
        ValidationError: If ``nodes`` is not a non-empty 1-D array.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.quad import trapezoid
        >>> rule = trapezoid(jnp.array([0.0, 1.0, 3.0]))
        >>> bool(jnp.allclose(rule.weights, jnp.array([0.5, 1.5, 1.0])))
        True
        >>> trapezoid(jnp.array([2.0])).weights.tolist()
        [0.0]
    """
    nodes = _node_vector(nodes)
    # hₙ = xₙ₊₁ − xₙ
    widths = jnp.diff(nodes)
    zero = jnp.zeros(1, dtype=nodes.dtype)
    # the right half of the interval each node starts
    right = jnp.concatenate([widths, zero])
    # the left half of the interval each node ends
    left = jnp.concatenate([zero, widths])
    # wₙ = (hₙ + hₙ₋₁)/2
    weights = 0.5 * (right + left)
    return Rule(nodes, weights)


def simpson(nodes: ArrayLike) -> Rule:
    """The composite Simpson rule on given nodes, as ``scipy.integrate.simpson``.

    On each pair of intervals (h₀, h₁) the non-uniform Simpson weights are
    (h₀ + h₁)/6 · (2 − h₁/h₀, (h₀ + h₁)²/(h₀h₁), 2 − h₀/h₁). An odd node
    count is all pairs; an even count ≥ 4 takes pairs over all but the last
    interval and adds Cartwright's correction for it, α f_N + β f_{N−1} −
    η f_{N−2}, as SciPy ≥ 1.11 does; two nodes give the trapezoid rule.
    Traceable, like `trapezoid`.

    Algorithm:
        1. The interval widths hₙ.
        2. Scatter the pair weights onto their three nodes.
        3. For an even node count, add the last-interval correction (or, with
           two nodes, use the trapezoid weights).

    Args:
        nodes: The nodes, a 1-D array of at least one entry. Integers become
            floats.

    Returns:
        A `Rule` on the nodes, in their floating dtype.

    Raises:
        ValidationError: If ``nodes`` is not a non-empty 1-D array.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.quad import simpson
        >>> rule = simpson(jnp.array([0.0, 0.5, 1.0]))
        >>> bool(jnp.allclose(rule.weights, jnp.array([1.0, 4.0, 1.0]) / 6.0))
        True
        >>> # exact for quadratics on uneven nodes: ∫₀¹ x² dx = 1/3
        >>> rule = simpson(jnp.array([0.0, 0.3, 1.0]))
        >>> bool(jnp.allclose(rule.weights @ rule.nodes**2, 1.0 / 3.0))
        True

    References:
        Cartwright, K. V. (2017). Simpson's rule cumulative integration with
        MS Excel and irregularly-spaced data. J. Math. Sci. Math. Educ. 12(2).
    """
    nodes = _node_vector(nodes)
    count = nodes.shape[0]
    if count == 2:
        return trapezoid(nodes)
    weights = jnp.zeros(count, dtype=nodes.dtype)
    if count < 3:
        return Rule(nodes, weights)
    # hₙ = xₙ₊₁ − xₙ
    widths = jnp.diff(nodes)
    # pairs of intervals (h₀, h₁) starting at even nodes; an even count leaves
    # the last interval to the correction
    pair_count = (count - 1) // 2
    starts = 2 * np.arange(pair_count)
    first = widths[starts]
    second = widths[starts + 1]
    # (h₀ + h₁)/6
    scale = (first + second) / 6.0
    # left node: (h₀ + h₁)/6 · (2 − h₁/h₀)
    weights = weights.at[starts].add(scale * (2.0 - second / first))
    # middle node: (h₀ + h₁)/6 · (h₀ + h₁)²/(h₀h₁)
    middle = scale * (first + second) ** 2 / (first * second)
    weights = weights.at[starts + 1].add(middle)
    # right node: (h₀ + h₁)/6 · (2 − h₀/h₁)
    weights = weights.at[starts + 2].add(scale * (2.0 - first / second))
    if count % 2 == 0:
        # Cartwright's correction for the last interval, on its two widths
        before, last = widths[-2], widths[-1]
        # α = (2h₁² + 3h₀h₁) / (6(h₀ + h₁))
        alpha = (2.0 * last**2 + 3.0 * before * last) / (6.0 * (last + before))
        # β = (h₁² + 3h₀h₁) / (6h₀)
        beta = (last**2 + 3.0 * before * last) / (6.0 * before)
        # η = h₁³ / (6h₀(h₀ + h₁))
        eta = last**3 / (6.0 * before * (before + last))
        weights = weights.at[-1].add(alpha)
        weights = weights.at[-2].add(beta)
        weights = weights.at[-3].add(-eta)
    return Rule(nodes, weights)


def _node_count(n: Any, *, smallest: int) -> int:
    """``n`` as a Python int of at least ``smallest``, or `ValidationError`."""
    message = f"'n' must be an integer of at least {smallest}, got {n!r}"
    if isinstance(n, bool):
        raise ValidationError(message)
    try:
        count = operator.index(n)
    except TypeError as err:
        raise ValidationError(message) from err
    if count < smallest:
        raise ValidationError(message)
    return count


def _interval(interval: Any) -> tuple[float, float]:
    """The two ends of an interval as finite Python floats."""
    try:
        lower, upper = (float(end) for end in interval)
    except (TypeError, ValueError) as err:
        msg = f"'interval' must be two finite numbers (a, b), got {interval!r}"
        raise ValidationError(msg) from err
    if not (math.isfinite(lower) and math.isfinite(upper)):
        msg = f"'interval' must be two finite numbers (a, b), got {interval!r}"
        raise ValidationError(msg)
    return lower, upper


def _mapped(
    reference_nodes: np.ndarray,
    reference_weights: np.ndarray,
    lower: float,
    upper: float,
) -> Rule:
    """Map a rule on [−1, 1] onto [a, b]: x = c + hξ, w = hω."""
    # c = (a + b)/2, h = (b − a)/2
    centre = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    nodes = centre + half_width * reference_nodes
    weights = half_width * reference_weights
    return Rule(nodes, weights)


def _node_vector(nodes: ArrayLike) -> Float[Array, " node"]:
    """Nodes as a non-empty 1-D floating JAX array."""
    array = promote_integers(jnp.asarray(nodes))
    if array.ndim != 1 or array.shape[0] == 0:
        msg = f"'nodes' must be a non-empty 1-D array, got shape {array.shape}"
        raise ValidationError(msg)
    return array
