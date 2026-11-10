"""Quadrature rules on curves: `CurveRule` and `polyline`, arc-length weights with
the unit normal at each node, so a line integral or a flux across a transect is
a rule like any other.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

from typing import Any

import einx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float, Real

from functax._src._dtypes import promote_integers
from functax._src.geometry._sphere import _safe_norm
from functax._src.quad._rules import Rule, _node_count, gauss_legendre
from functax.exceptions import ValidationError


__all__ = [
    "CurveRule",
    "polyline",
]


_SEGMENT_RULES: tuple[str, ...] = ("gauss_legendre", "midpoint")


class CurveRule(Rule):
    """A rule on a curve L: ∫_L c dℓ ≈ ∑ₖ Δsₖ c(xₖ), with the unit normal nₖ.

    A `Rule` whose weights are arc lengths Δsₖ, so integrix sums with it like
    any other rule and ``nodes, weights = rule`` still unpacks. The normals
    turn it into a flux across the curve, Φ = ∑ₖ Δsₖ (uₖ · nₖ) for a vector
    field u. A curve in 2-D has one normal per node, the unit tangent
    rotated by +90° (to the left of the direction of travel); in any other
    dimension a curve has no single normal, and ``normals`` is ``None``.

    Attributes:
        nodes: The nodes xₖ on the curve, one row of D coordinates per node.
        weights: The arc-length weights Δsₖ.
        normals: The unit normals nₖ, (K, 2), or ``None`` when D ≠ 2.

    Raises:
        ValidationError: As `Rule`, or if ``normals`` is given and does not
            have the shape of 2-D ``nodes``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.quad import polyline
        >>> # northward along x = 0: the normal points west, to −x
        >>> rule = polyline(jnp.array([[0.0, 0.0], [0.0, 2.0]]), 2)
        >>> nodes, weights = rule
        >>> weights.tolist(), rule.normals.tolist()
        ([1.0, 1.0], [[-1.0, 0.0], [-1.0, 0.0]])
    """

    normals: Float[Array, " node dim"] | None = None

    def __check_init__(self) -> None:
        if self.normals is None:
            return
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            msg = (
                f"'normals' are defined for 2-D nodes only, got nodes of shape "
                f"{self.nodes.shape}"
            )
            raise ValidationError(msg)
        if self.normals.shape != self.nodes.shape:
            msg = (
                f"'normals' must have the shape of 'nodes' {self.nodes.shape}, "
                f"got {self.normals.shape}"
            )
            raise ValidationError(msg)


def polyline(
    vertices: Real[ArrayLike, "vertex dim"],
    n: int,
    *,
    rule: str = "midpoint",
) -> CurveRule:
    """A rule along a polyline: n nodes per segment, arc-length weights and normals.

    Segment j, from vⱼ to vⱼ₊₁, has length ℓⱼ = ‖vⱼ₊₁ − vⱼ‖ and unit tangent
    tⱼ = (vⱼ₊₁ − vⱼ)/ℓⱼ. Its n nodes are xₖ = vⱼ + τᵢ (vⱼ₊₁ − vⱼ) with
    weights Δsₖ = ℓⱼ ωᵢ, from a rule (τᵢ, ωᵢ) on [0, 1]:

    - ``"midpoint"``: τᵢ = (i + ½)/n and ωᵢ = 1/n, the composite midpoint
      rule, exact for linear c along each segment;
    - ``"gauss_legendre"``: the n-point Gauss–Legendre rule, exact for c
      polynomial along each segment up to degree 2n − 1.

    So ∑ₖ Δsₖ = ∑ⱼ ℓⱼ, the polyline's length. In 2-D every node of segment j
    carries the normal nⱼ = R₉₀ tⱼ = (−tⱼ,y, tⱼ,x), so the vertices' order
    sets its sign: reverse them to flip a flux. Each segment gets the same n,
    so K = n (V − 1) is static and the rule traces and differentiates in the
    vertices. A zero-length segment has zero weights and a zero normal.

    Algorithm:
        1. The rule (τᵢ, ωᵢ) on [0, 1], a host-side constant.
        2. The edge vectors vⱼ₊₁ − vⱼ, their lengths ℓⱼ and unit tangents tⱼ.
        3. Nodes vⱼ + τᵢ (vⱼ₊₁ − vⱼ) and weights ℓⱼ ωᵢ, segments outer; in
           2-D the normals R₉₀ tⱼ, repeated over each segment's nodes.

    Args:
        vertices: The V vertices v₀, …, v_{V−1}, at least two, of D
            coordinates each. Integers become floats.
        n: The number of nodes per segment, at least 1.
        rule: The rule on each segment, ``"midpoint"`` or
            ``"gauss_legendre"``.

    Returns:
        A `CurveRule` of K = n (V − 1) nodes (K, D), weights (K,) and, in 2-D,
        normals (K, 2), in the vertices' floating dtype.

    Raises:
        ValidationError: If ``vertices`` is not (V, D) with V ≥ 2 and D ≥ 1,
            ``n`` is not an integer of at least 1, or ``rule`` is unknown.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.quad import polyline
        >>> # an L-shaped transect of length 3 + 4
        >>> vertices = jnp.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
        >>> rule = polyline(vertices, 4, rule="gauss_legendre")
        >>> rule.nodes.shape, bool(jnp.isclose(rule.weights.sum(), 7.0))
        ((8, 2), True)
        >>> # the normal of the second segment points to −x, left of +y
        >>> rule.normals[-1].tolist()
        [-1.0, 0.0]
        >>> # a line of sight in 3-D has no single normal
        >>> sight = polyline(jnp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]]), 3)
        >>> sight.normals is None, bool(jnp.isclose(sight.weights.sum(), 3.0))
        (True, True)
    """
    count = _node_count(n, smallest=1)
    if rule not in _SEGMENT_RULES:
        msg = f"'rule' must be one of {list(_SEGMENT_RULES)}, got {rule!r}"
        raise ValidationError(msg)
    points = promote_integers(jnp.asarray(vertices))
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 1:
        msg = (
            "'vertices' must be (V, D) with at least two vertices and one "
            f"coordinate, got shape {points.shape}"
        )
        raise ValidationError(msg)
    dtype = points.dtype

    # --- the rule on [0, 1] ---
    offsets, fractions = _unit_rule(count, rule)
    offsets = jnp.asarray(offsets, dtype=dtype)
    fractions = jnp.asarray(fractions, dtype=dtype)

    # --- segments ---
    # vⱼ₊₁ − vⱼ
    edges = points[1:] - points[:-1]
    # ℓⱼ = ‖vⱼ₊₁ − vⱼ‖, with a zero gradient at a zero-length segment
    squared_lengths = einx.sum("segment [dim]", edges**2)
    lengths = _safe_norm(squared_lengths)
    # tⱼ = (vⱼ₊₁ − vⱼ)/ℓⱼ, and 0 on a zero-length segment
    safe_lengths = jnp.where(lengths > 0, lengths, 1.0)
    tangents = einx.divide("segment dim, segment -> segment dim", edges, safe_lengths)

    # --- nodes and weights, segments outer ---
    # xₖ = vⱼ + τᵢ (vⱼ₊₁ − vⱼ)
    steps = einx.multiply("segment dim, node -> segment node dim", edges, offsets)
    nodes = einx.add(
        "segment dim, segment node dim -> (segment node) dim", points[:-1], steps
    )
    # Δsₖ = ℓⱼ ωᵢ
    weights = einx.multiply("segment, node -> (segment node)", lengths, fractions)

    normals = None
    if points.shape[1] == 2:
        # nⱼ = R₉₀ tⱼ = (−t_y, t_x)
        tangent_x = tangents[:, 0]
        tangent_y = tangents[:, 1]
        segment_normals = einx.id(
            "segment, segment -> segment (1 + 1)", -tangent_y, tangent_x
        )
        normals = einx.id(
            "segment dim -> (segment node) dim", segment_normals, node=count
        )
    return CurveRule(nodes, weights, normals)


def _unit_rule(count: int, rule: Any) -> tuple[np.ndarray, np.ndarray]:
    """The nodes τᵢ and weights ωᵢ of a segment rule on [0, 1], float64."""
    if rule == "gauss_legendre":
        reference = gauss_legendre(count, interval=(0.0, 1.0))
        return np.asarray(reference.nodes), np.asarray(reference.weights)
    # τᵢ = (i + ½)/n, ωᵢ = 1/n
    offsets = (np.arange(count, dtype=np.float64) + 0.5) / count
    fractions = np.full(count, 1.0 / count)
    return offsets, fractions
