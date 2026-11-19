"""Pairwise distances between point sets, in the metric the point sets carry."""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

from collections.abc import Callable

import einx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Shaped

from functax._src.geometry._sphere import (
    _check_radius,
    _check_unit,
    _safe_norm,
    chord_distance,
    geodesic_distance,
    lonlat_to_unit,
)
from functax._src.geometry._types import Points
from functax.exceptions import GeometryError


__all__ = [
    "pairwise_distance",
]


# M · N above which the targets are processed in chunks, so the (M, N, 3)
# intermediates of the sphere metrics stay near 2²² entries per chunk.
MAX_PAIRS = 2**22


def _euclidean_distance(
    u: Float[Array, "source dim"], v: Float[Array, "target dim"]
) -> Float[Array, "source target"]:
    """‖u − v‖₂ for every pair, guarded at u = v."""
    # u − v for every pair
    difference = einx.subtract("source dim, target dim -> source target dim", u, v)
    # (u − v)²
    squared_difference = difference**2
    # ‖u − v‖₂² = ∑_d (u_d − v_d)²
    squared = einx.sum("source target [dim]", squared_difference)
    # ‖u − v‖₂, with a finite gradient at u = v
    return _safe_norm(squared)


def _chunked(
    distance: Callable[
        [Shaped[Array, "source dim"], Shaped[Array, "block dim"]],
        Shaped[Array, "row block"],
    ],
    sources: Shaped[Array, "source dim"],
    targets: Shaped[Array, "target dim"],
    max_pairs: int,
) -> Shaped[Array, "row target"]:
    """``distance(sources, targets)``, over chunks of targets when M · N > max_pairs.

    The leading dim of ``distance``'s output is the sources for a distance
    matrix, or the taps of a nearest-neighbour search.
    """
    n_sources = sources.shape[0]
    n_targets = targets.shape[0]
    if n_sources * n_targets <= max_pairs:
        return distance(sources, targets)
    chunk_size = max(1, max_pairs // max(n_sources, 1))
    # ⌈N / chunk⌉
    n_chunks = -(-n_targets // chunk_size)
    # pad the targets with copies of the last one, so every chunk is full
    padding = n_chunks * chunk_size - n_targets
    padded = jnp.pad(targets, ((0, padding), (0, 0)), mode="edge")
    chunks = einx.id("(chunk block) dim -> chunk block dim", padded, chunk=n_chunks)
    blocks = jax.lax.map(lambda block: distance(sources, block), chunks)
    distances = einx.id("chunk row block -> row (chunk block)", blocks)
    # drop the padded targets
    return distances[:, :n_targets]


def pairwise_distance(
    sources: Points,
    targets: Points,
    *,
    radius: float | Float[Array, ""] = 1.0,
    unit: str = "radians",
    max_pairs: int = MAX_PAIRS,
) -> Float[Array, "source target"]:
    """The distance from every source to every target, in their shared metric.

    ``"euclidean"`` measures ‖x − x′‖₂ between a source x and a target x′ on
    ``xy`` as given, in any number of dimensions. ``"chord"`` and
    ``"geodesic"`` read ``xy`` as longitude and latitude (λ, φ) in ``unit``,
    lift both sets to unit vectors and return `chord_distance` or
    `geodesic_distance` at ``radius``. The vertical coordinate ``z`` is not
    used. A pair with a padding point (``valid`` False) gets +∞, so padding is
    never anyone's neighbour.

    Args:
        sources: The M source points.
        targets: The N target points, with the same metric as ``sources``.
        radius: R, the sphere radius, for the sphere metrics; ignored for
            ``"euclidean"``.
        unit: ``"radians"`` or ``"degrees"``, for the sphere metrics.
        max_pairs: The M · N above which the targets are processed in chunks
            with ``lax.map``, which bounds the memory of the intermediates.

    Returns:
        The M × N distances.

    Raises:
        GeometryError: If the two metrics differ, a sphere metric meets ``xy``
            that is not (λ, φ), or ``unit`` or a concrete ``radius`` is bad.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Points, pairwise_distance
        >>> grid = Points(jnp.array([[0.0, 0.0], [3.0, 4.0]]))
        >>> pairwise_distance(grid, grid).tolist()
        [[0.0, 5.0], [5.0, 0.0]]
        >>> poles = Points(
        ...     jnp.array([[0.0, 90.0], [0.0, -90.0]]), metric="geodesic"
        ... )
        >>> d = pairwise_distance(poles, poles, unit="degrees", radius=6371.0)
        >>> bool(jnp.allclose(d[0, 1], jnp.pi * 6371.0))
        True

        A padding point is nobody's neighbour:

        >>> padded = Points(
        ...     jnp.array([[0.0, 0.0], [3.0, 4.0]]), valid=jnp.array([True, False])
        ... )
        >>> pairwise_distance(grid, padded).tolist()
        [[0.0, inf], [5.0, inf]]
    """
    if sources.metric != targets.metric:
        msg = (
            "'sources' and 'targets' must share a metric, got "
            f"{sources.metric!r} and {targets.metric!r}"
        )
        raise GeometryError(msg)
    metric = targets.metric
    if metric == "euclidean":
        distances = _chunked(_euclidean_distance, sources.xy, targets.xy, max_pairs)
    else:
        _check_unit(unit)
        _check_radius(radius)
        for name, points in (("sources", sources), ("targets", targets)):
            if points.xy.shape[1] != 2:
                msg = (
                    f"'{name}.xy' must be (λ, φ) for the {metric!r} metric, "
                    f"got shape {points.xy.shape}"
                )
                raise GeometryError(msg)
        source_units = lonlat_to_unit(sources.xy, unit=unit)
        target_units = lonlat_to_unit(targets.xy, unit=unit)
        on_sphere = chord_distance if metric == "chord" else geodesic_distance

        def distance(
            u: Float[Array, "source dim"], v: Float[Array, "target dim"]
        ) -> Float[Array, "source target"]:
            return on_sphere(u, v, radius=radius)

        distances = _chunked(distance, source_units, target_units, max_pairs)

    valid = _pair_mask(sources, targets)
    if valid is None:
        return distances
    # d = +∞ wherever either point is padding
    return jnp.where(valid, distances, jnp.inf)


def _pair_mask(sources: Points, targets: Points) -> Bool[Array, "source target"] | None:
    """True where both points of a pair are real, or None when none is padding."""
    if sources.valid is None and targets.valid is None:
        return None
    source_valid = (
        jnp.ones(sources.xy.shape[0], dtype=bool)
        if sources.valid is None
        else sources.valid
    )
    target_valid = (
        jnp.ones(targets.xy.shape[0], dtype=bool)
        if targets.valid is None
        else targets.valid
    )
    # valid_mn = valid_m ∧ valid_n for source m and target n
    return einx.logical_and(
        "source, target -> source target", source_valid, target_valid
    )
