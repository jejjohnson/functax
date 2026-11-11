"""Neighbourhood schemes on scattered sources: inverse-distance weighting.

On scattered data the support of a target comes from a search, not from
arithmetic on an axis. The Q nearest sources of each target are found on the
host by a k-d tree (SciPy) when the coordinates are concrete, and by a chunked
brute force inside the trace when they are not; the weights then follow from
the distances in JAX, so both searches give the same `Stencil`.
"""

from __future__ import annotations

from functax._src._backend import import_scipy, is_tracer, require_jax


require_jax(__name__)

import math
from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from functax._src.geometry._affine import pixel_centres
from functax._src.geometry._metric import MAX_PAIRS, _chunked
from functax._src.geometry._sphere import _safe_norm, lonlat_to_unit
from functax._src.geometry._types import AffineGrid, Axis, Points, RectilinearGrid
from functax._src.interp._fit import Identity
from functax._src.interp._locate import check_outside
from functax._src.interp._plan import Plan
from functax._src.interp._policies import Policies
from functax._src.interp._schemes import Scheme
from functax._src.interp._stencil import Stencil
from functax._typing import Shape
from functax.exceptions import GeometryError, PolicyError, ValidationError


__all__ = [
    "InverseDistance",
    "check_scattered_policies",
    "nearest_sources",
    "scattered_targets",
]


# The metrics a neighbour search runs in: planar, or on unit vectors, where the
# chord and the great-circle angle order the neighbours the same way
_SEARCH_METRICS: tuple[str, ...] = ("euclidean", "chord", "geodesic")


def scattered_targets(
    target: Any,
) -> tuple[Float[Array, "target dim"], Shape | None, Bool[Array, " target"] | None]:
    """The (M, D) coordinates of a target, its grid shape, and its padding mask.

    `Points` give their ``xy`` and ``valid``; an `Axis`, a `RectilinearGrid`
    (nodes in row-major order) and an `AffineGrid` (pixel centres, as (x, y))
    give their nodes, and the two grids their shape, so a plan onto them keeps
    it as ``target_shape``.

    Raises:
        GeometryError: If ``target`` is none of these.
    """
    if isinstance(target, Points):
        return target.xy, None, target.valid
    if isinstance(target, Axis):
        return einx.id("target -> target 1", target.values), None, None
    if isinstance(target, RectilinearGrid):
        nodes = jnp.meshgrid(*(axis.values for axis in target.axes), indexing="ij")
        flat = [einx.id("... -> (...)", node) for node in nodes]
        return jnp.column_stack(flat), target.shape, None
    if isinstance(target, AffineGrid):
        x, y = pixel_centres(target)
        centres = einx.id("row col, row col -> (row col) (1 + 1)", x, y)
        return centres, target.shape, None
    msg = (
        "'target' must be Points, an Axis, a RectilinearGrid or an AffineGrid, "
        f"got {type(target).__name__}"
    )
    raise GeometryError(msg)


def check_scattered_policies(policies: Policies, what: str) -> None:
    """Raise unless ``outside`` is ``"nan"`` or ``"raise"``.

    A scattered scheme has no axis to clamp onto or extrapolate along: a target
    without support is invalid (or raises on the host).

    Raises:
        PolicyError: Under ``outside="clamp"`` or ``"extrapolate"``.
    """
    if policies.outside not in ("nan", "raise"):
        msg = (
            f"{what} supports outside='nan' or 'raise', got "
            f"outside={policies.outside!r}"
        )
        raise PolicyError(msg)


def _search_space(
    coordinates: Float[Array, "obs dim"], metric: str, unit: str | None
) -> Float[Array, "obs space"]:
    """The coordinates the search runs in: as given, or unit vectors on the sphere."""
    if metric == "euclidean":
        return coordinates
    if unit is None:
        msg = (
            f"'unit' must be 'degrees' or 'radians' for the {metric!r} metric, got None"
        )
        raise ValidationError(msg)
    if coordinates.shape[1] != 2:
        msg = (
            f"the {metric!r} metric needs (λ, φ) coordinates, got "
            f"{coordinates.shape[1]} per point"
        )
        raise GeometryError(msg)
    return lonlat_to_unit(coordinates, unit=unit)


def _tap_distances(
    sources: Float[Array, "source space"],
    targets: Float[Array, "target space"],
    indices: Int[Array, "target tap"],
) -> Float[Array, "target tap"]:
    """dᵢⱼ = ‖x_{iᵢⱼ} − x*ᵢ‖ in the search space, guarded at dᵢⱼ = 0."""
    neighbours = sources[indices]
    difference = einx.subtract(
        "target tap c, target c -> target tap c", neighbours, targets
    )
    squared = einx.sum("target tap [c]", difference**2)
    return _safe_norm(squared)


def _tap_angles(
    sources: Float[Array, "source 3"],
    targets: Float[Array, "target 3"],
    indices: Int[Array, "target tap"],
) -> Float[Array, "target tap"]:
    """θᵢⱼ = atan2(‖u × v‖, u · v), the great-circle angle to each tap.

    The atan2 form is accurate at every angle, as in `geodesic_distance`,
    and the cross product's norm is guarded at θ = 0, so the gradient there
    is zero.
    """
    neighbours = sources[indices]
    # u × v, one component at a time, for u a tap and v its target
    u_x, u_y, u_z = neighbours[..., 0], neighbours[..., 1], neighbours[..., 2]
    v_x, v_y, v_z = targets[:, 0], targets[:, 1], targets[:, 2]
    pair = "target tap, target -> target tap"
    cross_x = einx.multiply(pair, u_y, v_z) - einx.multiply(pair, u_z, v_y)
    cross_y = einx.multiply(pair, u_z, v_x) - einx.multiply(pair, u_x, v_z)
    cross_z = einx.multiply(pair, u_x, v_y) - einx.multiply(pair, u_y, v_x)
    cross_norm = _safe_norm(cross_x**2 + cross_y**2 + cross_z**2)
    dot = einx.dot("target tap c, target c -> target tap", neighbours, targets)
    return jnp.arctan2(cross_norm, dot)


def _chord_radius(radius: float | None) -> float | None:
    """The chord 2 sin(r/2) of an arc radius r, or no limit for r ≥ π."""
    if radius is None or radius >= math.pi:
        return None
    return 2.0 * math.sin(0.5 * radius)


def _tree_neighbours(
    sources: Float[Array, "source space"],
    targets: Float[Array, "target space"],
    source_valid: Bool[Array, " source"] | None,
    n_taps: int,
    radius: float | None,
) -> tuple[Int[Array, "target tap"], Bool[Array, "target tap"]]:
    """The Q nearest real sources of each target by a k-d tree, on the host.

    A missing neighbour (fewer than Q real sources, or none within the radius)
    is reported by ``cKDTree.query`` at index N; it comes back not found.
    """
    spatial = import_scipy("spatial", needed_by="InverseDistance.plan")
    source_array = np.asarray(sources)
    target_array = np.asarray(targets)
    real = (
        np.arange(source_array.shape[0])
        if source_valid is None
        else np.flatnonzero(np.asarray(source_valid))
    )
    n_targets = target_array.shape[0]
    finite = np.asarray(np.isfinite(target_array).all(axis=1, keepdims=True))
    if real.size == 0:
        indices = np.zeros((n_targets, n_taps), dtype=np.int32)
        return jnp.asarray(indices), jnp.zeros((n_targets, n_taps), dtype=bool)
    tree = spatial.cKDTree(source_array[real])
    # non-finite targets are searched at the origin and marked not found
    queries = np.where(finite, target_array, 0.0)
    bound = np.inf if radius is None else radius
    # k as the list 1, …, Q keeps the result 2-D when Q = 1
    _, neighbours = tree.query(
        queries, k=list(range(1, n_taps + 1)), distance_upper_bound=bound
    )
    found = (neighbours < real.size) & finite
    indices = real[np.where(found, neighbours, 0)].astype(np.int32)
    return jnp.asarray(indices), jnp.asarray(found)


def _brute_force_neighbours(
    sources: Float[Array, "source space"],
    targets: Float[Array, "target space"],
    source_valid: Bool[Array, " source"] | None,
    n_taps: int,
    radius: float | None,
) -> tuple[Int[Array, "target tap"], Bool[Array, "target tap"]]:
    """The Q nearest real sources of each target by brute force, inside the trace.

    The targets are taken in chunks with ``lax.map`` (as `pairwise_distance`
    does), so the (chunk, N) distances stay near 2²² entries per chunk. The
    neighbour set is piecewise constant in the coordinates, so the search
    sees them through ``stop_gradient``.
    """
    fixed_sources = jax.lax.stop_gradient(sources)
    fixed_targets = jax.lax.stop_gradient(targets)

    def nearest(
        block_sources: jax.Array, block_targets: jax.Array
    ) -> Int[Array, "tap target"]:
        difference = einx.subtract(
            "source c, target c -> target source c", block_sources, block_targets
        )
        squared = einx.sum("target source [c]", difference**2)
        if source_valid is not None:
            inf = jnp.asarray(jnp.inf, dtype=squared.dtype)
            squared = einx.where(
                "source, target source, -> target source", source_valid, squared, inf
            )
        # the Q smallest ‖x − x*‖², nearest first
        _, block_indices = jax.lax.top_k(-squared, n_taps)
        return einx.id("target tap -> tap target", block_indices)

    columns = _chunked(nearest, fixed_sources, fixed_targets, MAX_PAIRS)
    indices = einx.id("tap target -> target tap", columns).astype(jnp.int32)
    distances = _tap_distances(fixed_sources, fixed_targets, indices)
    bound = jnp.inf if radius is None else radius
    # cKDTree's rule: a neighbour counts when d < r (and is a real source)
    found = jnp.isfinite(distances) & (distances < bound)
    if source_valid is not None:
        found = found & source_valid[indices]
    finite = einx.all("target [c]", jnp.isfinite(fixed_targets))
    found = einx.logical_and("target tap, target -> target tap", found, finite)
    return indices, found


def nearest_sources(
    sources: Float[Array, "source space"],
    targets: Float[Array, "target space"],
    source_valid: Bool[Array, " source"] | None,
    n_taps: int,
    radius: float | None,
) -> tuple[Int[Array, "target tap"], Bool[Array, "target tap"]]:
    """The Q nearest real sources of each target, nearest first, and which were found.

    A SciPy k-d tree on the host when every coordinate is concrete, a chunked
    brute force inside the trace otherwise; both give the same neighbours. A
    neighbour counts when d < r; a missing one (fewer than Q real sources,
    none within the radius, or a non-finite target) comes back not found.

    Raises:
        MissingExtraError: If SciPy is missing and the coordinates are
            concrete.
    """
    traced = any(is_tracer(leaf) for leaf in (sources, targets, source_valid))
    search = _brute_force_neighbours if traced else _tree_neighbours
    return search(sources, targets, source_valid, n_taps, radius)


def _inverse_distance_stencil(
    distances: Float[Array, "target tap"],
    indices: Int[Array, "target tap"],
    found: Bool[Array, "target tap"],
    n_source: int,
    power: float,
    floor: float,
) -> Stencil:
    """wᵢⱼ = max(dᵢⱼ, ε)⁻ᵖ / ∑ⱼ′ max(dᵢⱼ′, ε)⁻ᵖ over the found taps."""
    floored = jnp.maximum(distances, floor)
    inf = jnp.asarray(jnp.inf, dtype=floored.dtype)
    row_ok = einx.any("target [tap]", found)
    # mᵢ = minⱼ max(dᵢⱼ, ε), the nearest found distance; vᵢⱼ is scaled by mᵢᵖ
    # so it lies in (0, 1] and cannot overflow (ε⁻ᵖ does in float32 for p ≥ 4)
    nearest = einx.min("target [tap]", jnp.where(found, floored, inf))
    safe_nearest = jnp.where(row_ok, nearest, jnp.ones((), floored.dtype))
    # vᵢⱼ = (mᵢ / max(dᵢⱼ, ε))ᵖ
    ratio = einx.divide("target, target tap -> target tap", safe_nearest, floored)
    zero = jnp.zeros((), floored.dtype)
    scaled = jnp.where(found, ratio**power, zero)
    total = einx.sum("target [tap]", scaled)
    safe_total = jnp.where(row_ok, total, jnp.ones((), total.dtype))
    # wᵢⱼ = vᵢⱼ / ∑ⱼ′ vᵢⱼ′
    weights = einx.divide("target tap, target -> target tap", scaled, safe_total)
    # a missing tap repeats the row's nearest index (index 0 in an empty row),
    # so it reads only a source the row reads anyway
    first = jnp.where(row_ok, indices[:, 0], jnp.zeros((), indices.dtype))
    padded = einx.where(
        "target tap, target tap, target -> target tap", found, indices, first
    )
    return Stencil(padded, weights, n_source=n_source, valid=row_ok, check=False)


class InverseDistance(Scheme):
    """Inverse-distance weighting of the Q nearest scattered sources.

    For target i with neighbour distances dᵢⱼ, j = 1, …, Q, power p and
    floor ε,

        vᵢⱼ = max(dᵢⱼ, ε)⁻ᵖ  and  wᵢⱼ = vᵢⱼ / ∑ⱼ′ vᵢⱼ′,

    so f̂(x*ᵢ) = ∑ⱼ wᵢⱼ y_{iᵢⱼ}. A target that coincides with a source takes
    that source's value, through the floor. The distance is the source
    `Points.metric`: ``"euclidean"`` on the coordinates as given, or, on
    (λ, φ), ``"chord"``, the straight line c between unit vectors, or
    ``"geodesic"``, the great-circle angle θ = 2 arcsin(c/2) (haversine IDW).
    Both sphere metrics search on unit vectors, across the dateline and the
    poles; c is monotone in θ, so they pick the same neighbours and differ
    only in the weights.

    Algorithm:
        1. Lift the coordinates into the search space (unit vectors for a
           chord).
        2. Find the Q = min(k, N) nearest real sources of each target: by a
           SciPy k-d tree on the host when the coordinates are concrete, by a
           chunked brute force (``lax.map`` over chunks of targets) inside the
           trace when they are traced. A neighbour counts when d < r.
        3. Recompute the distances of the neighbours in JAX and weigh them,
           so both searches give the same stencil and gradients reach the
           coordinates through the weights (the neighbour set is piecewise
           constant).

    A target with no neighbour (all beyond the radius, a non-finite
    coordinate, or padding) is invalid and reads ``fill_value``. Under
    ``missing="renormalise"`` a NaN source drops out of each row it is in.

    Args:
        k: The number of neighbours, at least 1.
        power: p ≥ 0; 0 gives the unweighted mean of the neighbours.
        radius: r, the search radius in the metric's units: coordinate units,
            the chord of the unit sphere (2 sin(θ/2) for an angle θ), or, for
            ``"geodesic"``, the angle θ in radians of arc (π or more for no
            limit); ``None`` for no limit.
        floor: ε > 0, the distance below which a neighbour counts as
            coincident (in radians under ``"geodesic"``).

    Raises:
        ValidationError: If ``k`` is not a positive integer, ``power`` is
            negative or not finite, or ``radius`` or ``floor`` is not positive
            and finite.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points, RectilinearGrid
        >>> from functax.interp import InverseDistance
        >>> sources = Points(jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
        >>> plan = InverseDistance(k=2).plan(
        ...     sources, Points(jnp.array([[0.25, 0.0]]))
        ... )
        >>> # weights 1/0.25² and 1/0.75², normalised: 0.9 and 0.1
        >>> [round(v, 12) for v in plan(jnp.array([10.0, 20.0, 30.0])).tolist()]
        [11.0]
        >>> grid = RectilinearGrid((Axis(jnp.arange(3.0)), Axis(jnp.arange(4.0))))
        >>> gridded = InverseDistance(k=3, radius=1.5).plan(sources, grid)
        >>> gridded.grid_shape, gridded.valid.tolist()[-1]
        ((3, 4), False)
        >>> # on the sphere: the two sources either side of the dateline
        >>> lonlat = Points(
        ...     jnp.array([[179.0, 0.0], [-179.0, 0.0], [0.0, 0.0]]), metric="chord"
        ... )
        >>> dateline = Points(jnp.array([[180.0, 0.0]]), metric="chord")
        >>> plan = InverseDistance(k=2).plan(lonlat, dateline, unit="degrees")
        >>> sorted(plan.read.indices[0].tolist())
        [0, 1]
        >>> # great-circle weights: the same neighbours, weighed by their angles
        >>> arcs = Points(lonlat.xy, metric="geodesic")
        >>> arc_plan = InverseDistance(k=2).plan(
        ...     arcs, Points(dateline.xy, metric="geodesic"), unit="degrees"
        ... )
        >>> sorted(arc_plan.read.indices[0].tolist())
        [0, 1]
    """

    k: int = eqx.field(static=True, default=8)
    # float | int rather than float, so an int reaches the checks below
    power: float | int = eqx.field(static=True, default=2.0)
    radius: float | int | None = eqx.field(static=True, default=None)
    floor: float | int = eqx.field(static=True, default=1e-12)

    def __check_init__(self) -> None:
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            msg = f"'k' must be a positive integer, got {self.k!r}"
            raise ValidationError(msg)
        if not _is_number(self.power) or not math.isfinite(self.power):
            msg = f"'power' must be a finite number of at least 0, got {self.power!r}"
            raise ValidationError(msg)
        if self.power < 0:
            msg = f"'power' must be a finite number of at least 0, got {self.power!r}"
            raise ValidationError(msg)
        for name in ("radius", "floor"):
            value = getattr(self, name)
            if name == "radius" and value is None:
                continue
            positive = _is_number(value) and math.isfinite(value) and value > 0
            if not positive:
                msg = f"'{name}' must be a positive, finite number, got {value!r}"
                raise ValidationError(msg)

    def plan(
        self,
        source: Any,
        target: Any,
        *,
        policies: Policies | None = None,
        unit: str | None = None,
    ) -> Plan:
        """H for scattered ``source`` data read at ``target``.

        Args:
            source: `Points` with a ``"euclidean"``, ``"chord"`` or
                ``"geodesic"`` metric; padding points (``valid`` False) are
                never a neighbour.
            target: `Points` (in the source's metric), or an `Axis`, a
                `RectilinearGrid` or an `AffineGrid` read at its nodes, in the
                source's coordinates; a grid keeps its shape as
                ``target_shape``.
            policies: ``outside`` (``"nan"`` or ``"raise"``) acts on targets
                with no neighbour; ``missing`` and ``fill_value`` as for any
                plan.
            unit: ``"degrees"`` or ``"radians"``, the unit of (λ, φ) under the
                ``"chord"`` and ``"geodesic"`` metrics; unused for
                ``"euclidean"``.

        Returns:
            The `Plan`, with an (M, Q) `Stencil` read-out and the identity fit.

        Raises:
            GeometryError: If the source is not `Points`, its metric is not
                searchable, the target's metric or dimension differs, or a
                sphere metric meets coordinates that are not (λ, φ).
            ValidationError: Under a sphere metric without ``unit``.
            PolicyError: Under ``outside="clamp"`` or ``"extrapolate"``.
            OutsideDomainError: Under ``outside="raise"`` when a target has no
                neighbour.
            MissingExtraError: If SciPy (the ``functax[scipy]`` extra) is
                missing and the coordinates are concrete.
        """
        policies = Policies() if policies is None else policies
        check_scattered_policies(policies, "InverseDistance")
        if not isinstance(source, Points):
            msg = f"'source' must be Points, got {type(source).__name__}"
            raise GeometryError(msg)
        if source.metric not in _SEARCH_METRICS:
            msg = (
                f"InverseDistance searches in {list(_SEARCH_METRICS)}, got the "
                f"{source.metric!r} metric"
            )
            raise GeometryError(msg)
        if isinstance(target, Points) and target.metric != source.metric:
            msg = (
                "'source' and 'target' must share a metric, got "
                f"{source.metric!r} and {target.metric!r}"
            )
            raise GeometryError(msg)
        coordinates, target_shape, target_valid = scattered_targets(target)
        if coordinates.shape[1] != source.xy.shape[1]:
            msg = (
                f"'target' has {coordinates.shape[1]} coordinates per point, but "
                f"'source' has {source.xy.shape[1]}"
            )
            raise GeometryError(msg)
        sources = _search_space(source.xy, source.metric, unit)
        targets = _search_space(coordinates, source.metric, unit)
        n_source = source.xy.shape[0]
        n_taps = min(self.k, n_source)
        geodesic = source.metric == "geodesic"
        # the search runs in chords, so an arc radius becomes 2 sin(r/2)
        radius = _chord_radius(self.radius) if geodesic else self.radius
        indices, found = nearest_sources(sources, targets, source.valid, n_taps, radius)
        if target_valid is not None:
            found = einx.logical_and(
                "target tap, target -> target tap", found, target_valid
            )
        if geodesic:
            distances = _tap_angles(sources, targets, indices)
        else:
            distances = _tap_distances(sources, targets, indices)
        stencil = _inverse_distance_stencil(
            distances, indices, found, n_source, self.power, self.floor
        )
        check_outside(stencil.valid, policies)
        return Plan(Identity(), stencil, policies, None, target_shape)


def _is_number(value: Any) -> bool:
    """Whether ``value`` is an int or a float, and not a bool."""
    return isinstance(value, int | float) and not isinstance(value, bool)
