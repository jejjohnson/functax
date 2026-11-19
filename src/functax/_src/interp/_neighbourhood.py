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
    (nodes in row-major order) and an `AffineGrid` (pixel centres, easting first)
    give their nodes, and the two grids their shape, so a plan onto them keeps
    it as ``target_shape``.

    Raises:
        GeometryError: If ``target`` is none of these.
    """
    if isinstance(target, Points):
        return target.xy, None, target.valid
    if isinstance(target, Axis):
        column = einx.id("target -> target 1", target.values)
        return column, None, None
    if isinstance(target, RectilinearGrid):
        nodes = jnp.meshgrid(*(axis.values for axis in target.axes), indexing="ij")
        flat = [einx.id("... -> (...)", node) for node in nodes]
        # one column per axis, however many axes the grid has
        coordinates = jnp.column_stack(flat)
        return coordinates, target.shape, None
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
    """dᵢⱼ = ‖x_{nᵢⱼ} − x_{*,i}‖ in the search space, guarded at dᵢⱼ = 0.

    nᵢⱼ is the source of tap j of target i.
    """
    neighbours = sources[indices]
    # x_{nᵢⱼ} − x_{*,i}
    difference = einx.subtract(
        "target tap space, target space -> target tap space", neighbours, targets
    )
    squared_difference = difference**2
    # dᵢⱼ² = ‖x_{nᵢⱼ} − x_{*,i}‖²
    squared = einx.sum("target tap [space]", squared_difference)
    # dᵢⱼ = √(dᵢⱼ²), with a zero gradient at dᵢⱼ = 0
    return _safe_norm(squared)


def _tap_angles(
    sources: Float[Array, "source 3"],
    targets: Float[Array, "target 3"],
    indices: Int[Array, "target tap"],
) -> Float[Array, "target tap"]:
    """dᵢⱼ = atan2(‖u × v‖, u · v), the great-circle angle to each tap.

    For u the unit vector of a tap and v that of its target, the atan2 form is
    accurate at every angle, as in `geodesic_distance`, and the cross
    product's norm is guarded at dᵢⱼ = 0, so the gradient there is zero.
    """
    neighbours = sources[indices]
    # u × v, one component at a time, for u a tap and v its target
    u_x, u_y, u_z = neighbours[..., 0], neighbours[..., 1], neighbours[..., 2]
    v_x, v_y, v_z = targets[:, 0], targets[:, 1], targets[:, 2]
    pair = "target tap, target -> target tap"
    # (u × v)_x = u_y v_z − u_z v_y
    yz = einx.multiply(pair, u_y, v_z)
    zy = einx.multiply(pair, u_z, v_y)
    cross_x = yz - zy
    # (u × v)_y = u_z v_x − u_x v_z
    zx = einx.multiply(pair, u_z, v_x)
    xz = einx.multiply(pair, u_x, v_z)
    cross_y = zx - xz
    # (u × v)_z = u_x v_y − u_y v_x
    xy = einx.multiply(pair, u_x, v_y)
    yx = einx.multiply(pair, u_y, v_x)
    cross_z = xy - yx
    # ‖u × v‖², summed in the order x, y, z
    squared_cross = cross_x**2 + cross_y**2 + cross_z**2
    # ‖u × v‖, with a zero gradient at dᵢⱼ = 0
    cross_norm = _safe_norm(squared_cross)
    # u · v
    dot = einx.dot("target tap c, target c -> target tap", neighbours, targets)
    # dᵢⱼ = atan2(‖u × v‖, u · v)
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
    is_finite = np.isfinite(target_array)
    finite = einx.all("target [space]", is_finite)
    if real.size == 0:
        indices = np.zeros((n_targets, n_taps), dtype=np.int32)
        none_found = jnp.zeros((n_targets, n_taps), dtype=bool)
        return jnp.asarray(indices), none_found
    real_sources = source_array[real]
    tree = spatial.cKDTree(real_sources)
    # non-finite targets are searched at the origin and marked not found
    origin = np.zeros((), target_array.dtype)
    queries = einx.where(
        "target, target space, -> target space", finite, target_array, origin
    )
    bound = np.inf if radius is None else radius
    # k as the list 1, …, Q keeps the result 2-D when Q = 1
    ranks = list(range(1, n_taps + 1))
    _, neighbours = tree.query(queries, k=ranks, distance_upper_bound=bound)
    in_tree = neighbours < real.size
    found = einx.logical_and("target tap, target -> target tap", in_tree, finite)
    tree_index = np.where(found, neighbours, 0)
    source_index = real[tree_index]
    indices = source_index.astype(np.int32)
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
        block_sources: Float[Array, "source space"],
        block_targets: Float[Array, "target space"],
    ) -> Int[Array, "tap target"]:
        # x − x_*
        difference = einx.subtract(
            "source space, target space -> target source space",
            block_sources,
            block_targets,
        )
        squared_difference = difference**2
        # ‖x − x_*‖²
        squared = einx.sum("target source [space]", squared_difference)
        if source_valid is not None:
            inf = jnp.asarray(jnp.inf, dtype=squared.dtype)
            squared = einx.where(
                "source, target source, -> target source", source_valid, squared, inf
            )
        # the Q smallest ‖x − x_*‖², nearest first
        negated = -squared
        _, block_indices = jax.lax.top_k(negated, n_taps)
        return einx.id("target tap -> tap target", block_indices)

    columns = _chunked(nearest, fixed_sources, fixed_targets, MAX_PAIRS)
    rows = einx.id("tap target -> target tap", columns)
    indices = rows.astype(jnp.int32)
    distances = _tap_distances(fixed_sources, fixed_targets, indices)
    bound = jnp.inf if radius is None else radius
    # cKDTree's rule: a neighbour counts when d < r (and is a real source)
    finite_distance = jnp.isfinite(distances)
    within = distances < bound
    found = finite_distance & within
    if source_valid is not None:
        tap_valid = source_valid[indices]
        found = found & tap_valid
    is_finite = jnp.isfinite(fixed_targets)
    finite = einx.all("target [space]", is_finite)
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
    neighbour counts when its distance is below the radius r; a missing one
    (fewer than Q real sources, none within the radius, or a non-finite
    target) comes back not found.

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
    """wᵢⱼ = max(dᵢⱼ, d_min)⁻ᵖ / ∑ⱼ′ max(dᵢⱼ′, d_min)⁻ᵖ over the found taps."""
    # max(dᵢⱼ, d_min)
    floored = jnp.maximum(distances, floor)
    inf = jnp.asarray(jnp.inf, dtype=floored.dtype)
    one = jnp.ones((), floored.dtype)
    zero = jnp.zeros((), floored.dtype)
    row_ok = einx.any("target [tap]", found)
    # d_{near,i} = minⱼ max(dᵢⱼ, d_min), the nearest found distance; w̃ᵢⱼ is
    # scaled by d_{near,i}ᵖ so it lies in (0, 1] and cannot overflow (d_min⁻ᵖ
    # does in float32 for p ≥ 4); the scale cancels in wᵢⱼ
    found_distances = jnp.where(found, floored, inf)
    nearest = einx.min("target [tap]", found_distances)
    safe_nearest = jnp.where(row_ok, nearest, one)
    # d_{near,i} / max(dᵢⱼ, d_min)
    ratio = einx.divide("target, target tap -> target tap", safe_nearest, floored)
    # d_{near,i}ᵖ w̃ᵢⱼ = (d_{near,i} / max(dᵢⱼ, d_min))ᵖ
    powered = ratio**power
    scaled = jnp.where(found, powered, zero)
    # d_{near,i}ᵖ ∑ⱼ′ w̃ᵢⱼ′
    total = einx.sum("target [tap]", scaled)
    safe_total = jnp.where(row_ok, total, one)
    # wᵢⱼ = w̃ᵢⱼ / ∑ⱼ′ w̃ᵢⱼ′
    weights = einx.divide("target tap, target -> target tap", scaled, safe_total)
    # a missing tap repeats the row's nearest index (index 0 in an empty row),
    # so it reads only a source the row reads anyway
    zero_index = jnp.zeros((), indices.dtype)
    first = jnp.where(row_ok, indices[:, 0], zero_index)
    padded = einx.where(
        "target tap, target tap, target -> target tap", found, indices, first
    )
    return Stencil(padded, weights, n_source=n_source, valid=row_ok, check=False)


class InverseDistance(Scheme):
    """Inverse-distance weighting of the Q nearest scattered sources.

    For target x_{*,i}, i over the M targets, with taps j = 1, …, Q at the
    sources x_{nᵢⱼ}, neighbour distances dᵢⱼ, power p and floor d_min,

        w̃ᵢⱼ = max(dᵢⱼ, d_min)⁻ᵖ  and  wᵢⱼ = w̃ᵢⱼ / ∑ⱼ′ w̃ᵢⱼ′,

    so f̂(x_{*,i}) = ∑ⱼ wᵢⱼ y_{nᵢⱼ}. A target that coincides with a source
    takes that source's value, through the floor. The distance is the source
    `Points.metric`: ``"euclidean"`` on the coordinates as given, or, on the
    longitude and latitude (λ, φ), ``"chord"``, the straight line c between
    unit vectors, or ``"geodesic"``, the great-circle angle 2 arcsin(c/2)
    (haversine IDW). Both sphere metrics search on unit vectors, across the
    dateline and the poles; the angle is monotone in c, so they pick the same
    neighbours and differ only in the weights.

    Algorithm:
        1. Lift the coordinates into the search space (unit vectors for a
           chord).
        2. Find the Q = min(``k``, N) nearest real sources of each target: by a
           SciPy k-d tree on the host when the coordinates are concrete, by a
           chunked brute force (``lax.map`` over chunks of targets) inside the
           trace when they are traced. A neighbour counts when dᵢⱼ < r.
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
            the chord of the unit sphere (2 sin(r/2) for an arc r), or, for
            ``"geodesic"``, the angle in radians of arc (π or more for no
            limit); ``None`` for no limit.
        floor: d_min > 0, the distance below which a neighbour counts as
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
        # the search runs in chords, so an arc radius r becomes 2 sin(r/2)
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
