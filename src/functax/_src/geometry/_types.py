"""CRS-free geometry types: an axis, affine, rectilinear and curvilinear grids, points.

Coordinates arrive already in one coordinate reference system; the CRS stays a
concern of the carrier (geoarray, xarray). Periodicity, the time epoch and the
distance metric travel with the geometry as data rather than as keywords, so
`locate` and every scheme read sources and targets the same way.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

from collections.abc import Sequence
from typing import Any, Literal

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, Shaped

from functax._src._backend import is_tracer
from functax._src._checks import check_values, require_floating
from functax._src.geometry._transform import as_transform
from functax.exceptions import GeometryError


__all__ = [
    "AffineGrid",
    "Axis",
    "CurvilinearGrid",
    "Metric",
    "Points",
    "RectilinearGrid",
]


type Metric = Literal["euclidean", "chord", "geodesic"]
"""The distance a point set's neighbours are measured in."""

_REPEATS: tuple[str, ...] = ("forbid", "allow")

_METRICS: tuple[str, ...] = ("euclidean", "chord", "geodesic")

# The resolutions `Axis.to_datetimes` rounds to; each divides one second.
_UNITS_PER_SECOND: dict[str, int] = {
    "s": 1,
    "ms": 1_000,
    "us": 1_000_000,
    "ns": 1_000_000_000,
}

# |Δₖ − Δ̄| ≤ 1e-12 |Δ̄| counts as equal spacing (Axis.is_regular)
_REGULAR_RTOL = 1e-12

# cells whose gap to the whole period is within 10⁻³ of a cell span it, so nodes
# or edges rounded by a file, or cast from float32, still close (Axis.closed)
_CLOSED_RTOL = 1e-3


def _check_values(
    array: Shaped[Array, "*shape"], bad: Bool[Array, ""], message: str
) -> Shaped[Array, "*shape"]:
    """`check_values` raising `GeometryError`."""
    return check_values(array, bad, message, error=GeometryError)


def _require_floating(array: Shaped[Array, "*shape"], name: str) -> None:
    """`require_floating` raising `GeometryError`."""
    require_floating(array, name, error=GeometryError)


def _as_epoch(epoch: Any) -> str:
    """An epoch as an ISO 8601 string at nanosecond resolution."""
    try:
        moment = np.datetime64(epoch, "ns")
    except (TypeError, ValueError) as err:
        msg = f"'epoch' must be a datetime64 or an ISO 8601 string, got {epoch!r}"
        raise GeometryError(msg) from err
    if np.isnat(moment):
        msg = f"'epoch' must not be NaT, got {epoch!r}"
        raise GeometryError(msg)
    return str(np.datetime_as_string(moment, unit="ns"))


def _host_array(array: Any) -> Shaped[np.ndarray, "*shape"] | None:
    """``array`` on the host, or ``None`` when it is (or holds) a tracer."""
    try:
        return np.asarray(array)
    except jax.errors.TracerArrayConversionError:
        return None


def _cells_span_period(
    nodes: Any, edges: Any, period: float, dtype: np.dtype
) -> bool | None:
    """Whether the cells of an axis span its period P, decided on the host.

    With edges, they do when P − |e_N − e₀| is zero to rounding; without, when
    the wrap gap P − |x_{N−1} − x₀| is at most the sum of the two end
    spacings, which one node has none of. Rounding is the larger of 10⁻³ of a
    cell and 4 ε max(|x|, P), with ε the machine epsilon of ``dtype``.
    ``None`` when the array that decides is traced.
    """
    ends = _host_array(nodes if edges is None else edges)
    if ends is None or ends.ndim != 1 or ends.shape[0] < 1:
        return None
    # the rounding of the axis' own dtype, subtracted in float64
    ends = ends.astype(dtype).astype(np.float64)
    # P − |x_last − x_first|
    gap = period - abs(ends[-1] - ends[0])
    magnitude = max(np.abs(ends).max(), period)
    # 4 ε max(|x|, P)
    rounding = 4.0 * float(jnp.finfo(dtype).eps) * magnitude
    if ends.shape[0] == 1:
        return bool(gap <= rounding)
    spacing = np.abs(np.diff(ends))
    if edges is None:
        # (x₁ − x₀) + (x_{N−1} − x_{N−2})
        reach = spacing[0] + spacing[-1]
        tolerance = max(_CLOSED_RTOL * reach, rounding)
        return bool(gap <= reach + tolerance)
    tolerance = max(_CLOSED_RTOL * spacing.mean(), rounding)
    return bool(gap <= tolerance)


class Axis(eqx.Module):
    """One strictly monotone coordinate axis, optionally periodic or in time.

    A periodic axis with period P treats x and x + mP as the same point for
    every integer m, so its nodes must span less than P. Descending axes are
    allowed (wavelength from wavenumber). Times are stored as float seconds
    since ``epoch``; `from_datetimes` converts them on the host.

    A periodic axis is closed when its cells span the period, as a global
    longitude does, and open otherwise, as a regional longitude across the
    date line. A closed axis has a wrap cell from its last node to its first
    one plus P, so every target is inside. An open axis only folds a target
    to its copy nearest the data, in [x₀ − (P − span)/2, x₀ + (P + span)/2)
    for increasing nodes with span = x_{N−1} − x₀; past its ends it is read
    as an axis without a period. With ``edges``, the cells span P when the
    edges do, to rounding. Without, they do when the wrap gap P − |x_{N−1} −
    x₀| is at most the sum of the two end spacings, so an axis one node short
    of the circle (nodes every 1° from 0° to 358°) is closed and one two nodes
    short is open. One node has no spacings, so without ``edges`` it is open.

    With ``repeats="allow"`` the nodes need only be non-decreasing (or
    non-increasing), so a repeat xₖ = xₖ₊₁ makes a cell of zero width, where
    a piecewise read-out may jump. Reads are right-continuous in x: a target
    at a repeated node reads the last node of the repeat, as ``numpy.interp``
    does. Not every scheme reads repeats; those that do not raise a
    `GeometryError` naming the repeat.

    An axis of one node x₀ (a single time step, a raster one row thick) has
    no spacing, so it carries no direction and `cell_edges` needs explicit
    ``edges``. A scheme reads it with weight 1 on its node: under
    ``extent="nodes"`` a target is inside when x = x₀, and under
    ``extent="cells"`` when x_{−½} ≤ x ≤ x_{½}.

    Args:
        values: x, the N node coordinates xₖ, k = 0, …, N − 1, at least one,
            strictly increasing or strictly decreasing. Kept in the dtype they
            arrive in.
        period: P, the period (360.0 for longitude in degrees), or ``None``
            for an axis that is not periodic.
        epoch: The time origin of ``values`` (a ``datetime64`` or an ISO 8601
            string), or ``None`` for an axis that is not time. Stored as an
            ISO 8601 string at nanosecond resolution.
        edges: The N + 1 cell edges x_{k±½}, node k owning the cell
            [x_{k−½}, x_{k+½}] (CF ``bounds``), strictly monotone in the
            direction of ``values`` and spanning at most P on a periodic axis;
            ``None`` for the midpoints between nodes (`cell_edges`). Kept in
            the dtype they arrive in. Conservative remapping reads them.
        repeats: ``"forbid"`` (the default) for strictly monotone nodes, or
            ``"allow"`` for non-decreasing or non-increasing ones with
            repeats. The nodes must still span a positive width, and a
            periodic axis cannot have repeats.
        closed: Whether a periodic axis closes on itself (above), stored as
            a static ``True`` or ``False``. ``None`` (the default) decides
            it on the host from concrete ``values`` or ``edges``, and takes
            traced ones as closed, so inside ``jit`` pass ``closed=False``
            for an axis built from traced regional nodes. An axis built
            from another's values carries its ``closed``. ``None``, whatever
            was passed, on an axis without a period.

    Raises:
        GeometryError: If ``values`` is not 1-D, is empty, is not floating,
            is not strictly monotone (monotone with a positive span under
            ``repeats="allow"``), or spans P or more; if ``repeats`` is
            unknown or set on a periodic axis; if ``period`` is
            not positive and finite; if ``epoch`` is not a time;
            if ``edges`` is not 1-D with N + 1 floating entries, or (when
            concrete) is not strictly monotone in the direction of ``values``
            or spans more than P; if ``closed`` is not a boolean or ``None``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis
        >>> longitude = Axis(jnp.linspace(0.0, 357.5, 144), period=360.0)
        >>> longitude.is_regular, longitude.is_descending, longitude.closed
        (True, False, True)
        >>> # a region across the date line is open
        >>> Axis(jnp.arange(170.0, 190.5), period=360.0).closed
        False
        >>> wavelength = Axis(jnp.array([2.5, 2.0, 1.2]))
        >>> wavelength.is_regular, wavelength.is_descending
        (False, True)
        >>> Axis(jnp.array([0.0, 2.0, 1.0]))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        functax.exceptions.GeometryError: 'values' must be strictly monotone, ...
        >>> # CF bounds: cells of 1° centred on half degrees
        >>> latitude = Axis(
        ...     jnp.arange(-89.5, 90.0, 1.0), edges=jnp.arange(-90.0, 90.5, 1.0)
        ... )
        >>> latitude.edges.shape
        (181,)
        >>> # a float32 CDF that repeats its last value
        >>> cdf = Axis(jnp.array([0.0, 0.5, 1.0, 1.0]), repeats="allow")
        >>> cdf.repeats
        'allow'
        >>> # one time step, as a grid one node thick
        >>> Axis(jnp.array([3600.0])).values.shape
        (1,)
    """

    values: Float[Array, " node"]
    period: float | None = eqx.field(static=True)
    epoch: str | None = eqx.field(static=True)
    edges: Float[Array, " edge"] | None
    repeats: str = eqx.field(static=True)
    closed: bool | None = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike | Sequence[float],
        *,
        period: float | int | None = None,
        epoch: Any = None,
        edges: ArrayLike | Sequence[float] | None = None,
        repeats: str = "forbid",
        closed: bool | None = None,
    ) -> None:
        # kept for `closed`: under jit even a concrete array becomes a tracer
        given_values = values
        values = jnp.asarray(values)
        if values.ndim != 1:
            msg = f"'values' must be 1-D, got shape {values.shape}"
            raise GeometryError(msg)
        if values.shape[0] < 1:
            msg = "'values' must have at least one node, got none"
            raise GeometryError(msg)
        _require_floating(values, "values")
        if period is not None:
            period = float(period)
            if not (np.isfinite(period) and period > 0.0):
                msg = f"'period' must be positive and finite, got {period!r}"
                raise GeometryError(msg)
        if repeats not in _REPEATS:
            msg = f"'repeats' must be one of {list(_REPEATS)}, got {repeats!r}"
            raise GeometryError(msg)
        if repeats == "allow" and period is not None:
            msg = "a periodic Axis cannot have repeats; use repeats='forbid'"
            raise GeometryError(msg)

        if not (closed is None or isinstance(closed, bool)):
            msg = f"'closed' must be a boolean or None, got {closed!r}"
            raise GeometryError(msg)

        # Δₖ = xₖ₊₁ − xₖ, all of one sign
        spacing = jnp.diff(values)
        if repeats == "forbid":
            # Δₖ > 0 for every k, or Δₖ < 0 for every k
            increasing = jnp.all(spacing > 0)
            decreasing = jnp.all(spacing < 0)
            message = "'values' must be strictly monotone"
        else:
            # Δₖ ≥ 0 (or ≤ 0) for every k, and the span x_last − x_first ≠ 0
            positive_span = values[-1] != values[0]
            non_decreasing = jnp.all(spacing >= 0)
            non_increasing = jnp.all(spacing <= 0)
            increasing = non_decreasing & positive_span
            decreasing = non_increasing & positive_span
            message = "'values' must be monotone with a positive span"
            if values.shape[0] == 1:
                increasing = jnp.ones((), dtype=bool)
        monotone = increasing | decreasing
        values = _check_values(values, ~monotone, message)
        if period is not None:
            # |x_last − x_first| < P, or two nodes would be the same point
            span = jnp.abs(values[-1] - values[0])
            values = _check_values(
                values, span >= period, f"'values' must span less than {period}"
            )

        self.values = values
        self.period = period
        self.epoch = None if epoch is None else _as_epoch(epoch)
        self.edges = None if edges is None else jnp.asarray(edges)
        self.repeats = repeats
        if period is None:
            closed = None
        elif closed is None:
            decided = _cells_span_period(given_values, edges, period, values.dtype)
            # traced nodes keep the wrap cell
            closed = True if decided is None else decided
        self.closed = closed

    def __check_init__(self) -> None:
        if self.edges is None:
            return
        n_edges = self.values.shape[0] + 1
        if self.edges.ndim != 1 or self.edges.shape[0] != n_edges:
            msg = (
                f"'edges' must be 1-D with one more entry than 'values' "
                f"({n_edges}), got shape {self.edges.shape}"
            )
            raise GeometryError(msg)
        _require_floating(self.edges, "edges")
        # the monotone check needs values, which a traced axis does not have;
        # a conservative plan reads the edges on the host and sees them then
        if is_tracer(self.edges) or is_tracer(self.values):
            return
        edges = np.asarray(self.edges)
        values = np.asarray(self.values)
        # x_{k+½} − x_{k−½}, all of the sign of xₖ₊₁ − xₖ (either sign for one node)
        widths = np.diff(edges)
        steps = np.sign(widths)
        if values.shape[0] == 1:
            direction = steps[0] if steps[0] != 0 else 1.0
        else:
            # the ends, since repeats may make the first step zero
            direction = np.sign(values[-1] - values[0])
        if not np.all(steps == direction):
            msg = (
                "'edges' must be strictly monotone in the direction of 'values', "
                f"got {edges!r}"
            )
            raise GeometryError(msg)
        if self.period is not None and abs(edges[-1] - edges[0]) > self.period:
            msg = f"'edges' must span at most {self.period}, got {edges!r}"
            raise GeometryError(msg)

    def cell_edges(self) -> Float[Array, " edge"]:
        """The N + 1 cell edges: ``edges``, or the midpoints between nodes.

        Without explicit edges, x_{k−½} = ½ (xₖ₋₁ + xₖ) between nodes, and the
        end cells are symmetric about their nodes, x_{−½} = x₀ − ½ (x₁ − x₀)
        (as xrtoolz's ``_bounds_from_centres``). On a closed axis the end
        edges are instead the midpoint of the wrap gap between x_{N−1} and
        x₀ ± P, so x_{N−½} = ½ (x_{N−1} + x₀ ± P) and x_{−½} = x_{N−½} ∓ P,
        and the cells tile the circle.

        Returns:
            The edges x_{k±½}, in the dtype of ``values`` (or of ``edges``).

        Raises:
            GeometryError: If the axis has one node and no ``edges``, since
                a single node has no spacing to put its edges at.

        Examples:
            >>> import jax.numpy as jnp
            >>> from functax.geometry import Axis
            >>> Axis(jnp.array([0.0, 1.0, 3.0])).cell_edges().tolist()
            [-0.5, 0.5, 2.0, 4.0]
            >>> longitude = Axis(jnp.array([10.0, 20.0, 340.0]), period=360.0)
            >>> longitude.cell_edges().tolist()
            [-5.0, 15.0, 180.0, 355.0]
            >>> region = Axis(jnp.array([170.0, 180.0, 190.0]), period=360.0)
            >>> region.cell_edges().tolist()
            [165.0, 175.0, 185.0, 195.0]
        """
        if self.edges is not None:
            return self.edges
        values = self.values
        if values.shape[0] == 1:
            msg = "a one-node Axis has no spacing, so its cell needs 'edges'"
            raise GeometryError(msg)
        # ½ (xₖ₋₁ + xₖ) between nodes
        midpoints = 0.5 * (values[:-1] + values[1:])
        if self.period is None or not self.closed:
            # x_{−½} = x₀ − ½ (x₁ − x₀)
            first = values[0] - 0.5 * (values[1] - values[0])
            # x_{N−½} = x_{N−1} + ½ (x_{N−1} − x_{N−2})
            last = values[-1] + 0.5 * (values[-1] - values[-2])
        else:
            # the wrap gap runs from x_{N−1} to x₀ ± P, the sign of the axis
            period = jnp.where(values[-1] > values[0], self.period, -self.period)
            period = period.astype(values.dtype)
            # x_{N−½} = ½ (x_{N−1} + x₀ ± P)
            last = 0.5 * (values[-1] + values[0] + period)
            # x_{−½} = x_{N−½} ∓ P
            first = last - period
        return einx.id(", edge, -> (1 + edge + 1)", first, midpoints, last)

    @classmethod
    def from_datetimes(
        cls,
        times: Any,
        *,
        epoch: Any = None,
    ) -> Axis:
        """An axis of float seconds since ``epoch``, built on the host.

        Float64 seconds resolve a nanosecond only within about ±48 days of the
        epoch, and a microsecond within about ±68 years, so pick an epoch
        near the times. JAX's 64-bit mode must be on; without it JAX warns and
        the seconds lose most of their precision.

        Args:
            times: The times, anything ``numpy.datetime64`` accepts.
            epoch: The time origin; the first time when ``None``.

        Returns:
            An `Axis` whose ``values`` are float64 seconds since ``epoch``.

        Raises:
            GeometryError: If a time or the epoch is NaT, or the times are not
                strictly monotone.

        Examples:
            >>> import numpy as np
            >>> from functax.geometry import Axis
            >>> hours = np.arange(
            ...     "2026-01-01T00", "2026-01-01T04", dtype="datetime64[h]"
            ... )
            >>> axis = Axis.from_datetimes(hours)
            >>> axis.values.tolist(), axis.epoch
            ([0.0, 3600.0, 7200.0, 10800.0], '2026-01-01T00:00:00.000000000')
            >>> bool((axis.to_datetimes() == hours).all())
            True

            An explicit epoch shifts the seconds:

            >>> axis = Axis.from_datetimes(hours, epoch="2025-12-31T23")
            >>> axis.values[:2].tolist()
            [3600.0, 7200.0]
        """
        moments = np.asarray(times).astype("datetime64[ns]")
        if moments.ndim != 1 or moments.shape[0] < 1:
            msg = f"'times' must be 1-D and non-empty, got shape {moments.shape}"
            raise GeometryError(msg)
        if np.isnat(moments).any():
            msg = f"'times' must not contain NaT, got {moments!r}"
            raise GeometryError(msg)
        origin_string = _as_epoch(moments[0] if epoch is None else epoch)
        origin = np.datetime64(origin_string)
        # Δt in integer nanoseconds
        elapsed = moments - origin
        offsets = elapsed.astype(np.int64)
        # t = s + n / 10⁹: whole seconds and the nanosecond remainder, so the
        # integer part is exact and only the sum rounds
        whole_seconds, nanoseconds = np.divmod(offsets, 1_000_000_000)
        seconds = whole_seconds.astype(np.float64) + nanoseconds / 1e9
        values = jnp.asarray(seconds, dtype=jnp.float64)
        return cls(values, epoch=origin)

    def to_datetimes(self, *, unit: str = "ns") -> Shaped[np.ndarray, " node"]:
        """The node times as ``datetime64[ns]``, rounded to ``unit``.

        Args:
            unit: The resolution to round to: ``"s"``, ``"ms"``, ``"us"`` or
                ``"ns"``.

        Returns:
            A ``datetime64[ns]`` NumPy array.

        Raises:
            GeometryError: If the axis has no epoch or ``unit`` is unknown.

        Examples:
            >>> import jax.numpy as jnp
            >>> from functax.geometry import Axis
            >>> axis = Axis(jnp.array([0.0, 1.5]), epoch="2026-01-01")
            >>> axis.to_datetimes(unit="s").astype(str).tolist()
            ['2026-01-01T00:00:00.000000000', '2026-01-01T00:00:02.000000000']

            An axis that is not time has no datetimes:

            >>> Axis(jnp.array([0.0, 1.0])).to_datetimes()
            ... # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            functax.exceptions.GeometryError: 'epoch' is None, ...
        """
        if self.epoch is None:
            msg = "'epoch' is None, so the axis has no times to return"
            raise GeometryError(msg)
        if unit not in _UNITS_PER_SECOND:
            msg = f"'unit' must be one of {sorted(_UNITS_PER_SECOND)}, got {unit!r}"
            raise GeometryError(msg)
        units_per_second = _UNITS_PER_SECOND[unit]
        nanoseconds_per_unit = 1_000_000_000 // units_per_second
        seconds = np.asarray(self.values, dtype=np.float64)
        # t = ⌊t⌋ + (t − ⌊t⌋): both parts exact in float64
        whole_seconds = np.floor(seconds)
        fraction = seconds - whole_seconds
        # ⌊(t − ⌊t⌋) u + ½⌋: round half up, so 1.5 s is 2 s at unit="s"
        rounded = np.floor(fraction * units_per_second + 0.5)
        fraction_units = rounded.astype(np.int64)
        # ⌊t⌋ u + the rounded fraction, in whole units
        total_units = whole_seconds.astype(np.int64) * units_per_second
        total_units = total_units + fraction_units
        total_nanoseconds = total_units * nanoseconds_per_unit
        offsets = total_nanoseconds.astype("timedelta64[ns]")
        origin = np.datetime64(self.epoch, "ns")
        return origin + offsets

    @property
    def is_regular(self) -> bool:
        """Whether the nodes are equally spaced, to 1e-12 relative.

        max |Δₖ − Δ̄| ≤ 10⁻¹² |Δ̄|, with Δₖ = xₖ₊₁ − xₖ and Δ̄ their mean;
        a one-node axis has no spacing and is not regular. Computed on the
        host, so it needs concrete values.
        """
        if self.values.shape[0] == 1:
            return False
        values = np.asarray(self.values)
        # Δₖ = xₖ₊₁ − xₖ
        spacing = np.diff(values)
        # Δ̄ = (1/(N − 1)) ∑ₖ Δₖ
        mean_spacing = spacing.mean()
        # max |Δₖ − Δ̄|
        absolute_deviation = np.abs(spacing - mean_spacing)
        deviation = np.max(absolute_deviation)
        # max |Δₖ − Δ̄| ≤ 10⁻¹² |Δ̄|
        return bool(deviation <= _REGULAR_RTOL * abs(mean_spacing))

    @property
    def is_descending(self) -> bool:
        """Whether the nodes decrease; never for one node. Needs concrete values."""
        # np.asarray first, so it also works on a concrete axis inside jit
        ends = np.asarray(self.values)[[0, -1]]
        return bool(ends[0] > ends[1])


class AffineGrid(eqx.Module):
    """A regular 2-D grid given by an affine transform and a shape.

    The pixel corner at column j and row i maps to x = a·j + b·i + c and
    y = d·j + e·i + f, so the centre of pixel (i, j) is T(j + ½, i + ½). For a
    north-up grid b = d = 0, a is the pixel width and e is minus the pixel
    height. The order is geoarray's and rasterio's, not GDAL's.

    Args:
        transform: T = (a, b, c, d, e, f), six numbers. Stored as float64 by
            geoarray's rule, so JAX's 64-bit mode must be on. `from_affine`
            accepts an ``Affine`` or a 3 × 3 matrix as well.
        shape: ``(rows, cols)``, both positive.

    Raises:
        GeometryError: If ``transform`` is not six numbers, is not float64
            (64-bit mode off), is not finite, or is singular (a·e − b·d = 0);
            if ``shape`` is not two positive integers.

    Examples:
        >>> from functax.geometry import AffineGrid
        >>> grid = AffineGrid(
        ...     [20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0], (600, 600)
        ... )
        >>> grid.shape, grid.transform.dtype
        ((600, 600), dtype('float64'))
        >>> AffineGrid(
        ...     [1.0, 2.0, 0.0, 2.0, 4.0, 0.0], (2, 2)
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        functax.exceptions.GeometryError: 'transform' must be invertible ...
    """

    transform: Float[Array, " 6"]
    shape: tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        transform: ArrayLike | Sequence[float],
        shape: Sequence[int | np.integer],
    ) -> None:
        transform = jnp.asarray(transform, dtype=jnp.float64)
        if transform.shape != (6,):
            msg = (
                "'transform' must have shape (6,), (a, b, c, d, e, f), "
                f"got {transform.shape}"
            )
            raise GeometryError(msg)
        if transform.dtype != jnp.float64:
            msg = (
                "'transform' must be float64, got "
                f"{transform.dtype}; enable jax_enable_x64"
            )
            raise GeometryError(msg)
        self.shape = _as_grid_shape(shape)

        a, b, _, d, e, _ = (transform[k] for k in range(6))
        # det A = a·e − b·d ≠ 0 for the linear part A, or the grid has no inverse
        determinant = a * e - b * d
        finite = jnp.all(jnp.isfinite(transform))
        transform = _check_values(transform, ~finite, "'transform' must be finite")
        transform = _check_values(
            transform,
            determinant == 0,
            "'transform' must be invertible (a·e − b·d ≠ 0)",
        )
        self.transform = transform

    @classmethod
    def from_affine(
        cls, transform: Any, shape: Sequence[int | np.integer]
    ) -> AffineGrid:
        """A grid from an ``Affine``, six numbers or a 3 × 3 affine matrix.

        Args:
            transform: Anything with attributes ``a`` to ``f``, six numbers,
                or a ``(3, 3)`` matrix whose last row is (0, 0, 1).
            shape: ``(rows, cols)``.

        Returns:
            The `AffineGrid`.

        Raises:
            GeometryError: As `AffineGrid`, or if ``transform`` describes no
                affine transform.

        Examples:
            >>> from functax.geometry import AffineGrid
            >>> matrix = [[0.5, 0.0, 10.0], [0.0, -0.5, 20.0], [0.0, 0.0, 1.0]]
            >>> AffineGrid.from_affine(matrix, (4, 8)).transform.tolist()
            [0.5, 0.0, 10.0, 0.0, -0.5, 20.0]
        """
        return cls(as_transform(transform), shape)


def _as_grid_shape(shape: Any) -> tuple[int, int]:
    """``(rows, cols)`` as two positive Python ints."""
    values = tuple(shape)
    valid = len(values) == 2 and all(
        isinstance(n, int | np.integer) and not isinstance(n, bool) and n > 0
        for n in values
    )
    if not valid:
        msg = f"'shape' must be two positive integers (rows, cols), got {shape!r}"
        raise GeometryError(msg)
    return (int(values[0]), int(values[1]))


class RectilinearGrid(eqx.Module):
    """A tensor-product grid with one `Axis` per dimension.

    A field on it has shape ``(*batch, N₁, …, N_D)``: the interpolated
    dimensions are last, in the order of ``axes``, and leading ones are batch.

    Args:
        axes: The D axes, at least one.

    Raises:
        GeometryError: If ``axes`` is empty or holds anything but `Axis`.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, RectilinearGrid
        >>> latitude = Axis(jnp.linspace(-90.0, 90.0, 73))
        >>> longitude = Axis(jnp.linspace(0.0, 357.5, 144), period=360.0)
        >>> RectilinearGrid((latitude, longitude)).shape
        (73, 144)

        One axis is a 1-D grid; no axis is an error:

        >>> RectilinearGrid([latitude]).shape
        (73,)
        >>> RectilinearGrid(())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        functax.exceptions.GeometryError: 'axes' must hold at least one Axis, ...
    """

    axes: tuple[Axis, ...] = eqx.field(converter=tuple)

    def __check_init__(self) -> None:
        if not self.axes:
            msg = "'axes' must hold at least one Axis, got ()"
            raise GeometryError(msg)
        for index, axis in enumerate(self.axes):
            if not isinstance(axis, Axis):
                msg = f"'axes[{index}]' must be an Axis, got {type(axis).__name__}"
                raise GeometryError(msg)

    @property
    def shape(self) -> tuple[int, ...]:
        """``(N₁, …, N_D)``, the number of nodes along each axis."""
        return tuple(axis.values.shape[0] for axis in self.axes)


class CurvilinearGrid(eqx.Module):
    """A 2-D grid whose node coordinates are given per node, such as a swath.

    Args:
        x: The x coordinate of each node, at least 2 × 2.
        y: The y coordinate of each node, the same shape as ``x``.

    Raises:
        GeometryError: If ``x`` or ``y`` is not 2-D, not floating, smaller than
            2 × 2, or the two shapes differ.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import CurvilinearGrid
        >>> rows, cols = jnp.meshgrid(
        ...     jnp.arange(3.0), jnp.arange(4.0), indexing="ij"
        ... )
        >>> CurvilinearGrid(x=cols + 0.1 * rows, y=rows).shape
        (3, 4)

        A single row is too thin to be a grid:

        >>> CurvilinearGrid(
        ...     x=jnp.zeros((1, 4)), y=jnp.zeros((1, 4))
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        functax.exceptions.GeometryError: 'x' must be at least 2 × 2, ...
    """

    x: Float[Array, "row col"] = eqx.field(converter=jnp.asarray)
    y: Float[Array, "row col"] = eqx.field(converter=jnp.asarray)

    def __check_init__(self) -> None:
        for name, array in (("x", self.x), ("y", self.y)):
            if array.ndim != 2:
                msg = f"'{name}' must be 2-D (row, col), got shape {array.shape}"
                raise GeometryError(msg)
            if min(array.shape) < 2:
                msg = f"'{name}' must be at least 2 × 2, got shape {array.shape}"
                raise GeometryError(msg)
            _require_floating(array, name)
        if self.x.shape != self.y.shape:
            msg = (
                f"'x' and 'y' must have the same shape, got {self.x.shape} "
                f"and {self.y.shape}"
            )
            raise GeometryError(msg)

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, cols)``."""
        rows, cols = self.x.shape
        return (rows, cols)


def _optional_array(value: ArrayLike | None) -> Shaped[Array, "*shape"] | None:
    """``jnp.asarray`` that passes ``None`` through."""
    return None if value is None else jnp.asarray(value)


class Points(eqx.Module):
    """Scattered sites, with the metric their distances use.

    The field names follow geoarray's ``GeoPoints``: ``xy``, a separate ``z``
    and a ``valid`` mask, so a fixed number of points can trace with padding.

    Args:
        xy: The site coordinates, one row per site, D columns.
        z: An optional vertical coordinate per site.
        valid: An optional boolean mask; ``False`` marks padding.
        metric: The distance neighbours are measured in: ``"euclidean"``
            in the coordinates as given, ``"chord"`` or ``"geodesic"`` on the
            sphere (#17).

    Raises:
        GeometryError: If ``xy`` is not 2-D or not floating; if ``z`` or
            ``valid`` does not have one entry per site, or ``valid`` is not
            boolean; if ``metric`` is unknown.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Points
        >>> sites = Points(
        ...     jnp.array([[0.0, 1.0], [2.0, 3.0], [0.0, 0.0]]),
        ...     valid=jnp.array([True, True, False]),
        ... )
        >>> sites.xy.shape, sites.metric
        ((3, 2), 'euclidean')
        >>> Points(
        ...     jnp.zeros((2, 2)), metric="manhattan"
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        functax.exceptions.GeometryError: 'metric' must be one of ...
    """

    xy: Float[Array, "obs dim"] = eqx.field(converter=jnp.asarray)
    z: Float[Array, " obs"] | None = eqx.field(default=None, converter=_optional_array)
    valid: Bool[Array, " obs"] | None = eqx.field(
        default=None, converter=_optional_array
    )
    # str rather than Metric, so a bad value reaches the GeometryError below
    # instead of a type checker's error
    metric: str = eqx.field(static=True, default="euclidean")

    def __check_init__(self) -> None:
        if self.xy.ndim != 2:
            msg = f"'xy' must be 2-D (obs, dim), got shape {self.xy.shape}"
            raise GeometryError(msg)
        _require_floating(self.xy, "xy")
        n_sites = self.xy.shape[0]
        if self.z is not None:
            if self.z.shape != (n_sites,):
                msg = f"'z' must have shape ({n_sites},), got {self.z.shape}"
                raise GeometryError(msg)
            _require_floating(self.z, "z")
        if self.valid is not None:
            if self.valid.shape != (n_sites,):
                msg = f"'valid' must have shape ({n_sites},), got {self.valid.shape}"
                raise GeometryError(msg)
            if self.valid.dtype != jnp.bool_:
                msg = f"'valid' must be boolean, got {self.valid.dtype}"
                raise GeometryError(msg)
        if self.metric not in _METRICS:
            msg = f"'metric' must be one of {list(_METRICS)}, got {self.metric!r}"
            raise GeometryError(msg)
