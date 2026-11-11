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

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float

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


def _check_values(array: jax.Array, bad: jax.Array, message: str) -> jax.Array:
    """`check_values` raising `GeometryError`."""
    return check_values(array, bad, message, error=GeometryError)


def _require_floating(array: jax.Array, name: str) -> None:
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


class Axis(eqx.Module):
    """One strictly monotone coordinate axis, optionally periodic or in time.

    A periodic axis with period P treats x and x + kP as the same point for
    every integer k, so its nodes must span less than P. Descending axes are
    allowed (wavelength from wavenumber). Times are stored as float seconds
    since ``epoch``; `from_datetimes` converts them on the host.

    An axis of one node x₀ (a single time step, a raster one row thick) has
    no spacing, so it carries no direction and `cell_edges` needs explicit
    ``edges``. A scheme reads it with weight 1 on its node: under
    ``extent="nodes"`` a target is inside when x = x₀, and under
    ``extent="cells"`` when e₀ ≤ x ≤ e₁.

    Args:
        values: x, the node coordinates, at least one, strictly increasing or
            strictly decreasing. Kept in the dtype they arrive in.
        period: P, the period (360.0 for longitude in degrees), or ``None``
            for an axis that is not periodic.
        epoch: The time origin of ``values`` (a ``datetime64`` or an ISO 8601
            string), or ``None`` for an axis that is not time. Stored as an
            ISO 8601 string at nanosecond resolution.
        edges: e, the n + 1 cell edges, node k owning the cell [eₖ, eₖ₊₁]
            (CF ``bounds``), strictly monotone in the direction of
            ``values`` and spanning at most P on a periodic axis; ``None``
            for the midpoints between nodes (`cell_edges`). Kept in the dtype
            they arrive in. Conservative remapping reads them.

    Raises:
        GeometryError: If ``values`` is not 1-D, is empty, is not floating,
            is not strictly monotone, or spans P or more; if ``period`` is
            not positive and finite; if ``epoch`` is not a time;
            if ``edges`` is not 1-D with n + 1 floating entries, or (when
            concrete) is not strictly monotone in the direction of ``values``
            or spans more than P.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis
        >>> longitude = Axis(jnp.linspace(0.0, 357.5, 144), period=360.0)
        >>> longitude.is_regular, longitude.is_descending
        (True, False)
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
        >>> # one time step, as a grid one node thick
        >>> Axis(jnp.array([3600.0])).values.shape
        (1,)
    """

    values: Float[Array, " node"]
    period: float | None = eqx.field(static=True)
    epoch: str | None = eqx.field(static=True)
    edges: Float[Array, " edge"] | None

    def __init__(
        self,
        values: ArrayLike | Sequence[float],
        *,
        period: float | int | None = None,
        epoch: Any = None,
        edges: ArrayLike | Sequence[float] | None = None,
    ) -> None:
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

        # Δₖ = xₖ₊₁ − xₖ, all of one sign
        spacing = jnp.diff(values)
        increasing = jnp.all(spacing > 0)
        decreasing = jnp.all(spacing < 0)
        values = _check_values(
            values, ~(increasing | decreasing), "'values' must be strictly monotone"
        )
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
        # eₖ₊₁ − eₖ, all of the sign of xₖ₊₁ − xₖ (either sign for one node)
        steps = np.sign(np.diff(edges))
        if values.shape[0] == 1:
            direction = steps[0] if steps[0] != 0 else 1.0
        else:
            direction = np.sign(values[1] - values[0])
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
        """The n + 1 cell edges: ``edges``, or the midpoints between nodes.

        Without explicit edges, eₖ = ½ (xₖ₋₁ + xₖ) between nodes, and the end
        cells are symmetric about their nodes, e₀ = x₀ − ½ (x₁ − x₀) (as
        xrtoolz's ``_bounds_from_centres``). On a periodic axis the end edges
        are instead the midpoint m of the wrap gap, between x_{n−1} and
        x₀ ± P, so e₀ = m ∓ P and e_n = m, and the cells tile the circle.

        Returns:
            e, in the dtype of ``values`` (or of ``edges``).

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
        """
        if self.edges is not None:
            return self.edges
        values = self.values
        if values.shape[0] == 1:
            msg = "a one-node Axis has no spacing, so its cell needs 'edges'"
            raise GeometryError(msg)
        # ½ (xₖ₋₁ + xₖ) between nodes
        midpoints = 0.5 * (values[:-1] + values[1:])
        if self.period is None:
            # x₀ − ½ (x₁ − x₀) and x_{n−1} + ½ (x_{n−1} − x_{n−2})
            first = values[0] - 0.5 * (values[1] - values[0])
            last = values[-1] + 0.5 * (values[-1] - values[-2])
        else:
            # the wrap gap runs from x_{n−1} to x₀ ± P, the sign of the axis
            period = jnp.where(values[-1] > values[0], self.period, -self.period)
            period = period.astype(values.dtype)
            last = 0.5 * (values[-1] + values[0] + period)
            first = last - period
        return jnp.concatenate([jnp.atleast_1d(first), midpoints, jnp.atleast_1d(last)])

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
        """
        moments = np.asarray(times).astype("datetime64[ns]")
        if moments.ndim != 1 or moments.shape[0] < 1:
            msg = f"'times' must be 1-D and non-empty, got shape {moments.shape}"
            raise GeometryError(msg)
        if np.isnat(moments).any():
            msg = f"'times' must not contain NaT, got {moments!r}"
            raise GeometryError(msg)
        origin = np.datetime64(_as_epoch(moments[0] if epoch is None else epoch))
        offsets = (moments - origin).astype(np.int64)
        # t = s + n / 10⁹: whole seconds and the nanosecond remainder, so the
        # integer part is exact and only the sum rounds
        whole_seconds, nanoseconds = np.divmod(offsets, 1_000_000_000)
        seconds = whole_seconds.astype(np.float64) + nanoseconds / 1e9
        values = jnp.asarray(seconds, dtype=jnp.float64)
        return cls(values, epoch=origin)

    def to_datetimes(self, *, unit: str = "ns") -> Any:
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
        # round half up, so 1.5 s is 2 s at unit="s"
        fraction_units = np.floor(fraction * units_per_second + 0.5).astype(np.int64)
        total_units = whole_seconds.astype(np.int64) * units_per_second
        total_units = total_units + fraction_units
        offsets = (total_units * nanoseconds_per_unit).astype("timedelta64[ns]")
        return np.datetime64(self.epoch, "ns") + offsets

    @property
    def is_regular(self) -> bool:
        """Whether the nodes are equally spaced, to 1e-12 relative.

        max |Δₖ − Δ̄| ≤ 10⁻¹² |Δ̄|, with Δₖ = xₖ₊₁ − xₖ and Δ̄ their mean;
        a one-node axis has no spacing and is not regular. Computed on the
        host, so it needs concrete values.
        """
        if self.values.shape[0] == 1:
            return False
        spacing = np.diff(np.asarray(self.values))
        mean_spacing = spacing.mean()
        deviation = np.max(np.abs(spacing - mean_spacing))
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
        # det = a·e − b·d ≠ 0, or the grid has no inverse
        determinant = a * e - b * d
        transform = _check_values(
            transform, ~jnp.all(jnp.isfinite(transform)), "'transform' must be finite"
        )
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

    A field on it has shape ``(*batch, n₁, …, n_D)``: the interpolated
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
        """``(n₁, …, n_D)``, the number of nodes along each axis."""
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


def _optional_array(value: ArrayLike | None) -> jax.Array | None:
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
