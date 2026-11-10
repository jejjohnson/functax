"""Cell-area rules on grids: `cell_area`, the cell centres weighted by their areas.

A field of cell means integrates as ∑ₖ |Cₖ| yₖ. The cells of an affine or a
rectilinear grid are in the plane, or are lon/lat cells on a sphere of a given
radius.
"""

from __future__ import annotations

from functax._src._backend import is_tracer, require_jax


require_jax(__name__)

import math
from typing import Any

import einx
import jax.numpy as jnp
import numpy as np

from functax._src.geometry._affine import pixel_centres
from functax._src.geometry._sphere import _check_unit
from functax._src.geometry._transform import as_transform
from functax._src.geometry._types import AffineGrid, RectilinearGrid
from functax._src.quad._rules import Rule
from functax.exceptions import GeometryError, ValidationError


__all__ = [
    "cell_area",
]


def cell_area(
    grid: AffineGrid | RectilinearGrid,
    *,
    sphere_radius: float | int | None = None,
    unit: str | None = None,
) -> Rule:
    """The cell-area rule of a grid: ∫ g dA ≈ ∑ₖ |Cₖ| g(xₖ) over its cells Cₖ.

    The nodes xₖ are the cell centres and the weights the cell areas |Cₖ|,
    one per cell, in the order of the field's flattened (row-major) values:

    - `AffineGrid` (a, b, c, d, e, f): every pixel has area |ae − bd|, and
      the nodes are the pixel centres (x, y), rows outer.
    - `RectilinearGrid` with D axes: cell k has the volume ∏_d |e_{d,k+1} −
      e_{d,k}| from each axis's `Axis.cell_edges` (its ``edges``, or the
      midpoints between nodes), and the nodes are the axis values, the
      first axis outer. Two axes give areas, one lengths.

    With ``sphere_radius`` a the coordinates are longitude λ and latitude φ
    in ``unit``, and a lon/lat cell has the area

        |C| = a² |Δλ| |sin φ_top − sin φ_bottom|,

    so a global grid sums to 4πa². An `AffineGrid` must then be north-up
    (b = d = 0), x being λ and y being φ; a `RectilinearGrid` must have two
    axes in (λ, φ) order, as `Conservative` reads it with ``sphere=True``.
    Latitude edges past a pole are clipped to it, as `Conservative` does.
    The radius is the caller's (an Earth radius is not functax's).

    Algorithm:
        1. The edges of each axis (from the transform, or `Axis.cell_edges`).
        2. The width of each cell along each axis (on the sphere, |Δλ| in
           radians and |Δ sin φ|).
        3. The outer product of the widths, flattened; the cell centres.

    Args:
        grid: The grid of cells.
        sphere_radius: a, the radius of the sphere the lon/lat cells lie on,
            positive; ``None`` for cells in the plane.
        unit: ``"degrees"`` or ``"radians"``, the unit of (λ, φ); required
            with ``sphere_radius`` and ``None`` without it.

    Returns:
        A `Rule` of nodes (cells, D) and weights (cells,): the grid's units
        squared in the plane, a²'s units on the sphere. An `AffineGrid` gives
        float64; a `RectilinearGrid` keeps the dtype of its axes.

    Raises:
        ValidationError: If ``unit`` is given without ``sphere_radius`` or is
            missing with it, or ``sphere_radius`` is not positive and finite.
        GeometryError: If ``grid`` is not an `AffineGrid` or a
            `RectilinearGrid`, ``unit`` is unknown, or on the sphere the
            `AffineGrid` is not north-up or the `RectilinearGrid` does not
            have two axes.

    Examples:
        >>> import math
        >>> import jax.numpy as jnp
        >>> from functax.geometry import AffineGrid, Axis, RectilinearGrid
        >>> from functax.quad import cell_area
        >>> # 20 m pixels: every cell is 400 m²
        >>> grid = AffineGrid([20.0, 0.0, 0.0, 0.0, -20.0, 0.0], (3, 4))
        >>> rule = cell_area(grid)
        >>> rule.nodes.shape, rule.weights[:2].tolist()
        ((12, 2), [400.0, 400.0])
        >>> # uneven cells: the areas sum to the domain's
        >>> axes = (
        ...     Axis(jnp.array([0.5, 2.0]), edges=jnp.array([0.0, 1.0, 3.0])),
        ...     Axis(
        ...         jnp.array([1.0, 3.0, 4.5]),
        ...         edges=jnp.array([0.0, 2.0, 4.0, 5.0]),
        ...     ),
        ... )
        >>> cell_area(RectilinearGrid(axes)).weights.tolist()
        [2.0, 2.0, 1.0, 4.0, 4.0, 2.0]
        >>> # a global 1° grid on the unit sphere sums to 4π
        >>> globe = AffineGrid([1.0, 0.0, -180.0, 0.0, -1.0, 90.0], (180, 360))
        >>> areas = cell_area(globe, sphere_radius=1.0, unit="degrees").weights
        >>> math.isclose(float(areas.sum()), 4 * math.pi, rel_tol=1e-12)
        True
    """
    _check_sphere(sphere_radius, unit)
    if isinstance(grid, AffineGrid):
        widths, nodes = _affine_cells(grid, unit=unit)
    elif isinstance(grid, RectilinearGrid):
        widths, nodes = _rectilinear_cells(grid, unit=unit)
    else:
        msg = (
            "'grid' must be an AffineGrid or a RectilinearGrid, got "
            f"{type(grid).__name__}"
        )
        raise GeometryError(msg)
    # |Cₖ| = ∏_d w_{d,k}, the first axis outer
    weights = widths[0]
    for width in widths[1:]:
        weights = einx.multiply("outer, inner -> (outer inner)", weights, width)
    if sphere_radius is not None:
        # a² |Δλ| |Δ sin φ|
        weights = weights * sphere_radius**2
    return Rule(nodes, weights)


def _check_sphere(sphere_radius: Any, unit: Any) -> None:
    """``unit`` goes with ``sphere_radius``; the radius is positive and finite."""
    if sphere_radius is None:
        if unit is not None:
            msg = f"'unit' must be None without 'sphere_radius', got {unit!r}"
            raise ValidationError(msg)
        return
    if unit is None:
        msg = "'unit' must be 'degrees' or 'radians' with 'sphere_radius', got None"
        raise ValidationError(msg)
    _check_unit(unit)
    if is_tracer(sphere_radius):
        return
    value = float(sphere_radius)
    if not (math.isfinite(value) and value > 0.0):
        msg = f"'sphere_radius' must be positive and finite, got {sphere_radius!r}"
        raise ValidationError(msg)


def _affine_cells(grid: AffineGrid, *, unit: str | None) -> tuple[list[Any], Any]:
    """The widths along rows, then columns, and the (cells, 2) pixel centres.

    In the plane every pixel has area |ae − bd|, carried as one width per row
    and ones along the columns. On the sphere (north-up) the rows are the
    latitude bands between edges f + e·i and the columns the longitude
    intervals between edges c + a·j.
    """
    a, b, c, d, e, f = as_transform(np.asarray(grid.transform)).tolist()
    rows, cols = grid.shape
    # (x, y) of every pixel centre, rows outer
    x, y = pixel_centres(grid)
    nodes = einx.id("row col, row col -> (row col) (1 + 1)", x, y)
    if unit is None:
        # |ae − bd|, the area of the parallelogram each pixel maps to
        pixel_area = abs(a * e - b * d)
        row_widths = jnp.full((rows,), pixel_area, dtype=jnp.float64)
        col_widths = jnp.ones((cols,), dtype=jnp.float64)
        return [row_widths, col_widths], nodes
    if b != 0.0 or d != 0.0:
        msg = (
            "'grid' must be a north-up AffineGrid (b = d = 0) on the sphere, "
            f"got b={b}, d={d}"
        )
        raise GeometryError(msg)
    # φ = f + e·i at row edge i, λ = c + a·j at column edge j
    lat_edges = f + e * jnp.arange(rows + 1, dtype=jnp.float64)
    lon_edges = c + a * jnp.arange(cols + 1, dtype=jnp.float64)
    lon_widths, band_heights = _lonlat_widths(lon_edges, lat_edges, unit=unit)
    return [band_heights, lon_widths], nodes


def _rectilinear_cells(
    grid: RectilinearGrid, *, unit: str | None
) -> tuple[list[Any], Any]:
    """The widths along each axis and the (cells, D) node coordinates."""
    if unit is not None and len(grid.axes) != 2:
        msg = f"'grid' must have two axes (λ, φ) on the sphere, got {len(grid.axes)}"
        raise GeometryError(msg)
    edges = [axis.cell_edges() for axis in grid.axes]
    # the node coordinates of every cell, the first axis outer
    names = " ".join(f"axis{index}" for index in range(len(grid.axes)))
    meshes = jnp.meshgrid(*(axis.values for axis in grid.axes), indexing="ij")
    stacked = jnp.stack(meshes)
    nodes = einx.id(f"dim {names} -> ({names}) dim", stacked)
    if unit is not None:
        lon_widths, band_heights = _lonlat_widths(*edges, unit=unit)
        return [lon_widths, band_heights], nodes
    # |eₖ₊₁ − eₖ|, positive on a descending axis too
    widths = [jnp.abs(jnp.diff(edge)) for edge in edges]
    return widths, nodes


def _lonlat_widths(lon_edges: Any, lat_edges: Any, *, unit: str) -> tuple[Any, Any]:
    """|Δλ| in radians and |sin φ_top − sin φ_bottom| from lon and lat edges."""
    if unit == "degrees":
        lon_edges = jnp.deg2rad(lon_edges)
        lat_edges = jnp.deg2rad(lat_edges)
    # a latitude edge past a pole would fold the band back over it
    half_pi = jnp.asarray(math.pi / 2, dtype=lat_edges.dtype)
    lat_edges = jnp.clip(lat_edges, -half_pi, half_pi)
    lon_widths = jnp.abs(jnp.diff(lon_edges))
    band_heights = jnp.abs(jnp.diff(jnp.sin(lat_edges)))
    return lon_widths, band_heights
