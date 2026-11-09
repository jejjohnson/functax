"""Affine raster maths on the array namespace of the input: pixels, centres, windows.

The transform T = (a, b, c, d, e, f) maps the pixel corner at column j and
row i to x = a·j + b·i + c, y = d·j + e·i + f. Pixel (row, col) coordinates
here follow the area convention: integers are pixel centres, so (0, 0) is the
centre of the top-left pixel and its corner is (−½, −½). This is geoarray's
convention, and `to_pixel` reproduces geoarray's arithmetic bit for bit.

The kernels run on NumPy alone and import JAX only when a JAX array arrives,
so NumPy-only consumers (geoarray, geoops, xrops) call them from the base
install (decisions 16 and 17). On JAX they trace under ``jit``, ``vmap`` and
``grad``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from functax._src._backend import is_tracer, namespace
from functax.exceptions import GeometryError


__all__ = [
    "from_pixel",
    "pixel_centres",
    "to_pixel",
    "window",
]


def _require_invertible(transform: Any) -> None:
    """Raise when a concrete transform is singular; traced ones are not checked."""
    if is_tracer(transform):
        return
    # np.asarray before indexing: under jit, indexing a concrete JAX array
    # stages a traced slice, which float() cannot read
    a, b, _, d, e, _ = np.asarray(transform, dtype=np.float64).tolist()
    if a * e - b * d == 0.0:
        msg = f"'transform' must be invertible (a·e − b·d ≠ 0), got {transform!r}"
        raise GeometryError(msg)


def to_pixel(transform: Any, xy: Any) -> Any:
    """Map coordinates (x, y) to fractional pixel (row, col), centres at integers.

    Inverting the 2 × 2 part of T,

        (j, i)ᵀ = [[a, b], [d, e]]⁻¹ (x − c, y − f)ᵀ,

    and subtracting ½ from both puts integers at pixel centres, so (0, 0) is
    the centre of the top-left pixel.

    Args:
        transform: T = (a, b, c, d, e, f), six numbers, NumPy or JAX.
        xy: The positions (x, y) in the transform's CRS, shape ``(..., 2)``.

    Returns:
        The fractional (row, col), shape ``(..., 2)``, in the namespace of
        the inputs (JAX if either is a JAX array).

    Raises:
        GeometryError: If a concrete ``transform`` is singular. A traced one
            is not checked; `AffineGrid` guarantees an invertible transform.

    Examples:
        >>> import numpy as np
        >>> from functax.geometry import to_pixel
        >>> north_up = np.array([20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0])
        >>> to_pixel(north_up, np.array([[600010.0, 3549990.0]])).tolist()
        [[0.0, 0.0]]
        >>> to_pixel(north_up, np.array([[600000.0, 3550000.0]])).tolist()
        [[-0.5, -0.5]]
    """
    xp = namespace(transform, xy)
    _require_invertible(transform)
    a, b, c, d, e, f = (transform[k] for k in range(6))
    # det A = a·e − b·d
    determinant = a * e - b * d
    # (x − c, y − f)
    dx = xy[..., 0] - c
    dy = xy[..., 1] - f
    # A⁻¹ = [[e, −b], [−d, a]] / det A, then − ½ for the centre convention
    col = (e * dx - b * dy) / determinant - 0.5
    row = (-d * dx + a * dy) / determinant - 0.5
    return xp.stack([row, col], axis=-1)


def from_pixel(transform: Any, rowcol: Any) -> Any:
    """Map fractional pixel (row, col) to coordinates (x, y): the inverse of `to_pixel`.

    x = a·(col + ½) + b·(row + ½) + c and y = d·(col + ½) + e·(row + ½) + f.

    Args:
        transform: T = (a, b, c, d, e, f), six numbers, NumPy or JAX.
        rowcol: The fractional (row, col), shape ``(..., 2)``, centres at
            integers.

    Returns:
        The positions (x, y), shape ``(..., 2)``, in the namespace of the
        inputs.

    Examples:
        >>> import numpy as np
        >>> from functax.geometry import from_pixel, to_pixel
        >>> north_up = np.array([20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0])
        >>> from_pixel(north_up, np.array([[0.0, 0.0]])).tolist()
        [[600010.0, 3549990.0]]
        >>> xy = np.array([[600123.0, 3549876.0]])
        >>> bool(np.allclose(from_pixel(north_up, to_pixel(north_up, xy)), xy))
        True
    """
    xp = namespace(transform, rowcol)
    a, b, c, d, e, f = (transform[k] for k in range(6))
    # the corner-convention (col, row) of the area-convention (row, col)
    col = rowcol[..., 1] + 0.5
    row = rowcol[..., 0] + 0.5
    # x = a·j + b·i + c, y = d·j + e·i + f
    x = a * col + b * row + c
    y = d * col + e * row + f
    return xp.stack([x, y], axis=-1)


def pixel_centres(grid: Any) -> tuple[Any, Any]:
    """The x and y of every pixel centre of a grid, each of shape (rows, cols).

    The centre of pixel (i, j) is T(j + ½, i + ½).

    Args:
        grid: An `AffineGrid`, or anything with a ``transform`` of six numbers
            and a ``shape`` of ``(rows, cols)`` (geoarray's ``Grid`` too).

    Returns:
        ``(x, y)``, each ``(rows, cols)``, in the namespace of the transform.

    Examples:
        >>> import types
        >>> from functax.geometry import pixel_centres
        >>> grid = types.SimpleNamespace(
        ...     transform=(10.0, 0.0, 0.0, 0.0, -10.0, 100.0), shape=(2, 3)
        ... )
        >>> x, y = pixel_centres(grid)
        >>> x.tolist()
        [[5.0, 15.0, 25.0], [5.0, 15.0, 25.0]]
        >>> y.tolist()
        [[95.0, 95.0, 95.0], [85.0, 85.0, 85.0]]
    """
    transform = grid.transform
    xp = namespace(transform)
    rows, cols = grid.shape
    a, b, c, d, e, f = (transform[k] for k in range(6))
    # j + ½ and i + ½ for every column and row
    col_centres = xp.arange(cols, dtype=xp.float64) + 0.5
    row_centres = xp.arange(rows, dtype=xp.float64) + 0.5
    col_grid, row_grid = xp.meshgrid(col_centres, row_centres, indexing="xy")
    # x = a·j + b·i + c, y = d·j + e·i + f
    x = a * col_grid + b * row_grid + c
    y = d * col_grid + e * row_grid + f
    return x, y


def _snap(value: float, atol: float) -> float:
    """``value`` rounded to the nearest integer when within ``atol`` of it."""
    nearest = round(value)
    return float(nearest) if abs(value - nearest) <= atol else value


def window(
    grid: Any,
    bounds: Sequence[float | int],
    *,
    atol: float = 1e-6,
) -> tuple[slice, slice]:
    """The row and column slices of the pixels that cover ``bounds``.

    The four corners of ``bounds`` map to pixel-corner coordinates, and the
    window rounds outward to whole pixels; a corner within ``atol`` pixels of a
    pixel edge snaps to it, so bounds on the grid select exactly the pixels
    inside them. These are geoarray's ``Grid.window`` semantics.

    The window is not clipped to the grid: a start can be negative and a stop
    can pass the shape, so clip before indexing an array with it (a negative
    start would otherwise count from the end).

    Args:
        grid: An `AffineGrid`, or anything with a ``transform`` and a
            ``shape``. Computed on the host, so the transform must be concrete.
        bounds: ``(left, bottom, right, top)``, that is ``(x_min, y_min,
            x_max, y_max)``, in the grid's CRS.
        atol: The snapping tolerance, in pixels.

    Returns:
        ``(rows, cols)``, two step-1 slices.

    Raises:
        GeometryError: If ``bounds`` is not four numbers with min ≤ max, or
            the transform is singular.

    Examples:
        >>> import types
        >>> from functax.geometry import window
        >>> grid = types.SimpleNamespace(
        ...     transform=(20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0),
        ...     shape=(600, 600),
        ... )
        >>> window(grid, (600100, 3549000, 600400, 3549900))
        (slice(5, 50, None), slice(5, 20, None))
    """
    values = tuple(float(v) for v in bounds)
    if len(values) != 4 or values[0] > values[2] or values[1] > values[3]:
        msg = (
            "'bounds' must be (x_min, y_min, x_max, y_max) with min ≤ max, "
            f"got {bounds!r}"
        )
        raise GeometryError(msg)
    x_min, y_min, x_max, y_max = values
    a, b, c, d, e, f = np.asarray(grid.transform, dtype=np.float64).tolist()
    # det A = a·e − b·d
    determinant = a * e - b * d
    if determinant == 0.0:
        msg = f"'transform' must be invertible (a·e − b·d ≠ 0), got {grid.transform!r}"
        raise GeometryError(msg)
    corners = ((x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max))
    # T⁻¹: (j, i) = A⁻¹ (x − c, y − f), written as six coefficients
    inverse_a = e / determinant
    inverse_b = -b / determinant
    inverse_c = (b * f - c * e) / determinant
    inverse_d = -d / determinant
    inverse_e = a / determinant
    inverse_f = (c * d - a * f) / determinant
    cols = [inverse_a * x + inverse_b * y + inverse_c for x, y in corners]
    rows = [inverse_d * x + inverse_e * y + inverse_f for x, y in corners]
    row_start = math.floor(_snap(min(rows), atol))
    row_stop = math.ceil(_snap(max(rows), atol))
    col_start = math.floor(_snap(min(cols), atol))
    col_stop = math.ceil(_snap(max(cols), atol))
    return slice(row_start, row_stop), slice(col_start, col_stop)
