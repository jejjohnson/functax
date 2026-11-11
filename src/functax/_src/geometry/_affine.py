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
    "slice_transform",
    "to_pixel",
    "window",
]


def _require_six(transform: Any) -> None:
    """Raise unless the last axis of an array transform has the six (a, … f)."""
    shape = getattr(transform, "shape", None)
    if shape is not None and (len(shape) == 0 or shape[-1] != 6):
        msg = f"'transform' must have shape (*batch, 6), got {tuple(shape)}"
        raise GeometryError(msg)


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


def _batched_coefficients(transform: Any, ndim: int) -> tuple[Any, ...]:
    """(a, b, c, d, e, f) of a (*batch, 6) transform, each shaped (*batch, 1, …, 1).

    The ``ndim`` − len(batch) trailing ones let each coefficient broadcast
    against an array of ``ndim`` dims whose leading dims are the batch.
    """
    trailing = (None,) * (ndim - (transform.ndim - 1))
    return tuple(transform[(..., k, *trailing)] for k in range(6))


def from_pixel(transform: Any, rowcol: Any) -> Any:
    """Map fractional pixel (row, col) to coordinates (x, y): the inverse of `to_pixel`.

    x = a·(col + ½) + b·(row + ½) + c and y = d·(col + ½) + e·(row + ½) + f.

    A batch of transforms reads a batch of pixel sets: transform t of the
    batch maps the pixels ``rowcol[t]``, as for the transforms of a batch of
    chips.

    Args:
        transform: T = (a, b, c, d, e, f): six numbers, or an array of shape
            ``(*batch, 6)``, NumPy or JAX.
        rowcol: The fractional (row, col), shape ``(*batch, ..., 2)``, centres
            at integers; its leading dims are the transform's batch.

    Returns:
        The positions (x, y), shape ``(*batch, ..., 2)``, in the namespace of
        the inputs.

    Raises:
        GeometryError: If an array ``transform`` does not end in six, or
            ``rowcol`` has fewer leading dims than its batch.

    Examples:
        >>> import numpy as np
        >>> from functax.geometry import from_pixel, to_pixel
        >>> north_up = np.array([20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0])
        >>> from_pixel(north_up, np.array([[0.0, 0.0]])).tolist()
        [[600010.0, 3549990.0]]
        >>> xy = np.array([[600123.0, 3549876.0]])
        >>> bool(np.allclose(from_pixel(north_up, to_pixel(north_up, xy)), xy))
        True
        >>> # two chips, three pixels each: (2, 6) transforms, (2, 3, 2) pixels
        >>> chips = np.stack([north_up, north_up + [0, 0, 100.0, 0, 0, 0]])
        >>> from_pixel(chips, np.zeros((2, 3, 2)))[:, 0].tolist()
        [[600010.0, 3549990.0], [600110.0, 3549990.0]]
    """
    xp = namespace(transform, rowcol)
    _require_six(transform)
    batch_ndim = getattr(transform, "ndim", 1) - 1
    if batch_ndim == 0:
        a, b, c, d, e, f = (transform[k] for k in range(6))
    else:
        ndim = rowcol.ndim - 1
        if ndim < batch_ndim:
            msg = (
                f"'rowcol' must lead with the transform's batch dims "
                f"{tuple(transform.shape[:-1])}, got shape {tuple(rowcol.shape)}"
            )
            raise GeometryError(msg)
        a, b, c, d, e, f = _batched_coefficients(transform, ndim)
    # the corner-convention (col, row) of the area-convention (row, col)
    col = rowcol[..., 1] + 0.5
    row = rowcol[..., 0] + 0.5
    # x = a·j + b·i + c, y = d·j + e·i + f
    x = a * col + b * row + c
    y = d * col + e * row + f
    return xp.stack([x, y], axis=-1)


def slice_transform(
    transform: Any,
    row0: Any,
    col0: Any,
    row_step: Any = 1,
    col_step: Any = 1,
) -> Any:
    """The transform of the slice ``[row0::row_step, col0::col_step]`` of a raster.

    The corner (i, j) of the slice's pixels is the corner (r₀ + sᵣ i,
    c₀ + s_c j) of the raster's, so T′ = (a′, b′, c′, d′, e′, f′) is

        a′ = a s_c,  b′ = b sᵣ,  c′ = c + a c₀ + b r₀,
        d′ = d s_c,  e′ = e sᵣ,  f′ = f + d c₀ + e r₀.

    With unit steps the slice's pixels are the raster's. With a step s > 1
    each pixel of the slice covers a block of s raster pixels, and its centre
    is the block's centre, (s − 1)/2 raster pixels past the centre of the
    pixel the slice samples. This is geoarray's ``shift_scale_transform``,
    with the same arithmetic, so the two agree bit for bit. Every operation is
    elementwise, so a batch of transforms takes a batch of offsets and steps,
    and the offsets may be traced.

    Args:
        transform: T = (a, b, c, d, e, f), shape ``(*batch, 6)``, NumPy or
            JAX (a sequence of six numbers counts as NumPy).
        row0: r₀, the first row of the slice; a number or ``(*batch,)``.
        col0: c₀, the first column of the slice; a number or ``(*batch,)``.
        row_step: sᵣ, the row step; a number or ``(*batch,)``.
        col_step: s_c, the column step; a number or ``(*batch,)``.

    Returns:
        T′, shape ``(*batch, 6)``, in the namespace of the inputs.

    Raises:
        GeometryError: If ``transform`` does not end in six.

    Examples:
        >>> import numpy as np
        >>> from functax.geometry import from_pixel, slice_transform
        >>> north_up = np.array([20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0])
        >>> chip = slice_transform(north_up, 5, 10, row_step=2, col_step=2)
        >>> chip.tolist()
        [40.0, 0.0, 600200.0, 0.0, -40.0, 3549900.0]
        >>> # the corner of chip pixel (1, 1) is the corner of raster pixel (7, 12)
        >>> bool(
        ...     np.array_equal(
        ...         from_pixel(chip, np.array([0.5, 0.5])),
        ...         from_pixel(north_up, np.array([6.5, 11.5])),
        ...     )
        ... )
        True
        >>> # a batch of two transforms, one offset each
        >>> slice_transform(np.stack([north_up] * 2), np.array([0, 1]), 0).shape
        (2, 6)
    """
    xp = namespace(transform, row0, col0, row_step, col_step)
    transform = xp.asarray(transform)
    _require_six(transform)
    a, b, c, d, e, f = (transform[..., k] for k in range(6))
    coefficients = (
        a * col_step,
        b * row_step,
        c + a * col0 + b * row0,
        d * col_step,
        e * row_step,
        f + d * col0 + e * row0,
    )
    return xp.stack(coefficients, axis=-1)


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
