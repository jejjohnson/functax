# Affine raster maths

Role: ① primitive (`role-ladder`, `references/primitive.md`).
Unit: geometry.

Open this when you index a raster by coordinates: which pixel holds a point, where the pixel centres are, or which window of rows and columns covers a bounding box.

## Names

The five functions live in `functax.geometry.affine`, which needs NumPy alone, and are exported again from `functax.geometry`, which needs the `functax[jax]` extra.

| Name | What it does |
|---|---|
| `to_pixel(transform, xy)` | Coordinates `(x, y)` to fractional `(row, col)`, with pixel centres at integers. |
| `from_pixel(transform, rowcol)` | Fractional `(row, col)` to coordinates `(x, y)`: the inverse of `to_pixel`. |
| `pixel_centres(grid)` | The x and the y of every pixel centre, each of shape `(rows, cols)`. |
| `window(grid, bounds, *, atol=1e-6)` | The row slice and the column slice of the pixels that cover `bounds = (xmin, ymin, xmax, ymax)`. |
| `slice_transform(transform, row0, col0, row_step=1, col_step=1)` | The transform of the slice `[row0::row_step, col0::col_step]` of a raster. |

## Use it

```python
import numpy as np

from functax.geometry.affine import (
    from_pixel,
    pixel_centres,
    slice_transform,
    to_pixel,
    window,
)

# (a, b, c, d, e, f): x = a col + b row + c, y = d col + e row + f, north up
transform = np.array([10.0, 0.0, 500.0, 0.0, -10.0, 900.0])


class Raster:  # anything with a transform and a shape
    def __init__(self, transform, shape):
        self.transform, self.shape = transform, shape


raster = Raster(transform, (4, 5))

rowcol = to_pixel(transform, np.array([[515.0, 885.0]]))  # (1, 2)
assert np.allclose(rowcol, [[1.0, 1.0]])
assert np.allclose(from_pixel(transform, rowcol), [[515.0, 885.0]])

x, y = pixel_centres(raster)  # each (4, 5)
assert x.shape == (4, 5) and np.isclose(x[0, 0], 505.0) and np.isclose(y[0, 0], 895.0)

rows, cols = window(raster, (510.0, 870.0, 530.0, 890.0))
assert (rows, cols) == (slice(1, 3), slice(1, 3))

cropped = slice_transform(transform, rows.start, cols.start)
assert np.allclose(from_pixel(cropped, np.array([[0.0, 0.0]])), [[515.0, 885.0]])
```

## Rules

- Import from `functax.geometry.affine` in code that must run without JAX; the same names from `functax.geometry` pull in the extra.
- The functions run on the array namespace of their input, so NumPy in gives NumPy out and JAX in gives JAX out.
- Pixel centres sit at integer `(row, col)`: the corner of the first pixel is at `(-0.5, -0.5)`.
- `pixel_centres` and `window` take any object with a `transform` and a `shape`, and `functax.geometry.AffineGrid` is one.
- After cropping or striding an array, carry its georeference with `slice_transform`, not by editing the six numbers by hand.
