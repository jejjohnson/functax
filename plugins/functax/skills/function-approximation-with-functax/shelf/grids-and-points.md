# Axes, grids and point sets

Role: ⓪ structure (`role-ladder`, `references/structure.md`).
Unit: geometry.

Open this when you have coordinates and need the object that every functax operation takes as its source or its target.

## Names

All of them live in `functax.geometry`.

| Name | What it is |
|---|---|
| `Axis(values, *, period=None, epoch=None, edges=None, repeats="forbid", closed=None)` | One strictly monotone coordinate axis. `period=` makes it periodic, `closed=` says whether its cells go round the whole period (decided from concrete values when `None`), `edges=` gives the cell edges when they are not midway between nodes, `epoch=` makes it a time axis in seconds. `axis.cell_edges()`, `axis.is_regular()` and `axis.is_descending()` describe it. |
| `AffineGrid(transform, shape)` | A regular 2-D grid from a six-number affine transform `(a, b, c, d, e, f)` and `(rows, cols)`: a raster. |
| `RectilinearGrid(axes)` | A tensor-product grid with one `Axis` per dimension, in any number of dimensions. |
| `CurvilinearGrid(x, y)` | A 2-D grid whose node coordinates are given per node, such as a satellite swath. |
| `Points(xy, z=None, valid=None, metric="euclidean")` | Scattered sites, one row per site, with the metric their distances use. `valid=False` marks padding, so a fixed number of points can trace. |
| `Metric` | The type of `metric=`: `"euclidean"`, `"chord"` or `"geodesic"`. |
| `pairwise_distance(sources, targets, *, radius=1.0, unit="radians")` | The `(N, M)` distances between two `Points` in the metric they share. |
| `chebyshev_nodes(n, *, kind="lobatto", interval=(-1.0, 1.0))` | The Chebyshev nodes of degree `n`, from the upper end of the interval down to the lower. |

## Use it

```python
import jax
import jax.numpy as jnp

from functax.geometry import (
    AffineGrid,
    Axis,
    CurvilinearGrid,
    Points,
    RectilinearGrid,
    chebyshev_nodes,
    pairwise_distance,
)

jax.config.update("jax_enable_x64", True)  # an AffineGrid holds float64

# a raster of 4 rows and 5 columns with 10 m pixels, north up
raster = AffineGrid([10.0, 0.0, 500.0, 0.0, -10.0, 900.0], (4, 5))

# a lon/lat grid: the longitude is periodic, the latitude is not
lon = Axis(jnp.arange(0.0, 360.0, 30.0), period=360.0)
lat = Axis(jnp.linspace(-75.0, 75.0, 6))
grid = RectilinearGrid((lon, lat))
assert grid.shape == (12, 6)
assert lat.cell_edges().shape == (7,)

# a swath: the coordinates of every node
rows, cols = jnp.meshgrid(jnp.arange(3.0), jnp.arange(4.0), indexing="ij")
swath = CurvilinearGrid(x=cols + 0.1 * rows, y=rows)
assert swath.shape == (3, 4)

# scattered sites, in degrees, with great-circle distances
sites = Points(jnp.array([[0.0, 0.0], [90.0, 0.0]]), metric="geodesic")
d = pairwise_distance(sites, sites, radius=6371.0, unit="degrees")  # (2, 2), km
assert jnp.allclose(d[0, 1], 6371.0 * jnp.pi / 2, rtol=1e-5)

nodes = chebyshev_nodes(8)  # (9,), from 1 down to -1
assert nodes.shape == (9,)
```

## Rules

- A field on a grid is one flat vector in row-major order: `(rows * cols,)` on an `AffineGrid`, the product of the axis lengths on a `RectilinearGrid`. Leading batch dimensions go in front.
- The axes of a `RectilinearGrid` come in the order of the columns of the target points: a grid of `(lon, lat)` axes is read at points whose columns are `(lon, lat)`.
- An `AffineGrid` needs `jax_enable_x64`: map coordinates in metres lose pixels in float32, so it raises `GeometryError` without it.
- The geometry carries no CRS. Reproject first, then build the grid in the coordinates you will interpolate in.
- A periodic axis must span less than its period, so do not repeat the first node at the end.
- A periodic axis is closed (a global longitude, with a wrap cell) or open (a region: targets in any convention fold to it, and `outside` applies past its ends). A traced axis counts as closed, so pass `closed=False` for a regional one built inside `jit`, and `closed=True` for one node that stands for the whole circle.
- Use `Points(metric="geodesic")` or `"chord"` for sites on the sphere; a kernel scheme then measures neighbours the same way.
- A malformed axis, grid or point set raises `GeometryError`, which is also a `ValueError`.
