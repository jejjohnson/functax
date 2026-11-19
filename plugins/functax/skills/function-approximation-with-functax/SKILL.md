---
name: function-approximation-with-functax
description: Write function approximation in JAX on functax instead of hand-rolled gather, searchsorted and lstsq code — CRS-free axes, affine, rectilinear and curvilinear grids, scattered points, sphere coordinates, frames and point sets, affine raster maths on NumPy alone, and the exception hierarchy. Use whenever a task interpolates, regrids, bins, integrates, fits a basis to or transforms a field on a grid, a swath, scattered points or the sphere, in a project that uses (or could use) functax. The front door to functax; it routes to functax:interpolation-with-functax and functax:bases-quadrature-and-transforms.
---

# Function approximation with functax

functax writes a field as one expansion, f̂(x) = ∑ⱼ cⱼ φⱼ(x), and gives five operations on it: interpolate, differentiate, integrate, project and transform.
Each operation takes the same geometry objects, so a grid is described once and reused by all of them.

This skill covers the lowest rung of functax, ⓪ structures: the geometry and the errors, in the vocabulary of the `role-ladder` skill.

## Where to enter

| The task | Open |
|---|---|
| Describe a grid, a swath, scattered sites or points on the sphere | this skill |
| Read a field at points, regrid it, bin samples into cells, krige | `functax:interpolation-with-functax` |
| Build a design matrix, fit it, integrate, differentiate a callable field, transform to coefficients | `functax:bases-quadrature-and-transforms` |

## The three rules

1. **A field is a flat vector.** On a grid it has shape `(..., N)` in row-major order, with batch dimensions in front. Every operation reads and writes that layout.
2. **Geometry carries no CRS.** Coordinates are numbers. Reproject before you build a grid, and state `unit=` wherever a longitude and latitude go in.
3. **Build once, apply many times.** A plan, a rule, a basis and a transform are built on the host, outside `jit`, and then applied inside it.

## Nouns and verbs

| Rung | Names |
|---|---|
| ⓪ Structures | `Axis`, `AffineGrid`, `RectilinearGrid`, `CurvilinearGrid`, `Points`, from `functax.geometry` |
| ① Verbs on them | `lonlat_to_unit`, `geodesic_distance`, `pairwise_distance`, `fibonacci_sphere`, `to_pixel`, `window` |
| Errors | `functax.FunctaxError` and its six subclasses |

## Worked example

```python
import jax
import jax.numpy as jnp

from functax.geometry import AffineGrid, Axis, Points, RectilinearGrid, lonlat_to_unit
from functax.interp import Linear
from functax.quad import cell_area

jax.config.update("jax_enable_x64", True)  # an AffineGrid holds float64

# one description of the grid: 4 rows and 5 columns of 10 m pixels, north up
grid = AffineGrid([10.0, 0.0, 500.0, 0.0, -10.0, 900.0], (4, 5))
field = jnp.arange(20.0)  # (N,), row-major

# interpolate: read it at two sites
sites = Points(jnp.array([[515.0, 885.0], [540.0, 870.0]]))
plan = Linear().plan(grid, sites)
assert plan(field).tolist() == [6.0, 16.0]

# integrate: the same grid gives its cell areas
areas = cell_area(grid).weights  # (4, 5), 100 m² each
assert jnp.allclose((areas.ravel() * field).sum(), 100.0 * field.sum())

# a lon/lat grid, with a periodic longitude, and sites on the sphere
globe = RectilinearGrid(
    (Axis(jnp.arange(0.0, 360.0, 30.0), period=360.0), Axis(jnp.linspace(-75.0, 75.0, 6)))
)
assert globe.shape == (12, 6)
unit = lonlat_to_unit(jnp.array([[2.35, 48.86]]), unit="degrees")  # (1, 3)
assert jnp.allclose(jnp.linalg.norm(unit, axis=-1), 1.0)
```

## Don't write it

| Don't write | Use |
|---|---|
| `jnp.searchsorted` and hand-made weights to read a grid at points | `Linear().plan(grid, points)` |
| A scatter-add for the transpose of an interpolation | `plan.adjoint(residual)` |
| `np.histogram2d` to grid scattered samples | `Binning("mean").plan(samples, grid)` |
| A haversine formula | `geodesic_distance(u, v, radius=...)` on unit vectors |
| `cos(lat)` weights on a lon/lat grid | `functax.quad.sphere_latlon`, or `cell_area(grid, sphere_radius=...)` |
| Inverting the six affine numbers to find a pixel | `to_pixel(transform, xy)` and `window(grid, bounds)` |
| `jnp.linalg.lstsq` onto a hand-built design matrix | a basis from `functax.basis` and `functax.project.least_squares` |

## The shelf

Each entry is one file with the names it covers, a worked example and the rules.
Open the one the task needs; do not read them all.

<!-- shelf:begin -->
<!-- shelf:end -->

## Before you finish

- Is every grid built once and passed to each operation, rather than rebuilt from raw arrays?
- Is `unit=` stated wherever a longitude and latitude go in?
- Does NumPy-only code import from `functax.geometry.affine` alone?
- Is every field flattened row-major, with batch dimensions in front?
