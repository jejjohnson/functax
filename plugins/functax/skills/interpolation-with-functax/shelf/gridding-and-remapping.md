# Gridding and conservative remapping

Role: ② strategy (`role-ladder`, `references/strategy.md`).
Unit: interp.

Open this when values move from scattered samples to grid cells, or from the cells of one grid to the cells of another, and the totals must be kept.

## Names

All of them live in `functax.interp`.

| Name | What it is |
|---|---|
| `Binning(statistic="mean", weights="nearest")` | Aggregates scattered samples into the cells of a grid. `statistic` is `"mean"`, `"sum"` or `"count"`; `weights="linear"` is cloud-in-cell, where a sample splits between its neighbours. |
| `BinningPlan` | The plan `Binning.plan(samples, grid)` returns: from N samples to M cells, with its `adjoint` and `operator`. |
| `TransposedStencil` | The read-out inside a `BinningPlan`: a sampling stencil used backwards and scaled per cell. |
| `Conservative(sphere=False, unit=None)` | First-order conservative remapping: each target cell's area-weighted mean of the source cells it overlaps. `sphere=True` weights by spherical area, with `unit=` for the longitudes and latitudes. |
| `overlap_weights_1d(source_edges, target_edges, *, period=None)` | The `(M, N)` overlap lengths of every target cell with every source cell on one axis, the building block of `Conservative` on tensor grids. |

## Use it

```python
import jax.numpy as jnp

from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import Binning, Conservative, Policies, overlap_weights_1d

# scattered samples into the cells of an axis
cells = Axis(jnp.arange(4.0))
samples = Points(jnp.array([[0.1], [0.4], [2.2], [9.0]]))  # the last is off the grid
values = jnp.array([1.0, 3.0, 5.0, 7.0])
mean = Binning("mean").plan(samples, cells)(values)
count = Binning("count").plan(samples, cells)(values)
assert count.tolist() == [2.0, 0.0, 1.0, 0.0]
assert float(mean[0]) == 2.0 and bool(jnp.isnan(mean[1]))  # an empty cell is NaN

# a 6 × 8 grid of unit cells onto a 2 × 3 grid of larger cells
source = RectilinearGrid((Axis(jnp.arange(6.0) + 0.5), Axis(jnp.arange(8.0) + 0.5)))
target = RectilinearGrid(
    (
        Axis(jnp.array([1.5, 4.5]), edges=jnp.array([0.0, 3.0, 6.0])),
        Axis(jnp.array([4 / 3, 4.0, 20 / 3]), edges=jnp.linspace(0.0, 8.0, 4)),
    )
)
field = jnp.arange(48.0)
plan = Conservative().plan(source, target)
remapped = plan(field)  # (6,)
# each target cell has area 8, each source cell area 1: the integral is kept
assert jnp.allclose(8.0 * remapped.sum(), field.sum())

# land in the first three columns: renormalise over the cells that have data
land = jnp.zeros((6, 8), bool).at[:, :3].set(True)
masked = jnp.where(land.ravel(), jnp.nan, field)
coastal = Conservative().plan(source, target, policies=Policies(missing="renormalise"))
assert int(jnp.isnan(coastal(masked)).sum()) == 2  # only the cells that are all land
assert int(jnp.isnan(plan(masked)).sum()) == 4  # the default lets NaN through

lengths = overlap_weights_1d(jnp.arange(7.0), jnp.array([0.0, 3.0, 6.0]))  # (2, 6)
assert lengths.sum(axis=1).tolist() == [3.0, 3.0]
```

## Rules

- Use `Conservative` when the quantity is a density or a flux and its integral must survive the regrid; use `Linear` when you sample a smooth field at points.
- Give the cell edges with `Axis(values, edges=...)` when cells are not centred midway between nodes. `Conservative` reads the edges, not the nodes.
- On a lon/lat grid pass `Conservative(sphere=True, unit="degrees")`, so that cells are weighted by their area on the sphere.
- `Policies(missing="renormalise")` averages over the source cells that have data, which is what a land mask or a cloud mask needs.
- A cell that receives no sample is NaN under `"mean"` and 0 under `"sum"` and `"count"`. Read `plan.valid` to tell an empty cell from a zero.
- A linear `Binning` covers the hull of the nodes, so a sample in the outer half of an end cell is dropped; pass `policies=Policies(extent="cells")` to keep it, on the end node (an `Axis` or a `RectilinearGrid` only).
- With `weights="nearest"` an axis with `edges` bins each sample between them, e_k ≤ x < e_{k+1}, as `numpy.histogram` does on stretched cells; an axis without edges bins by the nearest node.
- Binning is the adjoint of sampling: `Binning("sum")` on the samples of a nearest-neighbour plan is that plan's `adjoint`.
