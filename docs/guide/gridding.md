# Gridding scattered data

Scattered observations (stations, swaths, particles) carry no grid, so the support of each target comes from a search or a triangulation rather than from arithmetic on an axis. Once the support is known, the result is the same kind of plan as the local schemes in the [interpolation guide](./interpolation.md): a sparse read-out $\mat{H}$ built once on the host, applied to every field in JAX, with the adjoint $\mat{H}^\top$ that sends residuals back to the sources.

## Inverse distance

[`InverseDistance`](xref:api#functax.interp.InverseDistance) weighs the $Q$ nearest sources of each target by $\max(d_{ij}, \varepsilon)^{-p}$, normalised to sum to one. Planning onto a grid keeps its shape, so the result reshapes straight back:

```python
import einx
import jax.numpy as jnp
import numpy as np
from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import InverseDistance

rng = np.random.default_rng(0)
stations = Points(jnp.asarray(rng.uniform(0.0, 10.0, (200, 2))))
values = jnp.asarray(rng.normal(size=200))
grid = RectilinearGrid(
    (Axis(jnp.linspace(0.0, 10.0, 21)), Axis(jnp.linspace(0.0, 10.0, 31)))
)

plan = InverseDistance(k=8, power=2.0, radius=1.5).plan(stations, grid)
gridded = einx.id("(y x) -> y x", plan(values), y=21)
gridded.shape  # (21, 31)
```

A cell with no station within the radius is not valid and reads NaN (the `fill_value` of the plan's [`Policies`](xref:api#functax.interp.Policies)); with `radius=None` every cell reads its eight nearest stations. A target that coincides with a station takes that station's value, through the floor $\varepsilon$.

The distance is the source's metric. On longitude and latitude, `Points(lonlat, metric="chord")` measures the straight line between unit vectors, so a target at $180°$ finds the stations on both sides of the dateline; pass `unit="degrees"` (or `"radians"`) to `plan`. The radius is then a chord of the unit sphere, $2 \sin(\theta / 2)$ for an angle $\theta$.

The neighbour search runs on the host with SciPy's k-d tree, which needs the `functax[scipy]` extra. Inside `jit` or `grad`, where the coordinates are traced, the same plan searches by brute force over chunks of targets instead, and gives the same stencil; the neighbour set is piecewise constant, so gradients with respect to the coordinates come through the weights.

## Triangulation

[`Delaunay`](xref:api#functax.interp.Delaunay) triangulates the stations and reads each target as the barycentric average of its triangle's three corners. The weights are linear in the target position, so the plan reproduces any affine field inside the convex hull of the stations; outside it a target is not valid and reads NaN:

```python
from functax.interp import Delaunay

targets = Points(jnp.array([[5.0, 5.0], [12.0, 5.0]]))
plan = Delaunay().plan(stations, targets)
plan.read.indices.shape  # (2, 3)
plan.valid.tolist()  # [True, False]
```

A mesh that already has its triangles (an icosphere patch, a finite-element mesh) passes them in instead, `Delaunay(simplices=triangles)`, and the plan locates each target in those triangles rather than in a fresh triangulation; that path does not need SciPy. The same scheme works in three dimensions with tetrahedra, four taps per target.

## Binning and cloud-in-cell

When there are many samples per cell, aggregate them instead of interpolating. [`Binning`](xref:api#functax.interp.Binning) is sampling run backwards: the stencil $\mat{H}$ that would read the grid at the sample positions, applied as $\mat{H}^\top$, sends each sample to its cells. With `weights="nearest"` a sample lands in the cell of its nearest node, so the sums are a histogram and the means divide by the counts $\vect{n} = \mat{H}^\top \vect{1}$; with `weights="linear"` a sample spreads over the $2^D$ nodes around it, which is cloud-in-cell deposition:

```python
from functax.interp import Binning

particles = Points(jnp.asarray(rng.uniform(0.0, 10.0, (1000, 2))))
masses = jnp.asarray(rng.uniform(0.5, 2.0, 1000))

cic = Binning("sum", weights="linear").plan(particles, grid)
deposited = cic(masses)
bool(jnp.allclose(deposited.sum(), masses.sum()))  # True: the mass is conserved

means = Binning("mean").plan(particles, grid)
int(jnp.isnan(means(masses)).sum())  # 155 empty cells read NaN
```

Samples off the grid are dropped from every cell rather than clamped onto the edge. The counts depend only on the positions, so for fixed positions the sum and the mean are linear in the data, and `plan.adjoint` sends a residual on the cells back to the samples. The plan also traces in the positions, so `jax.grad` of a loss on the deposited field reaches the particle positions through the linear weights; with nearest weights that gradient is zero, as it is for any histogram.
