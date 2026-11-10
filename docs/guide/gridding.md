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
