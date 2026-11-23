# Schemes and policies

Role: ② strategy (`role-ladder`, `references/strategy.md`).
Unit: interp.

Open this when you choose how a field is interpolated, and what happens to a target off the domain, to a NaN in the data and to a stencil at the edge.

## Names

All of them live in `functax.interp`. A scheme is configuration only: it holds no data and no geometry.

| Name | Source | What it is |
|---|---|---|
| `Scheme` | | The base class: `plan(source, target, *, policies=None)`, and `get_config()` / `from_config(config)` for strict JSON. |
| `LocalScheme` | a grid | A scheme with one local basis per axis and the identity fit; it adds `fit(source, data)`. |
| `Nearest()`, `Linear()`, `Cubic(a=-0.5)` | a grid | Nearest neighbour, linear, and Keys cubic convolution: 1, 2 and 4 taps per axis, no solve. |
| `BSpline(order=3)` | a grid | A B-spline of order k, `k + 1` taps per axis, after a prefilter of the data. |
| `Tensor(schemes)` | a grid | One local scheme per source axis, such as nearest in time and linear in space. |
| `SeriesScheme` | one `Axis` | The base of the 1-D schemes with a cubic Hermite read-out. |
| `CubicSpline(bc="not-a-knot")`, `Pchip()`, `Akima()`, `Steffen()` | one `Axis` | The C² cubic spline, and three local cubics; `Pchip` and `Steffen` do not overshoot. |
| `Chebyshev(n_coefficients, kind="lobatto")` | Chebyshev nodes | The Chebyshev series of the samples, read at any points. |
| `Fourier()` | one periodic `Axis` | Band-limited resampling from N to M nodes by two FFTs. |
| `Delaunay(simplices=None)` | `Points` | Linear interpolation on a triangulation of scattered sources. |
| `InverseDistance(k=8, power=2.0, radius=None)` | `Points` | Inverse-distance weighting of the `k` nearest sources. |
| `Policies(outside="nan", missing="propagate", boundary="mirror", extent="nodes", fill_value=nan, spacing="coordinates")` | | The named rules a plan follows; see the table below. |

| Policy | Options |
|---|---|
| `outside` | `"nan"` marks the row invalid and writes `fill_value`; `"clamp"` moves the target to the nearest boundary; `"extrapolate"` keeps the scheme's weights; `"raise"` raises `OutsideDomainError` when the plan is built. |
| `missing` | `"propagate"` lets NaN through; `"renormalise"` drops those taps and divides by the remaining weight; `"fill"` replaces them by `fill_value` first. |
| `boundary` | `"mirror"`, `"wrap"`, `"edge"` or `"constant"`, for a tap past the end of an axis; a periodic axis always wraps. |
| `extent` | `"nodes"`: the domain ends at the first and last node; `"cells"`: half a cell wider at each end. |
| `spacing` | `"coordinates"`: `Cubic`, and `BSpline` from order 2, refuse an unevenly spaced axis; `"index"`: they read it in index space. |

## Use it

```python
import jax.numpy as jnp

from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import (
    BSpline,
    CubicSpline,
    InverseDistance,
    Linear,
    Nearest,
    Pchip,
    Policies,
    Scheme,
    Tensor,
)

# time, y, x: nearest in time, linear in space
cube = RectilinearGrid(
    (Axis(jnp.arange(24.0)), Axis(jnp.arange(4.0)), Axis(jnp.arange(5.0)))
)
target = Points(
    jnp.array([[3.4, 1.5, 2.5], [3.4, 9.0, 2.5]])
)  # the second is off the grid
plan = Tensor((Nearest(), Linear(), Linear())).plan(cube, target)
field = jnp.arange(24 * 4 * 5.0)
values = plan(field)  # (2,)
assert plan.valid.tolist() == [True, False] and bool(jnp.isnan(values[1]))

clamped = Linear().plan(cube, target, policies=Policies(outside="clamp"))
assert bool(jnp.isfinite(clamped(field)).all())

# one axis: a spline and a monotone cubic through the same samples
axis = Axis(jnp.linspace(0.0, 1.0, 6))
samples = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
fine = Axis(jnp.linspace(0.0, 1.0, 41))
smooth = CubicSpline().plan(axis, fine)(samples)
monotone = Pchip().plan(axis, fine)(samples)
assert float(smooth.max()) > 1.0  # the spline overshoots the step
assert float(monotone.max()) <= 1.0 + 1e-6  # PCHIP does not

# scattered sources
sites = Points(jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
centre = Points(jnp.array([[0.5, 0.5]]))
mean = InverseDistance(k=4).plan(sites, centre)(jnp.array([1.0, 2.0, 3.0, 4.0]))
assert jnp.allclose(mean, 2.5)

# a scheme round-trips through JSON
assert Scheme.from_config(BSpline(3).get_config()) == BSpline(3)
```

## Rules

- Pick the scheme from the source: a grid takes a `LocalScheme`, one axis takes a `SeriesScheme`, `Chebyshev` or `Fourier`, scattered points take `Delaunay`, `InverseDistance` or a kernel scheme.
- `Linear` is the default: two taps per axis, no overshoot, exact adjoint. Use `Cubic` or `BSpline(3)` on evenly spaced axes when the derivative of the read field must be continuous; on one uneven `Axis` use `CubicSpline` or `Pchip`.
- `Pchip`, `Akima` and `Steffen` choose their slopes from the data, so their plans are nonlinear in the data: they have no fixed operator, and `NonlinearSchemeError` says so.
- State the policies when a default would hide a problem. `outside="nan"` is the default, so always read `plan.valid` or expect NaN at targets off the domain.
- A raster's pixels are cells, not nodes: pass `Policies(extent="cells", boundary="edge")` to read up to the outer pixel edges.
- An unknown option raises `PolicyError` when `Policies` is built, not when the plan runs.
