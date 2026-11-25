---
name: interpolation-with-functax
description: Interpolate, regrid and grid fields in JAX with functax — nearest, linear, cubic and B-spline schemes on grids, cubic splines and monotone cubics on one axis, Chebyshev and Fourier resampling, Delaunay and inverse-distance weighting on scattered points, radial basis functions and kriging with a posterior variance, conservative remapping, binning — as plans with hand-written adjoints, derivatives, windows and lineax operators. Use when reading a gridded field at points or along a track, regridding between grids, gridding scattered samples, building an observation operator for data assimilation, or advecting particles through a field, in a project that uses (or could use) functax.
---

# Interpolation with functax

A scheme says how to interpolate. A plan freezes it for one source and one target, and reads a field as f̂(X*) = H 𝒮(y): a fit 𝒮 and a read-out H.
The plan also sends residuals back with the adjoint 𝒮ᵀ Hᵀ, written by hand, which is what an observation operator in a variational problem needs.

This skill covers the strategies of functax's interpolation, with the primitives they are built from, in the vocabulary of the `role-ladder` skill.
The geometry a plan takes is in `functax:function-approximation-with-functax`.

## The contract

- `plan = scheme.plan(source, target, policies=...)`, built once, on the host.
- `plan(field)` maps `(..., N)` to `(..., M)`; `plan.adjoint(residual)` maps `(..., M)` back to `(..., N)`.
- `plan.valid` is `(M,)`: false where a target could not be read, and the value there is `fill_value`.
- `plan.operator()` is the same map as a lineax operator, and `plan.derivative(axis)` reads a derivative at the same targets.
- `scheme.fit(source, data)` returns a field you call at positions chosen inside a traced function.

## Which scheme

| Source and need | Choose |
|---|---|
| A grid, read at points: the default | `Linear()` |
| An evenly spaced grid, with a continuous derivative | `Cubic()` or `BSpline(3)` |
| A grid, with a different rule per axis | `Tensor((Nearest(), Linear(), Linear()))` |
| One axis, even or not, smooth | `CubicSpline()` |
| One axis, no overshoot between knots | `Pchip()` or `Steffen()` |
| One periodic axis, band-limited | `Fourier()` |
| Samples on Chebyshev nodes | `Chebyshev(n_coefficients)` |
| Scattered points, piecewise linear | `Delaunay()` |
| Scattered points, cheap and local | `InverseDistance(k=8)` |
| Scattered points, smooth | `RBF(Polyharmonic(2))` |
| Scattered points, with a variance | `Kriging(Wendland(2, support=...))` |
| Cells onto cells, keeping the integral | `Conservative()` |
| Scattered samples into cells | `Binning("mean")` |

## Worked example

```python
import jax
import jax.numpy as jnp

from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import Linear, Policies

# a model state on a 21 × 21 grid, observed at three sites
grid = RectilinearGrid(
    (Axis(jnp.linspace(0.0, 10.0, 21)), Axis(jnp.linspace(0.0, 10.0, 21)))
)
x, y = jnp.meshgrid(grid.axes[0].values, grid.axes[1].values, indexing="ij")
truth = (jnp.sin(0.5 * x) + 0.1 * y).ravel()  # (441,)
sites = Points(jnp.array([[2.0, 3.0], [4.5, 1.25], [7.3, 8.8]]))

# the observation operator, built once
H = Linear().plan(grid, sites, policies=Policies(outside="raise"))
observations = H(truth)  # (3,)


def misfit(state):
    return 0.5 * jnp.sum((H(state) - observations) ** 2)


# the gradient of the misfit is Hᵀ(H x − y): the adjoint, not a traced gather
state = jnp.zeros(441)
by_hand = H.adjoint(H(state) - observations)
assert jnp.allclose(jax.grad(misfit)(state), by_hand, atol=1e-5)

# a few steps of gradient descent pull the state towards the observations
for _ in range(50):
    state = state - 0.5 * H.adjoint(H(state) - observations)
assert float(misfit(state)) < 1e-3
```

## Don't write it

| Don't write | Use |
|---|---|
| `jnp.searchsorted` plus weights, per axis | a `LocalScheme` and its plan |
| `jax.scipy.ndimage.map_coordinates` | `Linear()`, or `BSpline(order)` for higher orders |
| `scipy.interpolate.CubicSpline`, `PchipInterpolator` on the host | `CubicSpline()`, `Pchip()`: the same curves, traced |
| `jax.vjp` of an interpolation to get its transpose | `plan.adjoint(residual)` |
| A dense `(M, N)` weight matrix | `plan.operator()`, or `plan.to_gaussx()` for the stencil |
| An `if` on the target's position inside `jit` | `Policies(outside=...)` and `plan.valid` |
| An area-weighted loop over overlapping cells | `Conservative().plan(source, target)` |

## The shelf

Each entry is one file with the names it covers, a worked example and the rules.
Open the one the task needs; do not read them all.

<!-- shelf:begin -->
| Role | Entry | Covers | Names |
|---|---|---|---|
| ② strategy | [Schemes and policies](shelf/schemes.md) | which interpolation to use on a grid, an axis or scattered points, and what happens off the domain, at NaN and at the edges | 17 |
| setup | [Plans, adjoints and interpolants](shelf/plans-and-adjoints.md) | a scheme frozen for one source and one target, its adjoint, derivative, window and composition, and fields read at targets chosen in the trace | 10 |
| ② strategy | [Gridding and conservative remapping](shelf/gridding-and-remapping.md) | binning scattered samples into cells and area-weighted remapping between grids | 5 |
| ② strategy | [Kernel interpolation and kriging](shelf/kernels-and-kriging.md) | radial basis functions with a polynomial tail, kriging with a posterior covariance, and the kernels they take | 12 |
| ① primitive | [Stencils and extension points](shelf/stencils-and-extension-points.md) | the sparse read-out, the per-axis bases, fits, prefilters and locate functions a new scheme is built from | 26 |
<!-- shelf:end -->

<!-- shelf:below:begin -->
## The units below this skill

A card is enough to call a unit.
Load its skill only when the card's last line applies.

### Geometry · ⓪ · skill `functax:function-approximation-with-functax`

- **Provides.** CRS-free geometry: one axis, three kinds of grid, scattered points, and the sphere's coordinates, frames and point sets.
- **Nouns.** `functax.geometry.Axis`, `functax.geometry.AffineGrid`, `functax.geometry.RectilinearGrid`, `functax.geometry.CurvilinearGrid`, `functax.geometry.Points`.
- **Verbs.** `functax.geometry.lonlat_to_unit`, `functax.geometry.geodesic_distance`, `functax.geometry.pairwise_distance`; `functax.geometry.fibonacci_sphere` and `functax.geometry.gauss_legendre_grid` return points with quadrature weights; `functax.geometry.to_pixel`, `functax.geometry.from_pixel` and `functax.geometry.window` for a raster.
- **Adverbs.** `period=`, `closed=` and `edges=` on an axis; `metric=` on points (`"euclidean"`, `"chord"`, `"geodesic"`); `unit=` (`"radians"` or `"degrees"`) wherever a longitude and latitude go in.
- **Keep.** A grid is flattened row-major, so a field on it has shape `(..., N)`; an axis is strictly monotone; coordinates carry no CRS, so reproject before you build one.
- **Never.** Import `functax.geometry` on a NumPy-only path: only `functax.geometry.affine` runs without the `functax[jax]` extra.
- **One rung down.** Open `functax:function-approximation-with-functax` to choose between the grids, to work in a local tangent frame or to build a sphere point set.
<!-- shelf:below:end -->

## Before you finish

- Is each plan built once, outside `jit`, and reused across fields and time steps?
- Are the policies stated where a NaN, an edge or a target off the grid would matter, and is `plan.valid` read?
- Does the transpose come from `plan.adjoint`?
- Is a density or a flux regridded with `Conservative`, not `Linear`?

<!-- skill-tree:begin · generated by geostack scripts/skill_tree.py; do not edit by hand -->
## Builds on

Nothing else in the stack: this is a base skill.

Used by (pointers, not skills to load): gauss_flows, geonnax, geotoolz, integrix, kernellib, plumax, pyrox, sesmax, spectraldiffx, tracerax, vardax.

<!-- skill-tree:end -->
