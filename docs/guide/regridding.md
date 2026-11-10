# Regridding

Interpolation reads a field at points; regridding moves cell averages from one grid of cells to another. When the field is a flux, an emission or a bathymetry, the cell averages are what we care about, and a remap that reads point values loses mass at every coarsening. Conservative remapping keeps it: each target cell holds the area-weighted mean of the source cells it overlaps, so the integral over the domain is the same on both grids.

## Regular and rectilinear grids

Write $c_j$ for source cell $j$ of $N$, with value $y_j$, and $C_i$ for target cell $i$ of $M$. Target cell $C_i$ reads

$$
\hat{y}_i = \sum_{j=1}^{N} w_{ij} y_j \text{ with } w_{ij} = \frac{\lvert c_j \cap C_i \rvert}{\lvert C_i \rvert},
$$

so $\sum_i \lvert C_i \rvert \hat{y}_i = \sum_j \lvert c_j \rvert y_j$ whenever the target cells cover the source cells. On a tensor grid a cell is a product of intervals, so its overlap with another is the product of the per-axis overlaps. On one axis, with source edges $e_j$ and target edges $E_i$,

$$
\lvert c_j \cap C_i \rvert = \max(0, \min(e_{j+1}, E_{i+1}) - \max(e_j, E_i)),
$$

and both sets of edges are monotone, so the source cells a target overlaps are contiguous and two `searchsorted` sweeps find them. [`Conservative`](xref:api#functax.interp.Conservative) builds one band of overlaps per axis and takes their tensor product, which is a [`Stencil`](xref:api#functax.interp.Stencil) with $S = \prod_d S_d$ taps per target, $S_d$ being the most source cells a target cell overlaps on axis $d$. No polygon is clipped, and [`overlap_weights_1d`](xref:api#functax.interp.overlap_weights_1d) returns the dense per-axis overlaps for inspection.

```python
import einx
import jax.numpy as jnp
from functax.geometry import Axis, RectilinearGrid
from functax.interp import Conservative, Policies

# a 6 × 8 grid of unit cells, and a 2 × 3 grid of cells 3 by 8/3 wide
source = RectilinearGrid((Axis(jnp.arange(6.0) + 0.5), Axis(jnp.arange(8.0) + 0.5)))
target = RectilinearGrid(
    (
        Axis(jnp.array([1.5, 4.5]), edges=jnp.array([0.0, 3.0, 6.0])),
        Axis(jnp.array([4 / 3, 4.0, 20 / 3]), edges=jnp.linspace(0.0, 8.0, 4)),
    )
)
field = jnp.arange(48.0)
plan = Conservative().plan(source, target)
remapped = plan(field)
remapped.shape, plan.read.n_taps  # ((6,), 12)
einx.id("(y x) -> y x", remapped, y=2).round(4).tolist()
# [[8.875, 11.5, 14.125], [32.875, 35.5, 38.125]]
bool(jnp.isclose(8.0 * remapped.sum(), field.sum()))  # True
```

Each target cell has area 8, so the last line checks that the integral is kept. The factor need not be an integer; when it is, the plan is the block mean exactly.

Cell edges come from the geometry. An [`Axis`](xref:api#functax.geometry.Axis) built with `edges` (the $n + 1$ edges of its $n$ cells, as CF `bounds` give them) uses those, and one without uses the midpoints between its nodes, with the end cells symmetric about their nodes. A periodic source axis reads target cells modulo its period, so a source on 0–360° regrids onto a target on −180–180° and a target cell across the seam reads both ends. A north-up `AffineGrid` is a tensor grid too, with its rows and columns as the two axes.

The policies say what happens at cells the source does not cover. Under the default `outside="nan"` a target cell that is not inside the source cells is invalid and reads the fill value; `outside="extrapolate"` keeps $w_{ij}$, so a partly covered cell holds its covered integral over its whole area (ESMF's `destarea`); and `outside="clamp"` cuts the target cell to the source domain, dividing by the covered area. Under `missing="renormalise"` each row is divided by its weight on finite samples, so a cell holds the mean over its valid source area, which is the coastal-mask mean, and the area outside the source counts as missing too:

```python
land = jnp.zeros((6, 8), bool).at[:, :3].set(True)
masked = jnp.where(einx.id("y x -> (y x)", land), jnp.nan, field)
coastal = Conservative().plan(source, target, policies=Policies(missing="renormalise"))
einx.id("(y x) -> y x", coastal(masked), y=2).round(4).tolist()
# [[nan, 11.7143, 14.125], [nan, 35.7143, 38.125]]
einx.id("(y x) -> y x", plan(masked), y=2).round(4).tolist()
# [[nan, nan, 14.125], [nan, nan, 38.125]]
```

The first column of target cells lies on land, so it has no valid source and stays NaN; the second straddles the coast and holds the mean of its ocean part, where the plan without renormalising lets the NaN through.

The plan is linear (`missing="renormalise"` aside), so `plan.adjoint` sends target residuals back to the source cells, and the dot-product test holds to $10^{-12}$. On the sphere, a lon/lat cell has area $a^2 \Delta \lambda (\sin \varphi_+ - \sin \varphi_-)$ for longitude width $\Delta \lambda$ and latitude edges $\varphi_\pm$, so a latitude axis whose nodes and edges are in $\sin \varphi$, with a periodic longitude axis, gives the exact spherical weights (xrtoolz's `regrid_conservative` with `geometry="spherical"`).

## Curvilinear and spherical cells

A [`CurvilinearGrid`](xref:api#functax.geometry.CurvilinearGrid) (a swath, or a curvilinear ocean grid) gives a coordinate per node, so its cells are not products of intervals and their overlaps need polygon clipping. Each node owns the quadrilateral whose corners are the means of the four nodes around them, with the grid extended by linear extrapolation past its edges, so a rectilinear grid written as a curvilinear one gets the same cells as on the tensor path. `Conservative` takes this path whenever the source or the target is curvilinear, or with `sphere=True`.

For target cell $C_i$, the candidate source cells are those whose bounding circles meet its own, found with a k-d tree (SciPy's `cKDTree`, so this path needs the `functax[scipy]` extra). Each candidate $c_j$ is clipped to $C_i$ by Sutherland–Hodgman on the host, which needs the target cell to be convex, and $w_{ij} = \lvert c_j \cap C_i \rvert / \lvert C_i \rvert$ with the area of the clipped polygon. Rows have as many taps as their target cell has overlaps, and they are padded with zero weights to the largest count $S$. In the plane, coordinates are $(x, y)$ as given, so a swath across the dateline needs unwrapped longitudes.

With `sphere=True` (and `unit="degrees"` or `"radians"`), coordinates are longitude and latitude and cells have great-circle edges, as in ESMF. Each pair is clipped in the gnomonic tangent plane of its target cell, where great circles are straight lines, so the planar clip is exact, and the clipped polygon goes back to the sphere. Its area is the spherical excess $E = \sum_k \alpha_k - (n - 2) \pi$ of an $n$-gon with interior angles $\alpha_k$, times $a^2$ for radius $a$. [`spherical_polygon_area`](xref:api#functax.geometry.spherical_polygon_area) sums it over a fan of triangles by the formula of Van Oosterom and Strackee, because the angle sum loses most of its digits on small cells. A 2-D `RectilinearGrid` on this path has its axes in $(x, y)$ order, $(\lambda, \varphi)$ on the sphere, as when it is a target read on a curvilinear source.

```python
import jax.numpy as jnp
from functax.geometry import Axis, CurvilinearGrid, RectilinearGrid
from functax.interp import Conservative

# a 0.5° swath, sheared to the east as it runs north
rows, cols = jnp.meshgrid(jnp.arange(20.0), jnp.arange(24.0), indexing="ij")
swath = CurvilinearGrid(x=-5.0 + 0.5 * (cols + 0.15 * rows), y=40.0 + 0.5 * rows)
# a 2° lon/lat grid, axes in (λ, φ) order
grid = RectilinearGrid(
    (Axis(jnp.arange(-9.0, 18.0, 2.0)), Axis(jnp.arange(37.0, 56.0, 2.0)))
)

plan = Conservative(sphere=True, unit="degrees").plan(swath, grid)
plan.read.n_taps, int(plan.valid.sum()), plan.valid.shape  # (26, 20, (140,))
values = plan(jnp.ones(20 * 24))
bool(jnp.allclose(values[plan.valid], 1.0))  # True
```

Twenty of the 140 target cells lie inside the swath, and a constant field stays constant there. A target cell counts as covered when its overlaps sum to its area to $10^{-9}$ relative; the policies then act as on the tensor path. Cells with no area, and pairs that do not overlap, carry zero weight. On a tensor grid, a lon/lat cell bounded by latitude circles has the exact area $a^2 \Delta \lambda (\sin \varphi_+ - \sin \varphi_-)$ through a $\sin \varphi$ axis (see above), which differs from the great-circle cell of `sphere=True` near the poles; mixing the two on one grid would not conserve.
