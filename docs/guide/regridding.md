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
