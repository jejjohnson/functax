# Interpolation

A plan reads a gridded field at a set of targets and sends residuals at those targets back to the grid. You build it once from a scheme and the two geometries, which pays for the locate and the weights, and then apply it to every field.

## Read a grid at points

```python
import einx
import jax.numpy as jnp
from functax.geometry import AffineGrid, Points
from functax.interp import Linear

# a 4 × 5 north-up grid of 10 m pixels, and 24 hourly frames on it
grid = AffineGrid([10.0, 0.0, 500.0, 0.0, -10.0, 900.0], (4, 5))
field = jnp.arange(24 * 4 * 5.0).reshape(24, 4, 5)
# three observations, the last off the grid
points = Points(jnp.array([[515.0, 885.0], [540.0, 870.0], [700.0, 870.0]]))

plan = Linear().plan(grid, points)
values = plan(einx.id("t y x -> t (y x)", field))
values.shape  # (24, 3)
values[0].tolist()  # [6.0, 16.0, nan]
plan.valid.tolist()  # [True, True, False]
```

The plan works on the flat source, so a `(time, y, x)` field is flattened row-major first; every leading dimension is a batch. The first point sits on the centre of pixel $(1, 1)$, the second halfway between four pixels, and the third is off the grid, so it reads NaN and its row is not valid.

## Send residuals back

The adjoint applies $\mat{H}^\top$, the transpose of the read-out, with a scatter-add. It is the gradient of a misfit at the points with respect to the field:

```python
residual = jnp.ones((24, 3))
grid_residual = plan.adjoint(residual)
grid_residual.shape  # (24, 20)
einx.id("(y x) -> y x", grid_residual[0], y=4).tolist()
# [[0.0, 0.0, 0.0, 0.0, 0.0],
#  [0.0, 1.0, 0.0, 0.0, 0.0],
#  [0.0, 0.0, 0.0, 0.25, 0.25],
#  [0.0, 0.0, 0.0, 0.25, 0.25]]
```

The point off the grid contributes nothing. The same map is `plan.operator()`, a lineax operator whose transpose equals `plan.adjoint`.

## Choose the edge rules

`Policies` names what happens at the edges. geoops' `interp_at` reads the half-pixel border of a raster with the edge value; that is `Policies(extent="cells", boundary="edge")`:

```python
from functax.interp import Policies

geoops = Linear().plan(grid, points, policies=Policies(extent="cells", boundary="edge"))
geoops(field[0].ravel()).tolist()  # [6.0, 16.0, nan]
```

`outside="clamp"` reads the nearest boundary instead of NaN, `outside="extrapolate"` continues the edge cells, and `outside="raise"` refuses targets off the grid when the plan is built. `missing="renormalise"` drops non-finite samples and divides each row by the weight left. The [interpolation API page](https://jejjohnson.github.io/functax/reference/interp/) lists every policy.

## Mix schemes per axis

`Tensor` takes one scheme per source axis, so nearest in time and bilinear in space is one stencil:

```python
from functax.geometry import Axis, RectilinearGrid
from functax.interp import Nearest, Tensor

cube = RectilinearGrid(
    (Axis(jnp.arange(24.0)), Axis(jnp.arange(4.0)), Axis(jnp.arange(5.0)))
)
plan = Tensor((Nearest(), Linear(), Linear())).plan(
    cube, Points(jnp.array([[3.4, 1.5, 2.5]]))
)
plan.read.n_taps  # 4: one frame, then 2 × 2 in space
```
