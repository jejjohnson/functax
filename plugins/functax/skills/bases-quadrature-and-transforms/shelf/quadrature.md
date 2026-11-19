# Quadrature rules

Role: ⓪ structure (`role-ladder`, `references/structure.md`).
Unit: quad.

Open this when you integrate: over an interval, over the sphere, over the cells of a grid, or along a curve.

## Names

All of them live in `functax.quad`. A rule is data, nodes and weights, with ∫ g ≈ ∑ₙ wₙ g(xₙ).

| Name | What it is |
|---|---|
| `Rule(nodes, weights)` | A quadrature rule. It unpacks: `nodes, weights = rule`. |
| `gauss_legendre(n, interval=(-1.0, 1.0))` | The n-point Gauss–Legendre rule, exact for polynomials up to degree 2n − 1. |
| `clenshaw_curtis(n, interval=(-1.0, 1.0))` | The n-point Clenshaw–Curtis rule, at the Chebyshev extreme points. |
| `trapezoid(nodes)`, `simpson(nodes)` | Composite rules on nodes you already have, uniform or not. |
| `SphereRule` | A rule whose nodes are unit vectors, with its `radius` and its exactness `degree`. |
| `sphere_gauss_legendre(n_lat, n_lon=None, *, radius=1.0)` | Gauss–Legendre in latitude times uniform longitude, exact to a degree. |
| `sphere_healpix(nside, *, nest=True, radius=1.0)`, `sphere_fibonacci(n, *, radius=1.0)` | Equal-weight rules on HEALPix pixel centres and on the Fibonacci lattice. |
| `sphere_latlon(n_lat, n_lon, *, radius=1.0)` | The cell-centred lon/lat rule, weighted by exact cell areas. |
| `cell_area(grid, *, sphere_radius=None, unit=None)` | The cell centres of an `AffineGrid` or a `RectilinearGrid` weighted by their areas; with `sphere_radius=` the grid is in longitude and latitude. |
| `CurveRule`, `polyline(vertices, n, *, rule="midpoint")` | A rule along a polyline: `n` nodes per segment, arc-length weights, and the unit normal at each node in 2-D. |

## Use it

```python
import jax
import jax.numpy as jnp

from functax.geometry import AffineGrid
from functax.quad import (
    cell_area,
    gauss_legendre,
    polyline,
    simpson,
    sphere_gauss_legendre,
)

jax.config.update("jax_enable_x64", True)  # an AffineGrid holds float64

# ∫₀² x³ dx = 4, exactly, with two Gauss nodes
nodes, weights = gauss_legendre(2, interval=(0.0, 2.0))
assert jnp.allclose(weights @ nodes**3, 4.0)

# samples you already have, unevenly spaced: ∫₀¹ x² dx = 1/3
rule = simpson(jnp.array([0.0, 0.3, 1.0]))
assert jnp.allclose(rule.weights @ rule.nodes**2, 1.0 / 3.0)

# ∫ z² dA over a sphere of radius 2 is 4π a⁴ / 3
sphere = sphere_gauss_legendre(8, radius=2.0)
z = sphere.radius * sphere.nodes[:, 2]  # the nodes are unit vectors
assert jnp.allclose(sphere.weights @ z**2, 4 * jnp.pi * 2.0**4 / 3)

# the mass in an image: 20 m pixels, 400 m² each
grid = AffineGrid([20.0, 0.0, 0.0, 0.0, -20.0, 0.0], (3, 4))
areas = cell_area(grid).weights  # (3, 4)
column = jnp.full((3, 4), 0.5)  # kg m⁻²
assert jnp.allclose((areas * column).sum(), 0.5 * 400.0 * 12)

# the flux of a uniform wind across an L-shaped transect
transect = polyline(jnp.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]]), 4)
wind = jnp.array([1.0, 0.0])
flux = transect.weights @ (transect.normals @ wind)
assert jnp.allclose(transect.weights.sum(), 7.0) and jnp.allclose(abs(flux), 4.0)
```

## Rules

- A rule has no `integrate` method. Evaluate the integrand at `rule.nodes` and contract with `rule.weights`; integrix sums, scans and adapts with the same rules.
- Use `gauss_legendre` when you can choose the nodes, and `trapezoid` or `simpson` when the samples are given.
- On the sphere, pick the rule from the data's grid: `sphere_latlon` for a regular lon/lat grid, `sphere_healpix` for HEALPix maps, `sphere_gauss_legendre` when you choose the nodes and want exactness to a degree.
- The nodes of a `SphereRule` are unit vectors whatever the radius; the weights carry the radius and sum to 4πa².
- Never weight a lon/lat grid by `cos(lat)` by hand. `cell_area(grid, sphere_radius=..., unit=...)` and `sphere_latlon` use the exact cell areas.
- The normal of a `polyline` points to the left of the direction of travel, so reverse the vertices to flip the sign of a flux. A polyline in 3-D has no normal.
