# Quadrature

A quadrature rule approximates an integral by a weighted sum of samples. In functax a rule is data and nothing else: a [`Rule`](xref:api#functax.quad.Rule) holds the nodes $x_n$ and the weights $w_n$, so that

$$
\int g(x) \mathrm{d}x \approx \sum_{n=1}^{N} w_n g(x_n).
$$

functax builds the nodes and weights; integrix sums with them, scans over them and adapts them (integrix design decision 3), so a rule has no `integrate` method. Unpacking a rule gives the pair, `nodes, weights = rule`, and the sum itself is one contraction, `weights @ g(nodes)`. This page walks through the four families: rules on an interval, on the sphere, on the cells of a grid and along a curve.

## Rules on an interval

[`gauss_legendre`](xref:api#functax.quad.gauss_legendre) is the $n$-point Gauss–Legendre rule, exact for polynomials of degree up to $2n - 1$, and [`clenshaw_curtis`](xref:api#functax.quad.clenshaw_curtis) the $n$-point Clenshaw–Curtis rule on the Chebyshev extreme points, exact to degree $n - 1$ ($n$ when $n - 1$ is even). Both are built on the host in float64, so they can size static arrays, and both map affinely from $[-1, 1]$ onto any interval $[a, b]$. When the field already lives on its own nodes, uniform or not, [`trapezoid`](xref:api#functax.quad.trapezoid) and [`simpson`](xref:api#functax.quad.simpson) give the weights on those nodes; they equal `scipy.integrate.trapezoid` and `scipy.integrate.simpson` and are traceable, so their weights are differentiable in the nodes.

```python
import math
import jax.numpy as jnp
from functax.quad import clenshaw_curtis, gauss_legendre, trapezoid

# 4 Gauss–Legendre nodes on [0, 2] integrate x⁷ exactly
nodes, weights = gauss_legendre(4, interval=(0.0, 2.0))
bool(abs(weights @ nodes**7 - 2.0**8 / 8) < 1e-12)  # True

# ∫₀^π sin x dx = 2: 41 trapezoid nodes against 9 Clenshaw–Curtis nodes
x = jnp.linspace(0.0, math.pi, 41)
rule = trapezoid(x)
float(rule.weights @ jnp.sin(rule.nodes))  # 1.9989718104970657
cc = clenshaw_curtis(9, interval=(0.0, math.pi))
float(cc.weights @ jnp.sin(cc.nodes))  # 2.0000000026597644
```

## Rules on the sphere

On a sphere of radius $a$ a rule approximates $\int_{S^2} g \mathrm{d}A \approx \sum_q w_q g(\vect{u}_q)$, with $Q$ nodes $\vect{u}_q$ that are unit vectors (shape $(Q, 3)$) and weights that sum to $4 \pi a^2$. A [`SphereRule`](xref:api#functax.quad.SphereRule) is a `Rule` with two more fields: the `radius` its weights are scaled to, and the `degree` $L$ up to which it integrates every spherical harmonic $Y_l^m$ with $l \le L$ exactly. Four constructors wrap the point sets of `functax.geometry`:

- [`sphere_gauss_legendre`](xref:api#functax.quad.sphere_gauss_legendre) takes the sines of latitude at the $n_{\text{lat}}$ Gauss–Legendre nodes and $n_{\text{lon}}$ uniform longitudes, and is exact to degree $\min(2 n_{\text{lat}} - 1, n_{\text{lon}} - 1)$: the longitude sum removes every order $m \ne 0$ below $n_{\text{lon}}$, and Gauss–Legendre is exact for the Legendre polynomial $P_l$ up to degree $2 n_{\text{lat}} - 1$. Latitudes ascend from south to north and longitudes start at $-\pi$; spectraldiffx uses the same nodes ordered north to south, with longitudes from 0.
- [`sphere_latlon`](xref:api#functax.quad.sphere_latlon) weights the centres of equiangular cells by their exact areas, so a field constant on each cell integrates exactly, but the rule is exact only to degree 1.
- [`sphere_healpix`](xref:api#functax.quad.sphere_healpix) and [`sphere_fibonacci`](xref:api#functax.quad.sphere_fibonacci) give every node the same weight $4 \pi a^2 / Q$. Equal areas are not quadrature weights, so neither is exact beyond degree 1 (HEALPix, by its symmetry) or 0 (Fibonacci); their error falls as $Q$ grows.

```python
import math
import jax.numpy as jnp
from functax.basis.sphere import real_spherical_harmonics
from functax.quad import sphere_gauss_legendre

rule = sphere_gauss_legendre(8)
rule.nodes.shape, rule.degree  # ((128, 3), 15)
# ∫ Y_0^0 dΩ = √(4π), and every other Y_l^m with l ≤ 15 integrates to 0
integrals = rule.weights @ real_spherical_harmonics(rule.nodes, rule.degree)
bool(jnp.isclose(integrals[0], math.sqrt(4 * math.pi)))  # True
float(jnp.abs(integrals[1:]).max()) < 1e-12  # True

# the radius is an argument: functax holds no Earth constants
earth = sphere_gauss_legendre(8, radius=6.371e6)
bool(jnp.isclose(earth.weights.sum(), 4 * math.pi * 6.371e6**2))  # True
```

## Cell areas on grids

A field of cell means $y_k$ on a grid integrates as $\sum_k \lvert C_k \rvert y_k$, with $\lvert C_k \rvert$ the area of cell $k$. [`cell_area`](xref:api#functax.quad.cell_area) returns that sum as a rule: its nodes are the cell centres and its weights the cell areas, one per cell in the order of the field's flattened values. Every pixel of an [`AffineGrid`](xref:api#functax.geometry.AffineGrid) with transform $(a, b, c, d, e, f)$ has the area $\lvert ae - bd \rvert$, and a cell of a [`RectilinearGrid`](xref:api#functax.geometry.RectilinearGrid) is the product of its widths along each axis, from the edges of [`Axis.cell_edges`](xref:api#functax.geometry.Axis.cell_edges) (the axis's `edges`, or the midpoints between its nodes).

```python
import math
import jax.numpy as jnp
from functax.geometry import AffineGrid
from functax.quad import cell_area

# a 64 × 64 scene of 30 m pixels, and a uniform enhancement per unit area
plume_grid = AffineGrid([30.0, 0.0, 500000.0, 0.0, -30.0, 3600000.0], (64, 64))
rule = cell_area(plume_grid)
rule.nodes.shape, rule.weights[0].tolist()  # ((4096, 2), 900.0)
enhancement = jnp.full(64 * 64, 2e-3)
float(rule.weights @ enhancement)  # 7372.800000000001
```

With `sphere_radius=a` the coordinates are longitude $\lambda$ and latitude $\varphi$, and `unit=` (`"degrees"` or `"radians"`) is required, as it is for [`Conservative`](xref:api#functax.interp.Conservative) with `sphere=True`. A lon/lat cell then has the area $a^2 \lvert \Delta \lambda \rvert \lvert \sin \varphi_{\text{top}} - \sin \varphi_{\text{bottom}} \rvert$, so a global grid sums to $4 \pi a^2$. An `AffineGrid` must be north-up, with rows of latitude, and a `RectilinearGrid` has its two axes in $(\lambda, \varphi)$ order.

```python
globe = AffineGrid([1.0, 0.0, -180.0, 0.0, -1.0, 90.0], (180, 360))
areas = cell_area(globe, sphere_radius=6.371e6, unit="degrees").weights
areas.shape  # (64800,)
bool(jnp.isclose(areas.sum(), 4 * math.pi * 6.371e6**2, rtol=1e-12))  # True
```

## Rules on curves

A line integral or a flux across a transect is a rule too. For a polyline with vertices $\vect{v}_0, \dots, \vect{v}_{V-1}$, segment $j$ has length $\ell_j = \lVert \vect{v}_{j+1} - \vect{v}_j \rVert$ and unit tangent $\vect{t}_j = (\vect{v}_{j+1} - \vect{v}_j) / \ell_j$. [`polyline`](xref:api#functax.quad.polyline) puts $n$ nodes on each segment, from the midpoint rule or the Gauss–Legendre rule on $[0, \ell_j]$, so the arc-length weights $\Delta s_k$ sum to $\sum_j \ell_j$ and the Gauss–Legendre nodes integrate polynomials along each segment exactly up to degree $2n - 1$. It returns a [`CurveRule`](xref:api#functax.quad.CurveRule), a `Rule` with a `normals` field: in 2-D each node carries the tangent of its segment rotated by $+90°$, which points to the left of the direction of travel. The flux of a concentration $c$ carried by a wind $\vect{u}$ across the transect is then

$$
\Phi = \int_L c (\vect{u} \cdot \vect{n}) \mathrm{d}\ell \approx \sum_{k=1}^{K} \Delta s_k c(\vect{x}_k) (\vect{u}_k \cdot \vect{n}_k).
$$

Every segment gets the same $n$, so $K = n (V - 1)$ is static: the rule traces under `jit` and `vmap` and differentiates in the vertices, and the nodes are where functax's interpolation plans sample the field. A curve in any dimension other than 2 has no single normal, and `normals` is `None` there (a line of sight in 3-D still has its nodes and weights).

```python
import math
import jax.numpy as jnp
from functax.quad import polyline

# a 1 km transect heading north, across a Gaussian plume of width 100 m
vertices = jnp.array([[0.0, -500.0], [0.0, 500.0]])
transect = polyline(vertices, 16, rule="gauss_legendre")
transect.nodes.shape, transect.normals[0].tolist()  # ((16, 2), [-1.0, 0.0])
y = transect.nodes[:, 1]
concentration = jnp.exp(-0.5 * (y / 100.0) ** 2)
# a westward wind of 5 m/s crosses it along the normal
wind = jnp.array([-5.0, 0.0])
flux = transect.weights @ (concentration * (transect.normals @ wind))
float(flux)  # 1253.313295745834
5.0 * 100.0 * math.sqrt(2 * math.pi)  # 1253.3141373155001
```

The transect stops at five plume widths either side, which is why the two numbers differ in the seventh digit. Reversing the vertices flips the normals and so the sign of the flux.
