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

## RBF

Radial basis function interpolation writes the field as a sum of kernels centred on the $N$ sources plus a polynomial tail,

$$
\hat{f}(\vect{x}) = \sum_{j=1}^{N} c_j k(\vect{x}, \vect{x}_j) + \sum_{p=1}^{P} b_p p_p(\vect{x}),
$$

with $P$ monomials $p_p$ up to a chosen degree. With the $N \times N$ kernel matrix $\mat{K}$ on the sources, the $N \times P$ tail matrix $\mat{P}$, the samples $\vect{y} \in \mathbb{R}^N$ and a smoothing $\lambda \ge 0$, the weights $\vect{c} \in \mathbb{R}^N$ and $\vect{b} \in \mathbb{R}^P$ solve the saddle-point system

$$
\begin{pmatrix} \mat{K} + \lambda \mat{I} & \mat{P} \\ \mat{P}^\top & \mat{0} \end{pmatrix} \begin{pmatrix} \vect{c} \\ \vect{b} \end{pmatrix} = \begin{pmatrix} \vect{y} \\ \vect{0} \end{pmatrix},
$$

and the plan reads $\hat{\vect{f}}(\mat{X}_*) = \mat{K}_* \vect{c} + \mat{P}_* \vect{b}$ at $M$ targets, with the $M \times N$ cross-kernel $\mat{K}_*$ and the $M \times P$ monomials $\mat{P}_*$ there. With $\lambda = 0$ the interpolant passes through every sample; a positive $\lambda$ trades that for a smoother field.

```python
import jax
import jax.numpy as jnp
import lineax as lx
from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import RBF, Polyharmonic

# 50 scattered stations and a 20 × 30 target grid
stations = Points(jax.random.uniform(jax.random.key(0), (50, 2)))
grid = RectilinearGrid(
    (Axis(jnp.linspace(0.0, 1.0, 20)), Axis(jnp.linspace(0.0, 1.0, 30)))
)
temperature = jnp.sin(3.0 * stations.xy[:, 0]) + stations.xy[:, 1] ** 2

scheme = RBF(kernel=Polyharmonic(2), degree=1, smoothing=0.0, solver=lx.Cholesky())
plan = scheme.plan(stations, grid)
plan(temperature).shape  # (600,)
plan.grid_shape  # (20, 30)
plan.adjoint(jnp.ones(600)).shape  # (50,)
```

The kernel is anything that takes the targets and the sources and returns their cross-kernel, as an array or as a lineax operator: a plain function, or another library's kernel object, so functax never imports the kernel library ([`Kernel`](xref:api#functax.interp.Kernel)). Two families ship with functax because they are about interpolation rather than Gaussian processes.

- [`Polyharmonic(k)`](xref:api#functax.interp.Polyharmonic) is $\pm r^k$ for odd $k$ and $\pm r^k \log r$ for even $k$ ($k = 2$ is the thin-plate spline, $k = 3$ the cubic). It is only conditionally positive definite, so it needs a tail of degree at least `min_degree` (1 for both), which is what `degree=None` picks. Its signs are SciPy's, so the plan equals `scipy.interpolate.RBFInterpolator` for the same kernel, degree and smoothing.
- [`Wendland(smoothness, support)`](xref:api#functax.interp.Wendland) is the compact $C^2$ or $C^4$ Wendland function of $r / \rho$, zero beyond the support radius $\rho$. It is positive definite in up to three dimensions and needs no tail.

The plan's fit is a [`KernelSolve`](xref:api#functax.interp.KernelSolve), which factorises the saddle-point system once when the plan is built and solves it for each field. It projects onto the null space of $\mat{P}^\top$ (a QR factorisation of $\mat{P}$), where the kernel is positive definite, and factorises that block with the lineax solver you pass (Cholesky by default). The system is symmetric, so the adjoint of the fit is the same solve, and `plan.adjoint` and `plan.operator()` work as for any other plan. Building costs $O(N^3)$ and the dense $M \times (N + P)$ read-out; applying costs $O(N^2 + MN)$ per field, so RBF suits up to a few thousand sources.

## Kriging

Kriging reads the same system as a Gaussian process. The kernel is a covariance $k$, the tail is a drift, and the smoothing is the nugget $\tau^2$, the noise variance of the samples, so one plan gives the map and its uncertainty. With the $N \times N$ covariance $\mat{K}$ on the sources, the $N \times P$ drift matrix $\mat{F}$, and at one target $\vect{x}_*$ the cross-covariance $\vect{k}_* = k(\mat{X}, \vect{x}_*) \in \mathbb{R}^N$, the prior variance $k_{**} = k(\vect{x}_*, \vect{x}_*)$ and the drift $\vect{f}_* \in \mathbb{R}^P$, the mean is the RBF read-out with $\lambda = \tau^2$, and the variance is

$$
\sigma^2(\vect{x}_*) = k_{**} - \vect{k}_*^\top \mat{K}_\tau^{-1} \vect{k}_* + \vect{u}^\top (\mat{F}^\top \mat{K}_\tau^{-1} \mat{F})^{-1} \vect{u} \text{ with } \vect{u} = \vect{f}_* - \mat{F}^\top \mat{K}_\tau^{-1} \vect{k}_*,
$$

where $\mat{K}_\tau = \mat{K} + \tau^2 \mat{I}$. The last term is what estimating the drift costs. The `drift` picks the flavour:

| `drift` | Kriging | $\mat{F}$ |
|---|---|---|
| `None` | simple (known zero mean), the GP posterior | none |
| `"constant"` (the default) | ordinary, weights that sum to 1 | a column of ones |
| a degree $d$ | universal, reproduces polynomial trends up to $d$ | monomials up to $d$ |
| a callable $F(\vect{x})$ | universal, with your own drift | $F(\mat{X})$ |

```python
import jax
import jax.numpy as jnp
from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import Kriging

stations = Points(jax.random.uniform(jax.random.key(0), (50, 2)))
grid = RectilinearGrid(
    (Axis(jnp.linspace(0.0, 1.0, 20)), Axis(jnp.linspace(0.0, 1.0, 30)))
)
temperature = jnp.sin(3.0 * stations.xy[:, 0]) + stations.xy[:, 1] ** 2


def covariance(targets, sources):
    # σ² = 1, ℓ = 0.2; from a variogram fit, say
    squared = jnp.sum((targets[:, None, :] - sources[None, :, :]) ** 2, axis=-1)
    return jnp.exp(-0.5 * squared / 0.2**2)


plan = Kriging(kernel=covariance, drift="constant", nugget=1e-4).plan(stations, grid)
plan(temperature).shape  # (600,), the kriging mean
plan.variance().shape  # (600,), the kriging variance
fields = jnp.broadcast_to(temperature, (16, 50))
plan.sample(fields, key=jax.random.key(1)).shape  # (16, 600)
```

The plan is a [`GaussianPlan`](xref:api#functax.interp.GaussianPlan) whose covariance is a [`KernelCovariance`](xref:api#functax.interp.KernelCovariance): `variance()`, `covariance()` and `sample(y, key=…)` reuse the factorised system of the mean, and the variance is that of the latent field (the nugget sits on the sources only), so it is zero at a noiseless source. Rounding can push it a hair below zero there; anything below $-10^{-10} k_{**}$ raises, because it means the covariance is not positive definite on the sources, and the rest is clipped to zero. The covariance's hyperparameters come from outside, from a variogram fit in kernellib for example, and the solve uses no jitter but your nugget.
