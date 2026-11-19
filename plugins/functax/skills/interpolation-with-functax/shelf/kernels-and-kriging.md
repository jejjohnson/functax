# Kernel interpolation and kriging

Role: ② strategy (`role-ladder`, `references/strategy.md`).
Unit: interp.

Open this when the sources are scattered and you want a smooth interpolant through them, or a predictor that comes with a variance.

## Names

All of them live in `functax.interp`.

| Name | What it is |
|---|---|
| `Kernel` | The protocol of a kernel: a callable `k(targets, sources)` that returns the `(M, N)` cross-kernel, as an array or a lineax operator. |
| `RadialKernel` | A kernel of the Euclidean distance, k(x, x′) = φ(‖x − x′‖), with `profile(r)` and `min_degree`. |
| `Polyharmonic(order=2)` | φ(r) = rᵏ or rᵏ log r; order 2 is the thin-plate spline. |
| `Wendland(smoothness=2, support=1.0)` | A compactly supported kernel, zero beyond `support`. |
| `RBF(kernel, degree=None, smoothing=0.0, solver=None, neighbors=None)` | Radial basis function interpolation with a polynomial tail: one solve when the plan is built. `neighbors=Q` makes it local. |
| `Kriging(kernel, drift="constant", nugget=0.0)` | Simple (`drift=None`), ordinary or universal kriging, as a `GaussianPlan`. |
| `GaussianPlan` | A plan read as a Gaussian posterior: `plan(data)` is the mean, with `variance()`, `covariance()` and `sample(data, key=...)`. |
| `PosteriorCovariance` | The base of the posterior covariances Σ*: `variance()`, `covariance()`, `sample(...)`. |
| `KernelCovariance` | Σ* of a kernel fit, k(x, x′) − h(x)ᵀ A⁻¹ h(x′). |
| `BrownianBridge` | Σ* of linear interpolation on one axis read as a Brownian bridge between the knots; `BrownianBridge.from_positions(axis, positions, rate=...)`. |
| `KernelSolve` | The fit of a kernel scheme: the saddle-point solve from samples to coefficients. |
| `KernelReadout` | The dense read-out of a kernel scheme, f̂(X*) = K* c + P* b. |

## Use it

```python
import jax
import jax.numpy as jnp

from functax.geometry import Points
from functax.interp import RBF, GaussianPlan, Kriging, Polyharmonic, Wendland

sources = Points(jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
data = jnp.array([1.0, 2.0, 3.0, 5.0])

# thin-plate spline: it passes through the samples
exact = RBF(Polyharmonic(2)).plan(sources, sources)
assert jnp.allclose(exact(data), data, atol=1e-4)

# its degree-1 tail reproduces an affine field
affine = 1.0 + sources.xy[:, 0] + 2.0 * sources.xy[:, 1]
midpoint = RBF(Polyharmonic(2)).plan(sources, Points(jnp.array([[0.5, 0.25]])))
assert jnp.allclose(midpoint(affine), 2.0, atol=1e-4)

# ordinary kriging: a mean, a variance and draws
targets = Points(jnp.array([[0.0, 0.0], [0.5, 0.5]]))
plan = Kriging(Wendland(2, support=2.0)).plan(sources, targets)
assert isinstance(plan, GaussianPlan)
mean = plan(data)  # (2,)
variance = plan.variance()  # (2,)
assert jnp.allclose(mean[0], 1.0, atol=1e-4)  # through a noiseless sample
assert jnp.allclose(variance[0], 0.0, atol=1e-4) and float(variance[1]) > 0.0
assert jnp.allclose(plan(jnp.ones(4)), 1.0, atol=1e-4)  # the weights sum to 1

draws = plan.sample(jnp.zeros((3, 4)), key=jax.random.key(0))  # (3, 2)
assert draws.shape == (3, 2)
```

## Rules

- The solve happens once, when the plan is built. Applying the plan to a new field reuses it, so build one plan per pair of geometries.
- Leave `degree=None`: the tail then has the degree the kernel needs for the system to be solvable (1 for the thin-plate spline).
- For more than a few thousand sources, pass `neighbors=Q` to `RBF`, or use a `Wendland` kernel whose `support` covers a few neighbours.
- `Kriging` needs a positive definite covariance, so use `Wendland` or your own kernel, not `Polyharmonic`.
- `nugget=` is the noise variance of the samples: with it the mean no longer passes through them and the variance there is no longer zero.
- A kernel from kernellib or any callable `k(targets, sources)` is a `Kernel`. Kernels with hyperparameters to learn belong to kernellib; functax only reads them.
- `Points(metric=...)` sets how neighbours are found; a `RadialKernel` itself measures Euclidean distance in the coordinates as given.
