# Plans, adjoints and interpolants

Role: setup (`role-ladder`, `references/strategy.md`).
Unit: interp.

Open this when you apply an interpolation many times, need its adjoint or its derivative, or read a field at targets that are only known inside a traced function.

## Names

All of them live in `functax.interp`. A plan freezes a scheme for one source and one target: f̂(X*) = H 𝒮(y), with a fit 𝒮 and a read-out H.

| Name | What it is |
|---|---|
| `Plan` | The frozen scheme. `plan(field)` reads, `plan.adjoint(residual)` applies 𝒮ᵀ Hᵀ, `plan.operator()` is the `(M, N)` lineax operator, `plan.valid` marks the readable targets, `plan.to_gaussx()` exports the stencil, and `plan.serialize()` / `Plan.deserialize(data, like=plan)` store it as bytes. |
| `DerivativePlan` | `plan.derivative(axis, order=1)`: the read-out of ∂ᵐf̂/∂x_cᵐ at the same targets. |
| `WindowedPlan` | `plan[window]`: the rows of a grid target inside a window, with the sources unchanged. |
| `ComposedPlan` | `outer @ inner`: two plans as one, with the product stencil. |
| `NonlinearPlan`, `NonlinearDerivativePlan` | The plans of `Pchip`, `Akima` and `Steffen`, whose fit depends on the data: they read, and `linearize(data)` gives the linear map at that data. |
| `plan_key(scheme, source, target, *, policies=None)` | The sha256 of the scheme, the policies and both geometries: a cache key for a plan. |
| `Interpolant` | `scheme.fit(source, data)` on a grid: a fitted field `f(positions)` read at targets chosen later, with `derivative(axis)` and `plan(target)`. |
| `SeriesInterpolant`, `ChebyshevInterpolant` | The same from a 1-D series scheme and from `Chebyshev`. |

## Use it

```python
import jax
import jax.numpy as jnp

from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import BSpline, Linear, Plan, plan_key

grid = RectilinearGrid(
    (Axis(jnp.linspace(0.0, 1.0, 11)), Axis(jnp.linspace(0.0, 2.0, 21)))
)
x, y = jnp.meshgrid(grid.axes[0].values, grid.axes[1].values, indexing="ij")
field = (x**2 + 3.0 * y).ravel()  # (231,), row-major
targets = Points(jnp.array([[0.25, 0.5], [0.75, 1.5], [0.4, 1.1]]))

# build once, outside jit; apply to many fields
plan = Linear().plan(grid, targets)
frames = jnp.stack([field, 2.0 * field])  # (2, 231)
values = jax.jit(lambda fields: plan(fields))(frames)  # (2, 3)
assert values.shape == (2, 3)

# the adjoint is exact: ⟨H x, r⟩ = ⟨x, Hᵀ r⟩
residual = jnp.array([1.0, -2.0, 0.5])
back = plan.adjoint(residual)  # (231,)
assert jnp.allclose(plan(field) @ residual, field @ back, rtol=1e-5)

# the same map as a lineax operator, for a solver or for gaussx
operator = plan.operator()
assert operator.in_size() == 231 and operator.out_size() == 3

# ∂f/∂y at the targets is 3
slope = plan.derivative(1)(field)
assert jnp.allclose(slope, 3.0, atol=1e-3)

# a fitted field, read at positions chosen inside the trace
surface = BSpline(3).fit(grid, field)


def read(position):
    return surface(position[None, :])[0]


assert jnp.allclose(jax.jit(read)(jnp.array([0.5, 1.0])), 0.25 + 3.0, atol=1e-3)

# store and reload a plan, keyed by its inputs
key = plan_key(Linear(), grid, targets)
again = Plan.deserialize(plan.serialize(), like=plan)
assert len(key) == 64 and jnp.allclose(again(field), plan(field))
```

## Rules

- A plan is an Equinox module with array leaves: close over it in the function you `jit`, or use `equinox.filter_jit`; `jax.jit(plan)` itself fails because the plan is not hashable.
- Build the plan on the host, outside `jit`, `vmap` and `grad`. Locating the targets and checking the policies happen there once.
- Call `plan.adjoint` for the transpose. It is written by hand and exact; do not take `jax.vjp` of the plan for it.
- A field is one flat vector `(..., N)` in the source's row-major order, and the output is `(..., M)`. Reshape with `einx.id("t y x -> t (y x)", field)` or `.reshape`.
- Use an `Interpolant` when the targets move during the computation, as particle positions do; use a `Plan` when they are fixed.
- A linear-only operation on a `NonlinearPlan`, such as `operator()`, raises `NonlinearSchemeError`. Call `linearize(data)` for the tangent map.
- Cache plans by `plan_key`: equal schemes, policies and geometries give an equal key.
