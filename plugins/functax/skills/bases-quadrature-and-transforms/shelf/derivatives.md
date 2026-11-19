# Derivatives of fields

Role: ① primitive (`role-ladder`, `references/primitive.md`).
Unit: diffops.

Open this when a field is a function you can call, such as a neural field or a basis expansion, and you need its gradient, divergence, curl or Laplacian.

## Names

All of them live in `functax.diffops`. Each takes a callable of one point and returns a callable of one point.

| Name | Field | Returns |
|---|---|---|
| `gradient(f)` | scalar, `(D,) → ()` | ∇f, `(D,)` |
| `laplacian(f)` | scalar | Δf = ∑ ∂²f/∂x_d², `()` |
| `divergence(u)` | vector, `(D,) → (D,)` | ∇ · u, `()` |
| `curl2d(u)` | planar vector, `(2,) → (2,)` | ∂ₓu_y − ∂_yuₓ, `()` |
| `jacobian_field(u)` | vector, `(D,) → (K,)` | J with J_kd = ∂u_k/∂x_d, `(K, D)` |
| `surface_grad(f)` | scalar of a unit 3-vector | The surface gradient ∇_S f, a tangent 3-vector |
| `surface_grad_enu(f, *, unit)` | scalar of a unit 3-vector | Its east and north components, as a function of `(lon, lat)` |
| `laplace_beltrami(f)` | scalar of a unit 3-vector | Δ_S f on the unit sphere |

## Use it

```python
import jax
import jax.numpy as jnp

from functax.diffops import (
    curl2d,
    divergence,
    gradient,
    laplace_beltrami,
    laplacian,
    surface_grad_enu,
)


def potential(x):  # a scalar field of one point (2,)
    return x[0] ** 2 + 3.0 * x[0] * x[1]


def rotation(x):  # solid-body rotation, a vector field
    return jnp.array([-x[1], x[0]])


point = jnp.array([0.3, -0.2])
assert jnp.allclose(gradient(potential)(point), jnp.array([0.0, 0.9]))
assert jnp.allclose(laplacian(potential)(point), 2.0)
assert jnp.allclose(divergence(rotation)(point), 0.0)
assert jnp.allclose(curl2d(rotation)(point), 2.0)

# many points: vmap the returned function
points = jnp.array([[0.0, 0.0], [1.0, 2.0], [0.3, -0.2]])
vorticity = jax.vmap(curl2d(rotation))(points)  # (3,)
assert jnp.allclose(vorticity, 2.0)

# on the sphere: f = z is a degree-1 harmonic, so Δ_S f = −2 f
n = jnp.array([0.6, 0.0, 0.8])
assert jnp.allclose(laplace_beltrami(lambda m: m[2])(n), -2.0 * n[2], atol=1e-5)

# its gradient at (30°E, 0°N) points due north with unit length
east_north = surface_grad_enu(lambda m: m[2], unit="degrees")(jnp.array([30.0, 0.0]))
assert jnp.allclose(east_north, jnp.array([0.0, 1.0]), atol=1e-5)
```

## Rules

- These differentiate a callable by automatic differentiation. They do not take arrays of samples.
- Write the field as a function of one point of shape `(D,)`, and `jax.vmap` the result over points. Do not vmap the field first.
- For a field sampled on a grid, read the derivative with `plan.derivative(axis)` from `functax.interp`, or use a finite-volume or spectral library.
- The sphere operators take the field as a function of a 3-vector, defined near the sphere, and return derivatives along the surface only.
- `surface_grad_enu` returns a function of `(lon, lat)`, not of a unit vector, and needs `unit=`.
