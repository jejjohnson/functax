# Projection and gap filling

Role: ④ algorithm (`role-ladder`, `references/algorithm.md`).
Unit: project.

Open this when data must become coefficients of a basis, a smooth field on a grid, or an image without holes.

## Names

All of them live in `functax.project`.

| Name | What it does |
|---|---|
| `least_squares(design_matrix, y, *, ridge=0.0)` | c = argmin ‖Φc − y‖² + α‖c‖², the ridge least-squares coefficients. |
| `galerkin(design_matrix, y, *, weights)` | c = (ΦᵀWΦ)⁻¹ ΦᵀW y, the projection in the inner product of a quadrature rule. |
| `Variational(observation, roughness, observation_precision=None, smoothing_weight=1.0, solver=None)` | x = argmin (Hx − y)ᵀR⁻¹(Hx − y) + λ‖Lx‖², the smoothing fit, from lineax operators and matrix-free. Call it on `y`. |
| `fill_gaps(image, *, mask, method="laplace", periodic=(False, False), solver=None)` | An image with its holes filled by Δx = 0, or Δ²x = 0 with `method="biharmonic"`. `mask` is true where the image is observed. |

## Use it

```python
import jax.numpy as jnp
import lineax as lx

from functax.project import Variational, fill_gaps, galerkin, least_squares
from functax.quad import gauss_legendre

# the L² projection of x² onto {1, x} on [-1, 1] is the constant 1/3
nodes, weights = gauss_legendre(8)
design = jnp.stack([jnp.ones_like(nodes), nodes], axis=-1)  # (8, 2)
c = galerkin(design, nodes**2, weights=weights)
assert jnp.allclose(c, jnp.array([1.0 / 3.0, 0.0]), atol=1e-5)

# an overdetermined fit, for a batch of right-hand sides at once
t = jnp.linspace(0.0, 1.0, 20)
Phi = jnp.stack([jnp.ones_like(t), t], axis=-1)  # (20, 2)
y = jnp.stack([1.0 + 2.0 * t, 3.0 - t])  # (2, 20)
coefficients = least_squares(Phi, y)  # (2, 2)
assert jnp.allclose(coefficients, jnp.array([[1.0, 2.0], [3.0, -1.0]]), atol=1e-4)

# the smoothing fit with H = I and L = I: x = y / (1 + λ)
identity = lx.IdentityLinearOperator(jnp.zeros(4))
fit = Variational(identity, identity, smoothing_weight=1.0)
data = jnp.array([1.0, -2.0, 0.5, 3.0])
assert jnp.allclose(fit(data), data / 2.0, atol=1e-5)

# fill two missing pixels between 1 and 0: linear under the Laplace fill
image = jnp.array([[0.0, 1.0, jnp.nan, jnp.nan, 0.0]])
filled = fill_gaps(image, mask=jnp.isfinite(image))
assert jnp.allclose(filled, jnp.array([[0.0, 1.0, 2 / 3, 1 / 3, 0.0]]), atol=1e-5)
```

## Rules

- Use `least_squares` for scattered samples and `galerkin` when the samples sit on the nodes of a quadrature rule, whose weights make the fit an L² projection.
- `y` may carry leading batch dimensions, and the coefficients keep them, so fit every time step in one call.
- Add `ridge=` when the design matrix is ill-conditioned, which is the usual case for a basis with more columns than the data supports.
- In `Variational`, the observation operator H is a plan's `operator()` from `functax.interp`, and the roughness L is a difference operator. Both are lineax operators, so nothing is densified.
- The variational solve is differentiated implicitly by lineax, so a gradient in the data or in `smoothing_weight` costs one more solve.
- `fill_gaps` is for an image on a regular grid. For scattered samples onto a grid, bin them first with `functax.interp.Binning`, then fill.
