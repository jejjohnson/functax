# Quickstart

## Two installs

functax has a NumPy-only base install and a `jax` extra. The base exists so that NumPy-only consumers, such as geoarray, can call functax's affine raster kernels without pulling in JAX.

| Install | Dependencies | Modules |
|---|---|---|
| `functax` | NumPy | `functax`, `functax.geometry.affine` |
| `functax[jax]` | NumPy, JAX, Equinox, lineax, einx, jaxtyping | every module |

Importing `functax` never imports JAX:

```python
import functax as fx

fx.__version__  # '0.0.1'
```

## Without the extra

A module that needs JAX checks for the extra on import. Without it, the import fails with a [`MissingExtraError`](xref:api#functax.MissingExtraError) that names the extra to install, rather than with a bare `No module named 'jax'`:

```text
MissingExtraError: functax.interp needs the functax[jax] extra, but 'jax' could
not be imported; install it with: pip install 'functax[jax]'
```

`MissingExtraError` is also an `ImportError`, so an existing `except ImportError` still catches it.

## Three operations on one expansion

functax writes a field $x(\vect{s})$ over the coordinates $\vect{s}$ as one expansion, $x(\vect{s}) \approx \sum_{d=1}^{D_z} z_d \phi_d(\vect{s}; \gvect{\theta})$, with basis functions $\phi_d$ and coefficients $\vect{z} \in \mathbb{R}^{D_z}$.
Here are three of the five operations on it, on one small problem.
We fit the coefficients of a field on the sphere from scattered points with [`least_squares`](xref:api#functax.project.least_squares).
Then we read a grid at two points with a [`Linear`](xref:api#functax.interp.Linear) plan and its adjoint, and integrate the fitted field over the sphere with [`sphere_gauss_legendre`](xref:api#functax.quad.sphere_gauss_legendre).

```python
import jax
import jax.numpy as jnp
from functax.basis.sphere import real_spherical_harmonics
from functax.geometry import AffineGrid, Points, fibonacci_sphere
from functax.interp import Linear
from functax.project import least_squares
from functax.quad import sphere_gauss_legendre

jax.config.update("jax_enable_x64", True)

# Shapes: N_Ωy = 500 points on the sphere; D_z = 25 harmonics (ℓ ≤ 4);
#         N_Ωx = 20 grid pixels; N_* = 2 targets; N_q = 128 quadrature nodes

# A · fit the coefficients of a field on the sphere
# today (functax)
points, _ = fibonacci_sphere(500)                       # → (N_Ωy, 3)
# yₙ = x(sₙ) = s₃² + s₁ / 2
observations = points[:, 2] ** 2 + 0.5 * points[:, 0]   # (N_Ωy, 3) → (N_Ωy,)
# today (functax)
# [Φ]ₙd = Y_ℓᵐ(sₙ), the real spherical harmonics with ℓ ≤ 4
design = real_spherical_harmonics(points, 4)            # (N_Ωy, 3) → (N_Ωy, D_z)
# today (functax)
# ẑ = argmin_z ‖Φz − y‖²
coefficients = least_squares(design, observations)      # → (D_z,)

# B · read a grid at two points, and send residuals back
# a north-up affine transform (a, b, c, d, e, f) with 10 m pixels
transform = [10.0, 0.0, 500.0, 0.0, -10.0, 900.0]       # → (6,)
# today (functax)
grid = AffineGrid(transform, (4, 5))                    # 4 × 5 pixels, N_Ωx = 20
# today (functax)
targets = Points(jnp.array([[515.0, 885.0], [540.0, 870.0]]))  # → (N_*, 2)
# today (functax)
# H, the bilinear weights from the pixels to the targets
plan = Linear().plan(grid, targets)                     # (N_*, N_Ωx) operator
# y_* = H x
values = plan(jnp.arange(20.0))                         # (N_Ωx,) → (N_*,)
# Hᵀ r
back = plan.adjoint(jnp.ones(2))                        # (N_*,) → (N_Ωx,)

# C · integrate the fitted field over the sphere
# today (functax)
# nodes s_q and weights w_q, exact for every Y_ℓᵐ with ℓ ≤ 15
rule = sphere_gauss_legendre(8)                         # → (N_q, 3), (N_q,)
# today (functax)
design_nodes = real_spherical_harmonics(rule.nodes, 4)  # (N_q, 3) → (N_q, D_z)
# x̂(s_q) = [Φ_q ẑ]_q
fitted = design_nodes @ coefficients                    # → (N_q,)
# ∫ x̂ ds ≈ ∑_q w_q x̂(s_q)
integral = rule.weights @ fitted                        # → ()
```

The field $x(\vect{s}) = s_3^2 + s_1 / 2$ only has harmonics up to degree 2, so the 25 coefficients $\hat{\vect{z}}$ fit the 500 points to machine precision.
The plan reads `values = [6.0, 16.0]`, and the integral agrees with the exact $\int_{S^2} x(\vect{s}) \mathrm{d}\vect{s} = 4\pi/3$ to 15 significant digits.
The basis, the plan and the rule are built once.
The lines that apply them, the plan call, its adjoint and the contractions, all trace under `jit` and differentiate under `grad`.
The [interpolation](./interpolation.md) and [quadrature](./quadrature.md) guides go further into each of these.
