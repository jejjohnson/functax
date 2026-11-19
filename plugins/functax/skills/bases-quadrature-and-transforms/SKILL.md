---
name: bases-quadrature-and-transforms
description: Build and use basis expansions in JAX with functax — Laplacian eigenfunctions, Fourier and seasonal features, radial bases, wavelets and Gabor frames on boxes; real and vector spherical harmonics, needlets and Slepian functions on the sphere; EOFs and graph or mesh Laplacian modes from data; quadrature rules on intervals, spheres, grid cells and curves; gradient, divergence, curl and Laplacian of a callable field; ridge least squares, Galerkin projection, a variational smoothing fit and gap filling; Chebyshev, spherical-harmonic and non-uniform Fourier transforms. Use when fitting a field to a basis, integrating a field, computing a flux across a transect, differentiating a neural or analytic field, or moving between grid values and spectral coefficients, in a project that uses (or could use) functax.
---

# Bases, quadrature and transforms with functax

A field is an expansion, f̂(x) = ∑ⱼ cⱼ φⱼ(x).
This skill covers the four operations on it that are not interpolation: evaluate the basis, integrate, differentiate, and move between samples and coefficients by a fit or by a transform.

It covers rung ① of functax, the primitives, with the algorithms of `functax.project`, in the vocabulary of the `role-ladder` skill.
The geometry these take is in `functax:function-approximation-with-functax`.

## The contract

- A basis function returns a design matrix Φ of shape `(N, D)`: one row per point, one column per basis function.
- A rule is data: `nodes` and `weights`, with ∫ g ≈ ∑ₙ wₙ g(xₙ). It has no `integrate` method.
- A differential operator takes a callable of one point and returns a callable of one point.
- A fit returns coefficients and keeps the batch dimensions of the data.
- A transform is built once for a grid, and then goes both ways.

## Which one

| The task | Use |
|---|---|
| A smooth field on a box | `functax.basis.fourier.fourier_basis` |
| A time series with known periods | `functax.basis.features.seasonal_features` |
| Localised bumps at chosen centres | `functax.basis.local.rbf_basis` |
| A global field on the sphere | `functax.basis.sphere.real_spherical_harmonics` |
| A field known over one region of the sphere | `functax.basis.slepian.slepian_cap_basis` or `slepian_region_basis` |
| Modes of an ensemble of fields | `functax.basis.eigen.eof_basis` |
| Coefficients from scattered samples | `functax.project.least_squares(Φ, y, ridge=...)` |
| Coefficients from samples on quadrature nodes | `functax.project.galerkin(Φ, y, weights=rule.weights)` |
| An integral over an interval, the sphere, a grid's cells | `functax.quad.gauss_legendre`, `sphere_gauss_legendre`, `cell_area` |
| A flux across a transect | `functax.quad.polyline` and its normals |
| ∇f, ∇ · u, Δf of a callable field | `functax.diffops.gradient`, `divergence`, `laplacian` |
| Grid values to coefficients, repeatedly | a class of `functax.transform` |
| Holes in an image | `functax.project.fill_gaps` |

## Worked example

```python
import jax.numpy as jnp

from functax.basis.sphere import real_spherical_harmonics
from functax.diffops import laplace_beltrami
from functax.geometry import fibonacci_sphere
from functax.project import galerkin, least_squares
from functax.quad import sphere_gauss_legendre

# a field on the sphere, known at 500 scattered points
points, _ = fibonacci_sphere(500)  # (500, 3)
samples = points[:, 2] ** 2 + 0.5 * points[:, 0]

# evaluate the basis and fit: 25 harmonics, degree ≤ 4
Phi = real_spherical_harmonics(points, 4)  # (500, 25)
c = least_squares(Phi, samples)  # (25,)

# integrate the fitted field: ∫ (z² + x/2) dΩ = 4π/3
rule = sphere_gauss_legendre(8)
fitted = real_spherical_harmonics(rule.nodes, 4) @ c
assert jnp.allclose(rule.weights @ fitted, 4 * jnp.pi / 3, atol=1e-3)

# the same coefficients by projection on the rule's nodes
exact = rule.nodes[:, 2] ** 2 + 0.5 * rule.nodes[:, 0]
assert jnp.allclose(galerkin(real_spherical_harmonics(rule.nodes, 4), exact, weights=rule.weights), c, atol=1e-3)

# differentiate the field as a function: x is a degree-1 harmonic, Δ_S x = −2x
n = jnp.array([0.6, 0.0, 0.8])
assert jnp.allclose(laplace_beltrami(lambda m: m[0])(n), -2.0 * n[0], atol=1e-5)
```

## Don't write it

| Don't write | Use |
|---|---|
| `scipy.special.sph_harm` in a loop | `real_spherical_harmonics(unit_xyz, l_max)` |
| `np.polynomial.legendre.leggauss` and a rescale | `gauss_legendre(n, interval=(a, b))` |
| `jnp.linalg.lstsq`, or `(ΦᵀΦ)⁻¹Φᵀy` by hand | `least_squares(Φ, y, ridge=...)` |
| An SVD of a data matrix for PCA modes | `eof_basis(data, n_modes)` |
| `jax.jacfwd` and a trace for a divergence | `divergence(u)` |
| A DCT for Chebyshev coefficients | `ChebyshevTransform1D(n).to_spectral(u)` |
| A Jacobi iteration to inpaint an image | `fill_gaps(image, mask=...)` |

## The shelf

Each entry is one file with the names it covers, a worked example and the rules.
Open the one the task needs; do not read them all.

<!-- shelf:begin -->
<!-- shelf:end -->

<!-- shelf:below:begin -->
<!-- shelf:below:end -->

## Before you finish

- Are points on the sphere passed as unit vectors?
- Is each basis imported from its submodule, and its tuple unpacked?
- Does the integral use a rule's weights, not hand-made ones?
- Is each transform built once, on its own grid?
