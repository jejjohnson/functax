# Bases on lines and boxes

Role: ① primitive (`role-ladder`, `references/primitive.md`).
Unit: basis.

Open this when you need a design matrix Φ for a field on an interval or a box: f̂(x) = ∑ⱼ cⱼ φⱼ(x).

## Names

Each family lives in its own submodule of `functax.basis`.

| Module | Name | What it gives |
|---|---|---|
| `functax.basis.fourier` | `fourier_basis(x, num_basis_per_dim, L, *, boundary="dirichlet")` | The tensor-product Laplacian eigenfunctions on the box [−L, L]ᴰ, with their eigenvalues: `(Φ, λ)`. |
| | `fourier_basis_1d(x, num_basis, L)`, `fourier_eigenvalues(...)`, `fourier_eigenvalues_1d(...)` | One axis alone, and the eigenvalues without Φ. |
| `functax.basis.features` | `fourier_features(x, max_degree)` | Cosines and sines of a coordinate at dyadic frequencies. |
| | `seasonal_features(x, periods, harmonics)`, `seasonal_frequencies(periods, harmonics)` | Cosines and sines of a time at the harmonics of given periods, and those frequencies. |
| | `gaussian_window_features(t, centers, widths)` | Gaussian windows in time. |
| | `rff_forward(W, lengthscale, n_features, x)`, `rff_cosine_forward(W, b, lengthscale, n_features, x)` | Random Fourier features at given frequencies, for one point `x`. |
| `functax.basis.polynomial` | `legendre_polynomials(t, l_max)`, `legendre_series(x, coeffs)` | The Legendre polynomials, and a Legendre series evaluated elementwise. |
| | `associated_legendre(t, l_max)`, `associated_legendre_indices(l_max)` | The associated Legendre functions and the `(l, m)` of each column. |
| `functax.basis.local` | `rbf_basis(x, centers, widths, *, kernel="gaussian")`, `spherical_rbf_basis(unit_xyz, centers, widths)` | Placeable radial bumps, in Euclidean and in geodesic distance. |
| | `wendland_c2(r)`, `wendland_c4(r)` | The compact Wendland profiles. |
| | `gabor_frame(x, centers, scales, wavenumbers)`, `gabor_frame_grid(x, bounds, *, n_scales, base_scale)` | Gabor atoms, and a dyadic frame over a box. |
| | `wavelet_basis_1d(n, *, wavelet="haar")`, `wavelet_basis_2d(ny, nx)` | Orthonormal discrete-wavelet synthesis matrices. |
| `functax.basis.vector` | `divfree_basis(xy, num_basis_per_dim, L)`, `curlfree_basis(xy, num_basis_per_dim, L)` | Divergence-free and curl-free vector atoms on a box, `(N, D, 2)`, with eigenvalues. |

## Use it

```python
import jax.numpy as jnp

from functax.basis.features import seasonal_features
from functax.basis.fourier import fourier_basis
from functax.basis.local import rbf_basis, wavelet_basis_1d
from functax.basis.polynomial import legendre_series
from functax.basis.vector import divfree_basis
from functax.project import least_squares

# fit a field on the box [-1, 1]² with Laplacian eigenfunctions
x = jnp.stack(jnp.meshgrid(jnp.linspace(-0.9, 0.9, 12), jnp.linspace(-0.9, 0.9, 12)), -1)
x = x.reshape(-1, 2)  # (144, 2)
Phi, eigenvalues = fourier_basis(x, num_basis_per_dim=4, L=1.0)  # (144, 16), (16,)
y = jnp.cos(0.5 * jnp.pi * x[:, 0]) * jnp.cos(0.5 * jnp.pi * x[:, 1])  # the first mode
c = least_squares(Phi, y)  # (16,)
assert jnp.allclose(Phi @ c, y, atol=1e-3)

# seasonal columns for a daily series: weekly and yearly, two harmonics each
days = jnp.arange(730.0)
season = seasonal_features(days, [7.0, 365.25], [2, 2])  # (730, 8)
assert season.shape == (730, 8)

# radial bumps at chosen centres; each peaks at 1 on its centre
centres = jnp.array([[0.0, 0.0], [0.5, -0.5]])
bumps = rbf_basis(centres, centres, jnp.ones(2))
assert jnp.allclose(jnp.diag(bumps), 1.0)

# an orthonormal wavelet basis on a periodic grid of 8 samples
W = wavelet_basis_1d(8, wavelet="db2")
assert jnp.allclose(W.T @ W, jnp.eye(8), atol=1e-5)

# a divergence-free vector basis: (N, D, 2)
flow, _ = divfree_basis(x, num_basis_per_dim=3, L=1.0)
assert flow.shape == (144, 9, 2)

# P₂(x) = (3x² − 1)/2
assert jnp.allclose(legendre_series(jnp.array([0.0, 1.0]), jnp.array([0.0, 0.0, 1.0])), jnp.array([-0.5, 1.0]))
```

## Rules

- Import each family from its submodule. `functax.basis` exports only the submodules.
- A design matrix is `(N, D)`: one row per point, one column per basis function. Fit with `functax.project.least_squares` and evaluate with `Φ @ c`.
- Several constructors return a tuple, `(Φ, eigenvalues)` or more. Unpack it; the eigenvalues are what a spectral prior needs.
- `fourier_basis` works on the box [−L, L]ᴰ, so choose `L` larger than the data's extent, or the fit is pinned to zero at the data's edge under the default Dirichlet boundary.
- `rff_forward` and `rff_cosine_forward` take one point; wrap them in `jax.vmap` for many.
- For a vector field that must conserve mass, expand it in `divfree_basis` instead of fitting two scalar fields.
