# Transforms

Role: setup (`role-ladder`, `references/strategy.md`).
Unit: transform.

Open this when you move a field between its values on a grid and its spectral coefficients, many times, on the same grid.

## Names

All of them live in `functax.transform`. A transform is built once for a grid and holds its matrices.

| Name | What it is |
|---|---|
| `ChebyshevTransform1D(n, kind="lobatto")` | Values on the Chebyshev nodes of degree `n` to Chebyshev coefficients: `to_spectral(u)` and `from_spectral(a)`. |
| `ChebyshevTransform2D(nx, ny, kind="lobatto")` | The tensor-product transform on an `(N_y, N_x)` grid of nodes. |
| `SphericalHarmonicTransform.init(n_lat, n_lon, l_max)` | Fields on a Gauss–Legendre lon/lat grid, `(channel, lat, lon)`, to real spherical-harmonic coefficients: `forward(field)` and `inverse(coefficients)`. |
| `SphericalWaveletTransform.init(sht, *, n_scales=None)` | A field split into scale bands by needlet filters on the harmonic coefficients: `forward(field)` gives `(scale, channel, lat, lon)`, and `inverse(bands)` sums them back. |
| `nonuniform_rfft(u, xi, weights, n_modes)` | Scattered samples, with quadrature weights, to a truncated rfft spectrum. |
| `nonuniform_irfft(coeffs, xi, n_modes)` | The spectrum evaluated at scattered points. |

## Use it

```python
import jax.numpy as jnp
import jax.random as jr

from functax.geometry import chebyshev_nodes
from functax.transform import (
    ChebyshevTransform1D,
    SphericalHarmonicTransform,
    SphericalWaveletTransform,
    nonuniform_rfft,
)

# Chebyshev: T₃ on the Gauss nodes is the unit vector e₃
gauss = ChebyshevTransform1D(6, kind="gauss")
t3 = jnp.cos(3 * jnp.arccos(chebyshev_nodes(6, kind="gauss")))
assert jnp.allclose(gauss.to_spectral(t3), jnp.eye(6)[3], atol=1e-5)

# and the round trip on the Gauss–Lobatto nodes
lobatto = ChebyshevTransform1D(16)
u = jnp.sin(jnp.pi * chebyshev_nodes(16))  # (17,)
assert jnp.allclose(lobatto.from_spectral(lobatto.to_spectral(u)), u, atol=1e-5)

# spherical harmonics on a 12 × 24 Gauss–Legendre grid, up to degree 8
sht = SphericalHarmonicTransform.init(n_lat=12, n_lon=24, l_max=8)
coefficients = jr.normal(jr.key(0), (2, 81))  # (channel, (l_max + 1)²)
field = sht.inverse(coefficients)  # (2, 12, 24)
assert jnp.allclose(sht.forward(field), coefficients, atol=1e-4)

# the same field as a sum of scale bands
swt = SphericalWaveletTransform.init(sht)
bands = swt.forward(field)  # (scale, 2, 12, 24)
assert jnp.allclose(swt.inverse(bands), field, atol=1e-4)

# scattered samples to a 4 × 4 block of Fourier modes
xi = jr.uniform(jr.key(0), (500, 2))
signal = jnp.sin(2 * jnp.pi * xi[:, 0])[None]  # (1, 500)
blocks = nonuniform_rfft(signal, xi, jnp.full((500,), 1 / 500), (4, 4))
assert len(blocks) == 2 and blocks[0].shape == (1, 4, 4)
```

## Rules

- Build a transform once and reuse it. The spherical transforms hold dense synthesis and analysis matrices, so building one inside a loop is the expensive part.
- Build the spherical transforms with `init`, not with the dataclass fields.
- The values must sit on the transform's own grid: `functax.geometry.chebyshev_nodes(n, kind=...)` with the same `kind`, and the Gauss–Legendre grid of `functax.geometry.gauss_legendre_grid` for the spherical transform.
- The Chebyshev nodes run from the upper end of the interval down to the lower, so a field sampled in increasing order must be flipped first.
- `l_max` must be resolved by the grid. Analysis of a field with more detail than `l_max` aliases, as any truncated transform does.
- For a field at scattered points on the sphere there is no fast transform: evaluate `functax.basis.sphere.real_spherical_harmonics` there and fit with `functax.project.least_squares`.
