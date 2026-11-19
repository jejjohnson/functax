# Bases on the sphere

Role: ① primitive (`role-ladder`, `references/primitive.md`).
Unit: basis.

Open this when the field lives on the sphere, globally or over a region, and you need its design matrix.

## Names

| Module | Name | What it gives |
|---|---|---|
| `functax.basis.sphere` | `real_spherical_harmonics(unit_xyz, l_max)` | The real spherical harmonics up to degree `l_max`: `(N, (l_max + 1)²)`. |
| | `harmonic_degrees(l_max)` | The degree of each column. |
| | `vector_spherical_harmonics(unit_xyz, l_max)` | The gradient and toroidal vector harmonics, for tangent vector fields. |
| | `needlet_basis(unit_xyz, *, B=2.0, j_max)`, `needlet_window(B=2.0)` | Spherical needlets, a localised multiscale frame, with their levels and centres; and the window they are built from. |
| | `shell_basis(xyz, l_max, n_radial, *, r_inner, r_outer)` | Spherical harmonics times radial polynomials on a shell. |
| `functax.basis.slepian` | `slepian_cap_basis(l_max, cap_radius, *, n_modes=None, lonlat_centre=None)` | The Slepian functions of a spherical cap, band-limited and concentrated in it: a `SlepianCapBasis`. |
| | `SlepianCapBasis` | `evaluate(unit_xyz)`, `num_modes`, `eigenvalues`, `rotate_to(lonlat_centre)`. |
| | `slepian_region_basis(l_max, region_points_xyz, region_weights)`, `slepian_polar_gap_basis(l_max, gap_colatitude)` | The Slepian functions of a general region given by a quadrature over it, and of a polar gap: a `SlepianBasis`. |
| | `SlepianBasis` | `evaluate(unit_xyz)`, `num_modes`, `eigenvalues`. |
| | `shannon_number(l_max, area)` | N = (l_max + 1)² A / (4π), how many modes are well concentrated. |
| | `slepian_concentration_matrix(l_max, cap_radius)`, `slepian_cap_eigh_per_m(l_max, cap_radius)` | The concentration matrix of a cap and its eigenproblem solved per azimuthal order. |
| | `spherical_polygon_mask(points_xyz, polygon_lonlat, *, unit)` | Which points lie inside a spherical polygon. |

## Use it

```python
import jax.numpy as jnp

from functax.basis.slepian import shannon_number, slepian_cap_basis
from functax.basis.sphere import (
    harmonic_degrees,
    needlet_basis,
    real_spherical_harmonics,
)
from functax.geometry import fibonacci_sphere
from functax.project import least_squares

points, weights = fibonacci_sphere(600)  # (600, 3), (600,)

# a global field in spherical harmonics up to degree 4
Phi = real_spherical_harmonics(points, 4)  # (600, 25)
assert Phi.shape == (600, 25) and len(harmonic_degrees(4)) == 25
y = points[:, 2] ** 2 + 0.5 * points[:, 0]  # degrees 0, 1 and 2 only
c = least_squares(Phi, y)
assert jnp.allclose(Phi @ c, y, atol=1e-3)

# a regional field: Slepian functions of a 20° cap, band-limited to degree 12
cap = slepian_cap_basis(12, jnp.deg2rad(20.0), n_modes=6)
assert cap.num_modes == 6
area = 2 * jnp.pi * (1 - jnp.cos(jnp.deg2rad(20.0)))
assert 4.0 < float(shannon_number(12, area)) < 6.0  # about 5 well-concentrated modes
values = cap.evaluate(points)  # (600, 6)
assert values.shape == (600, 6)

# a localised multiscale frame
needlets, levels, centres = needlet_basis(points, j_max=1)
assert needlets.shape[0] == 600 and needlets.shape[1] == levels.shape[0]
```

## Rules

- Points go in as unit vectors `(N, 3)`. Convert with `functax.geometry.lonlat_to_unit(lonlat, unit=...)` first.
- A degree `l_max` gives `(l_max + 1)²` columns, so the count grows fast: degree 30 is 961 columns.
- For a field known only over a region, use a Slepian basis and keep about `shannon_number(l_max, area)` modes. Fitting global harmonics to regional data is ill-conditioned.
- Move a cap basis with `rotate_to(lonlat_centre)` or build it there with `lonlat_centre=`; the cap's eigenproblem does not depend on where it sits.
- To integrate against a basis, evaluate it at the nodes of a `functax.quad.SphereRule` and use `functax.project.galerkin` with the rule's weights.
- To analyse a field on a regular lon/lat grid, use `functax.transform.SphericalHarmonicTransform`, which builds the analysis matrix once.
