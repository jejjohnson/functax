# Points and frames on the sphere

Role: ① primitive (`role-ladder`, `references/primitive.md`).
Unit: geometry.

Open this when coordinates are longitudes and latitudes and the next step needs unit vectors, distances, a local frame or a point set with quadrature weights.

## Names

All of them live in `functax.geometry`. `unit=` is `"radians"` or `"degrees"` and has no default where a mistake would be silent.

| Name | What it does |
|---|---|
| `lonlat_to_unit(lonlat, *, unit="radians")`, `unit_to_lonlat(xyz, *, unit="radians")` | Longitude and latitude to unit vectors `(..., 3)` and back. |
| `lonlat_to_ecef(lonlat, altitude=0.0, *, semi_major, flattening=0.0, unit)`, `ecef_to_lonlat(xyz, *, semi_major, flattening=0.0, unit)` | Geodetic coordinates on an ellipsoid to Earth-centred, Earth-fixed and back. |
| `geodesic_distance(u, v, *, radius=1.0)`, `chord_distance(u, v, *, radius=1.0)` | The `(N, M)` great-circle and straight-line distances between two sets of unit vectors. |
| `spherical_polygon_area(vertices, *, radius=1.0)` | The signed area of a polygon with great-circle edges. |
| `enu_basis(lonlat, *, unit)` | The east, north and up unit vectors at each point, `(N, 3, 3)`. |
| `cartesian_to_tangent(lonlat, vec, *, unit)`, `tangent_to_cartesian(lonlat, uv, *, unit)` | A 3-vector to its east and north components, and back. |
| `local_tangent_plane(lonlat, origin, *, radius, projection="azimuthal_equidistant", unit)` | Points projected onto the tangent plane at `origin`, as `(east, north)`. |
| `rotation_to_pole(lonlat_centre, *, unit)` | The rotation that takes a centre to the north pole. |
| `fibonacci_sphere(n)`, `gauss_legendre_grid(n_lat, n_lon=None)`, `healpix_grid(nside, *, nest=True)`, `latlon_grid(n_lat, n_lon)` | Point sets on the unit sphere, each returned with its quadrature weights. `output="xyz"` by default; `output="lonlat"` needs `unit=`. |
| `icosphere(level, *, radius=1.0)` | The subdivided icosahedron: vertices and triangles. |
| `mesh_vertex_areas(vertices, triangles)` | The lumped vertex areas of any triangle mesh. |

## Use it

```python
import jax.numpy as jnp

from functax.geometry import (
    cartesian_to_tangent,
    enu_basis,
    fibonacci_sphere,
    geodesic_distance,
    icosphere,
    local_tangent_plane,
    lonlat_to_unit,
    mesh_vertex_areas,
    unit_to_lonlat,
)

lonlat = jnp.array([[2.35, 48.86], [-0.13, 51.51]])  # Paris, London, in degrees
xyz = lonlat_to_unit(lonlat, unit="degrees")  # (2, 3)
assert jnp.allclose(unit_to_lonlat(xyz, unit="degrees"), lonlat, atol=1e-3)

km = geodesic_distance(xyz, xyz, radius=6371.0)  # (2, 2)
assert 330.0 < float(km[0, 1]) < 350.0

# a wind given as a 3-vector, read as east and north components
frames = enu_basis(lonlat, unit="degrees")  # (2, 3, 3)
east = frames[:, 0]
uv = cartesian_to_tangent(lonlat, 5.0 * east, unit="degrees")  # (2, 2)
assert jnp.allclose(uv, jnp.array([[5.0, 0.0], [5.0, 0.0]]), atol=1e-4)

# metres east and north of Paris
plane = local_tangent_plane(lonlat, lonlat[0], radius=6.371e6, unit="degrees")
assert jnp.allclose(plane[0], 0.0, atol=1.0)

points, weights = fibonacci_sphere(500)  # (500, 3), (500,)
assert jnp.allclose(weights.sum(), 4 * jnp.pi, rtol=1e-5)

vertices, triangles = icosphere(2)
areas = mesh_vertex_areas(vertices, triangles)  # one per vertex
assert areas.shape == (vertices.shape[0],)
```

## Rules

- Pass unit vectors, not longitudes and latitudes, to the distance functions and to every basis on the sphere.
- State `unit=` on every call that takes a longitude and latitude. The frame functions have no default, so that degrees are never read as radians.
- The weights of a sphere point set sum to 4π on the unit sphere. Use them as quadrature weights, or build a `functax.quad.SphereRule` with the constructors of `functax.quad`, which carry the radius.
- `healpix_grid` and `gauss_legendre_grid` integrate spherical harmonics exactly up to a degree; `fibonacci_sphere` is near-uniform but not exact.
- `lonlat_to_ecef` and `ecef_to_lonlat` take the ellipsoid as `semi_major=` and `flattening=`: functax holds no datum.
