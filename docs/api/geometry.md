# Geometry

CRS-free geometry: axes, grids, point sets, the sphere and affine raster maths. Coordinates arrive already in one coordinate reference system; the CRS stays a concern of the carrier.

## Affine raster maths

The transform $T = (a, b, c, d, e, f)$ maps the pixel corner at column $j$ and row $i$ to

$$
x = a j + b i + c \text{ and } y = d j + e i + f,
$$

in the order of `affine.Affine`, rasterio and geoarray (not GDAL's). Pixel coordinates in functax follow the area convention: integers are pixel centres, so $(0, 0)$ is the centre of the top-left pixel and its outer corner is $(-\frac{1}{2}, -\frac{1}{2})$. `to_pixel` inverts the $2 \times 2$ part $A$ of $T$, with $\det A = ae - bd$, and subtracts $\frac{1}{2}$; `from_pixel` is its inverse. Both reproduce geoarray's arithmetic exactly.

`to_pixel`, `from_pixel`, `pixel_centres` and `window` are the one part of functax in the NumPy-only base install: they take the array namespace of their input, return NumPy for NumPy and JAX for JAX, and import JAX only when a JAX array arrives. On JAX they trace under `jit`, `vmap` and `grad`. `window` follows geoarray's `Grid.window`: it rounds outward to whole pixels, snaps corners within `atol` pixels of an edge, and is not clipped to the grid.

## The sphere

Longitude $\lambda$ and latitude $\varphi$ lift to the unit vector $\vect{u} = (\cos\varphi \cos\lambda, \cos\varphi \sin\lambda, \sin\varphi)$ with `lonlat_to_unit`, the convention of geonnax, and `unit_to_lonlat` goes back. For unit vectors $\vect{u}$ and $\vect{v}$ on a sphere of radius $a$, `chord_distance` and `geodesic_distance` return

$$
d_{\text{chord}} = a \lVert \vect{u} - \vect{v} \rVert \text{ and } d_{\text{geo}} = a \mathrm{atan2}(\lVert \vect{u} \times \vect{v} \rVert, \vect{u} \cdot \vect{v}),
$$

for every pair. The $\mathrm{atan2}$ form is accurate at every angle, where $\arccos(\vect{u} \cdot \vect{v})$ loses precision near $0$ and $\pi$ and has an infinite derivative at $\vect{u} = \vect{v}$; both norms are guarded so the gradient at zero distance is zero rather than NaN. The two distances satisfy $d_{\text{chord}} = 2a \sin(d_{\text{geo}} / 2a)$.

`pairwise_distance(sources, targets)` dispatches on the metric the two `Points` share: `"euclidean"` on `xy` as given, or `"chord"` and `"geodesic"` on `xy` read as $(\lambda, \varphi)$. A pair with a padding point gets $+\infty$, and large problems are processed in chunks of targets with `lax.map`. The radius is a plain argument; the WGS84 constants and the ellipsoid belong to geophysix.

## Sphere grids and meshes

`fibonacci_sphere`, `gauss_legendre_grid`, `latlon_grid` and `healpix_grid` return points on the unit sphere with quadrature weights $w_q$ for $\int_{S^2} f \mathrm{d}\Omega \approx \sum_q w_q f(\vect{u}_q)$, summing to $4\pi$. The Gauss–Legendre grid takes the sines of latitude from `gauss_legendre` and is exact for band-limited integrands of degree up to $\min(2 n_{\text{lat}} - 1, n_{\text{lon}} - 1)$; the lon/lat grid weights each cell by its exact area; HEALPix pixels have equal area and come in the ring or the nested order of healpy. Points come as unit vectors, or as $(\lambda, \varphi)$ with `output="lonlat"`, which needs an explicit `unit`.

`icosphere` subdivides the icosahedron into a quasi-uniform triangle mesh with $V = 10 \cdot 4^{\ell} + 2$ vertices at level $\ell$, and `mesh_vertex_areas` gives any triangle mesh its lumped vertex areas $a_v = \sum_{t \ni v} |t| / 3$, the diagonal of its mass matrix. The grids and meshes are built on the host with NumPy, as static geometry.

## Chebyshev nodes

`chebyshev_nodes(n, kind=..., interval=(a, b))` returns the nodes of the Chebyshev transforms on $[a, b]$, $x_j = c + h \xi_j$ with $c = (a + b)/2$ and $h = (b - a)/2$: the $N + 1$ Gauss–Lobatto nodes $\xi_j = \cos(j\pi/N)$, the extrema of $T_N$ with both ends, for `kind="lobatto"`, or the $N$ Gauss nodes $\xi_j = \cos((2j + 1)\pi/2N)$, the zeros of $T_N$, for `kind="gauss"`. They are the nodes of spectraldiffx's `ChebyshevGrid1D`, `ChebyshevGrid2D` and `ChebyshevGrid3D` at `d68f8f6` (its $[-L, L]$ is `interval=(-L, L)`), in its order, decreasing from the upper end, which is the order `ChebyshevTransform1D` expects. `clenshaw_curtis` orders the same Lobatto nodes ascending, so `clenshaw_curtis(n + 1, interval)` is the quadrature rule for these nodes reversed.

## Frames

`lonlat_to_ecef` maps geodetic $(\lambda, \varphi)$ and height $h$ on the ellipsoid with semi-major axis $a$ and flattening $f$ to Earth-centred, Earth-fixed coordinates,

$$
(X, Y, Z) = ((N + h)\cos\varphi\cos\lambda, (N + h)\cos\varphi\sin\lambda, (N(1 - e^2) + h)\sin\varphi),
$$

with $e^2 = f(2 - f)$ and $N = a / \sqrt{1 - e^2 \sin^2\varphi}$; $f = 0$ is the sphere of radius $a$, and `ecef_to_lonlat` inverts it with Bowring's latitude and two fixed refinement steps. The caller passes $a$ and $f$: WGS84 is $a = 6378137$ m and $f = 1/298.257223563$, constants that belong to geophysix. `enu_basis` returns the east, north and up unit vectors at each point, `tangent_to_cartesian` and `cartesian_to_tangent` convert between $(u, v)$ components and 3-D tangent vectors, `local_tangent_plane` projects onto the plane at an origin (azimuthal equidistant or gnomonic), and `rotation_to_pole` takes a centre to the north pole. Every function that reads or writes $(\lambda, \varphi)$ takes `unit="degrees"` or `unit="radians"` with no default.

## Reference

::: functax.geometry
