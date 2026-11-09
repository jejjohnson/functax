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

## Reference

::: functax.geometry
