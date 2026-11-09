# Geometry

CRS-free geometry: axes, grids, point sets, the sphere and affine raster maths. Coordinates arrive already in one coordinate reference system; the CRS stays a concern of the carrier.

## Affine raster maths

The transform $T = (a, b, c, d, e, f)$ maps the pixel corner at column $j$ and row $i$ to

$$
x = a j + b i + c \text{ and } y = d j + e i + f,
$$

in the order of `affine.Affine`, rasterio and geoarray (not GDAL's). Pixel coordinates in functax follow the area convention: integers are pixel centres, so $(0, 0)$ is the centre of the top-left pixel and its outer corner is $(-\frac{1}{2}, -\frac{1}{2})$. `to_pixel` inverts the $2 \times 2$ part $A$ of $T$, with $\det A = ae - bd$, and subtracts $\frac{1}{2}$; `from_pixel` is its inverse. Both reproduce geoarray's arithmetic exactly.

`to_pixel`, `from_pixel`, `pixel_centres` and `window` are the one part of functax in the NumPy-only base install: they take the array namespace of their input, return NumPy for NumPy and JAX for JAX, and import JAX only when a JAX array arrives. On JAX they trace under `jit`, `vmap` and `grad`. `window` follows geoarray's `Grid.window`: it rounds outward to whole pixels, snaps corners within `atol` pixels of an edge, and is not clipped to the grid.

## Reference

::: functax.geometry
