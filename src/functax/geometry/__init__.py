"""CRS-free geometry: axes, grids, point sets, the sphere and affine raster maths.

Coordinates arrive already in one coordinate reference system; the CRS itself
stays a carrier concern. This package is empty until its first feature lands.
Its ``__init__`` must stay importable without JAX, because
``functax.geometry.affine`` is part of the NumPy-only base install.
"""

from __future__ import annotations


__all__: list[str] = []
