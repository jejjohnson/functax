"""Affine raster maths on NumPy or JAX: pixels and back, pixel centres, windows.

This module is part of the NumPy-only base install: it imports JAX only when a
JAX array arrives. Pixel (row, col) coordinates put integers at pixel centres,
so (0, 0) is the centre of the top-left pixel (geoarray's area convention).
"""

from __future__ import annotations

from functax._src.geometry._affine import (
    from_pixel,
    pixel_centres,
    slice_transform,
    to_pixel,
    window,
)


__all__ = [
    "from_pixel",
    "pixel_centres",
    "slice_transform",
    "to_pixel",
    "window",
]
