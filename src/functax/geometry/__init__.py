"""CRS-free geometry: axes, grids, point sets, the sphere and affine raster maths.

Coordinates arrive already in one coordinate reference system; the CRS itself
stays a carrier concern. The types are Equinox modules and need the
``functax[jax]`` extra, but this package must import without it, because
``functax.geometry.affine`` is part of the NumPy-only base install. So the
affine kernels (`to_pixel`, `from_pixel`, `pixel_centres`, `window`) are
imported eagerly, and the types are re-exported lazily: the first access to
``functax.geometry.Axis`` imports them, and raises `MissingExtraError` when
JAX is missing.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from functax.geometry.affine import from_pixel, pixel_centres, to_pixel, window


if TYPE_CHECKING:
    from functax._src.geometry._metric import pairwise_distance
    from functax._src.geometry._sphere import (
        chord_distance,
        geodesic_distance,
        lonlat_to_unit,
        unit_to_lonlat,
    )
    from functax._src.geometry._types import (
        AffineGrid,
        Axis,
        CurvilinearGrid,
        Metric,
        Points,
        RectilinearGrid,
    )


__all__ = [
    "AffineGrid",
    "Axis",
    "CurvilinearGrid",
    "Metric",
    "Points",
    "RectilinearGrid",
    "chord_distance",
    "from_pixel",
    "geodesic_distance",
    "lonlat_to_unit",
    "pairwise_distance",
    "pixel_centres",
    "to_pixel",
    "unit_to_lonlat",
    "window",
]

# Every lazily re-exported name, and the private module that defines it.
_LAZY: dict[str, str] = {
    "AffineGrid": "functax._src.geometry._types",
    "Axis": "functax._src.geometry._types",
    "CurvilinearGrid": "functax._src.geometry._types",
    "Metric": "functax._src.geometry._types",
    "Points": "functax._src.geometry._types",
    "RectilinearGrid": "functax._src.geometry._types",
    "chord_distance": "functax._src.geometry._sphere",
    "geodesic_distance": "functax._src.geometry._sphere",
    "lonlat_to_unit": "functax._src.geometry._sphere",
    "pairwise_distance": "functax._src.geometry._metric",
    "unit_to_lonlat": "functax._src.geometry._sphere",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name])
        value = getattr(module, name)
        # Cache it, so later accesses skip __getattr__.
        globals()[name] = value
        return value
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
