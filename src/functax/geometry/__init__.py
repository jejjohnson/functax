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
    from functax._src.geometry._frames import (
        cartesian_to_tangent,
        ecef_to_lonlat,
        enu_basis,
        local_tangent_plane,
        lonlat_to_ecef,
        rotation_to_pole,
        tangent_to_cartesian,
    )
    from functax._src.geometry._mesh import (
        icosphere,
        mesh_vertex_areas,
    )
    from functax._src.geometry._metric import pairwise_distance
    from functax._src.geometry._sphere import (
        chord_distance,
        geodesic_distance,
        lonlat_to_unit,
        unit_to_lonlat,
    )
    from functax._src.geometry._sphere_grids import (
        fibonacci_sphere,
        gauss_legendre_grid,
        healpix_grid,
        latlon_grid,
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
    "cartesian_to_tangent",
    "chord_distance",
    "ecef_to_lonlat",
    "enu_basis",
    "fibonacci_sphere",
    "from_pixel",
    "gauss_legendre_grid",
    "geodesic_distance",
    "healpix_grid",
    "icosphere",
    "latlon_grid",
    "local_tangent_plane",
    "lonlat_to_ecef",
    "lonlat_to_unit",
    "mesh_vertex_areas",
    "pairwise_distance",
    "pixel_centres",
    "rotation_to_pole",
    "tangent_to_cartesian",
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
    "cartesian_to_tangent": "functax._src.geometry._frames",
    "chord_distance": "functax._src.geometry._sphere",
    "ecef_to_lonlat": "functax._src.geometry._frames",
    "enu_basis": "functax._src.geometry._frames",
    "fibonacci_sphere": "functax._src.geometry._sphere_grids",
    "gauss_legendre_grid": "functax._src.geometry._sphere_grids",
    "geodesic_distance": "functax._src.geometry._sphere",
    "healpix_grid": "functax._src.geometry._sphere_grids",
    "icosphere": "functax._src.geometry._mesh",
    "latlon_grid": "functax._src.geometry._sphere_grids",
    "local_tangent_plane": "functax._src.geometry._frames",
    "lonlat_to_ecef": "functax._src.geometry._frames",
    "lonlat_to_unit": "functax._src.geometry._sphere",
    "mesh_vertex_areas": "functax._src.geometry._mesh",
    "pairwise_distance": "functax._src.geometry._metric",
    "rotation_to_pole": "functax._src.geometry._frames",
    "tangent_to_cartesian": "functax._src.geometry._frames",
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
