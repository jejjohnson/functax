"""Slepian bases: band-limited functions concentrated in a cap, a gap or a region.

Needs the ``functax[jax]`` extra.
"""

from __future__ import annotations

from functax._src.basis._slepian_cap import (
    SlepianCapBasis,
    shannon_number,
    slepian_cap_basis,
    slepian_cap_eigh_per_m,
    slepian_concentration_matrix,
)
from functax._src.basis._slepian_region import (
    SlepianBasis,
    slepian_polar_gap_basis,
    slepian_region_basis,
    spherical_polygon_mask,
)


__all__ = [
    "SlepianBasis",
    "SlepianCapBasis",
    "shannon_number",
    "slepian_cap_basis",
    "slepian_cap_eigh_per_m",
    "slepian_concentration_matrix",
    "slepian_polar_gap_basis",
    "slepian_region_basis",
    "spherical_polygon_mask",
]
