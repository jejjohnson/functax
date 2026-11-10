"""Bases on the sphere: scalar and vector harmonics, needlets and shell bases.

Needs the ``functax[jax]`` extra.
"""

from __future__ import annotations

from functax._src.basis._needlets import needlet_basis, needlet_window
from functax._src.basis._shell import shell_basis
from functax._src.basis._spherical_harmonics import (
    harmonic_degrees,
    real_spherical_harmonics,
)
from functax._src.basis._vector_harmonics import vector_spherical_harmonics


__all__ = [
    "harmonic_degrees",
    "needlet_basis",
    "needlet_window",
    "real_spherical_harmonics",
    "shell_basis",
    "vector_spherical_harmonics",
]
