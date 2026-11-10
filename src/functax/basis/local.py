"""Local and multiscale bases: radial bumps, Wendland kernels, Gabor frames, wavelets.

Needs the ``functax[jax]`` extra.
"""

from __future__ import annotations

from functax._src.basis._gabor import gabor_frame, gabor_frame_grid
from functax._src.basis._rbf import (
    rbf_basis,
    spherical_rbf_basis,
    wendland_c2,
    wendland_c4,
)
from functax._src.basis._wavelet import wavelet_basis_1d, wavelet_basis_2d


__all__ = [
    "gabor_frame",
    "gabor_frame_grid",
    "rbf_basis",
    "spherical_rbf_basis",
    "wavelet_basis_1d",
    "wavelet_basis_2d",
    "wendland_c2",
    "wendland_c4",
]
