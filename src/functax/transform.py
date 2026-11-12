"""Transforms between bases: spherical-harmonic, wavelet, NUFFT and Chebyshev.

Needs the ``functax[jax]`` extra.
"""

from __future__ import annotations

from functax._src.transform._chebyshev import (
    ChebyshevTransform1D,
    ChebyshevTransform2D,
)
from functax._src.transform._nufft import nonuniform_irfft, nonuniform_rfft
from functax._src.transform._sht import SphericalHarmonicTransform
from functax._src.transform._spherical_wavelet import SphericalWaveletTransform


__all__ = [
    "ChebyshevTransform1D",
    "ChebyshevTransform2D",
    "SphericalHarmonicTransform",
    "SphericalWaveletTransform",
    "nonuniform_irfft",
    "nonuniform_rfft",
]
