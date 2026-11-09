"""Feature maps of coordinates: Fourier features at given frequencies, and windows.

The frequencies are arguments; drawing them from a kernel's spectral density
stays with the library that owns the kernel. Needs the ``functax[jax]`` extra.
"""

from __future__ import annotations

from functax._src.basis._features import (
    fourier_features,
    gaussian_window_features,
    rff_cosine_forward,
    rff_forward,
    seasonal_features,
    seasonal_frequencies,
)


__all__ = [
    "fourier_features",
    "gaussian_window_features",
    "rff_cosine_forward",
    "rff_forward",
    "seasonal_features",
    "seasonal_frequencies",
]
