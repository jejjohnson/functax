"""Bases φⱼ(x; θ) for the expansion f̂(x) = ∑ⱼ cⱼ φⱼ(x; θ), one submodule per family.

The submodules are the public homes; this package re-exports none of their
names: ``features`` (Fourier features at given frequencies, seasonal and
windowed features), ``fourier`` (box Laplacian eigenfunctions), ``local``
(radial bumps, Gabor frames, wavelets), ``vector``
(divergence-free and curl-free atoms), ``polynomial`` (Legendre), ``sphere``
(scalar and vector harmonics, needlets, shell bases), ``slepian`` (caps, polar
gaps, regions) and ``eigen`` (the cotangent Laplacian of a mesh). θ may be
learned: centres, widths and frequencies are ordinary arrays. Needs the
``functax[jax]`` extra.
"""

from __future__ import annotations

from functax.basis import (
    eigen,
    features,
    fourier,
    local,
    polynomial,
    slepian,
    sphere,
    vector,
)


__all__ = [
    "eigen",
    "features",
    "fourier",
    "local",
    "polynomial",
    "slepian",
    "sphere",
    "vector",
]
