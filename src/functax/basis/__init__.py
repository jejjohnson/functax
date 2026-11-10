"""Bases φⱼ(x; θ) for the expansion f̂(x) = ∑ⱼ cⱼ φⱼ(x; θ), one submodule per family.

The submodules are the public homes; this package re-exports none of their
names: ``features`` (Fourier features at given frequencies, seasonal and
windowed features), ``polynomial`` (Legendre), ``sphere`` (scalar and vector
harmonics, needlets, shell bases) and ``eigen`` (the cotangent Laplacian of a
mesh). θ may be learned: centres, widths and frequencies are ordinary arrays.
Needs the ``functax[jax]`` extra.
"""

from __future__ import annotations

from functax.basis import eigen, features, polynomial, sphere


__all__ = ["eigen", "features", "polynomial", "sphere"]
