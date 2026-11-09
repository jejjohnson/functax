"""Bases φⱼ(x; θ) for the expansion f̂(x) = ∑ⱼ cⱼ φⱼ(x; θ), one submodule per family.

The submodules are the public homes; this package re-exports none of their
names. ``features`` holds the feature maps of coordinates (Fourier features at
given frequencies, seasonal and windowed features). θ may be learned: centres,
widths and frequencies are ordinary arrays. Needs the ``functax[jax]`` extra.
"""

from __future__ import annotations

from functax.basis import features


__all__ = ["features"]
