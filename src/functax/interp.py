"""Interpolation: stencils, locate and policies, schemes and plans with adjoints.

A scheme is configuration; a plan freezes it for one source and one target
geometry. Every plan reads a field as f̂(X*) = H 𝒮(y), with a fit 𝒮 (the
identity for local schemes) and a read-out H, and sends residuals back with
the hand-written adjoint 𝒮ᵀ Hᵀ. Needs the ``functax[jax]`` extra.
"""

from __future__ import annotations

from functax._src.interp._stencil import Readout, Stencil


__all__ = [
    "Readout",
    "Stencil",
]
