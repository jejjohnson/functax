"""Interpolation: stencils, locate and policies, schemes and plans with adjoints.

A scheme is configuration; a plan freezes it for one source and one target
geometry. Every plan reads a field as f̂(X*) = H 𝒮(y), with a fit 𝒮 (the
identity for local schemes) and a read-out H, and sends residuals back with
the hand-written adjoint 𝒮ᵀ Hᵀ. Needs the ``functax[jax]`` extra.
"""

from __future__ import annotations

from functax._src.interp._bases import Basis, LinearBasis, NearestBasis
from functax._src.interp._fit import Fit, Identity
from functax._src.interp._locate import (
    check_outside,
    index_positions,
    locate,
    remap_taps,
)
from functax._src.interp._plan import Plan
from functax._src.interp._policies import Policies
from functax._src.interp._schemes import Linear, LocalScheme, Nearest, Scheme, Tensor
from functax._src.interp._stencil import Readout, Stencil
from functax._src.interp._tensor import tensor_stencil


__all__ = [
    "Basis",
    "Fit",
    "Identity",
    "Linear",
    "LinearBasis",
    "LocalScheme",
    "Nearest",
    "NearestBasis",
    "Plan",
    "Policies",
    "Readout",
    "Scheme",
    "Stencil",
    "Tensor",
    "check_outside",
    "index_positions",
    "locate",
    "remap_taps",
    "tensor_stencil",
]
