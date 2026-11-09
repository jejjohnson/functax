"""Interpolation: stencils, locate and policies, schemes and plans with adjoints.

A scheme is configuration; a plan freezes it for one source and one target
geometry. Every plan reads a field as f̂(X*) = H 𝒮(y), with a fit 𝒮 (the
identity for local schemes) and a read-out H, and sends residuals back with
the hand-written adjoint 𝒮ᵀ Hᵀ. Needs the ``functax[jax]`` extra.
"""

from __future__ import annotations

from functax._src.interp._bases import (
    Basis,
    BSplineBasis,
    CubicBasis,
    LinearBasis,
    NearestBasis,
    bspline,
)
from functax._src.interp._derivative import LocalGeometry
from functax._src.interp._fit import Fit, Identity
from functax._src.interp._locate import (
    check_outside,
    index_positions,
    locate,
    remap_taps,
)
from functax._src.interp._plan import DerivativePlan, Plan
from functax._src.interp._policies import Policies
from functax._src.interp._prefilter import BandedPrefilter, spline_poles
from functax._src.interp._schemes import (
    BSpline,
    Cubic,
    Linear,
    LocalScheme,
    Nearest,
    Scheme,
    Tensor,
)
from functax._src.interp._stencil import Readout, Stencil
from functax._src.interp._tensor import tensor_stencil


__all__ = [
    "BSpline",
    "BSplineBasis",
    "BandedPrefilter",
    "Basis",
    "Cubic",
    "CubicBasis",
    "DerivativePlan",
    "Fit",
    "Identity",
    "Linear",
    "LinearBasis",
    "LocalGeometry",
    "LocalScheme",
    "Nearest",
    "NearestBasis",
    "Plan",
    "Policies",
    "Readout",
    "Scheme",
    "Stencil",
    "Tensor",
    "bspline",
    "check_outside",
    "index_positions",
    "locate",
    "remap_taps",
    "spline_poles",
    "tensor_stencil",
]
