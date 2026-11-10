"""Interpolation: stencils, locate and policies, schemes and plans with adjoints.

A scheme is configuration; a plan freezes it for one source and one target
geometry. Every plan reads a field as f̂(X*) = H 𝒮(y), with a fit 𝒮 (the
identity for local schemes) and a read-out H, and sends residuals back with
the hand-written adjoint 𝒮ᵀ Hᵀ. Plans compose (``outer @ inner``), cut to
windows of a grid target (``plan[window]``) and export their stencil for
gaussx (``plan.to_gaussx()``). Schemes and policies round-trip
through strict JSON, plans through bytes, and `plan_key` names a plan by the
content hash of its inputs. Needs the ``functax[jax]`` extra.
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
from functax._src.interp._cache import plan_key
from functax._src.interp._compose import compose
from functax._src.interp._covariance import BrownianBridge, PosteriorCovariance
from functax._src.interp._curvilinear import locate_curvilinear
from functax._src.interp._derivative import LocalGeometry
from functax._src.interp._fit import Fit, Identity
from functax._src.interp._gaussian import GaussianPlan
from functax._src.interp._hermite import HermiteFit
from functax._src.interp._interpolant import Interpolant
from functax._src.interp._locate import (
    check_outside,
    index_positions,
    locate,
    remap_taps,
)
from functax._src.interp._neighbourhood import InverseDistance
from functax._src.interp._plan import ComposedPlan, DerivativePlan, Plan, WindowedPlan
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
from functax._src.interp._series import (
    Akima,
    CubicSpline,
    NonlinearPlan,
    Pchip,
    SeriesInterpolant,
    SeriesScheme,
    Steffen,
)
from functax._src.interp._stencil import Readout, Stencil
from functax._src.interp._tensor import tensor_stencil


__all__ = [
    "Akima",
    "BSpline",
    "BSplineBasis",
    "BandedPrefilter",
    "Basis",
    "BrownianBridge",
    "ComposedPlan",
    "Cubic",
    "CubicBasis",
    "CubicSpline",
    "DerivativePlan",
    "Fit",
    "GaussianPlan",
    "HermiteFit",
    "Identity",
    "Interpolant",
    "InverseDistance",
    "Linear",
    "LinearBasis",
    "LocalGeometry",
    "LocalScheme",
    "Nearest",
    "NearestBasis",
    "NonlinearPlan",
    "Pchip",
    "Plan",
    "Policies",
    "PosteriorCovariance",
    "Readout",
    "Scheme",
    "SeriesInterpolant",
    "SeriesScheme",
    "Steffen",
    "Stencil",
    "Tensor",
    "WindowedPlan",
    "bspline",
    "check_outside",
    "compose",
    "index_positions",
    "locate",
    "locate_curvilinear",
    "plan_key",
    "remap_taps",
    "spline_poles",
    "tensor_stencil",
]
