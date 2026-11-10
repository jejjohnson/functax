"""Projection: fits from data to coefficients, c = 𝒮(y).

Least squares onto any basis, Galerkin projection with quadrature weights,
the matrix-free variational fit with a pluggable roughness operator, and the
Laplace and biharmonic gap fills. Needs the ``functax[jax]`` extra.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

from functax._src.project._fill import fill_gaps
from functax._src.project._linear import galerkin, least_squares
from functax._src.project._variational import Variational


__all__ = [
    "Variational",
    "fill_gaps",
    "galerkin",
    "least_squares",
]
