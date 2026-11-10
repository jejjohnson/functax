"""Quadrature rules: the nodes and weights of a rule on a domain.

A rule is data. Procedures that adapt or march (adaptive quadrature, scans,
time stepping) belong to integrix. Needs the ``functax[jax]`` extra.
"""

from __future__ import annotations

from functax._src.quad._cell_area import cell_area
from functax._src.quad._curve import CurveRule, polyline
from functax._src.quad._rules import (
    Rule,
    clenshaw_curtis,
    gauss_legendre,
    simpson,
    trapezoid,
)
from functax._src.quad._sphere import (
    SphereRule,
    sphere_fibonacci,
    sphere_gauss_legendre,
    sphere_healpix,
    sphere_latlon,
)


__all__ = [
    "CurveRule",
    "Rule",
    "SphereRule",
    "cell_area",
    "clenshaw_curtis",
    "gauss_legendre",
    "polyline",
    "simpson",
    "sphere_fibonacci",
    "sphere_gauss_legendre",
    "sphere_healpix",
    "sphere_latlon",
    "trapezoid",
]
