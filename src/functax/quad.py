"""Quadrature rules: the nodes and weights of a rule on a domain.

A rule is data. Procedures that adapt or march (adaptive quadrature, scans,
time stepping) belong to integrix. Needs the ``functax[jax]`` extra.
"""

from __future__ import annotations

from functax._src.quad._rules import (
    Rule,
    clenshaw_curtis,
    gauss_legendre,
    simpson,
    trapezoid,
)


__all__ = [
    "Rule",
    "clenshaw_curtis",
    "gauss_legendre",
    "simpson",
    "trapezoid",
]
