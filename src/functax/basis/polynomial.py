"""Orthogonal polynomials: Legendre polynomials and associated Legendre functions.

Needs the ``functax[jax]`` extra.
"""

from __future__ import annotations

from functax._src.basis._legendre import (
    associated_legendre,
    associated_legendre_indices,
    legendre_polynomials,
)


__all__ = [
    "associated_legendre",
    "associated_legendre_indices",
    "legendre_polynomials",
]
