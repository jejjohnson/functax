"""Laplacian eigenfunctions of a box under Dirichlet, Neumann or periodic boundaries.

Needs the ``functax[jax]`` extra.
"""

from __future__ import annotations

from functax._src.basis._fourier import (
    fourier_basis,
    fourier_basis_1d,
    fourier_eigenvalues,
    fourier_eigenvalues_1d,
)


__all__ = [
    "fourier_basis",
    "fourier_basis_1d",
    "fourier_eigenvalues",
    "fourier_eigenvalues_1d",
]
