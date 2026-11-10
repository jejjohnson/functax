"""Eigenbases of data and discrete operators: EOFs, graph and mesh Laplacians.

Needs the ``functax[jax]`` extra.
"""

from __future__ import annotations

from functax._src.basis._eof import eof_basis
from functax._src.basis._graph_laplacian import graph_laplacian_eigpairs
from functax._src.basis._mesh_laplacian import cotangent_laplacian_eigpairs


__all__ = [
    "cotangent_laplacian_eigpairs",
    "eof_basis",
    "graph_laplacian_eigpairs",
]
