"""Eigenbases of discrete operators: the cotangent Laplacian of a triangle mesh.

Needs the ``functax[jax]`` extra.
"""

from __future__ import annotations

from functax._src.basis._mesh_laplacian import cotangent_laplacian_eigpairs


__all__ = ["cotangent_laplacian_eigpairs"]
