"""Vector bases on a box: divergence-free and curl-free atoms.

Needs the ``functax[jax]`` extra.
"""

from __future__ import annotations

from functax._src.basis._divfree import curlfree_basis, divfree_basis


__all__ = [
    "curlfree_basis",
    "divfree_basis",
]
