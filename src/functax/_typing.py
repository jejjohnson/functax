"""Shared type aliases.

The aliases are written with [PEP 695](https://peps.python.org/pep-0695/)
``type`` statements (Python 3.12+). Array annotations use jaxtyping and live
with the modules that need the ``functax[jax]`` extra, so this module stays
importable from the NumPy-only base install.
"""

from __future__ import annotations


__all__ = [
    "Shape",
]


type Shape = tuple[int, ...]
"""The shape of a grid or an array, one positive length per axis."""
