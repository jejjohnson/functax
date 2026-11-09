"""Private implementation of `functax.geometry`.

This package's ``__init__`` imports nothing, so the NumPy-only modules under
it (``_transform``, and the affine kernels of #16) import without JAX.
"""
