"""Shared pytest configuration.

The oracle comparisons run in float64, so JAX's 64-bit mode is enabled for the
whole suite, and jaxtyping checks every annotated array at runtime through its
import hook (`python-style`). The guard keeps the suite importable in a
NumPy-only install, where only the JAX-free modules can be tested.
"""

from __future__ import annotations


try:
    import jax
    from jaxtyping import install_import_hook
except ImportError:  # pragma: no cover - exercised only in a NumPy-only install
    pass
else:
    jax.config.update("jax_enable_x64", True)
    # Installed before pytest imports functax for doctest collection, so every
    # functax module imported afterwards is checked with beartype.
    install_import_hook("functax", "beartype.beartype")
