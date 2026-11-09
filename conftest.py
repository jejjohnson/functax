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


# The tier of a test is its directory under tests/: fast/ (unmarked), slow/
# and integration/. The marker follows from the path, so a test changes tier
# by moving, and `-m "not slow and not integration"` selects the same set as
# `tests/fast src/functax`.
_TIERS = ("slow", "integration")


def pytest_collection_modifyitems(config, items):  # type: ignore[no-untyped-def]
    """Mark every test in tests/slow or tests/integration with its tier."""
    import pytest

    for item in items:
        parts = item.path.parts
        if "tests" not in parts:
            continue
        tier = parts[parts.index("tests") + 1]
        declared = {name for name in _TIERS if item.get_closest_marker(name)}
        if tier in _TIERS:
            item.add_marker(getattr(pytest.mark, tier))
        if declared - {tier}:
            msg = (
                f"{item.nodeid} is marked {sorted(declared)} but lives in "
                f"tests/{tier}; move it to tests/{sorted(declared)[0]}/ instead"
            )
            raise pytest.UsageError(msg)
