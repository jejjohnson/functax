"""Tests for the ``functax[jax]`` extra check and the NumPy-only import rule."""

from __future__ import annotations

import subprocess
import sys

import pytest

from functax import MissingExtraError
from functax._src._backend import JAX_EXTRA, require_jax


def _run(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter, so no module is imported yet."""
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )


def test_require_jax_is_silent_when_the_extra_is_installed() -> None:
    require_jax("functax.interp")


@pytest.mark.parametrize("name", JAX_EXTRA)
def test_require_jax_names_the_extra_when_a_distribution_is_missing(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A ``None`` entry in sys.modules makes ``import name`` raise ImportError.
    monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(MissingExtraError, match=r"functax\[jax\]") as excinfo:
        require_jax("functax.interp")
    assert "functax.interp" in str(excinfo.value)
    assert repr(name) in str(excinfo.value)


@pytest.mark.parametrize("module", ["functax", "functax.geometry"])
def test_base_modules_never_import_jax(module: str) -> None:
    result = _run(
        f"import sys, {module}; "
        "loaded = sorted(n for n in ('jax', 'equinox', 'lineax', 'einx', "
        "'jaxtyping') if n in sys.modules); "
        "assert not loaded, loaded"
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module", ["functax", "functax.geometry"])
def test_base_modules_import_with_jax_blocked(module: str) -> None:
    result = _run(f"import sys; sys.modules['jax'] = None; import {module}")
    assert result.returncode == 0, result.stderr
