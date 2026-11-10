"""Tests for the ``functax[jax]`` and ``functax[scipy]`` extra checks and the
NumPy-only import rule. The fast tier: behaviour, validation and small closed-form
checks, each well under a second.
"""

from __future__ import annotations

import sys

import pytest

from functax import MissingExtraError
from functax._src._backend import JAX_EXTRA, import_scipy, require_jax


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


def test_import_scipy_returns_the_submodule() -> None:
    assert import_scipy("spatial", needed_by="InverseDistance.plan").__name__ == (
        "scipy.spatial"
    )


def test_import_scipy_names_the_extra_when_scipy_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A ``None`` entry in sys.modules makes ``import scipy.spatial`` raise.
    monkeypatch.setitem(sys.modules, "scipy.spatial", None)
    with pytest.raises(MissingExtraError, match=r"functax\[scipy\]") as excinfo:
        import_scipy("spatial", needed_by="InverseDistance.plan")
    assert "InverseDistance.plan" in str(excinfo.value)


def test_array_checks_never_import_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    from functax._src._backend import is_jax_array, is_tracer, namespace

    # With jax absent from sys.modules no JAX array can exist, so the checks
    # answer False without importing it.
    monkeypatch.delitem(sys.modules, "jax")
    assert not is_jax_array(object())
    assert not is_tracer(object())
    assert namespace([1.0, 2.0]).__name__ == "numpy"
    assert "jax" not in sys.modules
