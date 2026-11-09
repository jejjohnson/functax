"""Guards on the public API surface.

These tests are deliberately about *contracts*, not behaviour: if a symbol is
removed from ``__all__`` or stops being importable from the top level, that is
a breaking change and should fail loudly here.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

import functax


SUBMODULES = [
    "_typing",
    "exceptions",
]

# Packages: the public area modules that are packages, and the private tree.
SUBPACKAGES = [
    "_src",
    "geometry",
]


def test_version_is_a_dotted_string() -> None:
    assert isinstance(functax.__version__, str)
    major, minor, patch = functax.__version__.split(".")[:3]
    assert all(part.isdigit() for part in (major, minor, patch))


def test_all_is_sorted_and_unique() -> None:
    assert functax.__all__ == sorted(functax.__all__)
    assert len(functax.__all__) == len(set(functax.__all__))


@pytest.mark.parametrize("name", functax.__all__)
def test_every_exported_name_is_importable(name: str) -> None:
    assert hasattr(functax, name), f"{name} is in __all__ but not defined"


@pytest.mark.parametrize("module", [*SUBMODULES, *SUBPACKAGES])
def test_submodule_imports_cleanly(module: str) -> None:
    assert importlib.import_module(f"functax.{module}") is not None


def test_no_unexpected_submodules() -> None:
    """A new module must be added to SUBMODULES (and given an API doc page)."""
    found = {
        info.name for info in pkgutil.iter_modules(functax.__path__) if not info.ispkg
    }
    assert found == set(SUBMODULES)


def test_no_unexpected_subpackages() -> None:
    """A new subpackage must be added to SUBPACKAGES (and given an API doc page)."""
    found = {info.name for info in pkgutil.iter_modules(functax.__path__) if info.ispkg}
    assert found == set(SUBPACKAGES)


@pytest.mark.parametrize("module", [*SUBMODULES, "geometry"])
def test_submodules_declare_all(module: str) -> None:
    imported = importlib.import_module(f"functax.{module}")
    assert hasattr(imported, "__all__"), f"functax.{module} is missing __all__"
    assert imported.__all__ == sorted(imported.__all__)


def test_package_is_typed() -> None:
    """PEP 561 marker must ship so downstream type checkers see annotations."""
    marker = Path(functax.__path__[0]) / "py.typed"
    assert marker.is_file(), f"missing PEP 561 marker at {marker}"


def test_geometry_reexports_lazily() -> None:
    import functax.geometry as geometry
    from functax._src.geometry import _types

    assert geometry.Axis is _types.Axis
    assert set(geometry.__all__) <= set(dir(geometry))
    with pytest.raises(AttributeError, match="no attribute 'Grid'"):
        _ = geometry.Grid
