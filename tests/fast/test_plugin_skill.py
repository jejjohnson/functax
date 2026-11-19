"""Keep the Claude Code plugin's guidance current.

``plugins/functax/`` is what agents in downstream projects read before
writing interpolation, quadrature or basis code, so a stale name there teaches
them the wrong API. The plugin is a shelf (``tree/shelf.toml``): three listed
skills, each with a ``shelf/`` folder of entries, and one contract card per
unit.

These tests run the check of ``scripts/skill_shelf.py`` (the generated parts
of each ``SKILL.md`` are current, every public name belongs to exactly one
entry whose file names it, the line budgets hold), and check that every
``functax.module.name`` the plugin writes is a current public name.
``tests/slow/test_plugin_examples.py`` runs the plugin's ``python`` blocks.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "plugins" / "functax"
if not (PLUGIN / "tree" / "shelf.toml").is_file():
    # The sdist ships tests/ but not plugins/.
    pytest.skip("the plugin is not present", allow_module_level=True)

_spec = importlib.util.spec_from_file_location(
    "skill_shelf", ROOT / "scripts" / "skill_shelf.py"
)
assert _spec is not None and _spec.loader is not None
ss = importlib.util.module_from_spec(_spec)
sys.modules["skill_shelf"] = ss  # dataclasses look their module up
_spec.loader.exec_module(ss)

_SHELF = ss.load()
_PAGES = sorted(PLUGIN.rglob("*.md"))
# functax and the affine kernels import without the jax extra
_BASE = {"functax", "functax.exceptions", "functax.geometry.affine"}


def test_shelf_is_current_and_covers_the_public_api():
    found = ss.violations(_SHELF, ss.public_names(_SHELF))
    assert found == [], "run uv run poe skill-shelf"


@pytest.mark.parametrize("module", sorted(_SHELF.modules))
def test_public_names_read_from_source_match_the_module(module: str):
    if module not in _BASE:
        pytest.importorskip("jax")
    listed = {n for n in importlib.import_module(module).__all__}
    submodules = {n for n in listed if f"{module}.{n}" in _SHELF.modules}
    assert ss.module_names(_SHELF)[module] == listed - submodules


@pytest.mark.parametrize("path", _PAGES, ids=lambda p: str(p.relative_to(PLUGIN)))
def test_named_api_is_current(path: Path):
    public = ss.module_names(_SHELF)
    stale = sorted(
        f"{module}.{name}"
        for module, name in set(ss.cited(_SHELF, path.read_text(encoding="utf-8")))
        if name not in public[module]
    )
    assert not stale, f"{path.name} names functax objects that do not exist: {stale}"
