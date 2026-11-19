"""Run every ``python`` block of the Claude Code plugin.

An entry of the shelf (``plugins/functax/skills/<skill>/shelf/<entry>.md``)
and each listed ``SKILL.md`` carry one worked example with its own asserts.
An example that no longer runs teaches agents in downstream projects an API
that is gone, so each one runs here, in a fresh namespace.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "plugins" / "functax"
if not (PLUGIN / "tree" / "shelf.toml").is_file():
    # The sdist ships tests/ but not plugins/.
    pytest.skip("the plugin is not present", allow_module_level=True)
pytest.importorskip("jax")

_spec = importlib.util.spec_from_file_location(
    "skill_shelf", ROOT / "scripts" / "skill_shelf.py"
)
assert _spec is not None and _spec.loader is not None
ss = importlib.util.module_from_spec(_spec)
sys.modules["skill_shelf"] = ss  # dataclasses look their module up
_spec.loader.exec_module(ss)

_EXAMPLES = ss.examples()


def test_every_entry_and_every_listed_skill_has_an_example():
    with_example = {path for path, _ in _EXAMPLES}
    pages = set(PLUGIN.glob("skills/*/SKILL.md")) | set(
        PLUGIN.glob("skills/*/shelf/*.md")
    )
    assert pages <= with_example, sorted(p.name for p in pages - with_example)


@pytest.mark.parametrize(
    "block",
    [
        pytest.param(block, id=f"{path.relative_to(PLUGIN)}:{i}")
        for i, (path, block) in enumerate(_EXAMPLES)
    ],
)
def test_example_runs(block: str):
    exec(block, {"__name__": "__plugin_example__"})
