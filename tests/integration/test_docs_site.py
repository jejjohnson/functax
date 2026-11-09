"""The built documentation site: the inventory mkdocstrings writes matches what the
docs pipeline parses. Skipped until uv run poe docs-api has built site/.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType

import pytest
from _paths import REPO_ROOT


def _load() -> ModuleType:
    path = REPO_ROOT / "scripts" / "build_docs.py"
    spec = importlib.util.spec_from_file_location("build_docs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_docs = _load()


class TestBuiltSite:
    def test_matches_the_real_inventory_format(self) -> None:
        """Guards against drift in what mkdocstrings actually emits."""
        real = REPO_ROOT / "site" / "objects.inv"
        if not real.is_file():
            pytest.skip("run `uv run poe docs-api` first")
        entries = build_docs.parse_inventory(real.read_bytes())
        assert "functax.FunctaxError" in entries
        assert all("$" not in uri for uri in entries.values())
