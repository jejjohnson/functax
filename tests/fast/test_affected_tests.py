"""Tests for ``scripts/affected_tests.py``, the pull-request test selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from _paths import REPO_ROOT


def _load() -> ModuleType:
    path = REPO_ROOT / "scripts" / "affected_tests.py"
    spec = importlib.util.spec_from_file_location("affected_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


affected_tests = _load()


@pytest.fixture(scope="module")
def graph():  # type: ignore[no-untyped-def]
    return affected_tests.graph()


class TestSelect:
    @pytest.mark.parametrize(
        "path", ["conftest.py", "uv.lock", "tests/_harness.py", "pyproject.toml"]
    )
    def test_a_shared_file_selects_everything(self, path: str) -> None:
        assert affected_tests.select([path]) is None

    def test_a_quad_change_skips_unrelated_areas(self) -> None:
        selected = affected_tests.select(["src/functax/_src/quad/_cell_area.py"])
        assert selected is not None
        assert "tests/fast/quad/test_cell_area.py" in selected
        assert "tests/slow/quad/test_cell_area.py" in selected
        assert "src/functax/_src/quad/_cell_area.py" in selected
        # the spline tests never import cell_area, directly or not
        assert "tests/slow/interp/test_splines.py" not in selected
        assert "tests/slow/interp/test_series.py" not in selected

    def test_a_docs_change_selects_the_docs_tests_only(self) -> None:
        selected = affected_tests.select(["docs/guide/quadrature.md"])
        assert selected == sorted(affected_tests.DOCS_TESTS)

    def test_a_changed_test_file_selects_itself(self) -> None:
        path = "tests/slow/interp/test_splines.py"
        assert affected_tests.select([path]) == [path]

    def test_an_unrelated_file_selects_nothing(self) -> None:
        assert affected_tests.select(["CHANGELOG.md"]) == []


class TestGraph:
    def test_a_public_name_resolves_to_its_defining_module(self, graph) -> None:  # type: ignore[no-untyped-def]
        assert graph.resolve("functax.quad", "Rule") == {"functax._src.quad._rules"}

    def test_a_lazy_geometry_name_resolves_through_type_checking(self, graph) -> None:  # type: ignore[no-untyped-def]
        resolved = graph.resolve("functax.geometry", "Axis")
        assert resolved == {"functax._src.geometry._types"}

    def test_every_private_module_reaches_its_own_tests(self, graph) -> None:  # type: ignore[no-untyped-def]
        # a module whose tests the selection misses would never run on its PRs
        tests = sorted(Path(REPO_ROOT, "tests").rglob("test_*.py"))
        covered = set().union(*map(affected_tests.test_dependencies, tests))
        private = {
            name
            for name in graph.modules
            if name.startswith("functax._src.") and name.count(".") == 3
        }
        reached = {name for name in private if graph.affected({name}) & covered}
        assert reached == private
