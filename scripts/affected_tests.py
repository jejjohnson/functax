"""Select the tests a pull request needs, from the files it changes.

The selection follows functax's import graph module by module. A changed
module affects every module that imports it, directly or through others, and
a test file runs when it imports an affected module; so do the doctests of
the affected modules. A name imported from a public module is followed to the
module that defines it, so ``from functax.quad import Rule`` depends on
``functax._src.quad._rules`` only. A change to a shared file (the harness,
``conftest.py``, ``pyproject.toml``, the lock, this workflow) selects the whole
suite. Prints ``all``, ``none`` or the pytest paths, space-separated.
"""

from __future__ import annotations

import ast
import functools
import subprocess
import sys
from pathlib import Path

import cyclopts
from loguru import logger


ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = ROOT / "src"
TESTS = ROOT / "tests"

# Files every test depends on: a change here runs the whole suite.
SHARED = frozenset(
    {
        "conftest.py",
        "pyproject.toml",
        "uv.lock",
        ".github/workflows/ci.yml",
        "scripts/affected_tests.py",
        "tests/_harness.py",
        "tests/_paths.py",
        "tests/_geoarray_reference.py",
    }
)

# Pages and the docs pipeline are checked by these, not by a module's tests.
DOCS_TESTS = ("tests/fast/test_build_docs.py", "tests/integration/test_docs_site.py")

# Tests that check the whole package (public API, capability index, installs):
# they run whenever any module changes.
PACKAGE_TESTS = (
    "tests/fast/test_public_api.py",
    "tests/fast/test_capabilities.py",
    "tests/fast/test_exceptions.py",
    "tests/integration/test_install.py",
)

app = cyclopts.App(help=__doc__)


def module_name(path: Path) -> str:
    """``src/functax/_src/quad/_rules.py`` to ``functax._src.quad._rules``."""
    relative = path.relative_to(SOURCE_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def source_modules() -> dict[str, Path]:
    """Every module of the package, by dotted name."""
    paths = sorted((SOURCE_ROOT / "functax").rglob("*.py"))
    return {module_name(path): path for path in paths}


def _absolute(module: str | None, level: int, current: str, is_package: bool) -> str:
    """The absolute module of a (possibly relative) ``from`` import."""
    if level == 0:
        return module or ""
    package = current.split(".")
    if not is_package:
        package = package[:-1]
    if level > 1:
        package = package[: -(level - 1)]
    return ".".join([*package, module] if module else package)


def _imports(path: Path, current: str) -> list[tuple[str, str | None]]:
    """(module, name) pairs a file imports; name None for ``import module``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    is_package = path.name == "__init__.py"
    pairs: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            pairs += [(alias.name, None) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = _absolute(node.module, node.level, current, is_package)
            pairs += [(base, alias.name) for alias in node.names]
    return [(module, name) for module, name in pairs if module.startswith("functax")]


class Graph:
    """functax's modules and the modules each one depends on."""

    def __init__(self) -> None:
        self.modules = source_modules()
        self.raw = {name: _imports(path, name) for name, path in self.modules.items()}

    def resolve(self, module: str, name: str | None, depth: int = 0) -> set[str]:
        """The modules that define what ``from module import name`` binds."""
        if module not in self.modules:
            return set()
        if name is None or depth > 10:
            return {module}
        submodule = f"{module}.{name}"
        if submodule in self.modules:
            return {submodule}
        # a re-export, eager or under TYPE_CHECKING for a lazy __getattr__
        for source, imported in self.raw[module]:
            if imported == name:
                return self.resolve(source, imported, depth + 1)
        # defined in the module itself
        return {module}

    def dependencies(self, pairs: list[tuple[str, str | None]]) -> set[str]:
        """The modules a list of imports depends on directly."""
        found: set[str] = set()
        for module, name in pairs:
            found |= self.resolve(module, name)
        return found

    def affected(self, changed: set[str]) -> set[str]:
        """The changed modules and every module that imports one of them."""
        direct = {name: self.dependencies(pairs) for name, pairs in self.raw.items()}
        affected = set(changed)
        grew = True
        while grew:
            grew = False
            for name, uses in direct.items():
                if name not in affected and uses & affected:
                    affected.add(name)
                    grew = True
        return affected


def changed_files(base: str) -> list[str]:
    """The files that differ between ``base`` and HEAD (merge-base diff)."""
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


@functools.cache
def graph() -> Graph:
    """The import graph, built once per process."""
    return Graph()


@functools.cache
def test_dependencies(test: Path) -> frozenset[str]:
    """The functax modules a test file imports, followed to their definitions."""
    return frozenset(graph().dependencies(_imports(test, "tests")))


def select(changed: list[str]) -> list[str] | None:
    """The pytest paths for a set of changed files, or None for everything."""
    shared = [path for path in changed if path in SHARED]
    if shared:
        logger.info("{} is shared: the whole suite runs", shared[0])
        return None
    modules = graph()
    paths = {
        str(path.relative_to(ROOT)): name for name, path in modules.modules.items()
    }
    changed_modules: set[str] = set()
    selected: set[str] = set()
    for path in changed:
        if path in paths:
            changed_modules.add(paths[path])
        elif path.startswith("tests/") and path.endswith(".py"):
            if (ROOT / path).exists():
                selected.add(path)
        elif path.startswith("docs/") or path in ("mkdocs.yml", "README.md"):
            selected.update(DOCS_TESTS)
    if changed_modules:
        affected = modules.affected(changed_modules)
        # the doctests of every affected module
        selected |= {str(modules.modules[name].relative_to(ROOT)) for name in affected}
        for test in sorted(TESTS.rglob("test_*.py")):
            if test_dependencies(test) & affected:
                selected.add(str(test.relative_to(ROOT)))
        selected.update(PACKAGE_TESTS)
    return sorted(selected)


@app.default
def main(*, base: str = "origin/main") -> None:
    """Print the tests the changes since ``base`` need.

    Args:
        base: The ref the pull request merges into.
    """
    changed = changed_files(base)
    logger.info("{} files changed since {}", len(changed), base)
    selection = select(changed)
    if selection is None:
        sys.stdout.write("all\n")
    elif not selection:
        logger.info("no test depends on the changed files")
        sys.stdout.write("none\n")
    else:
        logger.info("{} test paths selected", len(selection))
        sys.stdout.write(" ".join(selection) + "\n")


if __name__ == "__main__":
    app()
