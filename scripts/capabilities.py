"""Generate, or check, the capability index: every public name in the package.

``docs/api/capabilities.md`` lists each public object of the package once, at
its public home (the top-level package first, then any public submodule that
exports something the top level does not), grouped by the module that defines
it, with the first sentence of its docstring. Agents and people search it
before writing a helper ("Reuse before you write" in ``AGENTS.md``).

Usage::

    uv run poe capabilities                             # rewrite the index
    uv run python scripts/capabilities.py --check       # fail if stale

``tests/test_capabilities.py`` runs the check, so a public API change that
skips ``uv run poe capabilities`` fails the test suite.

The script needs no editing when the template is adopted: the import package
is read from ``pyproject.toml`` and the public submodules are discovered. Two
settings are yours to fill in:

- ``UPSTREAM``: libraries this package builds on whose public API belongs in
  the index too (so nobody re-implements them here). Empty by default. Their
  sections depend on the installed versions, which the index records; when
  those differ from the installed ones, only this package's part is compared.
- ``ALLOWED_SHARED_NAMES``: names deliberately bound to different objects in
  two homes, each with the reason. ``--check`` fails on any other clash, and
  on a name that shadows an ``UPSTREAM`` name with a different object.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import pkgutil
import re
import sys
import tomllib
import types
import typing
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "api" / "capabilities.md"


def _package_name() -> str:
    """The import package, as the wheel build declares it in ``pyproject.toml``.

    Falls back to the distribution name with dashes turned into underscores.
    """
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    wheel = data.get("tool", {}).get("hatch", {}).get("build", {})
    packages = wheel.get("targets", {}).get("wheel", {}).get("packages", [])
    if packages:
        return Path(packages[0]).name
    return data["project"]["name"].replace("-", "_")


PACKAGE = _package_name()

# Libraries the package builds on, as {distribution: [public modules]}, with a
# one-paragraph blurb each, e.g. {"numpy": ["numpy"]}.
# template-example: empty because the example package is dependency-free.
UPSTREAM: dict[str, list[str]] = {}
UPSTREAM_BLURB: dict[str, str] = {}

# Names two homes may bind to different objects, and why, e.g.
# {"Mean": "stats.Mean is the statistic, aggregation.Mean the reducer"}.
ALLOWED_SHARED_NAMES: dict[str, str] = {}

_AUTOREF = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
# A full stop that ends a sentence (not "e.g." / "i.e." / "etc.").
_SENTENCE_END = re.compile(r"(?<!e\.g)(?<!i\.e)(?<!etc)\.\s+(?=[A-Z])")
UPSTREAM_MARKER = "<!-- upstream -->"


def _first_line(text: str) -> str:
    """The docstring's summary: its first sentence, on one line."""
    paragraph = text.strip().split("\n\n")[0]
    line = " ".join(part.strip() for part in paragraph.splitlines())
    # mkdocs-autorefs links resolve only in the docs that wrote them.
    line = _AUTOREF.sub(r"\1", line)
    sentence = _SENTENCE_END.split(line, maxsplit=1)[0]
    if sentence != line:
        line = sentence + "."
    if len(line) > 160:
        line = line[:157].rstrip() + "..."
    return line.replace("|", "\\|")


def _kind(obj: object) -> str:
    if isinstance(obj, typing.TypeAliasType):
        return "type alias"
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        return "function"
    return "constant"


def _origin(obj: object) -> str:
    module = getattr(obj, "__module__", None)
    defined = callable(obj) or isinstance(obj, typing.TypeAliasType)
    if not isinstance(module, str) or not defined:
        module = type(obj).__module__
    return module


def _summary(obj: object, *, home: str) -> str:
    origin = _origin(obj).split(".")[0]
    if origin != home:
        return f"Re-exported from `{origin}`."
    if isinstance(obj, typing.TypeAliasType):
        # A `type` statement's docstring is not attached to the alias.
        return f"Alias of `{obj.__value__!r}`.".replace("|", "\\|")
    if not callable(obj):
        value = repr(obj)
        return f"`{value if len(value) <= 60 else value[:57] + '...'}`".replace(
            "|", "\\|"
        )
    return _first_line(inspect.getdoc(obj) or "")


def _public(module: types.ModuleType) -> list[str]:
    names = getattr(module, "__all__", None)
    if names is None:
        names = [n for n in dir(module) if not n.startswith("_")]
    return [
        n
        for n in names
        if not n.startswith("__")
        and not isinstance(getattr(module, n, None), types.ModuleType)
    ]


def _module_title(name: str) -> str:
    module = sys.modules.get(name) or importlib.import_module(name)
    return _first_line(module.__doc__ or "")


def public_modules() -> list[str]:
    """The package, then every submodule with no ``_``-prefixed path component."""
    package = importlib.import_module(PACKAGE)
    found = [
        info.name
        for info in pkgutil.walk_packages(package.__path__, prefix=f"{PACKAGE}.")
        if not any(part.startswith("_") for part in info.name.split("."))
    ]
    return [PACKAGE, *sorted(found)]


def collect() -> tuple[dict[str, list[tuple]], list[str]]:
    """``{home: [(name, kind, summary, defining module)]}`` and the clashes.

    A name is listed under the first home that exports it, so a submodule
    appears only for what the top-level package does not re-export.
    """
    index: dict[str, list[tuple]] = defaultdict(list)
    owners: dict[str, dict[str, object]] = defaultdict(dict)
    seen: set[int] = set()
    for module_name in public_modules():
        module = importlib.import_module(module_name)
        for name in _public(module):
            obj = getattr(module, name)
            owners[name][f"{module_name}.{name}"] = obj
            if id(obj) in seen and not isinstance(obj, int | float | str):
                continue
            seen.add(id(obj))
            index[module_name].append(
                (name, _kind(obj), _summary(obj, home=PACKAGE), _origin(obj))
            )
    clashes = []
    for name, homes in sorted(owners.items()):
        if name in ALLOWED_SHARED_NAMES:
            continue
        if len({id(o) for o in homes.values()}) > 1:
            clashes.append(f"{name}: {', '.join(sorted(homes))}")
    for modules in UPSTREAM.values():
        for module_name in modules:
            try:
                upstream = importlib.import_module(module_name)
            except ImportError:
                continue
            for name in _public(upstream):
                if name in ALLOWED_SHARED_NAMES:
                    continue
                for path, obj in owners.get(name, {}).items():
                    if obj is not getattr(upstream, name):
                        clashes.append(f"{name}: {path} shadows {module_name}.{name}")
    return index, clashes


def _table(rows: list[tuple]) -> list[str]:
    out = ["| Name | Kind | What it does |", "|---|---|---|"]
    out += [f"| `{r[0]}` | {r[1]} | {r[2]} |" for r in rows]
    return [*out, ""]


def _package_part() -> tuple[list[str], int]:
    index, _ = collect()
    out: list[str] = []
    total = 0
    for module_name, rows in index.items():
        total += len(rows)
        out += [f"## `{module_name}`", ""]
        # A flat facade is grouped by the module that defines each name: one
        # concept per group (statistics, smoothing, ...).
        groups: dict[str, list[tuple]] = defaultdict(list)
        for row in rows:
            origin = row[3]
            local = origin.split(".")[0] == PACKAGE
            groups[origin if local else "re-exported"].append(row)
        if len(groups) == 1:
            out += _table(sorted(rows))
            continue
        for origin in sorted(groups, key=lambda o: (o == "re-exported", o)):
            title = (
                "Re-exported from other libraries"
                if origin == "re-exported"
                else f"{_module_title(origin)} (`{origin}`)"
            )
            out += [f"### {title}", "", *_table(sorted(groups[origin]))]
    return out, total


def _upstream_part() -> list[str]:
    out: list[str] = []
    for package, modules in UPSTREAM.items():
        out += [f"## Upstream: `{package}`", "", UPSTREAM_BLURB.get(package, ""), ""]
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            rows = []
            for name in sorted(_public(module)):
                obj = getattr(module, name)
                rows.append((name, _kind(obj), _summary(obj, home=package)))
            out += [f"### `{module_name}`", "", *_table(rows)]
    return out


def _versions() -> str:
    return ", ".join(f"{p} {importlib.metadata.version(p)}" for p in UPSTREAM)


def render() -> str:
    """The index, as Markdown."""
    package_part, total = _package_part()
    upstream = f"; then the public API of {', '.join(UPSTREAM)}" if UPSTREAM else ""
    out = [
        "# Capability index",
        "",
        "<!-- Generated by scripts/capabilities.py — do not edit by hand. -->",
        "",
        f"Every public name in `{PACKAGE}` ({total} of them), each listed once at",
        "its public home and grouped by the module that defines it, with the",
        f"first sentence of its docstring{upstream}. Search this page before",
        "writing a helper: if what you need is here, compose it; if it is almost",
        "here, extend it where it lives. Regenerate with `uv run poe capabilities`",
        "after changing a public API (`tests/test_capabilities.py` checks it is",
        "current).",
        "",
        *package_part,
    ]
    if UPSTREAM:
        out += [UPSTREAM_MARKER, "", f"Listed at {_versions()}.", ""]
        out += _upstream_part()
    return "\n".join(out).rstrip() + "\n"


def stale(current: str, text: str) -> bool:
    """Whether ``current`` differs from ``text`` where it is comparable.

    The upstream sections are compared only when the recorded versions match
    the installed ones; the package's own part always is.
    """
    if current == text:
        return False
    if not UPSTREAM or f"Listed at {_versions()}." in current:
        return True
    return current.partition(UPSTREAM_MARKER)[0] != text.partition(UPSTREAM_MARKER)[0]


def main() -> int:
    """Rewrite the index, or with ``--check`` fail if it is stale."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if stale")
    args = parser.parse_args()
    text = render()
    status = 0
    _, clashes = collect()
    if clashes:
        print("Names bound to different objects in two homes:")
        print("\n".join(f"  {c}" for c in clashes))
        print("Rename one, or add it to ALLOWED_SHARED_NAMES with a reason.")
        status = 1
    if args.check:
        current = INDEX.read_text(encoding="utf-8") if INDEX.exists() else ""
        if stale(current, text):
            rel = INDEX.relative_to(ROOT)
            print(f"{rel} is stale; run `uv run poe capabilities`")
            status = 1
    else:
        INDEX.write_text(text, encoding="utf-8")
        print(f"wrote {INDEX.relative_to(ROOT)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
