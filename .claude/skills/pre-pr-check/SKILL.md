---
name: pre-pr-check
description: Run the repository's full pre-PR verification — ruff lint and format on the whole repo, ty on the package and scripts, the targeted tests of the change (CI runs every tier and the coverage gate), the capability index, the lockfile and the assembled docs build. Use before committing, pushing or opening a pull request, and after any change to public API, dependencies, docstrings or docs.
---

# Pre-PR check

Run from the repo root and fix what fails before committing. Report results honestly: what ran, what passed, what was skipped and why.

## Always

```bash
uv run --group lint ruff check .          # entire repo: tests/, scripts/, notebook cells
uv run --group lint ruff format --check .
uv run poe typecheck                            # ty on src/functax and scripts (what CI runs)
uv run pytest --no-cov <the targeted tests>    # see "Which tests to run while you work" in AGENTS.md
```

The targeted tests are the fast and slow test files of each module the diff changes, the module's doctests, the direct consumers of a changed shared primitive and the contract test whose pinned behaviour changed; pass `--no-cov`, since a subset never reaches the gate. `uv run poe test-fast` (`-m fast`) is the broadest run worth doing locally. The slow and integration tiers and `uv run poe test-cov` (every tier with the 90 % gate) run in CI; run them locally only when CI cannot tell you what you need.

## When the public API changed

```bash
uv run poe capabilities
uv run pytest --no-cov tests/fast/test_capabilities.py tests/fast/test_public_api.py
```

…plus the API page (`docs/api/<module>.md`, a `nav` entry in `mkdocs.yml` for a new module) and the row in `docs/api/index.md`. `__all__` stays sorted; a new module joins `SUBMODULES` (a package, `SUBPACKAGES`) in `tests/fast/test_public_api.py`.

## When dependencies changed

`uv lock` and commit `uv.lock`. A runtime dependency needs discussion (`AGENTS.md`, "Boundaries"): the base install is NumPy alone, and everything else goes in the `jax` extra. Reproduce the `base-install` job of `ci.yml` (a fresh venv, `uv pip install .`, then its Python check) whenever `dependencies`, the extra or a base module changes.

## When docs or docstrings changed

- Every `Examples:` block runs under the test suite (`--doctest-modules`); check the expected output is what the code prints.
- Docstring maths is Unicode, never `$…$`. When the `eman-style-math` skill is installed, check each changed page: `uv run --with cyclopts --with loguru --with pyyaml python ~/.claude/skills/eman-style-math/scripts/check_math.py <file> --target myst`.
- `uv run poe docs-api` (Zensical, strict) for docstrings and `docs/api/`; `uv run poe docs-check` (strict, both halves) and `uv run poe docs` (the assembled site with link checking, what `docs.yml` runs) for prose and notebooks — both need the mystmd CLI (`npm install -g mystmd`); if the theme download is blocked, see `docs/README.md`.

## Before pushing

- `git status` shows no stray files (`.plans/`, notebook `.py` drafts, `coverage.xml`, `site/`, `public/` are ignored; check nothing else is).
- `uv run poe precommit` runs the git hooks (whitespace, end-of-file, YAML, large files, ruff); fix what it reports in the files you touched.
- Conventional Commits title with a lowercase subject; `!` and a `BREAKING CHANGE:` footer for a breaking change.
- Push only to your feature branch, only when asked.
- No AI attribution in commits or the PR description (no `Co-Authored-By:` trailer for an AI, no "Generated with …" line) unless the maintainer asks.
- The PR description is for a human who did not watch the work: why first, a small before/after table, what stays the same, one or two bullets on testing, one `Closes #N` line per issue.
