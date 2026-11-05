# pypackage_template

[![Tests](https://github.com/jejjohnson/pypackage_template/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/pypackage_template/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/pypackage_template/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/pypackage_template/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/pypackage_template/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/pypackage_template/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/pypackage_template/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/pypackage_template/actions/workflows/pages.yml)
[![codecov](https://codecov.io/gh/jejjohnson/pypackage_template/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/pypackage_template)
[![PyPI version](https://img.shields.io/pypi/v/mypackage.svg)](https://pypi.org/project/mypackage/)
[![Python versions](https://img.shields.io/pypi/pyversions/mypackage.svg)](https://pypi.org/project/mypackage/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

Author: J. Emmanuel Johnson
Repo: [https://github.com/jejjohnson/pypackage_template](https://github.com/jejjohnson/pypackage_template)
Website: [jejjohnson.netlify.com](https://jejjohnson.netlify.com)

An opinionated, modern Python package template with best-practice tooling already wired up. When you use this template, you get linting, formatting, type checking, testing with coverage gates, auto-generated documentation, automated releases, security scanning, and AI agent instructions — all configured and integrated from day one.

It also ships a **real worked example** rather than an empty `src/` directory. `mypackage` is a dependency-free library for descriptive statistics, smoothing filters, and composable transforms over numeric series — around 500 statements, 100% covered, with a console script, executable doctests, guide pages, and two executed example notebooks. Every piece of tooling in this repository is therefore *demonstrably working* on day one: the API reference renders real docstrings, the coverage gate measures real branches, and the notebooks import the real package. Delete it and drop in your own code when you adopt the template.

---

## 📂 Repository Layout

```
pypackage_template/
├── src/mypackage/                    # Main package code (src layout)
│   ├── _typing.py                    # Type aliases + structural protocols
│   ├── exceptions.py                 # Typed exception hierarchy
│   ├── utils.py                      # Validators, generic Window[T], timer()
│   ├── stats.py                      # summarize, quantile, RunningStats
│   ├── smoothing.py                  # moving_average, median_filter, Padding
│   ├── transforms.py                 # Standardize, Clip, Pipeline, chain
│   ├── cli.py                        # argparse console script
│   └── py.typed                      # PEP 561 typing marker
├── tests/                            # pytest test suite (100% coverage)
├── docs/                             # Documentation source
│   ├── myst.yml                      # MyST project (prose half)
│   ├── README.md                     # How the two-tool docs build works
│   ├── guide/                        # Guide pages (MyST Markdown)
│   ├── api/                          # mkdocstrings API reference (MkDocs)
│   └── notebooks/                    # Executed example notebooks
├── notebooks/                        # Scratch Jupyter notebooks
├── .github/
│   ├── workflows/                    # GitHub Actions CI/CD workflows (10 total)
│   ├── instructions/                 # Copilot custom instructions
│   ├── copilot-instructions.md       # Copilot behavioural config
│   ├── dependabot.yml                # Automated dependency PRs
│   └── labeler.yml                   # Automatic PR labelling rules
├── scripts/build_docs.py             # Builds, assembles & link-checks the docs
├── scripts/capabilities.py           # Generates docs/api/capabilities.md
├── pyproject.toml                    # Single source of truth for project metadata & tools
├── uv.lock                           # Fully reproducible lockfile
├── mkdocs.yml                        # API-reference site configuration
├── .pre-commit-config.yaml           # Git hook definitions
├── release-please-config.json        # Automated release & changelog config
├── .release-please-manifest.json     # Tracks the current released version
├── .env.example                      # Template for local environment variables
├── AGENTS.md                         # Standing instructions for AI coding agents
├── CLAUDE.md                         # Claude Code: imports AGENTS.md
├── CODE_REVIEW.md                    # Code review standards
└── CHANGELOG.md                      # Auto-generated changelog
```

---

## 🚀 Quick Start

```bash
# Prerequisites: uv (https://github.com/astral-sh/uv)
git clone https://github.com/jejjohnson/pypackage_template.git
cd pypackage_template
uv run poe install      # install all dependency groups
uv run poe test         # run tests + doctests
uv run poe docs-serve   # preview docs locally
```

Then try the bundled example:

```bash
echo "1 2 3 4 5" | uv run mypackage summarize
echo "1 1 99 1 1" | uv run mypackage smooth --method median --window 3
```

---

## ✨ Features

### 📦 Package Management — `uv` + `uv.lock`

**Files:** `pyproject.toml`, `uv.lock`

[uv](https://github.com/astral-sh/uv) is a Rust-based Python package manager that is 10–100× faster than pip. It handles virtual environments, dependency resolution, and installation in a single tool.

- `uv.lock` provides a fully reproducible environment (like `package-lock.json` but for Python) — every developer and every CI run installs the exact same versions.
- Dependency groups (`dev`, `lint`, `typecheck`, `docs`) let you install only what you need for a given task.

> **What:** A Rust-based package manager and virtual environment tool — runs `uv sync` once and every developer gets a bit-for-bit identical environment from `uv.lock`.

> **Why uv?** Eliminates "works on my machine" problems. Faster CI. A single binary replaces pip, pip-tools, virtualenv, and pyenv.

---

### 🏗️ Project Metadata — `pyproject.toml`

**File:** `pyproject.toml`

`pyproject.toml` is the single source of truth, replacing `setup.py`, `setup.cfg`, and `requirements.txt`. Everything lives in one place:

- **`[project]`** (PEP 621): name, version, description, authors, license, classifiers, `requires-python`
- **`[dependency-groups]`**: `dev`, `lint`, `typecheck`, `docs`
- **`[build-system]`**: `hatchling` as the build backend
- **`[tool.ruff]`**, **`[tool.ty]`**, **`[tool.pytest.ini_options]`**, **`[tool.coverage.*]`**: all tool configs co-located

> **What:** A single `pyproject.toml` replaces `setup.py`, `setup.cfg`, and every scattered `*.cfg` config file — one place for project metadata and every tool's configuration.

> **Why one file?** Standardised (PEP 517/518/621), no scattered config files, tooling reads from one canonical location.

---

### 📁 `src/` Layout

**Directory:** `src/mypackage/`

Source code lives under `src/mypackage/` rather than at the repo root. This forces the package to be properly installed before it can be imported, which means:

- Packaging bugs (missing files, incorrect paths) are caught early rather than hidden by the flat-layout import shortcut.
- Tests always exercise the installed package, not an accidentally-importable source directory.

> **What:** All source code lives under `src/mypackage/` instead of at the repo root, so the package must be installed before it can be imported.

> **Why `src/` layout?** Industry best practice. See writings by Brett Cannon and Hynek Schlawack on why flat layouts silently mask packaging errors.

---

### 🔨 Build Backend — `hatchling`

**File:** `pyproject.toml` (`[build-system]`, `[tool.hatch.build.targets.wheel]`)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mypackage"]
```

Zero-config, PEP 517 compliant. No `MANIFEST.in`, no surprises. Integrates cleanly with uv.

The demo package also exercises two packaging features you will want in a real
project:

- **`[project.scripts]`** installs a console script:
  `mypackage = "mypackage.cli:main"`. `uv run poe build` produces a wheel whose
  `mypackage` command works on any machine that installs it.
- **`src/mypackage/py.typed`** is the [PEP 561](https://peps.python.org/pep-0561/) marker that tells downstream
  type checkers your annotations are real and should be honoured. Without it,
  every annotation you write is invisible to your users.

> **What:** `hatchling` is the PEP 517 build backend — run `uv build` and it produces a wheel and sdist with zero extra config.

> **Why hatchling?** Minimal configuration, actively maintained, excellent uv integration, and handles the `src/` layout without extra config.

---

### 🧹 Linting & Formatting — `ruff`

**Files:** `pyproject.toml` (`[tool.ruff]`), `.pre-commit-config.yaml`, `.github/workflows/lint.yml`

[ruff](https://github.com/astral-sh/ruff) replaces flake8, isort, and black in a single Rust binary.

Rule sets enabled:

| Code | What it checks |
|------|---------------|
| `E`, `F` | pycodestyle + pyflakes (core correctness) |
| `I` | Import sorting (replaces isort) |
| `UP` | pyupgrade — modernise syntax automatically |
| `B` | flake8-bugbear — common gotchas |
| `SIM` | flake8-simplify — simplify expressions |
| `RUF` | Ruff-native rules |

Pre-commit hooks run both `ruff` (lint + autofix) and `ruff-format`. CI enforces with `ruff check` and `ruff format --check`.

> **What:** A single Rust binary that lints, sorts imports, and formats Python code — replaces flake8 + isort + black with one tool and one config block.

> **Why ruff?** ~100× faster than the black + flake8 + isort combination; single config block; same results.

---

### 🔬 Type Checking — `ty`

**Files:** `pyproject.toml` (`[tool.ty]`), `.github/workflows/typecheck.yml`

```toml
[tool.ty.environment]
python-version = "3.12"
```

[ty](https://github.com/astral-sh/ty) is Astral's new Rust-based type checker (same team as ruff and uv). It is extremely fast and designed to integrate with the rest of the Astral toolchain.

> **What:** A Rust-based static type checker from the Astral team — run `uv run poe typecheck` to catch type errors without executing any code.

> **Why ty?** Catches whole classes of bugs before runtime. Faster than mypy or pyright. Part of the same ecosystem as ruff and uv.

---

### 🧪 Testing — `pytest` + `pytest-cov`

**Files:** `pyproject.toml` (`[tool.pytest.ini_options]`, `[tool.coverage.*]`), `.github/workflows/ci.yml`

```toml
[tool.pytest.ini_options]
testpaths = ["tests", "src/mypackage"]
addopts = "--doctest-modules --cov=src/mypackage --cov-report=term-missing ..."

[tool.coverage.report]
fail_under = 90
```

- Tests live in `tests/`.
- **Doctests are part of the suite.** `src/mypackage` is a testpath and
  `--doctest-modules` is in `addopts`, so every `Examples:` block in a
  Google-style docstring is executed. Documentation that drifts from the code
  fails the build. `ruff format` is configured with `docstring-code-format`
  so those examples are formatted like real source.
- Fast local run (no coverage): `uv run poe test` (skips coverage collection for speed)
- Coverage runs (with reports + gate): `uv run pytest` or `uv run poe test-cov`, and in CI
- Coverage report (when enabled): terminal summary + `coverage.xml` for Codecov upload
- **Coverage gate:** `fail_under = 90` — coverage-enabled runs and CI fail if coverage drops below 90% (the bundled example sits at 100%)
- CI matrix: Python 3.12 and 3.13

> **What:** pytest with a 90% coverage gate and executable doctests wired in — `uv run poe test` is fast (no coverage), while CI and `uv run poe test-cov` enforce the gate and upload results to Codecov.

> **Why a coverage gate?** Prevents silent regressions. The matrix catches version-specific bugs that single-version CI misses.

---

### 🪝 Pre-commit — `.pre-commit-config.yaml`

**Files:** `.pre-commit-config.yaml`, `.github/workflows/pre-commit-autoupdate.yml`

Hooks that run on every `git commit`:

| Hook | What it does |
|------|-------------|
| `end-of-file-fixer` | Ensures files end with a newline |
| `trailing-whitespace` | Removes trailing whitespace |
| `check-yaml` | Validates YAML syntax |
| `check-added-large-files` | Blocks accidentally committed large files |
| `ruff` | Lint + autofix |
| `ruff-format` | Format |

Run manually: `uv run poe precommit`. Hook versions are bumped automatically weekly via `.github/workflows/pre-commit-autoupdate.yml`.

> **What:** Six git hooks that run automatically on every `git commit` to catch formatting, whitespace, and YAML issues before they reach CI.

> **Why pre-commit?** Stops bad commits at the source rather than relying on CI to catch them after a push. Weekly autoupdate keeps hooks on patched versions.

---

### 📖 Documentation — mystmd (prose) + MkDocs/mkdocstrings (API)

**Files:** `docs/myst.yml`, `mkdocs.yml`, `scripts/build_docs.py`, `docs/README.md`, `.github/workflows/docs.yml`

The documentation is built by **two generators and deployed as one site**,
because neither tool is good at both halves:

| Half | Tool | Source | Deployed at |
|---|---|---|---|
| Home, guides, notebooks | [mystmd](https://mystmd.org) | `docs/*.md`, `docs/guide/`, `docs/notebooks/` | `/` |
| API reference | MkDocs + mkdocstrings | `docs/api/` | `/reference/` |

MyST is far better for prose — real cross-references, first-class notebook
handling, PDF/LaTeX export, and a directive syntax that beats admonition
soup. What it has no answer for is autodoc: there is no mature way to render
Python docstrings into a MyST site today. mkdocstrings does that well and
publishes a Sphinx-compatible `objects.inv`, which is exactly what mystmd
needs to cross-reference *into* it. So each tool does the half it is good at.

```bash
uv run poe docs          # build both halves, assemble into public/, verify links
uv run poe docs-api      # API reference only (fast; needs no Node)
uv run poe docs-serve    # build, then serve the assembled site at :8000
```

`mystmd` is a Node CLI (`npm install -g mystmd`), not a uv dependency.

Prose links into the API with the `xref:` protocol:

```markdown
[`summarize`](xref:api#mypackage.summarize)
```

A target that is not in the inventory fails `myst build --strict`.

> **What:** Two documentation generators stitched into one site, with every
> link between them verified on each build.

> **Why two?** MyST wins on prose and notebooks; mkdocstrings wins on
> docstrings. The `objects.inv` that mkdocstrings already publishes is the
> bridge, so you do not have to compromise on either half.

---

### 🔗 Verified cross-generator links — `scripts/build_docs.py`

**File:** `scripts/build_docs.py`, tested by `tests/test_build_docs.py`

Splitting the docs across two tools creates a class of bug neither tool can
catch: a link from one generator's output into the other's. The build script
closes that gap. It

1. builds the API reference with Zensical (the successor to Material for MkDocs, reading `mkdocs.yml`),
2. serves it locally so mystmd can read the inventory — mystmd only loads
   inventories over http, rejecting a filesystem path and *silently ignoring*
   a `file://` URL,
3. builds the prose with mystmd,
4. assembles both into `public/`,
5. repairs anchors mangled by a mystmd bug (it lowercases an object's name
   when expanding the `$` anchor abbreviation in an inventory), and
6. **verifies that every internal link in the assembled site resolves** — the
   target file must exist and, when the link carries a fragment, that anchor
   must really be in it.

Step 6 is the one that matters: it is the only check that sees both halves at
once, and it turns a silent broken deep-link into a failed pull request.

> **What:** A build script that assembles the two documentation halves and
> fails CI on any broken link between them.

> **Why:** Cross-generator links are exactly the links no single tool
> validates, which makes them the ones that rot silently.

---

### 🏷️ Conventional Commits

**File:** `.github/workflows/conventional-commits.yml`

All commit messages must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>
```

Valid types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`

The `conventional-commits.yml` workflow validates every **PR title** automatically using [`amannn/action-semantic-pull-request`](https://github.com/amannn/action-semantic-pull-request).

> **What:** A CI check that enforces Conventional Commits format on every PR title, ensuring the commit history is machine-readable for automated changelogs and version bumps.

> **Why Conventional Commits?** Enables automated changelogs, drives semantic versioning decisions, and makes `git log` readable at a glance.

---

### 🚀 Automated Releases — Release Please

**Files:** `release-please-config.json`, `.release-please-manifest.json`, `.github/workflows/release-please.yml`

On every push to the default branch, [Release Please](https://github.com/googleapis/release-please) opens or updates a "release PR" that:

1. Bumps the version in `pyproject.toml` based on commit types
2. Updates `CHANGELOG.md` with grouped entries

Merging that PR creates a GitHub Release and a git tag — no manual version bumping or changelog editing required.

Changelog sections are driven by `release-please-config.json`. The current mapping:

| Commit type | CHANGELOG section | Visible |
|-------------|------------------|---------|
| `feat` | Features | ✅ |
| `fix` | Bug Fixes | ✅ |
| `perf` | Performance Improvements | ✅ |
| `revert` | Reverts | ✅ |
| `docs` | Documentation | hidden |
| `style` | Styles | hidden |
| `refactor` | Code Refactoring | hidden |
| `test` | Tests | hidden |
| `build` | Build System | hidden |
| `ci` | Continuous Integration | hidden |
| `chore` | Miscellaneous | hidden |

> **What:** Fully automated releases — every push to the default branch keeps a "release PR" up to date; merging it tags a release, bumps the version, and publishes a changelog.

> **Why Release Please?** Zero-friction releases. The changelog writes itself from your commit messages.

---

### 🔒 Security — CodeQL

**File:** `.github/workflows/codeql.yml`

GitHub's free static analysis tool scans for security vulnerabilities on every push, every PR, and on a weekly schedule. Zero configuration required.

> **What:** GitHub's built-in static analysis — scans the codebase for security vulnerabilities on push, PR, and a weekly schedule.

> **Why CodeQL?** Catches common security issues automatically. Free for public repos. No maintenance overhead.

---

### 🤖 Dependabot — `.github/dependabot.yml`

Dependabot monitors both **GitHub Actions** versions and **Python (pip) dependencies**, and opens automated PRs to update them whenever a new version is released.

> **What:** Automated dependency PRs every week — both GitHub Actions and Python packages are kept up to date without manual intervention.

> **Why Dependabot?** Keeps CI actions and Python dependencies on recent, patched versions without any manual tracking.

---

### 🏷️ Auto PR Labeling — `.github/labeler.yml` + `label-pr.yml`

File path patterns are mapped to labels from the standard taxonomy (`area:*`, `layer:*`, `dependencies`). For example:

- Changes in `src/**/_primitives/` → `layer:0-primitives`
- Changes in `tests/` → `area:testing`
- Changes in `docs/`, `mkdocs.yml`, `*.md` → `area:docs`
- Changes in `.github/`, `pyproject.toml` → `area:engineering`
- Changes in dependency files → `dependencies`

> **What:** Automatic label application on every PR based on which files changed — no manual labelling required.

> **Why auto-labeling?** At-a-glance PR categorization in the GitHub UI with zero manual effort.

---

### 🧭 Issue Templates + Epic Hierarchy — `.github/ISSUE_TEMPLATE/`

**Files:** `.github/ISSUE_TEMPLATE/*.md`, `.github/scripts/create-labels.sh`, `docs/contributing.md`, `CONTRIBUTING.md`

Six opinionated issue templates (plus a `config.yml`) enforce a two-layer epic model (**Wave → Theme → Issue**) and a consistent body format: Problem / User Story / Motivation / Proposed API / References / Implementation Steps / Definition of Done / Testing / Documentation / Relationships.

| Template | Purpose |
|---|---|
| `Epic — Wave (L1)` | Release-scoped mega-epic; owns a milestone |
| `Epic — Theme (L2)` | Parallel-safe group of issues under a wave |
| `Feature / Enhancement` | One substantial deliverable |
| `Design / ADR` | Resolves a design question for a new API |
| `Bug report` | Something isn't working |
| `Research / Comparative Analysis` | Investigate prior art and produce a prioritized roadmap of follow-up issues |

The `Feature` and `Design` templates include two **optional** sections:

- **Design Snapshot** — paste API sketches or excerpts from private / external design docs so the issue is self-contained.
- **Mathematical Notes** — equations, sign conventions, numerical considerations.

Both exist so that an implementer (human or AI agent) can work on an issue without opening other repos or chats.

A companion label taxonomy (`type:*`, `area:*`, `layer:*`, `wave:*`, `priority:*`) is created by running:

```bash
uv run poe gh-labels       # idempotent; edit .github/scripts/create-labels.sh to customise
```

For planning a whole wave of issues at once, copy [`docs/templates/wave-backlog.md`](docs/templates/wave-backlog.md) into your project's `.plans/` directory and draft the wave as one reviewable file before opening GitHub issues. See [`docs/contributing.md`](docs/contributing.md) → "Drafting a wave backlog" for the workflow.

Once issues are open, apply GitHub's **native sub-issue and blocked-by links** on top of the prose `## Relationships` block. A helper script and three poe tasks wrap the REST endpoints:

```bash
uv run poe gh-sub   7 42 43 44   # link sub-issues
uv run poe gh-block 44 43          # mark dependency
uv run poe gh-show  44                         # inspect parent / sub-issues / blocking / blocked-by
```

Two Claude Code skills guide these workflows end-to-end:

- [`create-gh-issue`](.claude/skills/create-gh-issue/SKILL.md) — picks the right template, drafts the body with required sections, applies labels + milestone, opens via `gh issue create`, and chains to the linking skill for relationships. Handles single issues and bulk publishing of a `.plans/<wave>-backlog.md` file.
- [`link-gh-issues`](.claude/skills/link-gh-issues/SKILL.md) — bulk-apply sub-issue and blocked-by links from a drafted backlog or theme epic's Issues checklist.

See [`docs/contributing.md`](docs/contributing.md) (or [`CONTRIBUTING.md`](CONTRIBUTING.md) at the repo root) for the full taxonomy, epic model, and conventions.

> **What:** Six issue templates + a 24-label taxonomy + a `Wave → Theme → Issue` hierarchy + a wave-backlog drafting template under `docs/templates/` + `create-gh-issue` and `link-gh-issues` skills with helper scripts for filing issues and applying native sub-issue / blocked-by links, all documented in `docs/contributing.md`.

> **Why an epic model?** Consistent issue structure lets humans and AI agents collaborate on planning without renegotiating conventions each time. Templates encode the conventions; `uv run poe gh-labels` bootstraps the labels; `uv run poe gh-sub` / `uv run poe gh-block` wire the native hierarchy.

---

### 🛠️ Poe the Poet — Task Runner in `pyproject.toml`

**File:** `pyproject.toml` (`[tool.poe]` and `[tool.poe.tasks]`)

Every common task is a [Poe the Poet](https://poethepoet.natn.io) task defined in `pyproject.toml`; there is no Makefile. Poe is in the `dev` dependency group, so `uv run poe <task>` works from a fresh clone (`uv run` syncs the group first). For a shorter `poe <task>`, install it globally with `uv tool install poethepoet`.

| Task | Description |
|--------|-------------|
| `uv run poe` | List every task with its description |
| `uv run poe install` | Install all dependency groups via uv + pre-commit hooks |
| `uv run poe init` | Bootstrap `.env` from `.env.example` |
| `uv run poe test` | Run tests (no coverage) |
| `uv run poe test-cov` | Run tests with coverage report and the 90 % gate |
| `uv run poe lint` | Lint with ruff (no autofix) |
| `uv run poe format` | Format with ruff (format + autofix) |
| `uv run poe format-check` | Check formatting without changing files |
| `uv run poe typecheck` | Type-check with ty |
| `uv run poe check` | Everything CI checks: lint, format-check, typecheck, test-cov |
| `uv run poe precommit` | Run pre-commit hooks on all files |
| `uv run poe build` | Build wheel and sdist |
| `uv run poe clean` | Remove build artefacts and caches |
| `uv run poe docs` | Build documentation site |
| `uv run poe docs-check` | Validate docs sources without the themed render |
| `uv run poe docs-api` | Build only the Zensical API reference |
| `uv run poe docs-serve` | Preview documentation locally |
| `uv run poe capabilities` | Regenerate the capability index (`docs/api/capabilities.md`) |
| `uv run poe gh-labels` | Bootstrap the GitHub label taxonomy (idempotent) |
| `uv run poe gh-sub` | Link children as sub-issues of a parent (`uv run poe gh-sub 7 42 43`) |
| `uv run poe gh-block` | Mark an issue as blocked by another (`uv run poe gh-block 44 43`) |
| `uv run poe gh-show` | Show parent / sub-issues / blocking / blocked-by for an issue |
| `uv run poe version` | Display package version and git hash |

Extra arguments after a task name are passed through to its command, e.g. `uv run poe test tests/test_stats.py -k summarize`. Poe loads `.env` when present, and `[tool.poe.env]` gives `PKGROOT` and `TYPECHECK_EXTRA` defaults that `.env` or the shell can override.

> **What:** A task runner configured in `pyproject.toml` — `uv run poe` prints every task with its description, so you never need to remember toolchain invocations.

> **Why Poe instead of a Makefile?** One config file instead of two, works on Windows (no `make`), real argument parsing for tasks like `gh-sub`, and task help text that lives next to the task. The commands themselves are the same `uv run …` invocations CI uses.

---

### 🔐 Environment Variables — `.env.example`

**File:** `.env.example`

`.env.example` is committed to git as a template. The actual `.env` is gitignored. Run `uv run poe init` to copy the example to `.env`.

Documented variables:

| Variable | Purpose |
|----------|---------|
| `PKGROOT` | Path to package source (default: `src/mypackage`) |
| `PYPI_TOKEN` | PyPI token for publishing |
| `GITHUB_TOKEN` | GitHub personal access token |

> **What:** A committed `.env.example` documents every supported environment variable; the real `.env` is gitignored and auto-loaded by poe.

> **Why `.env.example`?** Local overrides without committing secrets. Poe loads it before every task, so all tasks see the variables automatically.

---

### 🤖 AI Agent Support — `AGENTS.md` + a capability index

**Files:** `AGENTS.md`, `CLAUDE.md`, `.claude/skills/`, `.claude/agents/`, `.github/copilot-instructions.md`, `.github/instructions/`, `scripts/capabilities.py`, `docs/api/capabilities.md`

`AGENTS.md` is the single source of truth for every coding agent (Claude Code
imports it from `CLAUDE.md`; Copilot is pointed at it): the module map,
"reuse before you write", the package's contracts and the tests that enforce
them, the commands, the pre-commit checklist, test tiers, the docs toolchain,
and the git / PR rules. `docs/api/capabilities.md` lists every public name
with a one-line summary, generated by `uv run poe capabilities` and checked by
`tests/test_capabilities.py`, so agents find what exists before writing it
again.

Recipes for the common jobs (adopting the template, adding a function,
transform, CLI command, guide page or notebook, the pre-PR check, review,
squash commits, GitHub issues) live in `.claude/skills/`, and two read-only
reviewer subagents (reuse, correctness) in `.claude/agents/`; any agent can
follow them as plain Markdown.

The house style comes from the maintainer's personal skills in [jejagents](https://github.com/jejjohnson/jejagents): `python-style` for Python, `eman-style-math` for maths (Unicode in docstrings, macros in the docs), `code-sketch` for pseudocode, `eman-prose-tech` for prose, `report` and `gh-issue` for issues. The "House style" and "Personal skills" sections of `AGENTS.md` carry their essentials, so an agent without them installed still follows the same rules.

For projects that build on the package, the repository is also a Claude
Code plugin marketplace (`.claude-plugin/marketplace.json`, `plugins/`) and
the docs site serves an `llms.txt`; `docs/agents.md` explains both.

Passages that describe the example package are marked
`<!-- template-example: … -->`; rewrite them from your own code after
adopting the template (`grep -rn "template-example" .` lists them).

> **What:** One agent-instruction file, written from the real code, plus a generated index of every public name.

> **Why?** Agents re-implement what they cannot see and break rules nobody wrote down. Explicit, tested instructions fix both.

---

### 📋 Code Review Standards — `CODE_REVIEW.md`

**File:** `CODE_REVIEW.md`

Defines the review checklist (reuse and boundaries, the package's contracts, public API and docs, tests, modern Python, security), package-specific checks with runnable examples, output format, and suggestion priorities. Referenced by both `AGENTS.md` and `copilot-instructions.md` so that humans and AI agents apply the same criteria.

> **What:** A shared code-review rubric that both human reviewers and AI agents reference, ensuring consistent feedback quality across the project.

---

## 🔄 CI/CD Workflows

| Workflow | File | Trigger | What it does |
|----------|------|---------|-------------|
| Tests | `ci.yml` | push / PR to default branch | pytest matrix (3.12, 3.13) + Codecov upload |
| Lint | `lint.yml` | push / PR to default branch | `ruff check` + `ruff format --check` |
| Type Check | `typecheck.yml` | push / PR to default branch | `ty check` |
| Docs | `docs.yml` | push / PR to default branch | build both halves + verify every link |
| Deploy Docs | `pages.yml` | push to default branch | deploy the assembled site to Pages |
| Release Please | `release-please.yml` | push to default branch | automated release PR + changelog |
| CodeQL | `codeql.yml` | push / PR / schedule | security static analysis |
| Conventional Commits | `conventional-commits.yml` | PR | validates PR title format |
| PR Labeler | `label-pr.yml` | PR | applies path-based labels |
| Pre-commit Autoupdate | `pre-commit-autoupdate.yml` | weekly schedule | bumps hook revisions, opens PR |

---

## 🛠️ Adapting This Template

Follow this checklist when using this repo as a base for a new project.
With Claude Code, the `adopt-template` skill
(`.claude/skills/adopt-template/SKILL.md`) walks through the same steps,
including the rename commands and rewriting the agent files from your code.

1. **Decide what to do with the demo package.** `src/mypackage/` is a worked
   example, not scaffolding you must keep. Either delete its modules and start
   fresh (also clearing `tests/`, `docs/guide/`, `docs/api/`, and
   `docs/notebooks/`), or keep one module as a reference while you build. The
   coverage gate is set to 90%, so an empty package with no tests will fail CI
   until you either add tests or lower `fail_under`.
2. **Search-and-replace** `mypackage` with your package name everywhere (source, config, docs)
3. **Update `[project]` in `pyproject.toml`**: name, description, authors, keywords, classifiers, `requires-python`
4. **Update `mkdocs.yml`**: `site_name`, `site_description`, `repo_url`, `repo_name`
5. **Rename `src/mypackage/`** to `src/<yourpackage>/`
6. **Copy `.env.example` to `.env`** (`uv run poe init`) and fill in values
7. **Run `uv run poe install`** then **`uv run poe test`** to verify the baseline works
8. **Set up Codecov** and add `CODECOV_TOKEN` to GitHub Secrets if you want coverage tracking
9. **Update badge URLs** in this README to point at your repository
10. **Update `[project.urls]`** in `pyproject.toml` to your repository URL
11. Delete or update `CHANGELOG.md` and `.release-please-manifest.json` to start fresh
12. **Bootstrap the GitHub label taxonomy**: `uv run poe gh-labels` (edits `.github/scripts/create-labels.sh` if you need to customise `area:*` / `layer:*` / `wave:*` for your project)
13. **Review `docs/contributing.md`** and adjust the label taxonomy / epic model / contact links for your project; update `.github/ISSUE_TEMPLATE/config.yml` discussions URL

---

## 📚 Further Reading

| Tool | Documentation |
|------|--------------|
| uv | <https://docs.astral.sh/uv/> |
| ruff | <https://docs.astral.sh/ruff/> |
| ty | <https://github.com/astral-sh/ty> |
| hatchling | <https://hatch.pypa.io/latest/> |
| pytest | <https://docs.pytest.org/> |
| mystmd | <https://mystmd.org> |
| Zensical | <https://zensical.org> |
| mkdocstrings | <https://mkdocstrings.github.io/> |
| pre-commit | <https://pre-commit.com/> |
| Release Please | <https://github.com/googleapis/release-please> |
| Conventional Commits | <https://www.conventionalcommits.org/> |
