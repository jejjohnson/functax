# AGENTS.md

Standing instructions for **every** coding agent working in this repository (Claude Code, Copilot, Codex, Gemini, …). This is the single source of truth: `CLAUDE.md` and `.github/copilot-instructions.md` point here.

The house style (Python, maths notation, prose, pseudocode, reports) comes from the maintainer's personal skills in [jejagents](https://github.com/jejjohnson/jejagents); "House style" and "Personal skills" below say which skill covers what and carry the essentials, so an agent without those skills installed can still follow them.

> **This repository is a template.** Passages that describe the bundled example package, `mypackage`, start with a `<!-- template-example: … -->` comment (`# template-example: …` in code and config). Each marker covers the section it opens, down to the next heading of the same level, unless it names a smaller scope. Marked passages are true of the example and must be rewritten from your own code when you adopt the template (the `adopt-template` skill walks through it; `grep -rn "template-example" .` lists them). Everything unmarked is a permanent rule.

## What this repo is

<!-- template-example: describe your package and its module map. -->

An opinionated Python package template (Python 3.12+, uv, ruff, ty, pytest, mystmd and MkDocs, release-please) that ships a real worked example, `mypackage`: a **dependency-free** library of descriptive statistics, smoothing filters and composable transforms over one-dimensional numeric series, with an `argparse` console script. It exists so every tool here has real code to operate on: mkdocstrings renders its docstrings, ty checks its annotations, the coverage gate measures its branches, and every docstring example runs as a doctest.

People build on a package like this, so **the package is a library of primitives**: new code composes what is here, and anything genuinely new lands where the next person will find and reuse it.

It is one package, `src/mypackage/`. The public API is `mypackage.__all__`, re-exported from `src/mypackage/__init__.py`, plus the `mypackage.cli` module (`main`, `build_parser`). Imports only point down this table:

| Module | Owns |
|---|---|
| `_typing.py` | PEP 695 `type Series = Sequence[float]`; the runtime-checkable `Transform` and `Fittable` protocols |
| `exceptions.py` | `MypackageError` and its subclasses `ValidationError` (also a `ValueError`), `EmptySeriesError`, `NotFittedError` (also a `RuntimeError`) |
| `utils.py` | The boundary validators `as_floats`, `require_non_empty`, `require_positive`; the generic `Window[T]` ring buffer; `timer` / `Elapsed` |
| `stats.py` | `summarize` → frozen `Summary`, `quantile`, `zscores`, streaming `RunningStats` (Welford) |
| `smoothing.py` | `moving_average`, `exponential_moving_average`, `median_filter`, the `Padding` boundary strategies |
| `transforms.py` | `Standardize`, `MinMaxScale` (fitted), `Clip`, `MovingAverage` (stateless), `Pipeline`, `chain` |
| `cli.py` | The `mypackage` console script (`summarize`, `smooth` subcommands), wired through `[project.scripts]` |

### Boundaries

<!-- template-example: this list only; replace it with your runtime dependencies, who builds on the package, and what it must never import. -->

- **No runtime dependencies** (`dependencies = []` in `pyproject.toml`): everything is the standard library. A new runtime dependency needs discussion; an optional one goes behind an extra and an in-function import that names the extra when it is missing.
- **The CLI is a thin shell.** `cli.py` parses arguments and formats output; the computation lives in the library modules, which never import `cli`.

Whatever the package, dependencies point one way. When other projects build on it, list them here and keep their imports working: a rename or removal of a public name goes through a deprecation (the old name keeps working with a `DeprecationWarning` that names the replacement for at least one release).

## What it is built on

<!-- template-example: one row per library your package uses, with the rule it brings. -->

Each foundation brings a rule. Breaking one runs fine on your example and fails in someone else's pipeline or in CI.

| Foundation | The package uses it for | The rule it brings |
|---|---|---|
| **Python ≥ 3.12 standard library** | PEP 695 `type` aliases and generics (`class Window[T]`), `typing.Self`, `enum.StrEnum`, `math.fsum`, `statistics.median` | No runtime dependency for what the standard library already does. |
| **`dataclasses`** | `Summary`, every transform, `RunningStats`, `Elapsed` | `slots=True` always; `frozen=True` for values and stateless objects; a fitted transform keeps what it learned in `init=False` fields with a trailing underscore (`mean_`) defaulting to `None`; validation in `__post_init__`. |
| **`typing.Protocol`** (`runtime_checkable`) | `Transform`, `Fittable` | Structural, not nominal: no base class, no registry. `isinstance` only checks that the attribute exists, so code that relies on a protocol also checks the attribute is callable (as `Pipeline.__post_init__` does). |
| **`argparse`** | `mypackage.cli` | Output goes to the injected streams, `main` returns an exit status instead of calling `sys.exit`. |
| **pytest + `--doctest-modules`** | The suite and every docstring `Examples:` block | An example's expected output is what the code prints; a stale example is a failing test. |

## Reuse before you write

Before writing a helper, a validator, a statistic, a filter or a transform, find out whether it exists:

1. **Search the capability index.** [`docs/api/capabilities.md`](docs/api/capabilities.md) lists every public name once, at its public home, grouped by the module that defines it, with a one-line summary. It is generated (`uv run poe capabilities`) and checked by `tests/test_capabilities.py`, so it is current.
2. **Search the private helpers** in the table below: several invariants live in exactly one place each.
3. **If it is missing, add it to the module that owns the concept**, at the lowest layer every caller can reach, not inline in the one caller you have today; a helper two modules need goes into `utils.py`.
4. **One object, one name.** A name means one thing across the package and its submodules (`tests/test_capabilities.py`); a deliberate exception is listed with its reason in `ALLOWED_SHARED_NAMES` in `scripts/capabilities.py`.

<!-- template-example: this table maps hand-rolled code to the example's names; rebuild it for your package. -->

| You are about to write… | Use instead |
|---|---|
| `[float(v) for v in series]`, an emptiness check, an `if window <= 0` guard | `as_floats`, `require_non_empty`, `require_positive` (`mypackage.utils`); they raise `ValidationError` / `EmptySeriesError` with the uniform `'name' must …` message |
| A mean, variance, min / max or median loop | `summarize` (→ `Summary`, with `std`, `spread`, `to_dict()`); `quantile`; `zscores`; one value at a time, `RunningStats` |
| `math.fsum(values) / len(values)` that must survive overflow | `stats._mean` (rescales by a power of two when the sum overflows) |
| A rolling mean, rolling median or exponential smoother | `moving_average`, `median_filter`, `exponential_moving_average` |
| Padding a series at its ends | the `Padding` strategies (`edge`, `zero`, `reflect`, `none`) through `smoothing._pad`, `_reflect_index`; `_validate_window`, `_as_padding` for the arguments |
| A `collections.deque(maxlen=…)` buffer | `Window[T]` |
| `(x - mean) / std`, min–max scaling, clamping | `Standardize` (with `inverse`), `MinMaxScale`, `Clip` |
| A list of steps applied in order, a base class for "things with `apply`" | `Pipeline`, `chain`, a pipeline composed with the `or` operator (`Pipeline.__or__`); satisfy the `Transform` protocol structurally |
| A new exception for bad input or a missing `fit` | `ValidationError`, `EmptySeriesError`, `NotFittedError` |
| `time.perf_counter()` bookkeeping | `timer()` → `Elapsed` |
| Reading numbers from a file or stdin for the CLI, a non-negative-int option | `cli._read_values` (finite values only), `cli._non_negative_int` |

## The contracts

<!-- template-example: the contracts below are the example's extension points; replace them with your package's. -->

Everything composes because it keeps these contracts; the tests in "What enforces them" check most of them.

### 1. Public functions

- **Validate once, at the boundary.** A public function coerces and checks its inputs through the `utils` validators (`require_non_empty(series)`, `require_positive(window, name="window")`) and raises `ValidationError` (or `EmptySeriesError`) whose message names the argument: `'window' must be a positive integer, got 0`. Internal helpers assume well-formed data.
- **Pure.** Return a **new** `list[float]` and never mutate the input (the `SMOOTHERS` tests in `tests/test_smoothing.py`, `test_does_not_mutate_input` in the stats and transforms tests). Any `collections.abc.Sequence` of numbers is accepted (`Series`).
- **Honest numerics.** `math.fsum` for sums; an overflow becomes `inf`, not an exception, when the true result is not representable (see `summarize`); degenerate input (a constant series to standardise, `ddof` ≥ the count) raises `ValidationError` rather than returning `nan`.
- **Documented.** Google-style docstring in the `python-style` layout: a one-line summary, the defining equation in Unicode (`eman-style-math`, no `$…$` in docstrings), `Algorithm:` for a function that implements a method, `Args:` (meaning and symbol, no type), `Returns:`, `Raises:` and an executable `Examples:` block.

### 2. Transforms

- A transform satisfies `Transform`: `apply(series) -> list[float]`, returning a new list. No base class.
- **Stateless** (`Clip`, `MovingAverage`): `@dataclass(frozen=True, slots=True)`; `fit` returns `self` unchanged; `fit_apply` is `apply`.
- **Fitted** (`Standardize`, `MinMaxScale`): `@dataclass(slots=True)`; learned state in `init=False` trailing-underscore fields defaulting to `None`; `fit(series)` validates, sets them and returns `self` (so it chains); `apply` before `fit` raises `NotFittedError("<Name> is not fitted; call .fit(series) first")`, never a silent identity; `fit_apply` is `fit(series).apply(series)`.
- Constructor arguments that can be checked without data are validated in `__post_init__` (`ValidationError`), as `MinMaxScale`, `Clip` and `MovingAverage` do; the rest are checked by `fit`.
- `Pipeline` fits each step on the previous step's output, calls `fit` only when the step has a callable one, and rejects a step without a callable `apply`; `|` returns a new pipeline.

### 3. The CLI

- A subcommand is a subparser in `build_parser()` plus a `_run_<name>(args, out, stdin) -> int` handler registered in the `handlers` dict in `main`.
- Write only to the injected `out` stream; read numbers through `_read_values`; validate options with an `argparse` `type=` (as `_non_negative_int` does), so a bad option is a usage error (exit 2).
- Library errors propagate as `MypackageError`, which `main` reports as `error: …` on `err` with exit status 1 (`EXIT_ERROR`); `OSError` likewise. Never call `sys.exit` or `print` in a handler.

### 4. Exceptions

- Every error derives from `MypackageError` **and** the closest standard-library exception, so `except ValueError` keeps working. A new error goes into `exceptions.py`, is exported, and joins the parametrised lists in `tests/test_exceptions.py`.

### The public API

- Export a public name from its module's `__all__` and from `src/mypackage/__init__.py` (import and `__all__`); both stay sorted (ruff's `RUF022` and `test_all_is_sorted_and_unique` agree on the order).
- A new module declares a sorted `__all__`, joins `SUBMODULES` in `tests/test_public_api.py`, and gets its own page in `docs/api/` (a `::: mypackage.<module>` block), a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`; a name on an existing module's page is picked up by its `:::` block.
- Run `uv run poe capabilities` after any public API change.
- Link prose to the API by the **top-level** name (``[`summarize`](xref:api#mypackage.summarize)``), see "Documentation".

## What enforces them

Most rules here are tests, so CI tells you when one breaks. Read a test's docstring before changing what it checks.

<!-- template-example: the test files and registries are the example's; list your own. -->

| Test | Enforces |
|---|---|
| `tests/test_public_api.py` | `__all__` sorted, unique and importable; `SUBMODULES` is exactly the set of modules; every submodule declares a sorted `__all__`; the `py.typed` marker ships; `__version__` is dotted |
| `tests/test_capabilities.py` | `docs/api/capabilities.md` is current; no name is bound to two different objects |
| `--doctest-modules` on `src/mypackage` (`addopts`) | Every docstring `Examples:` block runs and prints what it says |
| `tests/test_smoothing.py` (`SMOOTHERS`) | Smoothers never mutate their input, return a new list, leave constants unchanged |
| `tests/test_transforms.py` (`TestProtocols`, `TestPipeline`) | Every transform satisfies `Transform` (fitted ones `Fittable`); `NotFittedError` before `fit`; sequential fitting; no mutation |
| `tests/test_exceptions.py` | The hierarchy: every error is a `MypackageError` and its stdlib base |
| `tests/test_cli.py` | Injected streams, exit codes 0 / 1 / 2, finite-only input, the `[project.scripts]` entry point |
| `tests/test_build_docs.py` | The two-tool docs pipeline: inventory parsing, anchor repair, link checking, the inventory port and the absolute nav URL in `docs/myst.yml`, the files copied to the site root (`ROOT_FILES`) |
| `tests/test_plugin_skill.py` | The downstream plugin's worked example runs and its quoted numbers hold; every name the plugin skill and reviewer mention exists |
| `[tool.coverage.report] fail_under = 90` | Coverage of `src/mypackage` in every coverage-enabled run (CI, `uv run poe test-cov`) |
| `strict_markers = true` | Only the registered tier markers (`slow`, `integration`) exist |
| ruff (`select = E, F, I, UP, B, SIM, RUF`), ty | Lint (the whole repo, notebook code cells included), format, types on `src/mypackage` and `scripts` |

## Recipes

Step-by-step recipes for the common jobs live as plain Markdown in `.claude/skills/<name>/SKILL.md` (Claude Code loads them automatically; any agent can read and follow them), and two read-only review checklists in `.claude/agents/`:

<!-- template-example: this table only; `add-function`, `add-transform`, `add-cli-command` and `mypackage-review` are the example package's, replace them with your extension points. -->

| Job | Recipe |
|---|---|
| Turn a fresh copy of the template into your project | `adopt-template` |
| Add a public function, class or module | `add-function` |
| Add a transform (the `Transform` protocol) | `add-transform` |
| Add a CLI subcommand or option | `add-cli-command` |
| Add or update a guide page (MyST prose) | `add-guide-page` |
| Add or update an example notebook | `add-notebook` |
| Verify before a PR | `pre-pr-check` |
| Review a change | `mypackage-review` (+ the read-only `.claude/agents/reuse-reviewer.md` and `correctness-reviewer.md`) |
| Write a squash commit message | `squash-commit` |
| Open or link GitHub issues | `create-gh-issue`, `link-gh-issues` (templates in `.github/ISSUE_TEMPLATE/`; `uv run poe gh-labels`, `gh-sub`, `gh-block`, `gh-show`) |

<!-- template-example: this paragraph only; the plugin describes the example package. -->

Downstream users get the package's guidance through the Claude Code plugin in `plugins/mypackage/` (published by `.claude-plugin/marketplace.json`) and `docs/llms.txt` (copied to the site root by `ROOT_FILES` in `scripts/build_docs.py`); see `docs/agents.md`. When the public API or the headline usage changes, update `plugins/mypackage/skills/series-analysis-with-mypackage/` too: `tests/test_plugin_skill.py` runs its worked example, checks the numbers its prose quotes, and checks that every `mp.X` / `mypackage.X` it and the plugin reviewer name still exists.

## Working in the repo

Always run Python tools through `uv run` (never the system Python); `git`, `ls` and other non-Python commands need no `uv run`.

```bash
uv run poe                  # list every task (they live in [tool.poe.tasks] in pyproject.toml)
uv run poe install          # uv sync --all-groups + pre-commit hooks
uv run poe test             # every test + doctests, no coverage (fast)
uv run poe test-cov         # the same with coverage and the 90 % gate (what CI runs)
uv run poe lint             # ruff check .   (entire repo)
uv run poe format           # ruff format . && ruff check --fix .
uv run poe typecheck        # ty check src/mypackage scripts (what CI runs)
uv run poe check            # lint + format-check + typecheck + test-cov (everything CI checks)
uv run poe capabilities     # regenerate docs/api/capabilities.md
uv run poe docs             # build both docs halves into public/ and check every link
uv run poe docs-check       # strict validation of both halves, no themed render
uv run poe docs-api         # Zensical API reference only (zensical build --strict)
uv run poe docs-serve       # build, then serve public/ at :8000
uv run poe precommit        # every pre-commit hook on every file
uv run poe build            # wheel + sdist into dist/
```

Extra arguments after a task name are passed to its command (`uv run poe test tests/test_stats.py -k summarize`).

`addopts` turns coverage on, and a partial run never reaches the 90 % gate, so pass `--no-cov` when running part of the suite:

```bash
uv run pytest --no-cov tests/test_stats.py::TestSummarize::test_known_values -v
```

### Test tiers

The markers are registered in `pyproject.toml` (`strict_markers = true`):

- **Unmarked (fast):** unit tests and doctests, each well under a second.
- **`@pytest.mark.slow`:** a test over ~1 s (a large sweep, a long fit). For a parametrised test, mark only the expensive cases (`pytest.param(..., marks=pytest.mark.slow)`).
- **`@pytest.mark.integration`:** end-to-end workflows, optional dependencies, external tools.

CI ("Tests", `ci.yml`) runs every tier with coverage on Python 3.12 and 3.13 (`uv run pytest -v`). Select the fast tier with `uv run pytest -m "not slow and not integration"`; when the slow tiers grow, give them a CI job of their own rather than dropping them.

### Tests that assert on random draws

<!-- template-example: this paragraph only. The example draws no random numbers; its tests use fixed fixtures in tests/conftest.py. -->

`mypackage` draws no random numbers; its tests use the fixed series in `tests/conftest.py`.

- **Incidental randomness** (any draw would do): seed it (`random.Random(0)`, `numpy.random.default_rng(0)`, a fixed key) so a failure reproduces.
- **Sampling behaviour under test** (a Monte Carlo estimate, a statistical property): bound the statistic by its own sampling distribution (a number of standard errors, a test's p-value floor), not by a tolerance tuned on one draw, and say in a comment where the bound comes from.

### Before every commit

All of these must pass, from the repo root:

1. `uv run poe test`: zero failures (doctests included).
2. `uv run --group lint ruff check .`: the **entire** repo, which includes `tests/`, `scripts/` and the code cells of `docs/notebooks/*.ipynb`. Never lint a subdirectory: `ruff check src/mypackage/` misses errors CI catches.
3. `uv run --group lint ruff format --check .`
4. `uv run poe typecheck` (`ty check src/mypackage scripts`, as CI runs it).
5. After changing a public API: `uv run poe capabilities`, and the `docs/api` page.
6. After changing docs, docstrings, `mkdocs.yml` or `docs/myst.yml`: `uv run poe docs` (or at least `uv run poe docs-api` when only docstrings changed).
7. After changing a dependency: `uv lock`, and commit `uv.lock`.

## Coding principles

1. **Think before coding.** State assumptions; if a request has several readings, name them instead of picking one silently; if something is unclear, stop and ask; push back when a simpler approach exists.
2. **Simplicity first.** The minimum code that solves the problem: no speculative features, no single-use abstractions, no configurability nobody asked for, no error handling for impossible cases.
3. **Surgical changes.** Touch only what the task needs; match the existing style; don't refactor or add docstrings to code you didn't change; remove only what your change made unused, and mention (don't delete) unrelated dead code.
4. **Goal-driven.** Turn the task into a check (a failing test, a reproduced bug, a closed form to match) and loop until it passes; for multi-step work, state the plan with a check per step.

These are the `karpathy-coding-principles` skill in four lines. When you suggest a change, propose the code; when you fix a bug or add a feature, add the test that checks it.

## House style

### Python

`python-style` is the house style. Its general rules hold in every package built from this template:

- Python 3.12+, `from __future__ import annotations` in every module, type hints on every public function, `X | None` over `Optional`, specific exceptions with `raise … from …`, pure functions with side effects isolated and explicit.
- Every module opens with a docstring saying what it holds, in one or two sentences.
- Docstrings follow the layout in "The contracts" above; `Examples:` gives two or three cases (typical, edge, composition), each building its own small data, and `References:` names the paper when one defines the method.
- Comments go above the line, never beside it. A mathematical line has its Unicode equation above it (`# x̄ = (1/N) ∑ᵢ xᵢ`), a long function is split into stages with step headers (`# --- whiten ---`), and every other comment says why, not what.
- Names are descriptive (`posterior_precision`, not `Lam`); the symbol lives only in the equation comment.
- One operation per line, no nested calls that hide an intermediate result, 88 columns.
- One package base error; subclasses are named by what failed.
- Extract only real duplicates (code that must change together); code that merely looks alike stays separate.

Its JAX/Equinox rules apply when the package uses JAX: logic under `pkg/_src/` with public modules that only re-export, jaxtyping with the shared dim vocabulary, einx instead of `axis=` / `.T` / `reshape` / `[:, None]`, lineax for solves, arrays in `eqx.Module`s with `__check_init__` and `eqx.error_if`, `key` keyword-only and last, no silent dtype casts, `get_config()` / `from_config()`, and the four tests per public forward function (closed-form oracle, hypothesis properties, gradient check, transform invariance).

<!-- template-example: this paragraph only; it records where the stdlib example departs from python-style on purpose. -->

The bundled example is pure standard library, so it keeps `dataclasses` (`slots=True`) where `python-style` uses `eqx.Module`, makes every error inherit the closest standard-library exception as well as the package base (an open question in `python-style`, settled here), and lets doctests print exact floats such as `2.5`.

**CLIs and scripts** use `cyclopts` for arguments and `loguru` for output; never argparse, click, typer, progress `print`s or stdlib `logging`. For a repo script they are in the `dev` group already; for a package CLI they are runtime dependencies and need the usual discussion. Two exceptions remain: the example's `mypackage` CLI uses `argparse` to keep the package dependency-free (`template-example`), and `scripts/capabilities.py` still uses `argparse` and `print`; move it over when you next change it (`scripts/build_docs.py` shows the pattern: a `cyclopts.App` with `@app.default`, and `loguru` for every message).

### Maths notation

Maths follows `eman-style-math` in every format: the letter says what kind of object a symbol is, the subscript says which variable it belongs to, and every symbol is defined, with its shape, before it is used.

- **Code and docstrings:** Unicode only (x̄, σ², Σ⁻¹, ∑, 𝔼, ⊙), never `$…$`; identifiers stay ASCII.
- **MyST prose and MkDocs pages:** `$…$` and `$$…$$` with the kind-named macros (`\vect{x}`, `\gvect{\mu}`, `\mat{K}`, `\gmat{\Sigma}`). The macros are installed in `docs/myst.yml` (`math:`) and `docs/api/javascripts/mathjax.js`; regenerate both with the skill's `scripts/emit_macros.py` when the macros change.
- **GitHub issues and PRs:** expand the macros first (`scripts/expand_macros.py`), display maths in a `math` fence.
- **Everywhere:** each `$…$` expression on one source line; no spacing or sizing macros (`\,`, `\;`, `\quad`, `\qquad`, `\big`, `\tfrac`, `\boxed`, `\cdots`, `\operatorname`). Check a page with `uv run --with cyclopts --with loguru --with pyyaml python ~/.claude/skills/eman-style-math/scripts/check_math.py <file> --target myst` (the skill's scripts need those three packages, which this repo does not install).

### Prose and diagrams

- Markdown prose is never hard-wrapped: one line per paragraph or list item in docs, skills and agent instructions; issue bodies follow `report` (one sentence per line); PR text, one line per paragraph. Don't rewrap untouched text in a file you edit.
- Docs, READMEs and release notes written in the maintainer's voice follow `eman-prose-tech` (the docs register): plain words, "we" for the maintainers, parentheses rather than em dashes.
- Pseudocode in design pages, issues and READMEs follows `code-sketch`: a shapes header, the equation above each line, an aligned shape column and a status marker (`today (pkg)`, `proposed (X)`, `user`) on every call into the stack.
- Never draw ASCII box-and-arrow diagrams; use Mermaid (both docs halves render it). Directory trees stay plain text.

## Personal skills

The maintainer's skills live in [jejagents](https://github.com/jejjohnson/jejagents) and install into `~/.claude/skills/` with its `install.sh`. Use them when they are installed; the sections above carry what an agent without them needs. Where a skill and this file disagree, this file wins: it records the template's deliberate departures.

| Job | Skill |
|---|---|
| Write, review or refactor Python | `python-style` |
| Maths in docstrings, docs pages, issues or PRs | `eman-style-math` |
| Pseudocode in a design page, issue or README | `code-sketch` |
| Prose in the maintainer's voice (docs, README, release notes, PR descriptions) | `eman-prose-tech` |
| Write up a research, feature, bug, design, epic or chore report | `report`, then file it here with `create-gh-issue` |
| File or link issues in another of the maintainer's repos | `gh-issue` (in this repo, `create-gh-issue` and `link-gh-issues` follow it) |
| A derivation walkthrough for the docs | `derivation`, with `derivation-myst` for a page in `docs/` |
| A learning explainer page | `explainer` |
| General coding discipline | `karpathy-coding-principles` ("Coding principles" above) |

## Git, commits and pull requests

- **Never** push to or merge into `main` unless explicitly told to ("push to main", "merge to main"). Work on a feature branch and commit locally; never run `git push` unless asked. "Merge the branch" means push the feature branch, not merge into `main`. Confirm before any action that affects a shared branch.
- Commit messages and PR titles follow [Conventional Commits](https://www.conventionalcommits.org/) with a lowercase subject (`feat(stats): add …`); CI validates PR titles (`conventional-commits.yml`). Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`. Breaking changes use `!` and a `BREAKING CHANGE:` footer.
- **No AI attribution.** No `Co-Authored-By:` trailer for an AI on any commit and no "Generated with …" line in a PR description, unless the maintainer asks for one; keep human co-authors' trailers.
- **PR descriptions are for a human who did not watch the work:** why first, a small before/after table, what stays the same, one or two bullets on how it was tested. Label behaviour as "Today (bug): …" and "After this PR: …". Link review comments rather than naming reviewers.
- Write one `Closes #N` line per issue.
- Releases are cut by release-please (`release-please-config.json`): it bumps the version and writes `CHANGELOG.md` from the `feat`, `fix`, `perf` and `revert` commits; don't bump versions by hand.
- **Never replace or remove an existing PR title or description.** An agent called for a small follow-up tends to write a fresh description scoped to its own work and silently discard the rest: read the existing description first and only append checklist items or update their status. Change the title only to fix a Conventional Commits violation.
- Code review follows [`CODE_REVIEW.md`](CODE_REVIEW.md); issues follow the label taxonomy and epic model in [`docs/contributing.md`](docs/contributing.md).

### Pull Request Review Comments

Fix a review comment in code, push the fix to the PR branch, then resolve its thread; don't post a reply comment. Don't resolve threads you didn't address.

```bash
# 1. List the review threads and their IDs
gh api graphql -f query='
  query($owner: String!, $repo: String!, $pr: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $pr) {
        reviewThreads(first: 100) {
          nodes { id isResolved comments(first: 1) { nodes { body path line } } }
        }
      }
    }
  }' -f owner=OWNER -f repo=REPO -F pr=PR_NUMBER

# 2. Resolve an addressed thread
gh api graphql -f query='mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) { thread { isResolved } } }' \
  -f threadId=THREAD_ID
```

GraphQL is blocked in Claude sessions; there, and whenever the `gh` CLI is unavailable, use the GitHub MCP tools for the same operations (they can resolve review threads).

## Documentation

The docs are built by **two tools** and deployed as one site (`docs/README.md` has the rationale); `docs.yml` builds them on every PR and push to `main`, and `pages.yml` deploys `public/` to GitHub Pages from `main`.

| Half | Tool | Source | Deployed at |
|---|---|---|---|
| Prose — home, guides, example notebooks | mystmd | `docs/*.md`, `docs/guide/`, `docs/notebooks/` (toc in `docs/myst.yml`) | `/` |
| API reference | Zensical + mkdocstrings | `docs/api/` (`docs_dir` in `mkdocs.yml`) | `/reference/` |

`uv run poe docs` runs `scripts/build_docs.py`: it builds the API with `zensical build --strict` (Zensical reads `mkdocs.yml`), serves `site/` on port 8910 so mystmd can read its `objects.inv` (mystmd loads inventories only over http), runs `myst build --html --strict`, rewrites the localhost API URLs to `/reference/`, repairs anchors lowercased by mystmd's `$`-expansion bug, and fails if any internal link in the assembled `public/` site does not resolve. mystmd is a Node CLI (`npm install -g mystmd`), not a uv dependency; if its theme download is blocked, `docs/README.md` says how to build against a local clone of the theme.

- **Prose is MyST Markdown**, not MkDocs-Material syntax: `:::{note}`, `:::{tab-set}`, `:::{dropdown}` directives, not `!!!` / `===` / `???`.
- **Link to the API** with the `xref:` protocol and the **top-level** exported name: ``[`summarize`](xref:api#mypackage.summarize)``, not `mypackage.stats.summarize` (mystmd lowercases `$`-abbreviated anchors; see `docs/README.md`). A missing target fails `myst build --strict`.
- **URLs are flat**: a page's URL is its basename, with `_` turned into `-` (`guide/quickstart.md` → `/quickstart/`, `notebooks/smoothing_deep_dive.ipynb` → `/smoothing-deep-dive/`), so keep basenames unique across `docs/`, `docs/guide/` and `docs/notebooks/`; frontmatter `slug:` is ignored.
- **Site-nav URLs stay absolute** (`site.nav` in `docs/myst.yml`): the theme prefixes root-relative ones with `BASE_URL` a second time (`tests/test_build_docs.py` checks it).
- **API pages** (`docs/api/*.md`) are MkDocs-format pages, built by Zensical (which runs only the plugins it supports and ignores the rest silently; check its plugin list before adding one): prose plus `:::` blocks, `!!!` admonitions, listed in the `nav` of `mkdocs.yml`; docstrings are Google style (`docstring_style: google`) with Unicode maths; maths in the API pages themselves goes through MathJax (`pymdownx.arithmatex`), with the `eman-style-math` macros loaded by `docs/api/javascripts/mathjax.js`.
- **Notebooks** in `docs/notebooks/` are committed as executed `.ipynb` files, authored as jupytext percent `.py` and deleted after conversion; mystmd renders the stored outputs without re-executing, so each must be listed in the `docs/myst.yml` toc. Figures render inline with `plt.show()`: no `savefig`, no committed PNGs. `notebooks/` at the root holds scratch notebooks, outside the docs. Full standards: `.github/instructions/docs-examples.instructions.md`.

## Plans

Plans and scratch design notes go in `.plans/` (gitignored, never committed); track work in GitHub issues. Draft a whole wave of issues from `docs/templates/wave-backlog.md` (see `docs/contributing.md`).
