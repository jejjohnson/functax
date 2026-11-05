# Copilot Instructions

Read [`AGENTS.md`](../AGENTS.md) at the repository root first: it is the single source of truth for every coding agent working here (the module map, "reuse before you write", the contracts, the tests that enforce them, commands, the pre-commit checklist, the two-tool docs, git and PR rules). Passages marked `<!-- template-example: … -->` there describe the bundled example package and are rewritten when the template is adopted.

The essentials, in case you only read this file:

<!-- template-example: the module map and contracts of the example package. -->

- One package, `src/mypackage/`, dependency-free, with modules that import only downward: `_typing` → `exceptions` → `utils` → `stats` / `smoothing` → `transforms` → `cli`. Search [`docs/api/capabilities.md`](../docs/api/capabilities.md) before writing a helper.
- Keep the contracts in `AGENTS.md`:
  - **public functions** validate once, at the boundary, with the `utils` validators, raise `ValidationError` naming the argument, and return a new list without mutating their input;
  - **transforms** satisfy the `Transform` protocol structurally; fitted ones raise `NotFittedError` before `fit`, and `fit` returns `self`;
  - **CLI** subcommands write to the injected streams and return an exit status; library errors become exit 1, usage errors exit 2;
  - every exception derives from `MypackageError` and its stdlib base.

The rules for every project:

- Every docstring `Examples:` block runs (`--doctest-modules`): the expected output must be what the code prints.
- Before committing, from the repo root: `uv run poe test`, `uv run --group lint ruff check .`, `uv run --group lint ruff format --check .`, `uv run poe typecheck`; `uv run poe capabilities` after a public API change. Lint the whole repo, never a subdirectory.
- When you suggest a change, propose the code; when you fix a bug, add the test that reproduces it. Don't nitpick what ruff and the formatter handle.
- House style ("House style" in `AGENTS.md`): the maintainer's `python-style` for Python (comments above the line, a Unicode equation above each mathematical line, descriptive names), `eman-style-math` for maths (Unicode in code and docstrings, macros in MyST and MkDocs pages), `cyclopts` and `loguru` for new CLIs and scripts, Markdown never hard-wrapped, Mermaid rather than ASCII diagrams.
- Partial test runs need `--no-cov` (the 90 % coverage gate fails a subset).
- Never push or merge to `main` unless told to; never replace an existing PR description, only append to it.
- No AI attribution: no `Co-Authored-By:` trailer for an AI and no "Generated with …" line, unless asked. PR descriptions open with why, then a small before/after table; one `Closes #N` line per issue.
- Step-by-step recipes (adopt the template, add a function, transform, CLI command, guide page or notebook; pre-PR check; review) are plain Markdown in `.claude/skills/<name>/SKILL.md`, and the two review checklists in `.claude/agents/`; follow them as written.
- Path-scoped standards live in `.github/instructions/`; code review follows [`CODE_REVIEW.md`](../CODE_REVIEW.md).
