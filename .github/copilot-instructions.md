# Copilot Instructions

Read [`AGENTS.md`](../AGENTS.md) at the repository root first: it is the single source of truth for every coding agent working here (the module map, "reuse before you write", the contracts, the tests that enforce them, commands, the pre-commit checklist, the two-tool docs, git and PR rules).

The essentials, in case you only read this file:

- One package, `src/functax/`, at Core tier 0 of the GeoStack: it imports only NumPy, JAX, Equinox, lineax, einx and jaxtyping, never another stack library. Logic lives in `functax._src.<area>`, and each public area module (`geometry`, `basis`, `diffops`, `transform`, `interp`, `project`, `quad`) only re-exports it. Search [`docs/api/capabilities.md`](../docs/api/capabilities.md) before writing a helper.
- Keep the contracts in `AGENTS.md`:
  - **two installs**: the base depends on NumPy alone, so `functax`, `functax.geometry` and `functax.geometry.affine` import with JAX absent; every other module calls `require_jax(__name__)` first, which raises `MissingExtraError` naming `functax[jax]`;
  - **public functions and modules** validate once, at the boundary, raise `ValidationError` naming the argument, never mutate their input, preserve the input dtype, and trace under `jit`, `vmap` and `grad`;
  - **linear read-outs** pass the dot-product test ⟨Hx, r⟩ = ⟨x, Hᵀr⟩ and equal their SciPy oracle to 1e-12 relative in float64;
  - every exception derives from `FunctaxError` and its stdlib base.

The rules for every project:

- Every docstring `Examples:` block runs (`--doctest-modules`): the expected output must be what the code prints.
- Before committing, from the repo root: `uv run poe test`, `uv run --group lint ruff check .`, `uv run --group lint ruff format --check .`, `uv run poe typecheck`; `uv run poe capabilities` after a public API change. Lint the whole repo, never a subdirectory.
- When you suggest a change, propose the code; when you fix a bug, add the test that reproduces it. Don't nitpick what ruff and the formatter handle.
- House style ("House style" in `AGENTS.md`): the maintainer's `python-style` for Python (comments above the line, a Unicode equation above each mathematical line, descriptive names), `eman-style-math` for maths (Unicode in code and docstrings, macros in MyST and MkDocs pages), `cyclopts` and `loguru` for new CLIs and scripts, Markdown never hard-wrapped, Mermaid rather than ASCII diagrams.
- Partial test runs need `--no-cov` (the 90 % coverage gate fails a subset).
- Never push or merge to `main` unless told to; never replace an existing PR description, only append to it.
- No AI attribution: no `Co-Authored-By:` trailer for an AI and no "Generated with …" line, unless asked. PR descriptions open with why, then a small before/after table; one `Closes #N` line per issue.
- Step-by-step recipes (add a scheme, guide page or notebook; pre-PR check; review) are plain Markdown in `.claude/skills/<name>/SKILL.md`, and the two review checklists in `.claude/agents/`; follow them as written.
- Path-scoped standards live in `.github/instructions/`; code review follows [`CODE_REVIEW.md`](../CODE_REVIEW.md).
