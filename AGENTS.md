# AGENTS.md

Standing instructions for **every** coding agent working in this repository (Claude Code, Copilot, Codex, Gemini, …). This is the single source of truth: `CLAUDE.md` and `.github/copilot-instructions.md` point here.

The house style (Python, maths notation, prose, pseudocode, reports) comes from the maintainer's personal skills in [jejagents](https://github.com/jejjohnson/jejagents); "House style" and "Personal skills" below say which skill covers what and carry the essentials, so an agent without those skills installed can still follow them.

## What this repo is

functax is the GeoStack's function-approximation library. It is built around one representation, an expansion f̂(x) = ∑ⱼ cⱼ φⱼ(x; θ) with coefficients c and basis functions φ, and five operations on it: interpolate, differentiate, integrate, project and transform. An interpolation plan exposes its read-out H, the adjoint Hᵀ and its derivative, so one object samples a field at observations and sends residuals back to the grid.

The design lives in the geostack repository, in [`docs/design/libraries/functax.md`](https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/functax.md) and [`functax_details.md`](https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/functax_details.md). The backlog is six wave epics, [#8](https://github.com/jejjohnson/functax/issues/8) to [#13](https://github.com/jejjohnson/functax/issues/13), under [geostack#129](https://github.com/jejjohnson/geostack/issues/129). The issues are newer than the design page where the two disagree, so read the live issue before starting work.

People build on this package, so **the package is a library of primitives**: new code composes what is here, and anything genuinely new lands where the next person will find and reuse it.

It is one package, `src/functax/`. Logic lives in `functax._src.<area>`, and each area has one public module that only re-exports it. The top-level `functax.__all__` holds the exceptions and `__version__`; everything else is reached through its area module (`functax.geometry`, `functax.interp`, …). Imports only point down this table, and each area module appears with the first feature issue that needs it:

| Module | Owns | Lands in |
|---|---|---|
| `_typing.py` | PEP 695 aliases shared across areas (`Shape`) | #14 |
| `exceptions.py` | `FunctaxError` and its subclasses `ValidationError` (also a `ValueError`) and `MissingExtraError` (also an `ImportError`) | #14 |
| `_src/_backend.py` | `require_jax` and `JAX_EXTRA`, the import-time check for the `functax[jax]` extra | #14 |
| `geometry` | CRS-free axes, grids and point sets; the sphere; the affine raster kernels in `geometry.affine` | Wave 0 (#15, #16, #17) |
| `basis` | the basis zoo and Fourier features, moved in from geonnax | Wave 1 (#19, #21) |
| `diffops` | derivatives of continuous fields | Wave 1 (#22) |
| `transform` | spherical-harmonic transform, NUFFT, Chebyshev transforms | Waves 1 and 4 (#23, #46) |
| `interp` | `Stencil`, `Scheme`, `Plan` and every interpolation scheme | Waves 2 and 3 |
| `project` | least-squares, Galerkin and variational fits, and gap fill | Wave 3 (#42) |
| `quad` | quadrature rules on intervals and the sphere | Wave 4 (#44, #45) |

### Boundaries

- **Two installs** (decision 17 in the design page, [#48](https://github.com/jejjohnson/functax/issues/48)). The base install depends on NumPy alone; JAX, Equinox, lineax, einx and jaxtyping are the `functax[jax]` extra. `functax`, `functax.geometry` (its `__init__`) and `functax.geometry.affine` must import with JAX absent, so NumPy-only consumers (geoarray, geoops, xrops) can call the affine kernels. Every other module calls `require_jax(__name__)` before its first JAX import, which raises `MissingExtraError` naming the extra.
- **Core tier 0.** functax imports only NumPy, JAX, Equinox, lineax, einx and jaxtyping. It never imports pipekit, kernellib, gaussx, finitevolX, geoarray or any other stack library, and it neither imports nor depends on interpax or orthax (decision 14): their outputs appear only as references in tests. A kernel or a structured solver arrives as an argument (a callable or a lineax operator), never as an import.
- **The fit may use the host; the apply must trace** (decision 1). Building a plan may use NumPy (and, once a scheme needs it, SciPy, which needs discussion as a runtime dependency); applying one runs in JAX under `jit`, `vmap` and `grad`.
- **Coordinates are CRS-free.** Geometry takes coordinates already in one CRS; the CRS stays a carrier concern of geoarray and xarray.

Dependencies point one way. The consumers are geotoolz (geoarray, geoops), xrtoolz, geonnax, vardax, kernellib, pyrox, filterax, sesmax, plumax, radtrax, spectraldiffx, finitevolX and gauss_flows (their issues sit under [#13](https://github.com/jejjohnson/functax/issues/13)). Keep their imports working: a rename or removal of a public name goes through a deprecation (the old name keeps working with a `DeprecationWarning` that names the replacement for at least one release).

## What it is built on

Each foundation brings a rule. Breaking one runs fine on your example and fails in someone else's pipeline or in CI.

| Foundation | The package uses it for | The rule it brings |
|---|---|---|
| **NumPy** | the base install: the affine raster kernels, and host-side fits | The affine kernels take the array namespace of their input and import JAX only when a JAX array arrives (decision 16). No NumPy call inside a traced apply. |
| **JAX** | every apply | Traceable: static shapes, no Python control flow on traced values, no host round-trips. `jit`, `vmap` and `grad` agree with an eager loop. |
| **Equinox** | geometry types, stencils, plans and bases as `eqx.Module`s | Arrays are always leaves; configuration is `eqx.field(static=True)`. Field checks in `__check_init__`, checks on traced values with `eqx.error_if`. |
| **lineax** | the lineax view of a plan (`operator()`), and every solve | Solve, never invert; never `jnp.linalg.solve` or `inv`. |
| **einx** | reductions, contractions, broadcasts and rearranges | No `axis=`, `.T`, `reshape` or `[:, None]`. |
| **jaxtyping** | array annotations | Named dims from the shared vocabulary ("House style"); checked at runtime in tests through the import hook in `conftest.py`. |
| **SciPy** (test group) | the oracle every scheme is tested against | A test oracle, not a runtime dependency. |
| **pytest + `--doctest-modules`** | the suite and every docstring `Examples:` block | An example's expected output is what the code prints; a stale example is a failing test. |

## Reuse before you write

Before writing a helper, a validator, a sampler, a basis or a rule, find out whether it exists:

1. **Search the capability index.** [`docs/api/capabilities.md`](docs/api/capabilities.md) lists every public name once, at its public home, grouped by the module that defines it, with a one-line summary, and the top-level API of JAX, Equinox, lineax and einx. It is generated (`uv run poe capabilities`) and checked by `tests/test_capabilities.py`, so it is current.
2. **Search the private helpers** in the table below: several invariants live in exactly one place each.
3. **If it is missing, add it to the area that owns the concept**, at the lowest layer every caller can reach, not inline in the one caller you have today; a helper two areas need goes into `functax._src`.
4. **One object, one name.** A name means one thing across the package and its submodules (`tests/test_capabilities.py`); a deliberate exception is listed with its reason in `ALLOWED_SHARED_NAMES` in `scripts/capabilities.py`.

| You are about to write… | Use instead |
|---|---|
| `try: import jax` with a hand-written message | `require_jax(__name__)` (`functax._src._backend`), which raises `MissingExtraError` naming `functax[jax]` |
| A new exception for bad input or a missing extra | `ValidationError`, `MissingExtraError` |
| `tuple[int, ...]` for a grid shape | `Shape` (`functax._typing`) |
| Reading an `Affine`, six numbers or a 3 × 3 matrix into (a, b, c, d, e, f) | `as_transform` (`functax._src.geometry._transform`, NumPy only) |
| A malformed axis, grid or point set | `GeometryError` (a `ValidationError`) |
| `jnp.linalg.solve`, `jnp.linalg.inv`, a hand-rolled CG | lineax (`lx.linear_solve` with the right solver) |
| The transpose of a linear map | the plan's hand-written adjoint, checked against `jax.linear_transpose` by the dot-product test |
| A SciPy comparison, a dot-product test, a `jit`/`vmap` agreement check or a finite-difference gradient check in a test | `assert_matches_oracle`, `assert_adjoint`, `assert_transpose_matches`, `assert_traces`, `assert_grad_matches_fd` (`tests/_harness.py`; "Testing a scheme" in `docs/contributing.md`) |
| `x.T`, `x.reshape(...)`, `x[:, None]`, `jnp.sum(x, axis=…)` | `einx.rearrange`, `einx.sum` and friends |
| Bilinear weights, a cell locate, a nearest-neighbour sampler | nothing yet: `Stencil`, `locate` and `Plan` land in Wave 2 (#24, #25, #26). Don't add a private copy in the meantime. |

## The contracts

Everything composes because it keeps these contracts; the tests in "What enforces them" check most of them.

### 1. Public functions and modules

- **Validate once, at the boundary.** A public function or module checks its inputs and raises `ValidationError` whose message names the argument: `'shape' must have one positive length per axis, got (0, 3)`. Static checks go in `__check_init__`; checks on traced arrays use `eqx.error_if`, so they survive `jit`. Internal helpers assume well-formed data.
- **Pure and traceable.** Never mutate an input. An apply runs under `jit`, `vmap` and `grad` with static shapes.
- **Precision is the caller's.** Preserve the input dtype and never cast silently; create constants in the input's dtype (`jnp.asarray(c, dtype=x.dtype)`) or as Python scalars.
- **Documented.** Google-style docstring in the `python-style` layout: a one-line summary, the defining equation in Unicode (`eman-style-math`, no `$…$` in docstrings), `Algorithm:` for a function that implements a method, `Args:` (meaning and symbol, no type), `Returns:`, `Raises:` and an executable `Examples:` block whose outputs are shapes, `allclose` booleans or reprs, never raw floats.

### 2. The two installs

- A module in the NumPy-only base (`functax`, `functax.geometry`, `functax.geometry.affine`, `functax._src._backend`, `functax._src.geometry._affine`, `functax._src.geometry._transform`) imports nothing from the `jax` extra at module level. A JAX type it needs to re-export is reached lazily, through a module `__getattr__`.
- Every other module calls `require_jax(__name__)` before its first JAX import.
- A module that joins the base adds itself to `BASE_MODULES` in `tests/test_backend.py` and to the imports in the `base-install` job of `ci.yml`.

### 3. Linear read-outs (from Wave 2)

- A linear scheme passes the dot-product test, ⟨Hx, r⟩ = ⟨x, Hᵀr⟩ to 1e-12 relative in float64, on random x and r (gate G2-adjoint).
- It equals its SciPy oracle (`scipy.ndimage.map_coordinates`, `RegularGridInterpolator`, or the one its issue names) to 1e-12 relative in float64 on interior targets (G2-oracle).
- `jit`, `vmap` and `grad` agree with an eager loop to 1e-12 relative (G2-trace).

### 4. Exceptions

- Every error derives from `FunctaxError` **and** the closest standard-library exception, so `except ValueError` keeps working. A new error goes into `exceptions.py`, is exported, and joins the parametrised lists in `tests/test_exceptions.py`.

### The public API

- Logic lives in `functax._src.<area>`; the public area module only re-exports, from its own sorted `__all__`. The top-level `functax.__all__` holds only the exceptions and `__version__`; both stay sorted (ruff's `RUF022` and `test_all_is_sorted_and_unique` agree on the order).
- A new top-level module joins `SUBMODULES` (a subpackage joins `SUBPACKAGES`) in `tests/test_public_api.py`, and gets its own page in `docs/api/` (a `::: functax.<module>` block), a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`; a name on an existing module's page is picked up by its `:::` block.
- Run `uv run poe capabilities` after any public API change.
- Link prose to the API by the public name (``[`FunctaxError`](xref:api#functax.FunctaxError)``), see "Documentation".

## What enforces them

Most rules here are tests, so CI tells you when one breaks. Read a test's docstring before changing what it checks.

| Test | Enforces |
|---|---|
| `tests/test_public_api.py` | `__all__` sorted, unique and importable; `SUBMODULES` and `SUBPACKAGES` are exactly the modules and packages; every module declares a sorted `__all__`; the `py.typed` marker ships; `__version__` is dotted |
| `tests/test_backend.py` | `require_jax` names `functax[jax]` when any distribution of the extra is missing; the base modules never import JAX and import with JAX blocked |
| The `base-install` job in `ci.yml` | A fresh `pip install .` has no `jax`, and the base modules import there |
| `tests/test_capabilities.py` | `docs/api/capabilities.md` is current; no name is bound to two different objects |
| `--doctest-modules` on `src/functax` (`addopts`) | Every docstring `Examples:` block runs and prints what it says |
| `tests/test_exceptions.py` | The hierarchy: every error is a `FunctaxError` and its stdlib base |
| `tests/test_harness.py` | Each helper in `tests/_harness.py` passes on a correct case and fails, naming the worst index, on one perturbed by 1e-6 |
| `tests/test_build_docs.py` | The two-tool docs pipeline: inventory parsing, anchor repair, link checking, the inventory port and the absolute nav URL in `docs/myst.yml`, the files copied to the site root (`ROOT_FILES`) |
| `[tool.coverage.report] fail_under = 90` | Coverage of `src/functax` in every coverage-enabled run (CI, `uv run poe test-cov`) |
| `strict_markers = true` | Only the registered tier markers (`slow`, `integration`) exist |
| ruff (`select = E, F, I, UP, B, SIM, RUF`), ty | Lint (the whole repo, notebook code cells included), format, types on `src/functax` and `scripts` |

## Recipes

Step-by-step recipes for the common jobs live as plain Markdown in `.claude/skills/<name>/SKILL.md` (Claude Code loads them automatically; any agent can read and follow them), and two read-only review checklists in `.claude/agents/`:

| Job | Recipe |
|---|---|
| Add an interpolation scheme, a basis, a rule or a transform | `add-scheme` |
| Add or update a guide page (MyST prose) | `add-guide-page` |
| Add or update an example notebook | `add-notebook` |
| Verify before a PR | `pre-pr-check` |
| Review a change | `functax-review` (+ the read-only `.claude/agents/reuse-reviewer.md` and `numerics-reviewer.md`) |
| Write a squash commit message | `squash-commit` |
| Open or link GitHub issues | `create-gh-issue`, `link-gh-issues` (templates in `.github/ISSUE_TEMPLATE/`; `uv run poe gh-labels`, `gh-sub`, `gh-block`, `gh-show`) |

There is no downstream Claude Code plugin or `llms.txt` yet. Add them (a `plugins/functax/` plugin with a `use-functax` skill, published by `.claude-plugin/marketplace.json`) once Wave 2 gives consumers something to learn.

## Working in the repo

Always run Python tools through `uv run` (never the system Python); `git`, `ls` and other non-Python commands need no `uv run`.

```bash
uv run poe                  # list every task (they live in [tool.poe.tasks] in pyproject.toml)
uv run poe install          # uv sync --all-groups + pre-commit hooks
uv run poe test             # every test + doctests, no coverage (fast)
uv run poe test-cov         # the same with coverage and the 90 % gate (what CI runs)
uv run poe lint             # ruff check .   (entire repo)
uv run poe format           # ruff format . && ruff check --fix .
uv run poe typecheck        # ty check src/functax scripts (what CI runs)
uv run poe check            # lint + format-check + typecheck + test-cov (everything CI checks)
uv run poe capabilities     # regenerate docs/api/capabilities.md
uv run poe docs             # build both docs halves into public/ and check every link
uv run poe docs-check       # strict validation of both halves, no themed render
uv run poe docs-api         # Zensical API reference only (zensical build --strict)
uv run poe docs-serve       # build, then serve public/ at :8000
uv run poe precommit        # every pre-commit hook on every file
uv run poe build            # wheel + sdist into dist/
```

Extra arguments after a task name are passed to its command (`uv run poe test tests/test_backend.py -k require_jax`).

`addopts` turns coverage on, and a partial run never reaches the 90 % gate, so pass `--no-cov` when running part of the suite:

```bash
uv run pytest --no-cov tests/test_backend.py::test_require_jax_is_silent_when_the_extra_is_installed -v
```

### Test tiers

The markers are registered in `pyproject.toml` (`strict_markers = true`):

- **Unmarked (fast):** unit tests and doctests, each well under a second.
- **`@pytest.mark.slow`:** a test over ~1 s (a large sweep, a long fit). For a parametrised test, mark only the expensive cases (`pytest.param(..., marks=pytest.mark.slow)`).
- **`@pytest.mark.integration`:** end-to-end workflows, optional dependencies, external tools.

CI ("Tests", `ci.yml`) runs every tier with coverage on Python 3.12 and 3.13 (`uv run pytest -v`). Select the fast tier with `uv run pytest -m "not slow and not integration"`; when the slow tiers grow, give them a CI job of their own rather than dropping them.

### Tests that assert on random draws

functax's tests draw random fields, targets and the vectors of the dot-product test; JAX's 64-bit mode is on for the whole suite (`conftest.py`).

- **Incidental randomness** (any draw would do): seed it (`random.Random(0)`, `numpy.random.default_rng(0)`, a fixed key) so a failure reproduces.
- **Sampling behaviour under test** (a Monte Carlo estimate, a statistical property): bound the statistic by its own sampling distribution (a number of standard errors, a test's p-value floor), not by a tolerance tuned on one draw, and say in a comment where the bound comes from.

### Before every commit

All of these must pass, from the repo root:

1. `uv run poe test`: zero failures (doctests included).
2. `uv run --group lint ruff check .`: the **entire** repo, which includes `tests/`, `scripts/` and the code cells of `docs/notebooks/*.ipynb`. Never lint a subdirectory: `ruff check src/functax/` misses errors CI catches.
3. `uv run --group lint ruff format --check .`
4. `uv run poe typecheck` (`ty check src/functax scripts`, as CI runs it).
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

`python-style` is the house style. Its general rules:

- Python 3.12+, `from __future__ import annotations` in every module, type hints on every public function, `X | None` over `Optional`, specific exceptions with `raise … from …`, pure functions with side effects isolated and explicit.
- Every module opens with a docstring saying what it holds, in one or two sentences.
- Docstrings follow the layout in "The contracts" above; `Examples:` gives two or three cases (typical, edge, composition), each building its own small data, and `References:` names the paper when one defines the method.
- Comments go above the line, never beside it. A mathematical line has its Unicode equation above it (`# x̄ = (1/N) ∑ᵢ xᵢ`), a long function is split into stages with step headers (`# --- whiten ---`), and every other comment says why, not what.
- Names are descriptive (`posterior_precision`, not `Lam`); the symbol lives only in the equation comment.
- One operation per line, no nested calls that hide an intermediate result, 88 columns.
- One package base error; subclasses are named by what failed.
- Extract only real duplicates (code that must change together); code that merely looks alike stays separate.

### JAX and Equinox

functax uses JAX, so the JAX/Equinox rules of `python-style` apply in full:

- **Layout.** Logic lives only under `functax/_src/`; a public module only re-exports (`from functax._src.geometry.grids import RegularGrid`). One concept per file: a protocol and its implementations share a file, and a file splits when a second concept appears, not at a line count.
- **Imports.** Modules under fixed aliases, never bare function imports: `import einx`, `import equinox as eqx`, `import jax`, `import jax.numpy as jnp`, `import lineax as lx`. jaxtyping names (`from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray`) are the one exception.
- **Types and shapes.** Every array argument and return is annotated with jaxtyping and named dims (`Float[Array, "obs channel"]`); shapes live in types, never in comments. Dims come from the shared vocabulary of `python-style` (`sample`, `test`, `member`, `batch`, `time`, `cell`, `obs`, `channel`, `var`, `level`, `inducing`, `state`, `latent`, `param`); functax adds `node` (the nodes of an `Axis`) and `dim` (the D coordinates of a point), and a new functax-specific dim is added here with the first module that needs it. Types are checked at runtime in tests only, through the jaxtyping import hook in `conftest.py`.
- **Array operations.** einx for reductions, contractions, broadcasts and rearranges, never `axis=`, `.T`, `reshape` or `[:, None]`. einx is for dense arrays: a linear operator keeps its own methods (`.mv`, `.T`, the solve primitives) and is never materialised to use einx. Two axes of the same dim are named by role (`row col`, `next prev`), never `i j`. lineax for linear solves: solve, never invert.
- **Modules and functions.** Arrays live in `eqx.Module`s and are always leaves, never static fields; host-side records that are never traced are frozen dataclasses. Modules carry state, functions carry transforms. Fixed parameters (a scheme's order, a basis's θ) are fields; data goes into `__call__`. One parameterisation per module. Inputs that travel together form their own small module. Protocols are abstract modules with `eqx.AbstractVar`. Few positionals, the rest keyword-only after `*`. Updates go through public `with_*` methods: config fields with `dataclasses.replace` (so `__check_init__` re-runs), array leaves with `eqx.tree_at`. `vmap` sits outside `jacrev`.
- **No `jax.jit` inside the library.** Functions and modules stay un-jitted, and the caller compiles once at the top (around a whole step, loss or solve). An internal jit is inlined under the caller's jit anyway; on its own it fixes which arguments must be static and hides compile time in an ordinary call. Tests and `tests/_harness.py` may jit, as callers do. Eager calls that are slow because einx compiles each new shape are a test-tier matter (`@pytest.mark.slow`), not a reason to jit.
- **PRNG keys** are keyword-only, named `key`, and last; split once at the top into one named key per stochastic stage.
- **Validation.** Static checks on fields in `__check_init__`; value checks on traced arrays with `eqx.error_if` (`_check_values` in `functax._src.geometry._types` raises `GeometryError` on concrete arrays and defers to `eqx.error_if` under `jit`). The jaxtyping hook in `conftest.py` rejects anything outside an annotation before the module's own check runs, so: annotate a non-array input with what the code accepts (`ArrayLike | Sequence[float]` for a constructor that coerces, `str` for an option checked against a fixed set), not the narrower type it is coerced to; keep the jaxtyping shape annotation on an array argument, and test the module's own shape and dtype checks through `inspect.unwrap(fn)`, which skips the hook (as `tests/geometry/test_sphere.py` does).
- **Precision.** The caller decides: preserve the input dtype, never cast silently, and create constants in the input's dtype or as Python scalars. Tests that need float64 get it from `conftest.py`.
- **Configuration.** Objects round-trip through `get_config()` / `from_config()`, with arrays excluded (strict JSON for schemes, #33).
- **Tests.** Every public forward function gets four: a closed-form or SciPy oracle at a stated tolerance (`rtol=1e-12` in float64), hypothesis properties, a gradient check (`jacrev` against central differences), and transform invariance (eager, `jit` and `vmap` agree with a Python loop). A linear read-out adds the dot-product test.

functax departs from `python-style` in two places, on purpose. Every error inherits the closest standard-library exception as well as `FunctaxError` (an open question in `python-style`, settled here). The NumPy-only base modules (`geometry.affine`, `_src._backend`) use plain NumPy-namespace code without `eqx.Module` or jaxtyping, because they must import without the extra.

**CLIs and scripts** use `cyclopts` for arguments and `loguru` for output; never argparse, click, typer, progress `print`s or stdlib `logging`. For a repo script they are in the `dev` group already; for a package CLI they are runtime dependencies and need the usual discussion. One exception remains: `scripts/capabilities.py` still uses `argparse` and `print`; move it over when you next change it (`scripts/build_docs.py` shows the pattern: a `cyclopts.App` with `@app.default`, and `loguru` for every message).

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
- Commit messages and PR titles follow [Conventional Commits](https://www.conventionalcommits.org/) with a lowercase subject (`feat(interp): add …`); CI validates PR titles (`conventional-commits.yml`). Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`. Breaking changes use `!` and a `BREAKING CHANGE:` footer.
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
- **Link to the API** with the `xref:` protocol and the **top-level** exported name: ``[`FunctaxError`](xref:api#functax.FunctaxError)``, not `functax.exceptions.FunctaxError` (mystmd lowercases `$`-abbreviated anchors; see `docs/README.md`). A missing target fails `myst build --strict`.
- **URLs are flat**: a page's URL is its basename, with `_` turned into `-` (`guide/quickstart.md` → `/quickstart/`, `notebooks/gap_fill.ipynb` → `/gap-fill/`), so keep basenames unique across `docs/`, `docs/guide/` and `docs/notebooks/`; frontmatter `slug:` is ignored.
- **Site-nav URLs stay absolute** (`site.nav` in `docs/myst.yml`): the theme prefixes root-relative ones with `BASE_URL` a second time (`tests/test_build_docs.py` checks it).
- **API pages** (`docs/api/*.md`) are MkDocs-format pages, built by Zensical (which runs only the plugins it supports and ignores the rest silently; check its plugin list before adding one): prose plus `:::` blocks, `!!!` admonitions, listed in the `nav` of `mkdocs.yml`; docstrings are Google style (`docstring_style: google`) with Unicode maths; maths in the API pages themselves goes through MathJax (`pymdownx.arithmatex`), with the `eman-style-math` macros loaded by `docs/api/javascripts/mathjax.js`.
- **Notebooks** in `docs/notebooks/` are committed as executed `.ipynb` files, authored as jupytext percent `.py` and deleted after conversion; mystmd renders the stored outputs without re-executing, so each must be listed in the `docs/myst.yml` toc. Figures render inline with `plt.show()`: no `savefig`, no committed PNGs. `notebooks/` at the root holds scratch notebooks, outside the docs. Full standards: `.github/instructions/docs-examples.instructions.md`.

## Plans

Plans and scratch design notes go in `.plans/` (gitignored, never committed); track work in GitHub issues. Draft a whole wave of issues from `docs/templates/wave-backlog.md` (see `docs/contributing.md`).
