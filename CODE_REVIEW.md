# Code Review Agent Instructions

Standing instructions for **all** agents performing code reviews on this repository. Read "Reuse before you write", "The contracts" and "What enforces them" in [`AGENTS.md`](AGENTS.md) first; this file is the checklist and the report format. Most defects worth finding are a contract broken in a way the tests cannot see (a gather that silently clamps targets outside the grid, a `where` whose untaken branch poisons the gradient, a silent dtype cast, an adjoint that is not the transpose, a docstring that promises what the code does not do) or a re-implementation of something the package already has, not style.

---

## How to Obtain the Diff

Use the following command to get the diff for review:

```bash
BASE_BRANCH="$(git rev-parse --verify main >/dev/null 2>&1 && echo main || echo master)"
git --no-pager diff --no-prefix --unified=100000 --minimal $(git merge-base --fork-point "$BASE_BRANCH")...HEAD
```

If that fails (e.g. detached HEAD, shallow clone), fall back to:

```bash
git --no-pager diff --no-prefix --unified=100000 --minimal "$BASE_BRANCH"...HEAD
```

### Reading the diff

| Prefix | Meaning |
|--------|---------|
| `+` | Added line |
| `-` | Removed line |
| ` ` (space) | Unchanged context |
| `@@` | Hunk header |

---

## Review Checklist

Skip anything ruff, ty or the tests already enforce (formatting, import order, `__all__` ordering, doctest output, the coverage gate); review what they cannot see.

### 1. Reuse and boundaries

- Every function, class or module the diff **adds** has been checked against [`docs/api/capabilities.md`](docs/api/capabilities.md) and the private helpers in "Reuse before you write" in `AGENTS.md`. A re-implemented sampler, locate, basis, quadrature rule, adjoint or backend check is a **High** finding, with the existing name to use.
- No new runtime dependency without discussion; an optional one sits behind an extra and an in-function import that names the extra. `uv.lock` is updated with any dependency change.
- Imports keep pointing one way (the module table in `AGENTS.md`). functax is Core tier 0: an import of pipekit, kernellib, gaussx, finitevolX, geoarray, interpax, orthax or any other stack library is a **Critical** finding.
- The NumPy-only base (`functax`, `functax.geometry`, `functax.geometry.affine`, `functax._src._backend`, `functax._src.geometry._affine`, `functax._src.geometry._transform`) imports nothing from the `jax` extra at module level; every other module calls `require_jax(__name__)` before its first JAX import.
- A rename or removal of a public name keeps the old name working with a `DeprecationWarning` for at least one release.

### 2. The contracts

- **Public functions and modules** validate once, at the boundary (`__check_init__` for fields, `eqx.error_if` for traced values), raise `ValidationError` whose message names the argument, never mutate their input, preserve the input dtype, and trace under `jit`, `vmap` and `grad` with static shapes.
- **Gathers** mask targets outside the source explicitly: JAX clamps an out-of-bounds index to the edge, so an unmasked target off the grid reads the edge value.
- **`jnp.where`** guards the untaken branch too, or its `nan` reaches the gradient.
- **Linear read-outs** (from Wave 2) pass the dot-product test ⟨Hx, r⟩ = ⟨x, Hᵀr⟩ and equal their SciPy oracle, both to 1e-12 relative in float64, and `jit`, `vmap` and `grad` agree with an eager loop.
- **Solves** go through lineax; `jnp.linalg.inv` and `jnp.linalg.solve` are findings.
- **Exceptions** derive from `FunctaxError` and the closest stdlib exception; errors are chained (`raise … from err`).

### 3. Public API and documentation

- New names: logic in `functax._src.<area>`, re-exported from the area module's `__all__`; a new module in `SUBMODULES` or a new package in `SUBPACKAGES` (`tests/fast/test_public_api.py`), with its `docs/api/<module>.md` page, a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`; `docs/api/capabilities.md` regenerated (`uv run poe capabilities`).
- Docstrings in the `python-style` layout: summary, the defining equation in Unicode (`eman-style-math`; `$…$` in a docstring is a finding), `Algorithm:` for a method, `Args:` with meaning and symbol, `Returns:` and `Raises:` matching the code, and two or three executable `Examples:` (typical, edge, composition) whose output is what the code prints (shapes, `allclose` booleans or reprs, never raw floats).
- Maths in MyST and MkDocs pages uses the `eman-style-math` macros, defines every symbol with its shape, keeps each expression on one line and has no spacing macros (`\,`, `\quad`, …).
- Prose pages: MyST directives, not MkDocs-Material syntax; API links by the top-level name (`xref:api#functax.<Name>`); a new page or notebook in the `docs/myst.yml` toc with a basename unique across `docs/`.
- Notebooks committed executed, with no error outputs, and every number quoted in the prose matching the printed output.

### 4. Tests

- Each behaviour is checked against a closed form, a hand computation or a published value; edge cases (targets on the boundary and outside it, a single cell, a descending axis, float32 inputs, invalid arguments) have their own tests. A public forward function has the four `python-style` tests: oracle, hypothesis properties, gradient check, transform invariance.
- New code is covered (the 90 % gate is global, so a well-covered package can hide an untested module: look at the diff, not the total).
- Randomness, if any, is seeded; a statistical assertion is bounded by the statistic's own sampling distribution, with the bound's source in a comment.
- Tier markers right: unmarked under ~1 s, `slow` above, `integration` for end-to-end workflows and optional dependencies.

### 5. Modern Python and dependencies

- `from __future__ import annotations`; type hints on every public function; `X | None`, built-in generics, PEP 695 `type` aliases.
- `eqx.Module` for anything holding arrays (arrays as leaves, configuration as static fields); frozen dataclasses for host-side records that are never traced; `Enum` / `StrEnum` for fixed sets of options; `pathlib`; f-strings; context managers for resources.
- Specific exceptions, chained with `raise … from …`; guard clauses over deep nesting.
- `python-style` comments and naming: comments above the line, never beside it; a Unicode equation above each mathematical line; descriptive names, with the symbol only in the comment; one operation per line. Its JAX rules too ("JAX and Equinox" in `AGENTS.md`): jaxtyping dims, einx, lineax, `eqx.error_if`, `key` keyword-only and last.
- New CLIs and scripts use `cyclopts` and `loguru`, not argparse, click, typer, `print` progress or stdlib `logging`.
- A dependency, when one is justified, follows the preferences in `.github/instructions/python-coding.instructions.md`.

### 6. Security

- No secrets or credentials in code, tests, notebooks or docs.
- `subprocess` without `shell=True`; file paths taken from users are not joined into places they could escape.

---

## Package-Specific Checks

Every example here runs, with `jax_enable_x64` on as in the test suite, except the last, whose module lands with #15.

### Mask targets outside the source

```python
import jax.numpy as jnp

field = jnp.array([10.0, 20.0, 30.0])
index = jnp.array([1, 5])

# ❌ JAX clamps the out-of-bounds index: the target at 5 reads the edge, 30.0
field[index]


# ✅ An explicit `inside` mask, and a safe index for the gather
def read_nearest(field, index):
    inside = (index >= 0) & (index < field.shape[0])
    safe_index = jnp.where(inside, index, 0)
    value = field[safe_index]
    return jnp.where(inside, value, jnp.nan)
```

### Guard both branches of `where`

```python
import jax
import jax.numpy as jnp


# ❌ The untaken branch still runs: jax.grad(...)(0.0) is nan, not 0.0
def inverse_distance(d):
    return jnp.where(d > 0, 1.0 / d, 0.0)


# ✅ Feed the untaken branch a safe value: jax.grad(...)(0.0) is 0.0
def inverse_distance(d):
    positive = d > 0
    safe_d = jnp.where(positive, d, 1.0)
    return jnp.where(positive, 1.0 / safe_d, 0.0)
```

### Preserve the input dtype

```python
import jax.numpy as jnp


# ❌ A float32 field comes back as float64
def lerp(a, b, t):
    return a + jnp.asarray(t, dtype=jnp.float64) * (b - a)


# ✅ Constants in the input's dtype
def lerp(a, b, t):
    t = jnp.asarray(t, dtype=a.dtype)
    return a + t * (b - a)
```

### The adjoint is the transpose

```python
import jax
import jax.numpy as jnp


def apply(x):
    # H x: the mean of each pair of neighbours, (N,) → (N − 1,)
    return 0.5 * (x[:-1] + x[1:])


# ❌ Looks plausible and fails the dot-product test
def adjoint(r):
    return jnp.concatenate([0.5 * r, jnp.zeros(1)])


# ✅ Each output spreads back to both inputs it read
def adjoint(r):
    padded_right = jnp.concatenate([r, jnp.zeros(1)])
    padded_left = jnp.concatenate([jnp.zeros(1), r])
    return 0.5 * (padded_right + padded_left)


key_x, key_r = jax.random.split(jax.random.key(0))
x = jax.random.normal(key_x, (6,))
r = jax.random.normal(key_r, (5,))
# ⟨Hx, r⟩ = ⟨x, Hᵀr⟩, and the hand-written adjoint equals jax.linear_transpose
assert jnp.allclose(jnp.vdot(apply(x), r), jnp.vdot(x, adjoint(r)), rtol=1e-12)
(transpose,) = jax.linear_transpose(apply, x)(r)
assert jnp.allclose(transpose, adjoint(r))
```

### Solve, never invert

```python
import jax.numpy as jnp
import lineax as lx

matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
rhs = jnp.array([1.0, 2.0])

# ❌ Forms the inverse: slower, less accurate, and blind to structure
solution = jnp.linalg.inv(matrix) @ rhs

# ✅ A lineax solve, with the structure declared
operator = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
solution = lx.linear_solve(operator, rhs, solver=lx.Cholesky()).value
```

### Keep JAX out of the base install

```python
# ❌ In functax/geometry/__init__.py: every NumPy-only consumer now needs JAX
from functax._src.geometry.grids import RegularGrid


# ✅ Re-export the Equinox types lazily, so the package imports without JAX
def __getattr__(name):
    if name == "RegularGrid":
        from functax._src.geometry.grids import RegularGrid

        return RegularGrid
    raise AttributeError(name)
```

---

## Output Format

Format each review using this structure:

````
# Code Review for ${feature_description}

Overview of the changes, including the purpose, context, and files involved.

## Suggestions

### ${emoji} ${Summary of suggestion with necessary context}

* **Priority**: ${priority_emoji} ${priority_label}
* **File**: `${relative/path/to/file.py}`
* **Line(s)**: ${line_numbers}
* **Details**: Explanation of the issue and why it matters
* **Current Code**:
  ```python
  # problematic code
  ```
* **Suggested Change**:
  ```python
  # improved code with explanation
  ```

### (additional suggestions…)

## Summary

Brief summary of overall code quality and key action items.
````

---

## Priority Levels

| Emoji | Level | Use when |
|-------|-------|----------|
| 🔥 | **Critical** | Bugs, security issues, or code that will fail |
| ⚠️ | **High** | Significant issues affecting maintainability or correctness |
| 🟡 | **Medium** | Improvements for code quality or consistency |
| 🟢 | **Low** | Minor polish or optional enhancements |

## Suggestion Type Emojis

Prefix each suggestion title with a type indicator:

| Emoji | Type |
|-------|------|
| 🐛 | Bug or potential bug |
| 🔒 | Security concern |
| 🔧 | Change request (must fix) |
| ♻️ | Refactor suggestion |
| 📝 | Documentation improvement |
| 🎨 | Style / formatting issue |
| ⚡ | Performance consideration |
| 🧪 | Testing suggestion |
| ❓ | Question or clarification needed |
| ⛏️ | Nitpick (very minor) |
| 💭 | Design consideration |
| 👍 | Positive feedback (highlight good patterns) |
| 🌱 | Future consideration (not blocking) |

---

## Review Tone

- Be **constructive** and **specific**
- **Acknowledge** good patterns and decisions (use 👍 liberally)
- Explain the *why* behind every suggestion
- Offer **concrete alternatives**, not just criticism
- Recognize that context matters — ask clarifying questions when needed
- Keep feedback **actionable**: every suggestion should have a clear next step
