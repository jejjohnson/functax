# Code Review Agent Instructions

Standing instructions for **all** agents performing code reviews on this repository. Read "Reuse before you write", "The contracts" and "What enforces them" in [`AGENTS.md`](AGENTS.md) first; this file is the checklist and the report format. Most defects worth finding are a contract broken in a way the tests cannot see (an input mutated on one path, a validator skipped, an unfitted transform that quietly does nothing, a docstring that promises what the code does not do) or a re-implementation of something the package already has, not style.

<!-- template-example: this file's sections 2 and "Package-Specific Checks" describe the example package. -->

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

- Every function, class or module the diff **adds** has been checked against [`docs/api/capabilities.md`](docs/api/capabilities.md) and the private helpers in "Reuse before you write" in `AGENTS.md`. A re-implemented validator, statistic, filter, padding rule, buffer or transform is a **High** finding, with the existing name to use.
- No new runtime dependency without discussion; an optional one sits behind an extra and an in-function import that names the extra. `uv.lock` is updated with any dependency change.
- Imports keep pointing one way (the module table in `AGENTS.md`); the library never imports the CLI.
- A rename or removal of a public name keeps the old name working with a `DeprecationWarning` for at least one release.

### 2. The contracts

<!-- template-example: the example package's contracts; replace with your package's. -->

- **Public functions** validate once, at the boundary, with `as_floats` / `require_non_empty` / `require_positive`; raise `ValidationError` / `EmptySeriesError` whose message names the argument; return a **new** list and never mutate the input; use `math.fsum` for sums; turn degenerate input into an error, not a `nan`.
- **Transforms** satisfy `Transform` structurally (no base class); stateless ones are frozen dataclasses whose `fit` returns `self`; fitted ones keep learned state in `init=False` trailing-underscore fields, return `self` from `fit`, and raise `NotFittedError` from `apply` before `fit`; every transform has `fit_apply`; constructor arguments are validated in `__post_init__`; the transform joins `TestProtocols` in `tests/test_transforms.py`.
- **CLI** subcommands are a subparser plus a `_run_<name>` handler in `main`'s `handlers`; they write only to the injected `out`, read through `_read_values`, validate options with an `argparse` `type=`, and never `print` or `sys.exit`.
- **Exceptions** derive from `MypackageError` and the closest stdlib exception; errors are chained (`raise … from err`).

### 3. Public API and documentation

- New names: in the module's `__all__` and the top-level `__all__`; a new module in `SUBMODULES` (`tests/test_public_api.py`), with its `docs/api/<module>.md` page, a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`; `docs/api/capabilities.md` regenerated (`uv run poe capabilities`).
- Docstrings in the `python-style` layout: summary, the defining equation in Unicode (`eman-style-math`; `$…$` in a docstring is a finding), `Algorithm:` for a method, `Args:` with meaning and symbol, `Returns:` and `Raises:` matching the code, and two or three executable `Examples:` (typical, edge, composition) whose output is what the code prints (rounded values, not 17-digit floats).
- Maths in MyST and MkDocs pages uses the `eman-style-math` macros, defines every symbol with its shape, keeps each expression on one line and has no spacing macros (`\,`, `\quad`, …).
- Prose pages: MyST directives, not MkDocs-Material syntax; API links by the top-level name (`xref:api#mypackage.<Name>`); a new page or notebook in the `docs/myst.yml` toc with a basename unique across `docs/`.
- Notebooks committed executed, with no error outputs, and every number quoted in the prose matching the printed output.

### 4. Tests

- Each behaviour is checked against a closed form, a hand computation or a published value; edge cases (empty input, a single value, a constant series, invalid arguments, overflow) have their own tests.
- New code is covered (the 90 % gate is global, so a well-covered package can hide an untested module: look at the diff, not the total).
- Randomness, if any, is seeded; a statistical assertion is bounded by the statistic's own sampling distribution, with the bound's source in a comment.
- Tier markers right: unmarked under ~1 s, `slow` above, `integration` for end-to-end workflows and optional dependencies.

### 5. Modern Python and dependencies

- `from __future__ import annotations`; type hints on every public function; `X | None`, built-in generics, PEP 695 `type` aliases.
- `dataclasses` with `slots=True` for data containers (`frozen=True` when immutable); `Enum` / `StrEnum` for fixed sets of options; `pathlib`; f-strings; context managers for resources.
- Specific exceptions, chained with `raise … from …`; guard clauses over deep nesting.
- `python-style` comments and naming: comments above the line, never beside it; a Unicode equation above each mathematical line; descriptive names, with the symbol only in the comment; one operation per line. In a JAX package, its JAX rules too (jaxtyping dims, einx, lineax, `eqx.error_if`, the four tests per public forward).
- New CLIs and scripts use `cyclopts` and `loguru`, not argparse, click, typer, `print` progress or stdlib `logging` (the example CLI's `argparse` is a documented exception).
- A dependency, when one is justified, follows the preferences in `.github/instructions/python-coding.instructions.md`.

### 6. Security

- No secrets or credentials in code, tests, notebooks or docs.
- `subprocess` without `shell=True`; file paths taken from users are not joined into places they could escape.

---

## Package-Specific Checks

<!-- template-example: these patterns are the example package's; replace them with the defects your package is prone to. Every example here runs. -->

### Validate once, at the boundary

```python
# ❌ No validation: window=0 or an empty series fails deep inside max() with
#    "max() iterable argument is empty", and a string slips through as text
def rolling_max(series, window):
    return [max(series[i : i + window]) for i in range(len(series) - window + 1)]


# ✅ The package's validators, and the package's error messages
from mypackage import Series, require_non_empty, require_positive


def rolling_max(series: Series, window: int) -> list[float]:
    values = require_non_empty(series)
    window = require_positive(window, name="window")
    return [max(values[i : i + window]) for i in range(len(values) - window + 1)]
```

### Never mutate the input

```python
# ❌ Sorts the caller's list in place
def upper_quartile(series: list[float]) -> float:
    series.sort()
    return series[(3 * len(series)) // 4]


# ✅ Work on a fresh, validated copy, or reuse the existing statistic
from mypackage import Series, quantile


def upper_quartile(series: Series) -> float:
    return quantile(series, 0.75)
```

### A fitted transform fails loudly before `fit`

```python
from dataclasses import dataclass, field
from typing import Self

from mypackage import NotFittedError, Series, as_floats, require_non_empty


# ❌ Unfitted, `apply` silently behaves as the identity
@dataclass(slots=True)
class CenterSilently:
    mean_: float | None = field(default=None, init=False)

    def apply(self, series: Series) -> list[float]:
        return [value - (self.mean_ or 0.0) for value in as_floats(series)]


# ✅ The package's fit / apply lifecycle
@dataclass(slots=True)
class Center:
    mean_: float | None = field(default=None, init=False)

    def fit(self, series: Series) -> Self:
        values = require_non_empty(series)
        self.mean_ = sum(values) / len(values)
        return self

    def apply(self, series: Series) -> list[float]:
        if self.mean_ is None:
            msg = "Center is not fitted; call .fit(series) first"
            raise NotFittedError(msg)
        return [value - self.mean_ for value in as_floats(series)]

    def fit_apply(self, series: Series) -> list[float]:
        return self.fit(series).apply(series)
```

### Typed, chained errors that name the argument

```python
from mypackage import ValidationError


# ❌ A bare except catches everything (KeyboardInterrupt included), and the
#    error is a RuntimeError naming neither the argument nor the bad value
def parse_alpha(text: str) -> float:
    try:
        return float(text)
    except:  # noqa: E722
        raise RuntimeError("bad input")


# ✅ The package's error (also a ValueError), the argument's name, the cause
def parse_alpha(text: str) -> float:
    try:
        alpha = float(text)
    except ValueError as err:
        msg = f"'alpha' must be a number, got {text!r}"
        raise ValidationError(msg) from err
    if not 0.0 < alpha <= 1.0:
        msg = f"'alpha' must lie in (0, 1], got {alpha!r}"
        raise ValidationError(msg)
    return alpha
```

### Doctest output the code really prints

```python
# ❌ Expected output copied from a calculator: the doctest fails on the
#    17th digit, or passes only on one platform
#     >>> zscores([1.0, 2.0, 4.0])
#     [-0.87, -0.22, 1.09]

# ✅ Round in the example, and paste what it prints
#     >>> [round(z, 4) for z in zscores([1.0, 2.0, 4.0])]
#     [-0.8729, -0.2182, 1.0911]
from mypackage import zscores

assert [round(z, 4) for z in zscores([1.0, 2.0, 4.0])] == [-0.8729, -0.2182, 1.0911]
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
