---
name: add-function
description: Add a public function, class or module to the package — validation at the boundary with the utils validators, a pure result, an executable Google-style docstring, exports in the module and top-level __all__, a new module in SUBMODULES with its API page, the capability index, and tests. Use when asked to add, port or implement a statistic, a filter, a helper or a new module in src/mypackage.
---

<!-- template-example: this recipe is written for the example package's modules, validators and test files; rewrite it for your package's extension points. -->

# Add a public function, class or module

Read "1. Public functions" and "The public API" under "The contracts" in `AGENTS.md` first; this is the step-by-step. A transform has its own recipe (`add-transform`), a CLI subcommand too (`add-cli-command`).

## 1. Make sure it does not exist yet

- Search `docs/api/capabilities.md` for the operation and its synonyms (a "rolling mean" is `moving_average`, "standard score" is `zscores`, a "percentile" is `quantile`), and the private helpers in "Reuse before you write" in `AGENTS.md` (`smoothing._pad`, `stats._mean`, …).
- A combination of existing names (a pipeline, a `summarize` field) needs no new function.

## 2. Pick its home

| It is… | Module | Exemplar |
|---|---|---|
| A statistic of a whole series, or a streaming accumulator | `src/mypackage/stats.py` | `quantile`, `RunningStats` |
| A filter that returns a smoothed series | `src/mypackage/smoothing.py` | `moving_average` (with `Padding`), `exponential_moving_average` |
| A validator or a helper two modules need | `src/mypackage/utils.py` | `require_positive`, `Window` |
| A new exception | `src/mypackage/exceptions.py` | `NotFittedError` (derive from `MypackageError` **and** the closest stdlib exception) |
| A new family of its own | a new `src/mypackage/<name>.py` | `smoothing.py` |

Imports only point down the module table in `AGENTS.md`: a library module never imports `cli` or `transforms`.

## 3. Write it

- Coerce and check inputs once: `values = require_non_empty(series)`, `window = require_positive(window, name="window")`; other checks raise `ValidationError(f"'<arg>' must …, got {value!r}")`.
- Return a new `list[float]` (or a frozen dataclass such as `Summary`); never mutate the input; sums through `math.fsum`; degenerate input raises instead of returning `nan`.
- Comments above the line, never beside it, with the Unicode equation above each mathematical line; descriptive names, the symbol only in the comment; one operation per line.
- Type hints on every parameter and the return; `Series` for series inputs; keyword-only options after `*` where the existing functions use them.
- Google-style docstring in the `python-style` layout ("House style" in `AGENTS.md`): one-line summary (it becomes the capability-index entry), the defining equation in Unicode (`x̄ = (1/N) ∑ᵢ xᵢ`, `eman-style-math` notation, never `$…$`), `Algorithm:` with numbered steps if it implements a method, `Args:` with each name's meaning and symbol, `Returns:`, `Raises:` and an `Examples:` block of two or three `>>>` cases (typical, edge, composition). It runs under `--doctest-modules`: print rounded values (`[round(v, 4) for v in …]`), and paste what the code really prints.

## 4. Export and document

- Add the name to the module's `__all__` and to `src/mypackage/__init__.py` (import and `__all__`), both sorted (`tests/test_public_api.py` and ruff's `RUF022`).
- A function on an existing module appears on its API page automatically (`::: mypackage.<module>` in `docs/api/<module>.md`); add it to that module's row in `docs/api/index.md`.
- A **new module**: a module docstring (its first sentence titles the capability-index group), a sorted `__all__`, an entry in `SUBMODULES` in `tests/test_public_api.py`, a `docs/api/<module>.md` page with a `::: mypackage.<module>` block, a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`; a guide page if it deserves prose (`add-guide-page`).
- `uv run poe capabilities`.

## 5. Tests

In `tests/test_<module>.py` (a new module gets a new file), one class per function:

- values against a hand computation or a closed form (the fixtures in `tests/conftest.py`: `simple_series`, `wikipedia_series` with mean 5 and population variance 4, `spiky_series`, `sine_series`);
- the edge cases: empty input (`EmptySeriesError`), a single value, a constant series, each invalid argument (`ValidationError` with `match=`);
- the input is not mutated and a new list comes back (a smoother joins `SMOOTHERS` in `tests/test_smoothing.py`, which checks both for every entry);
- a new exception joins the parametrised lists in `tests/test_exceptions.py`.

## 6. Verify

```bash
uv run pytest --no-cov tests/test_<module>.py src/mypackage/<module>.py tests/test_public_api.py tests/test_capabilities.py -v
uv run poe test-cov       # the whole suite with the 90 % coverage gate
```

then the `pre-pr-check` skill.
