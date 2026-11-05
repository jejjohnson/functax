---
applyTo: "src/**/*.py,tests/**/*.py,scripts/**/*.py"
---

# Python Coding Standards

The house style is the maintainer's `python-style` skill; `AGENTS.md` ("House style") carries its essentials and the template's documented departures from it. This file is the short form for Copilot.

## Modern Python (3.12+)

- `from __future__ import annotations` at the top of every module
- Type hints on **all** public functions, methods, and module-level variables
- Modern union syntax: `X | None` not `Optional[X]`, `X | Y` not `Union[X, Y]`
- Built-in generics: `list[int]`, `dict[str, Any]` not `List[int]`, `Dict[str, Any]`
- `pathlib.Path` over `os.path`
- f-strings for string formatting
- `dataclasses` (with `slots=True`; `frozen=True` when immutable) or `attrs` for data containers
- `Enum` for fixed sets of constants
- Context managers (`with` statements) for resource handling
- Specific exception types (never bare `except:`)
- Proper exception chaining (`raise ... from ...`)
- Early returns / guard clauses to reduce nesting

## Package Preferences

| Purpose | Preferred Package |
|---------|-------------------|
| Logging and script output | `loguru` (never stdlib `logging` or progress `print`s) |
| CLI and script arguments | `cyclopts` (never argparse, click or typer) |
| Data containers | `dataclasses` (stdlib) or `attrs` |
| Configuration | `hydra-core` / `omegaconf` |
| Path handling | `pathlib` (stdlib) |
| HTTP | `httpx` |
| Testing | `pytest` |

<!-- template-example: this note is about the bundled example package. -->

> The bundled demo package uses stdlib `argparse` for its CLI rather than `cyclopts`, because the template advertises zero runtime dependencies. Prefer `cyclopts` in a real project where a CLI dependency is acceptable.

## Comments and naming

- Comments go above the line, never beside it; they say *why*, not *what*
- A Unicode equation above each mathematical line (`# x̄ = (1/N) ∑ᵢ xᵢ`), in the `eman-style-math` notation
- Descriptive names (`posterior_precision`, not `Lam`); the symbol lives only in the equation comment
- One operation per line; step headers (`# --- whiten ---`) split a long function into stages
- In a JAX package: jaxtyping with named dims, einx (no `axis=`, `.T`, `reshape`, `[:, None]`), lineax for solves, `eqx.Module` with `__check_init__` and `eqx.error_if`

## Documentation

- Module-level docstrings explaining purpose
- Function/method docstrings for all public APIs (Google style)
- Docstrings open with a one-line summary and the defining equation in Unicode (`σ² = ∑ᵢ (xᵢ − μ)² / N`), never `$…$`; a function that implements a method adds a numbered `Algorithm:` section; `Args:` give each name's meaning and symbol, without the type
- Public classes and functions include an executable `Examples:` block (run by `--doctest-modules`; the expected output is what the code prints)
