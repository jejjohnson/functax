---
name: add-cli-command
description: Add a subcommand or an option to the package's argparse console script — a subparser in build_parser, a _run_<name> handler registered in main, output only to the injected streams, input through _read_values, argparse types for option validation, the exit-code contract (0 / 1 / 2), in-process tests through the run() helper, and the CLI guide page. Use when asked to add, change or expose functionality on the command line in src/mypackage/cli.py.
---

<!-- template-example: this recipe is written for the example package's argparse CLI; rewrite it for your package's command line (or delete it if there is none). -->

# Add a CLI subcommand

Read "3. The CLI" under "The contracts" in `AGENTS.md` first; this is the step-by-step. The CLI is a thin shell: if the computation does not exist in the library yet, add it there first (`add-function`) and call it from here.

## 1. Parser (`src/mypackage/cli.py`, `build_parser`)

- `subparsers.add_parser("<name>", help="…")`, then its arguments. Copy the `path` positional of `summarize` / `smooth` (`nargs="?"`, `default="-"`) when it reads a series.
- Choices from an enum come from the enum (`choices=tuple(member.value for member in Padding)`), defaults from its members, so the CLI and the library cannot drift.
- Validate an option with an `argparse` `type=` (as `_non_negative_int` does for `--precision`): a bad value is then a usage error, exit 2, printed to the injected `err`.
- Every option says its default in its `help`.

## 2. Handler

- `_run_<name>(args: argparse.Namespace, out: TextIO, stdin: TextIO | None) -> int`, registered in the `handlers` dict in `main`.
- Read numbers with `_read_values(args.path, stdin=stdin)` (whitespace- or comma-separated, finite values only).
- Write only to `out`; return `EXIT_OK`. Never `print`, never `sys.exit`.
- Let library errors propagate: `main` turns a `MypackageError` into `error: <message>` on `err` and `EXIT_ERROR` (1), and an `OSError` into `error: could not read …`. Raise `ValidationError` for a CLI-only failure (as `--json` does for a non-finite summary).

## 3. Tests (`tests/test_cli.py`)

Use the module's `run(*argv, stdin=...)` helper, which returns `(status, stdout, stderr)`:

- the parser's defaults (`TestParser`);
- the output for a small input, and each option forwarded to the library;
- each error path: status 1 and an `error:` line for bad input, status 2 and the option's name on `err` for a bad option, nothing on `out`;
- nothing leaks to the real process streams (see `test_usage_errors_are_written_to_the_injected_stream`).

## 4. Document

- `docs/guide/cli.md`: a `## \`<name>\`` section with the usage line, the options and an example; update "Exit codes" if the meaning of a code changes.
- `docs/api/cli.md` lists `main` and `build_parser` only; a new public helper there needs adding to its `members:`, `cli.__all__` and `uv run poe capabilities`.

## 5. Verify

```bash
uv run pytest --no-cov tests/test_cli.py src/mypackage/cli.py -v
echo "1 2 3 4" | uv run mypackage <name>
uv run poe test-cov
```

then the `pre-pr-check` skill.
