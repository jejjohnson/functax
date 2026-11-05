"""Command-line interface for ``mypackage``.

Installed as the ``mypackage`` console script via the ``[project.scripts]``
entry point in ``pyproject.toml``:

```bash
mypackage summarize data.txt
mypackage smooth data.txt --method median --window 5
echo "1 2 3 4 5" | mypackage summarize - --json
```

The whole CLI is built on `argparse` from the standard library, so the package
still declares zero runtime dependencies. [`main`][mypackage.cli.main] returns
an exit status rather than calling `sys.exit`, which makes it directly
testable.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from collections.abc import Sequence
from typing import TextIO

from mypackage import __version__
from mypackage.exceptions import MypackageError, ValidationError
from mypackage.smoothing import (
    Padding,
    exponential_moving_average,
    median_filter,
    moving_average,
)
from mypackage.stats import summarize
from mypackage.utils import as_floats


__all__ = ["build_parser", "main"]

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


def _read_values(path: str, *, stdin: TextIO | None = None) -> list[float]:
    """Read whitespace- or comma-separated numbers from ``path``.

    Args:
        path: A file path, or ``"-"`` to read from standard input.
        stdin: Stream used when ``path`` is ``"-"``. Defaults to `sys.stdin`.

    Returns:
        The parsed values.

    Raises:
        ValidationError: If a token cannot be parsed as a number, or parses
            to a non-finite value such as ``nan`` or ``inf``.
        OSError: If the file cannot be read.
    """
    if path == "-":
        text = (stdin if stdin is not None else sys.stdin).read()
    else:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()

    tokens = text.replace(",", " ").split()
    values = as_floats(tokens, name="input")
    # `float()` happily parses "nan" and "inf", but a summary of them is
    # meaningless and `--json` cannot represent them in standard JSON.
    for index, (token, value) in enumerate(zip(tokens, values, strict=True)):
        if not math.isfinite(value):
            msg = (
                f"'input' must contain only finite numbers, "
                f"got {token!r} at index {index}"
            )
            raise ValidationError(msg)
    return values


def _non_negative_int(text: str) -> int:
    """Argparse type for options that must be a non-negative integer."""
    try:
        value = int(text)
    except ValueError:
        msg = f"invalid int value: {text!r}"
        raise argparse.ArgumentTypeError(msg) from None
    if value < 0:
        msg = f"must be a non-negative integer, got {value}"
        raise argparse.ArgumentTypeError(msg)
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Exposed separately from [`main`][mypackage.cli.main] so documentation
    tooling and tests can introspect the CLI without running it.

    Returns:
        A fully configured `argparse.ArgumentParser`.

    Examples:
        >>> parser = build_parser()
        >>> args = parser.parse_args(["summarize", "data.txt"])
        >>> args.command, args.path, args.json
        ('summarize', 'data.txt', False)
    """
    parser = argparse.ArgumentParser(
        prog="mypackage",
        description="Descriptive statistics and smoothing for numeric series.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"mypackage {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Print descriptive statistics for a series.",
    )
    summarize_parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="File of numbers, or '-' for standard input (default: '-').",
    )
    summarize_parser.add_argument(
        "--ddof",
        type=int,
        default=1,
        help="Delta degrees of freedom for the variance (default: 1).",
    )
    summarize_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of an aligned table.",
    )

    smooth_parser = subparsers.add_parser(
        "smooth",
        help="Smooth a series and print the result.",
    )
    smooth_parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="File of numbers, or '-' for standard input (default: '-').",
    )
    smooth_parser.add_argument(
        "--method",
        choices=("mean", "median", "ewma"),
        default="mean",
        help="Smoothing filter to apply (default: mean).",
    )
    smooth_parser.add_argument(
        "--window",
        type=int,
        default=3,
        help="Window length for the 'mean' and 'median' methods (default: 3).",
    )
    smooth_parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Smoothing factor for the 'ewma' method (default: 0.3).",
    )
    smooth_parser.add_argument(
        "--padding",
        choices=tuple(member.value for member in Padding),
        default=Padding.EDGE.value,
        help="Boundary strategy for windowed methods (default: edge).",
    )
    smooth_parser.add_argument(
        "--precision",
        type=_non_negative_int,
        default=6,
        help="Number of decimals in the printed output (default: 6).",
    )

    return parser


def _run_summarize(args: argparse.Namespace, out: TextIO, stdin: TextIO | None) -> int:
    values = _read_values(args.path, stdin=stdin)
    summary = summarize(values, ddof=args.ddof)
    if args.json:
        try:
            payload = json.dumps(
                summary.to_dict(), indent=2, sort_keys=True, allow_nan=False
            )
        except ValueError as exc:
            # Finite inputs can still overflow, e.g. the variance of
            # [1e308, -1e308]. Standard JSON has no token for that.
            msg = "summary contains non-finite values, which JSON cannot represent"
            raise ValidationError(msg) from exc
        out.write(payload + "\n")
    else:
        for key, value in sorted(summary.to_dict().items()):
            out.write(f"{key:<10}{value:>14.6g}\n")
    return EXIT_OK


def _run_smooth(args: argparse.Namespace, out: TextIO, stdin: TextIO | None) -> int:
    values = _read_values(args.path, stdin=stdin)
    if args.method == "mean":
        smoothed = moving_average(values, args.window, padding=args.padding)
    elif args.method == "median":
        smoothed = median_filter(values, args.window, padding=args.padding)
    else:
        smoothed = exponential_moving_average(values, args.alpha)

    for value in smoothed:
        out.write(f"{value:.{args.precision}f}\n")
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    *,
    out: TextIO | None = None,
    err: TextIO | None = None,
    stdin: TextIO | None = None,
) -> int:
    """Run the command-line interface.

    Args:
        argv: Argument list, excluding the program name. Defaults to
            `sys.argv[1:]`.
        out: Stream for normal output, including ``--help`` and
            ``--version``. Defaults to `sys.stdout`.
        err: Stream for error messages, including argparse usage errors.
            Defaults to `sys.stderr`.
        stdin: Stream used when the input path is ``"-"``. Defaults to
            `sys.stdin`.

    Returns:
        ``0`` on success, ``1`` on a package-level or I/O error, and ``2`` on
        a usage error (propagated from `argparse`).

    Examples:
        >>> import io
        >>> buffer = io.StringIO()
        >>> main(
        ...     ["summarize", "-", "--json"],
        ...     out=buffer,
        ...     stdin=io.StringIO("1 2 3 4"),
        ... )
        0
        >>> json.loads(buffer.getvalue())["mean"]
        2.5
    """
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr

    parser = build_parser()
    # argparse prints --help, --version, and usage errors straight to
    # sys.stdout / sys.stderr, so redirect them to honour the injected streams.
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            args = parser.parse_args(argv)
    except SystemExit as exc:  # --help / --version / usage error
        return int(exc.code or EXIT_OK)

    handlers = {"summarize": _run_summarize, "smooth": _run_smooth}
    try:
        return handlers[args.command](args, out, stdin)
    except MypackageError as exc:
        err.write(f"error: {exc}\n")
        return EXIT_ERROR
    except OSError as exc:
        err.write(f"error: could not read {args.path!r}: {exc}\n")
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
