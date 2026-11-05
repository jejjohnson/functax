# Command-line interface

Installing the package puts a `mypackage` executable on your `PATH` via the
`[project.scripts]` entry point:

```toml
[project.scripts]
mypackage = "mypackage.cli:main"
```

It is built on `argparse` from the standard library, so the package still
declares zero runtime dependencies.

```bash
mypackage --help
mypackage --version
```

## `summarize`

Print descriptive statistics for a series of numbers.

```bash
mypackage summarize [PATH] [--ddof N] [--json]
```

`PATH` is a file of whitespace- or comma-separated numbers, or `-` (the
default) to read standard input. Every value must be finite: `nan` and `inf`
are rejected, because a summary of them is meaningless and `--json` could not
represent them in standard JSON.

::::{tab-set}
:::{tab-item} Table output

```console
$ echo "1 2 3 4 5" | mypackage summarize
count                  5
ddof                   1
maximum                5
mean                   3
median                 3
minimum                1
spread                 4
std              1.58114
variance             2.5
```
:::
:::{tab-item} JSON output

```console
$ echo "1 2 3 4 5" | mypackage summarize --json
{
  "count": 5,
  "ddof": 1,
  "maximum": 5.0,
  "mean": 3.0,
  "median": 3.0,
  "minimum": 1.0,
  "spread": 4.0,
  "std": 1.5811388300841898,
  "variance": 2.5
}
```
:::
:::{tab-item} From a file

```console
$ mypackage summarize measurements.txt --ddof 0
```
:::
::::

:::{tip} Piping into `jq`
`--json` makes the output composable:

```bash
mypackage summarize data.txt --json | jq '.mean, .std'
```
:::

## `smooth`

Apply a smoothing filter and print one value per line.

```bash
mypackage smooth [PATH] [--method {mean,median,ewma}] [--window N]
                        [--alpha A] [--padding STRATEGY] [--precision D]
```

| Flag | Default | Applies to | Meaning |
|---|---|---|---|
| `--method` | `mean` | — | `mean`, `median`, or `ewma` |
| `--window` | `3` | `mean`, `median` | samples per window |
| `--alpha` | `0.3` | `ewma` | smoothing factor in $(0, 1]$ |
| `--padding` | `edge` | `mean`, `median` | `edge`, `reflect`, `zero`, `none` |
| `--precision` | `6` | — | decimals in the output; must be non-negative |

```console
$ printf "1 1 99 1 1" | mypackage smooth --method median --window 3 --precision 1
1.0
1.0
1.0
1.0
1.0
```

With `--padding none` the output is `window - 1` samples shorter:

```console
$ echo "1 2 3 4 5" | mypackage smooth --padding none --precision 1
2.0
3.0
4.0
```

## Exit codes

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | a package error (empty input, bad or non-finite numbers) or an I/O error |
| `2` | a usage error — unknown flag, missing subcommand |

Errors go to standard error, prefixed with `error: `, leaving standard output
clean for pipes:

```console
$ echo "1 2 banana" | mypackage summarize
error: 'input' must contain only numbers, got 'banana' at index 2
$ echo $?
1
```

## Testing a CLI

[`main`](xref:api#mypackage.cli.main) accepts its streams as arguments and **returns**
an exit status rather than calling `sys.exit`. That is what makes the whole
CLI testable in-process, with no subprocess and no monkeypatching:

```python
import io

from mypackage.cli import main

out = io.StringIO()
status = main(["summarize", "-", "--json"], out=out, stdin=io.StringIO("1 2 3 4"))

status                                 # 0
json.loads(out.getvalue())["mean"]     # 2.5
```

[`build_parser`](xref:api#mypackage.cli.build_parser) is exposed separately so tests
and documentation tooling can introspect the argument structure without
running anything.

## API

Full signatures live in the [CLI API reference](xref:api#mypackage.cli).
