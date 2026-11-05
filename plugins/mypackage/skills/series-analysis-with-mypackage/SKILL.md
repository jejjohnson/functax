---
name: series-analysis-with-mypackage
description: Analyse one-dimensional numeric series with mypackage — descriptive statistics (summarize, quantile, z-scores) and streaming accumulators, moving-average, median and exponential smoothing with explicit boundary padding, and fit / apply transforms (standardise, min-max scale, clip, smooth) composed into pipelines, plus its command-line interface. Use whenever a task summarises, smooths, rescales or cleans a numeric series in a project that uses (or could use) mypackage.
---

<!-- template-example: this whole plugin describes the bundled example package; on adoption, rewrite it for your package (rename the skill to <verb>-with-<yourpackage>, keep the structure: find it, what lives where, the rules, a worked example that runs, a "don't write it" table, a self-check). -->

# Series analysis on mypackage

mypackage turns the usual loops over a list of numbers — a mean and variance, a rolling median, a z-score scaler fitted on one split and applied to another — into small, validated, pure functions and composable transforms, with no runtime dependencies. Before writing a statistic, a smoothing loop, a validator or a scaler, look it up:

1. **The capability index** lists every public name with a one-line summary:
   <https://jejjohnson.github.io/pypackage_template/reference/capabilities/>.
   Or list the installed version:

   ```python
   import importlib
   import inspect

   for name in ("mypackage", "mypackage.cli"):
       module = importlib.import_module(name)
       for attr in getattr(module, "__all__", []):
           if attr.startswith("_"):
               continue  # __version__
           doc = (inspect.getdoc(getattr(module, attr)) or "").split("\n")[0]
           print(f"{name}.{attr}: {doc}")
   ```

2. Compose what exists. Install it from GitHub: `uv add "mypackage @ git+https://github.com/jejjohnson/pypackage_template.git"`.

## What lives where

| You need… | Use |
|---|---|
| Count, mean, variance, standard deviation, min / max, median | `mp.summarize(series, ddof=1)` → a frozen `Summary` (`.std`, `.spread`, `.to_dict()`) |
| A quantile, z-scores | `mp.quantile(series, q)` (linear interpolation, NumPy's default), `mp.zscores` |
| Statistics over a stream, in constant memory | `mp.RunningStats().update(x)` / `.update_many(xs)` (Welford) |
| A rolling mean, a rolling median, an exponential smoother | `mp.moving_average`, `mp.median_filter` (removes isolated spikes), `mp.exponential_moving_average` (causal) |
| Control over the ends of a smoothed series | `padding=` one of `mp.Padding` (`"edge"`, `"zero"`, `"reflect"`, `"none"`) |
| Standardise or rescale with parameters learned on one split | `mp.Standardize` (with `inverse`), `mp.MinMaxScale`; `.fit(train)` then `.apply(test)` |
| Clamp, or smooth inside a pipeline | `mp.Clip`, `mp.MovingAverage` |
| Several steps in order, fitted step by step | `mp.chain(a, b, ...)` or `mp.Pipeline([...])`; append a step with the pipe operator |
| Validate input in your own code | `mp.as_floats`, `mp.require_non_empty`, `mp.require_positive` |
| A fixed-size sliding buffer, a wall-clock timer | `mp.Window`, `mp.timer` |
| The same from the shell | `mypackage summarize data.txt --json`, `mypackage smooth data.txt --method median --window 5` |

## The rules your code must keep

- **Inputs are never mutated.** No function or method changes the series it is given, and any sequence of numbers (`list`, `tuple`, `range`) is accepted. The series-to-series functions and every `apply` return a new list; `summarize` returns a `Summary`, `quantile` a float, and `fit` the transform itself.
- **Fit, then apply.** `Standardize` and `MinMaxScale` learn from `fit` and raise `mp.NotFittedError` if applied before it; `fit` returns the transform, so `scaler = mp.Standardize().fit(train)` keeps it fitted. Fit on training data only, and reuse the fitted object on new data.
- **Transforms are structural.** Anything with an `apply(series) -> list[float]` method is a `mp.Transform` and composes in a pipeline: no base class to inherit.
- **Errors are typed.** Bad input raises `mp.ValidationError` (a `ValueError`), empty input `mp.EmptySeriesError`; catch `mp.MypackageError` for all of them.
- **Degenerate data is an error, not a `nan`**: a constant series cannot be standardised, `ddof` must be smaller than the count, `window` no longer than the series.

## Worked example

```python
from dataclasses import dataclass

import mypackage as mp


# Hourly temperatures (°C) with one sensor glitch at index 10
readings = [20.1, 20.4, 20.2, 20.8, 21.0, 20.7, 21.3, 21.1, 21.6, 21.4]
readings += [95.0, 21.9, 22.1, 21.8, 22.4, 22.6, 22.3, 22.9, 23.0, 22.8]

# 1. Describe the raw series: the glitch drags the mean, not the median
raw = mp.summarize(readings)  # frozen Summary: count, mean, variance, ...

# 2. Remove the spike: the median filter discards it, a moving average
#    would smear it over three samples
cleaned = mp.median_filter(readings, window=3)  # new list, same length
smeared = mp.moving_average(readings, window=3)
clean = mp.summarize(cleaned)

# 3. Fit a pipeline on the first 14 hours, apply it to the last 6
train, test = cleaned[:14], cleaned[14:]
pipeline = mp.chain(mp.Standardize(), mp.Clip(lower=-3.0, upper=3.0))
z_train = pipeline.fit_apply(train)
z_test = pipeline.apply(test)  # reuses the training mean and std

# 4. The same statistics, streamed one value at a time
running = mp.RunningStats().update_many(cleaned)


# 5. Your own transform composes without a base class
@dataclass(frozen=True, slots=True)
class Difference:
    def apply(self, series: mp.Series) -> list[float]:
        values = mp.as_floats(series)
        return [b - a for a, b in zip(values, values[1:])]


steps = (pipeline | Difference()).apply(test)  # hour-to-hour change, in std units

# 6. Using a fitted transform before fit is an error, not a silent no-op
try:
    mp.Standardize().apply(test)
except mp.NotFittedError as err:
    message = str(err)
```

The glitch drags the raw mean to 25.27 °C and the standard deviation to 16.4, while the median stays at 21.7. The median filter removes it outright (the cleaned maximum is 22.9, the standard deviation 0.90), where the moving average smears it over three samples near 46. The last six hours sit 1.8 to 2.5 training standard deviations above the training mean, so the ±3 clip leaves them alone, and `RunningStats` agrees with `summarize` on the cleaned series.

## Don't write it — use mypackage

| Don't write… | Use |
|---|---|
| `sum(xs) / len(xs)`, a variance loop, `sorted(xs)[len(xs) // 2]` | `mp.summarize(xs)` (`.mean`, `.variance`, `.median`) |
| `[sum(xs[i - 1 : i + 2]) / 3 for i in ...]` and edge cases | `mp.moving_average(xs, window=3, padding="edge")` |
| A rolling `statistics.median` over a `deque` | `mp.median_filter(xs, window)` |
| `(x - mean) / std` with the mean and std kept in variables | `mp.Standardize().fit(train)`, then `.apply(test)` and `.inverse(...)` |
| `max(lo, min(hi, x))` in a comprehension | `mp.Clip(lower=lo, upper=hi).apply(xs)` |
| A list of functions applied in a loop | `mp.chain(...)` / `mp.Pipeline([...])` |
| `if not xs: raise ValueError(...)`, `[float(x) for x in xs]` | `mp.require_non_empty(xs)`, `mp.as_floats(xs)` |

## Self-check before you finish

- No hand-written mean, variance, quantile or rolling loop remains where a mypackage function does the job.
- Every scaler is fitted on training data only, and the fitted object (not a new one) is applied to new data.
- The caller's lists are unchanged after your code runs.
- Errors you catch are `mp.MypackageError` subclasses, not a bare `except`.

If mypackage lacks what you need, keep your addition small and shaped like it (a pure function that validates with `mp.require_non_empty`, or a class with `apply`) and consider proposing it upstream at
<https://github.com/jejjohnson/pypackage_template/issues>.
