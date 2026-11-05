"""Descriptive statistics, smoothing filters, and composable transforms.

``mypackage`` is the worked example that ships with this project template. It
is deliberately **dependency-free** — everything here is built on the standard
library — so the template stays domain-neutral while still giving every piece
of tooling something real to work on: `mkdocstrings` renders these docstrings,
`ty` checks these annotations, `pytest` exercises these code paths, and the
doctests below are verified on every test run.

## The three layers

| Layer | Module | What it gives you |
|---|---|---|
| Statistics | `mypackage.stats` | `summarize`, `quantile`, `RunningStats` |
| Smoothing | `mypackage.smoothing` | `moving_average`, `median_filter` |
| Transforms | `mypackage.transforms` | `Standardize`, `Clip`, `Pipeline` |

## Quickstart

```python
import mypackage as mp

series = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]

summary = mp.summarize(series)
smoothed = mp.moving_average(series, window=3)
scaled = mp.chain(mp.Standardize(), mp.Clip(-2.0, 2.0)).fit_apply(
    series
)
```

Everything listed in ``__all__`` is importable straight from the top-level
package; the submodules are an implementation detail you are free to reach
into, but never have to.
"""

from __future__ import annotations

from mypackage._typing import Fittable, Series, Transform
from mypackage.exceptions import (
    EmptySeriesError,
    MypackageError,
    NotFittedError,
    ValidationError,
)
from mypackage.smoothing import (
    Padding,
    exponential_moving_average,
    median_filter,
    moving_average,
)
from mypackage.stats import RunningStats, Summary, quantile, summarize, zscores
from mypackage.transforms import (
    Clip,
    MinMaxScale,
    MovingAverage,
    Pipeline,
    Standardize,
    chain,
)
from mypackage.utils import (
    Elapsed,
    Window,
    as_floats,
    require_non_empty,
    require_positive,
    timer,
)


__version__ = "0.1.7"

__all__ = [
    "Clip",
    "Elapsed",
    "EmptySeriesError",
    "Fittable",
    "MinMaxScale",
    "MovingAverage",
    "MypackageError",
    "NotFittedError",
    "Padding",
    "Pipeline",
    "RunningStats",
    "Series",
    "Standardize",
    "Summary",
    "Transform",
    "ValidationError",
    "Window",
    "__version__",
    "as_floats",
    "chain",
    "exponential_moving_average",
    "median_filter",
    "moving_average",
    "quantile",
    "require_non_empty",
    "require_positive",
    "summarize",
    "timer",
    "zscores",
]
