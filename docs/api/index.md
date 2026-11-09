# API Reference

Everything listed in `functax.__all__` is importable from the top level:

```python
import functax as mp

mp.summarize(...)  # same object as functax.stats.summarize
```

The submodules below are an implementation detail you are welcome to reach
into, but never have to. The [capability index](capabilities.md) lists every
public name on one page, with a one-line summary each.

## Modules

| Module | Contents |
|---|---|
| [Statistics](stats.md) | `summarize`, `Summary`, `quantile`, `zscores`, `RunningStats` |
| [Smoothing](smoothing.md) | `moving_average`, `exponential_moving_average`, `median_filter`, `Padding` |
| [Transforms](transforms.md) | `Standardize`, `MinMaxScale`, `Clip`, `MovingAverage`, `Pipeline`, `chain` |
| [Utilities](utils.md) | `as_floats`, `require_non_empty`, `require_positive`, `Window`, `timer` |
| [Protocols](typing.md) | `Series`, `Transform`, `Fittable` |
| [Exceptions](exceptions.md) | `FunctaxError`, `ValidationError`, `EmptySeriesError`, `NotFittedError` |
| [CLI](cli.md) | `main`, `build_parser` |

## Package overview

::: functax
    options:
      members: false
      show_root_heading: false
      show_source: false
