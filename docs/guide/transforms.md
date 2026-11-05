# Transforms

A *transform* is anything with an `apply` method that maps a series to a new
series. That is the entire contract:

```python
from mypackage import Series

class Square:
    def apply(self, series: Series) -> list[float]:
        return [value**2 for value in series]
```

`Square` inherits from nothing and registers with nothing, yet it composes
with the built-ins, satisfies `isinstance` checks, and works inside a
[`Pipeline`](xref:api#mypackage.Pipeline):

```python
import mypackage as mp

isinstance(Square(), mp.Transform)                 # True
mp.chain(mp.Standardize(), Square()).fit_apply(data)
```

This works because [`Transform`](xref:api#mypackage.Transform) is a
`runtime_checkable` `Protocol` — *structural* typing. The static checker
verifies the shape; `isinstance` verifies the method exists at runtime.

:::{note} Protocol `isinstance` checks are shallow
A `runtime_checkable` protocol only checks that the attribute *exists* —
not its signature. Static type checking (`uv run poe typecheck`) is what catches
an `apply` with the wrong parameters. The runtime check exists to give a
clear [`ValidationError`](xref:api#mypackage.ValidationError) when someone drops a
string into a pipeline, not to replace the type checker.
:::

## Stateless versus fitted

| Transform | Learns from data? | Protocol |
|---|---|---|
| [`Clip`](xref:api#mypackage.Clip) | no | `Transform` |
| [`MovingAverage`](xref:api#mypackage.MovingAverage) | no | `Transform` |
| [`Standardize`](xref:api#mypackage.Standardize) | yes — mean, std | `Transform`, `Fittable` |
| [`MinMaxScale`](xref:api#mypackage.MinMaxScale) | yes — min, max | `Transform`, `Fittable` |

Stateless transforms implement `fit` as a no-op returning `self`, so they drop
into a pipeline next to fitted ones without any special casing.

Fitted transforms raise [`NotFittedError`](xref:api#mypackage.NotFittedError) when used
before `fit` — never a silent identity, never a `None`-propagating `nan`:

```python
mp.Standardize().apply([1.0, 2.0])
# NotFittedError: Standardize is not fitted; call .fit(series) first
```

## Leakage

The reason `fit` and `apply` are separate steps at all:

```python
train = [10.0, 12.0, 14.0]
test = [16.0, 18.0]

scaler = mp.Standardize().fit(train)   # statistics come from train only
scaler.apply(train)                    # [-1.0, 0.0, 1.0]
scaler.apply(test)                     # [2.0, 3.0]  — same scale, not re-fitted
```

Calling `fit_apply(test)` instead would re-learn the mean and standard
deviation from the test data, quietly leaking information about it into the
transform. The shortcut exists — `fit_apply` — precisely so that the leaking
version has to be *typed out*, rather than being what you get by default.

[`Standardize`](xref:api#mypackage.Standardize) also provides `inverse`, which maps
standardised values back to the original scale:

```python
scaler.inverse(scaler.apply(train)) == train   # True (to floating point)
```

## Pipelines

[`Pipeline`](xref:api#mypackage.Pipeline) applies steps left to right. Fitting is
sequential: each step is fitted on the *output of the previous step*, exactly
as it will see the data at apply time.

```python
pipeline = mp.Pipeline([
    mp.Clip(upper=50.0),
    mp.Standardize(),
    mp.MovingAverage(window=3),
])

clean = pipeline.fit_apply(raw)
```

:::{dropdown} Why sequential fitting matters
Consider `Clip(upper=10.0)` followed by `Standardize()` on the series
`[0, 10, 1000]`. The clip turns it into `[0, 10, 10]`, so the scaler must
learn a mean of `6.67` — not the `336.67` of the raw series. Fitting each
step on raw data instead would produce a pipeline whose `apply` behaves
completely differently from its `fit`.

```python
pipeline = mp.Pipeline([mp.Clip(upper=10.0), mp.Standardize()])
pipeline.fit([0.0, 10.0, 1000.0])
pipeline[1].mean_          # 6.666..., not 336.666...
```
:::

### Composition

Pipelines compose with `|`. The operands are never mutated — you always get a
new pipeline:

```python
base = mp.chain(mp.Clip(lower=0.0), mp.Standardize())
extended = base | mp.MovingAverage(3)
nested = base | mp.chain(mp.Clip(-3.0, 3.0), Square())

len(base), len(extended), len(nested)   # (2, 3, 4)
```

They are also sequences — iterable, indexable, sized:

```python
for step in pipeline:
    print(type(step).__name__)

pipeline[0]      # Clip(lower=None, upper=50.0)
len(pipeline)    # 3
```

An empty pipeline is the identity, which makes `Pipeline(steps)` safe to build
from a config file that happens to list no steps.

## Writing your own

Make it a frozen dataclass, take configuration in `__init__`, validate there,
and keep `apply` pure:

```python
from dataclasses import dataclass

from mypackage import Series, ValidationError

@dataclass(frozen=True, slots=True)
class Detrend:
    """Subtract a rolling baseline estimated by a median filter."""

    window: int

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValidationError(f"'window' must be positive, got {self.window}")

    def fit(self, series: Series) -> "Detrend":
        return self

    def apply(self, series: Series) -> list[float]:
        baseline = mp.median_filter(series, self.window)
        return [x - b for x, b in zip(series, baseline, strict=True)]
```

Checklist for a well-behaved transform:

- [x] `apply` returns a **new** list and never mutates its input
- [x] configuration is validated at construction, not at apply time
- [x] `fit` returns `self` so it chains
- [x] stateful transforms raise
      [`NotFittedError`](xref:api#mypackage.NotFittedError) before `fit`
- [x] errors derive from [`MypackageError`](xref:api#mypackage.MypackageError)

## API

Full signatures live in the [transforms API reference](xref:api#mypackage.transforms).
