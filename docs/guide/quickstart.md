# Quickstart

Everything in this page is a real, runnable snippet. If you prefer an
executable version with plots, jump to the
[guided tour notebook](../notebooks/tour.ipynb).

## Install

```bash
uv add mypackage      # or: pip install mypackage
```

There are no runtime dependencies, so this pulls in nothing else.

## Describe a series

[`summarize`](xref:api#mypackage.summarize) returns a frozen
[`Summary`](xref:api#mypackage.Summary) — immutable, hashable, comparable.

```python
import mypackage as mp

series = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
summary = mp.summarize(series)

summary.count      # 8
summary.mean       # 5.0
summary.median     # 4.5
summary.std        # 2.1380899352993947
summary.spread     # 7.0
```

The variance uses the unbiased estimator by default. Pass `ddof=0` for the
population variance:

```python
mp.summarize(series, ddof=0).variance   # 4.0
```

:::{note} Why `fsum`?
Summation uses `math.fsum`, and the variance is computed from deviations
rather than from $\sum x^2 - (\sum x)^2 / N$. The naive formula loses
every significant digit when the mean is large — a series centred on
$10^9$ returns a *negative* variance. See
[Statistics](statistics.md#numerical-stability).
:::

## Stream it instead

When the data doesn't fit in memory, [`RunningStats`](xref:api#mypackage.RunningStats)
consumes it one value at a time in constant space:

```python
stats = mp.RunningStats()
for value in series:
    stats.update(value)

stats.mean == mp.summarize(series).mean   # True
```

## Smooth it

```python
mp.moving_average(series, window=3)              # linear, symmetric
mp.exponential_moving_average(series, alpha=0.3) # linear, causal
mp.median_filter(series, window=3)               # non-linear, spike-proof
```

The filters differ in how they treat an isolated outlier:

```python
spike = [1.0, 1.0, 99.0, 1.0, 1.0]

mp.moving_average(spike, 3)   # [1.0, 33.67, 33.67, 33.67, 1.0] — smeared
mp.median_filter(spike, 3)    # [1.0,   1.0,   1.0,   1.0, 1.0] — removed
```

## Compose transforms

[`Pipeline`](xref:api#mypackage.Pipeline) fits each step on the output of the previous
one, then applies them in order:

```python
pipeline = mp.chain(
    mp.Clip(upper=50.0),      # drop implausible spikes first
    mp.Standardize(),         # then centre and scale
    mp.MovingAverage(3),      # finally smooth
)

clean = pipeline.fit_apply(series)
```

Pipelines compose with `|`, and the result is a new pipeline — neither operand
is mutated:

```python
extended = pipeline | mp.Clip(-3.0, 3.0)
len(pipeline), len(extended)   # (3, 4)
```

## Handle failures

Every error derives from [`MypackageError`](xref:api#mypackage.MypackageError):

```python
try:
    mp.summarize([])
except mp.EmptySeriesError as err:
    print(err)      # 'series' must contain at least one value, got 0
```

…and from the closest standard-library exception, so existing `except` clauses
keep working:

```python
isinstance(mp.EmptySeriesError(), ValueError)   # True
isinstance(mp.NotFittedError(), RuntimeError)   # True
```

## Next steps

- [Statistics](statistics.md) — the estimators and their numerics
- [Smoothing](smoothing.md) — filter maths and boundary handling
- [Transforms](transforms.md) — writing your own transform
- [CLI](cli.md) — the `mypackage` console script
