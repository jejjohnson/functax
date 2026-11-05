# Statistics

Two complementary APIs cover the same ground:

| | Batch | Streaming |
|---|---|---|
| Entry point | [`summarize`](xref:api#mypackage.summarize) | [`RunningStats`](xref:api#mypackage.RunningStats) |
| Input | the whole series | one value at a time |
| Memory | $O(N)$ | $O(1)$ |
| Gives you | mean, variance, extrema, **median** | mean, variance, extrema |

The median is the reason the streaming API is not a drop-in replacement:
order statistics cannot be maintained exactly in constant space.

## Definitions

For observations $x_1, \dots, x_N$:

$$\bar{x} = \frac{1}{N} \sum_{i=1}^{N} x_i,
\qquad
s^2 = \frac{1}{N - \delta} \sum_{i=1}^{N} (x_i - \bar{x})^2$$

where $\delta$ is `ddof`. Use $\delta = 1$ (the default) when the series is a
*sample* from a larger population, and $\delta = 0$ when it *is* the
population.

:::{dropdown} Why is the default `ddof=1` rather than `0`?
With $\delta = 0$ the estimator is biased low: it divides by $N$ but the
deviations are taken from the *sample* mean, which is itself fitted to the
data and therefore sits closer to the points than the true mean does.
Bessel's correction ($\delta = 1$) compensates exactly. NumPy defaults to
$\delta = 0$; pandas and R default to $\delta = 1$. This package follows
pandas, because "I have a sample" is the more common situation.
:::

## Quantiles

[`quantile`](xref:api#mypackage.quantile) uses linear interpolation between order
statistics — the same as NumPy's default `method="linear"`. With
$h = q (N-1)$ and $j = \lfloor h \rfloor$:

$$Q(q) = x_{(j+1)} + (h - j) (x_{(j+2)} - x_{(j+1)})$$

```python
import mypackage as mp

data = [1.0, 2.0, 3.0, 4.0]
mp.quantile(data, 0.0)    # 1.0
mp.quantile(data, 0.25)   # 1.75
mp.quantile(data, 0.5)    # 2.5
mp.quantile(data, 1.0)    # 4.0
```

The input is sorted internally, so order doesn't matter, and the original
list is never touched.

## Standardising

[`zscores`](xref:api#mypackage.zscores) is the functional form; the stateful
[`Standardize`](xref:api#mypackage.Standardize) transform is the one you want when the
statistics must be *learned once and reused* — see
[Transforms](transforms.md#leakage).

```python
mp.zscores([1.0, 2.0, 3.0])    # [-1.0, 0.0, 1.0]
```

:::{warning} Constant series
A constant series has $s = 0$, so standardising it is undefined. Both
`zscores` and `Standardize.fit` raise
[`ValidationError`](xref:api#mypackage.ValidationError) rather than silently
returning zeros or `nan`.
:::

## Streaming with Welford's algorithm

[`RunningStats`](xref:api#mypackage.RunningStats) folds each observation into a running
mean and a running sum of squared deviations:

$$\bar{x}_n = \bar{x}_{n-1} + \frac{x_n - \bar{x}_{n-1}}{n},
\qquad
M_{2,n} = M_{2,n-1} + (x_n - \bar{x}_{n-1})(x_n - \bar{x}_n)$$

with $s^2 = M_{2,N} / (N - \delta)$.

```python
stats = mp.RunningStats()
stats.update_many([1.0, 2.0, 3.0, 4.0])

stats.count      # 4
stats.mean       # 2.5
stats.std        # 1.2909944487358056
stats.minimum    # 1.0
```

Updates are chainable, and the accumulator can be fed indefinitely:

```python
stats = mp.RunningStats(ddof=0).update(1.0).update(2.0).update(3.0)
```

## Numerical stability

This is the reason both APIs avoid the textbook "sum of squares" formula.
Consider four values offset by $10^9$:

```python
offset = 1e9
values = [offset + d for d in (1.0, 2.0, 3.0, 4.0)]

mp.summarize(values).variance                  # 1.6666666666666667
mp.RunningStats().update_many(values).variance # 1.6666666666666667
```

Both give the exact answer $5/3$. The naive alternative does not:

```python
n = len(values)
naive = (sum(v * v for v in values) - sum(values) ** 2 / n) / (n - 1)
# -21845.333333333332   ← catastrophic cancellation, and negative
```

:::{danger} Never use the naive formula
It is not a rounding-error problem you can shrug off: the result has the
wrong *sign*. `math.fsum` plus deviation-based accumulation costs nothing
and is always correct.
:::

## API

- [`summarize`](xref:api#mypackage.summarize)
- [`Summary`](xref:api#mypackage.Summary)
- [`quantile`](xref:api#mypackage.quantile)
- [`zscores`](xref:api#mypackage.zscores)
- [`RunningStats`](xref:api#mypackage.RunningStats)

Full signatures live in the [statistics API reference](xref:api#mypackage.stats).
