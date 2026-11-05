# Smoothing

Three filters, one decision: **what do you want to preserve?**

| Filter | Kind | Support | Preserves | Destroys |
|---|---|---|---|---|
| [`moving_average`](xref:api#mypackage.moving_average) | linear | finite, symmetric | slow trends | edges, spikes |
| [`exponential_moving_average`](xref:api#mypackage.exponential_moving_average) | linear | infinite, causal | streaming latency | phase |
| [`median_filter`](xref:api#mypackage.median_filter) | non-linear | finite, symmetric | **edges** | fine texture |

## Moving average

With window length $w$ and $l = \lfloor w/2 \rfloor$:

$$y_i = \frac{1}{w} \sum_{k=-l}^{w - 1 - l} x_{i+k}$$

This is a convolution with a rectangular kernel. In the frequency domain that
kernel is a sinc, so it does not cleanly separate "signal" from "noise" — it
attenuates some high frequencies and *inverts* others. It is nonetheless the
right default: cheap, unbiased on a locally linear trend, and easy to reason
about.

```python
import mypackage as mp

mp.moving_average([1.0, 2.0, 3.0, 4.0, 5.0], window=3)
# [1.333..., 2.0, 3.0, 4.0, 4.666...]
```

A window of `1` is the identity, and the window may not exceed the series
length.

## Exponential moving average

$$y_0 = x_0, \qquad y_i = \alpha x_i + (1 - \alpha)\, y_{i-1}$$

Expanding the recursion shows the weights decay geometrically into the past:

$$y_i = \alpha \sum_{k=0}^{i-1} (1-\alpha)^k x_{i-k} + (1-\alpha)^i x_0$$

:::{note} Causality is the point
`y[i]` depends only on samples up to `i`, so this filter runs on a live
stream where a centred window cannot. The price is a phase lag of roughly
$(1-\alpha)/\alpha$ samples — with $\alpha = 0.1$, features arrive about
nine samples late.
:::

```python
mp.exponential_moving_average([1.0, 2.0, 3.0], alpha=0.5)
# [1.0, 1.5, 2.25]
```

$\alpha = 1$ is the identity; smaller $\alpha$ smooths harder.

## Median filter

The median has a **breakdown point of 50%**: up to half the samples in a
window can be arbitrarily corrupted without moving the output at all. The mean
has a breakdown point of $0$ — a single infinite sample takes the whole window
with it.

::::{tab-set}
:::{tab-item} Median filter

```python
mp.median_filter([1.0, 1.0, 99.0, 1.0, 1.0], window=3)
# [1.0, 1.0, 1.0, 1.0, 1.0]     ← spike gone
```
:::
:::{tab-item} Moving average

```python
mp.moving_average([1.0, 1.0, 99.0, 1.0, 1.0], window=3)
# [1.0, 33.67, 33.67, 33.67, 1.0]   ← spike smeared over 3 samples
```
:::
::::

The same property makes it preserve step edges that a mean filter blurs:

```python
step = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

mp.median_filter(step, 3)     # [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]  ← sharp
mp.moving_average(step, 3)    # [0.0, 0.0, 0.33, 0.67, 1.0, 1.0] ← ramped
```

The trade-off is that the median is non-linear, so it does not commute with
scaling and addition, and it erases genuine fine texture along with the noise.

## Boundary handling

Any finite-support filter needs values past the ends of the series.
[`Padding`](xref:api#mypackage.Padding) makes that choice explicit instead of hiding it.

| Strategy | Extension | Output length | Use when |
|---|---|---|---|
| `Padding.EDGE` *(default)* | repeat first/last value | `len(series)` | the signal is roughly flat at the ends |
| `Padding.REFLECT` | mirror without repeating endpoints | `len(series)` | the ends are oscillatory rather than monotone |
| `Padding.ZERO` | pad with `0.0` | `len(series)` | the signal is genuinely mean-zero |
| `Padding.NONE` | no extension | `len(series) - window + 1` | you'd rather have fewer, honest samples |

```python
from mypackage import Padding

mp.moving_average([0.0, 0.0, 9.0], window=3, padding=Padding.ZERO)
# [0.0, 3.0, 3.0]

mp.moving_average([1.0, 2.0, 3.0, 4.0, 5.0], window=3, padding="none")
# [2.0, 3.0, 4.0]      ← two samples shorter
```

`Padding` is a `StrEnum`, so the plain strings `"edge"`, `"reflect"`,
`"zero"`, and `"none"` work anywhere the enum does — handy for CLI flags and
config files.

:::{note} Reflection does not preserve slope
Mirroring is *even* reflection: the padded values rise where the signal
falls. On a monotone ramp that puts them on the wrong side of the truth,
producing exactly **twice** the boundary error of edge padding. Reflect
earns its keep when the ends are oscillatory, where repeating a single
value is the worse approximation. The
[deep dive notebook](../notebooks/smoothing_deep_dive.ipynb) measures both.
:::

:::{warning} Zero padding is rarely what you want
Unless the series really is centred on zero, `Padding.ZERO` drags both
ends of the output toward the origin, creating an artefact that looks
exactly like a real trend. It is included because signal-processing code
often assumes it, not because it is a good default.
:::

## Choosing a window

There is no universal answer, but a serviceable rule:

1. Start with the smallest odd window that spans the noise correlation
   length.
2. Check that the *residual* (input minus output) looks like noise, not like
   signal. If the residual has structure, the window is too wide.
3. If isolated spikes survive, you want the median filter, not a wider mean.

The [smoothing deep dive notebook](../notebooks/smoothing_deep_dive.ipynb)
walks through this with plots and a quantitative error sweep.

## API

Full signatures live in the [smoothing API reference](xref:api#mypackage.smoothing).
