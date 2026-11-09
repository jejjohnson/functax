# Interpolation

Every plan reads a field as $\hat{\vect{f}}(\mat{X}_*) = \mat{H} \mathcal{S}(\vect{y})$: a fit $\mathcal{S}$ turns samples $\vect{y}$ into coefficients (the identity for local schemes), and a read-out $\mat{H}$ evaluates them at the targets. The adjoint $\mathcal{S}^\top \mat{H}^\top$ sends residuals back to the source.

## Stencils

A `Stencil` stores $\mat{H}$ row by row: row $k$ reads $S$ source values at flat indices $i_{ks}$ with weights $w_{ks}$, so

$$
\hat{f}_k = \sum_{s=1}^{S} w_{ks} c_{i_{ks}} \text{ and } (\mat{H}^\top \vect{r})_j = \sum_{k=1}^{M} \sum_{s=1}^{S} [i_{ks} = j] w_{ks} r_k.
$$

`apply` is a gather and a contraction, and `apply_adjoint` a scatter-add, so repeated indices in one row add. The hand-written adjoint is checked against `jax.linear_transpose` by the dot-product test, and `operator()` returns the same map as a lineax operator. Padding taps carry weight 0 and repeat an index their row already reads, so shapes stay static.

## Locate and policies

`locate` turns each target coordinate into a cell index, an offset and an inside flag on one `Axis`, through its fractional index position $u$ (node $j$ sits at $u = j$). A regular axis uses arithmetic, a rectilinear one `searchsorted`, a descending axis its reversed nodes, and a periodic axis of period $P$ maps targets into one period and adds a wrap cell between its last node and the first one plus $P$. `remap_taps` brings the taps that run off an end back onto the axis.

`Policies` names what happens at the edges. `outside` acts on targets off the domain, `missing` on non-finite source samples, `boundary` on taps past the end of an axis, and `extent` says whether the domain ends at the end nodes or half a cell beyond them. The conventions the stack used before functax map onto them as follows.

| Convention | `Policies` |
|---|---|
| geoops `interp_at`: NaN and `valid=False` outside, the half-pixel border reads the edge | `Policies(extent="cells", boundary="edge")` |
| geopatcher `PointDomain`: NaN outside | `Policies()` |
| a geom sampler writing its fill value (or 0) outside | `Policies(fill_value=fill)` |
| vardax `InterpObs`: clamp without flagging | `Policies(outside="clamp")` |
| the hand-written "raise if outside" checks | `Policies(outside="raise")` |

## Schemes and plans

A scheme is configuration only; `scheme.plan(source, target, policies=…)` freezes it for one source and one target geometry into a `Plan`. Sources are an `Axis`, a `RectilinearGrid` or an `AffineGrid` (whose targets are map coordinates, read through `to_pixel`), with data flattened row-major; targets are `Points` or a grid read at its nodes. `Nearest` reads one tap per axis (a tie goes to the lower node, as in SciPy's `RegularGridInterpolator`), `Linear` two, and `Tensor` takes one scheme per axis, so $S = \prod_d S_d$ taps per target at flat index $\sum_d j_d \prod_{d' > d} n_{d'}$.

A plan applies $\hat{\vect{f}} = \mat{H} \mathcal{S}(\vect{y})$ over any leading batch dims and writes `fill_value` at invalid rows; `adjoint` applies $\mathcal{S}^\top \mat{H}^\top$, ignoring residuals at invalid rows; `operator()` is the same linear map as a lineax operator, with zero rows where the plan is invalid. The [interpolation guide](../interpolation/) walks through reading a grid at points and sending residuals back.

## Schemes

| Scheme | Taps per axis | Fit | Reproduces | Oracle |
|---|---|---|---|---|
| `Nearest()` | 1 | identity | constants | `RegularGridInterpolator(method="nearest")` |
| `Linear()` | 2 | identity | affine functions | `RegularGridInterpolator`, `map_coordinates(order=1)` |
| `Cubic(a=-0.5)` | 4 (Keys) | identity | quadratics | gaussx `cubic_interpolation_weights` |
| `BSpline(order=k)`, $k \le 5$ | $k + 1$ | the `BandedPrefilter` from $k = 2$ | degree-$k$ polynomials away from the ends | `map_coordinates(order=k, mode="mirror")` |

A B-spline of order $k \ge 2$ does not pass through its samples, so its plan first solves $\vect{y} = \mat{B} \vect{c}$ with the banded matrix $\mat{B}$ of B-spline values at the knots. `BandedPrefilter` does this per axis by recursive filtering, a causal and an anticausal pass for each pole $z_p$ after the gain $\lambda = \prod_p (1 - z_p)(1 - z_p^{-1})$, with the mirror boundary (or the wrap on a periodic axis), as `scipy.ndimage.spline_filter` does. The plan's adjoint applies $\mathcal{S}^\top$ after $\mat{H}^\top$.

## Derivatives

`plan.derivative(axis=c, order=m)` returns a `DerivativePlan` whose read-out holds $\partial^m \phi_j / \partial x_c^m$ at each target, so it evaluates $\partial^m \hat{f} / \partial x_c^m$ with the plan's fit, taps and validity and has its own adjoint. The coordinate $c$ is a column of the targets' coordinates: per source axis on an `Axis` or a `RectilinearGrid`, and $(x, y)$ map coordinates on an `AffineGrid`, so a rotated grid differentiates along the map axes. The weights are the forward-mode derivative of the weight construction, so the chain rule through the spacing, the cell widths and the transform is exact, and `jax.grad` through a plan built on traced targets gives the same values. Plans on one source add (`plan.derivative(axis=-1, order=2) + plan.derivative(axis=-2, order=2)` is a Laplacian). Nearest has zero derivative, linear a piecewise-constant one, and a B-spline of order $k$ is $k - 1$ times continuously differentiable.

## Composition and windows

`outer @ inner` folds two plans into one `ComposedPlan`, with the product stencil computed when it is built: an instrument response after spectral interpolation, or a warp followed by a sample, becomes one sparse $\mat{H}$ with one adjoint. For an inner stencil $\mat{H}_a$ ($P \times N$, $S_a$ taps per row) and an outer stencil $\mat{H}_b$ ($M \times P$, $S_b$ taps per row), row $k$ of the product is

$$
(\mat{H}_b \mat{H}_a)_{kj} = \sum_{s=1}^{S_b} w^b_{ks} (\mat{H}_a)_{i^b_{ks} j},
$$

so it has $S_a S_b$ taps, at indices $i^a_{i^b_{ks} r}$ with weights $w^b_{ks} w^a_{i^b_{ks} r}$. `compose(outer, inner, merge=True)` sums repeated indices on the host. A row of $\mat{H}_b$ that reads an invalid row of $\mat{H}_a$ is invalid, at any weight, and on every row it marks valid the composed plan equals `outer(inner(y))`. Its adjoint is $\mat{H}_a^\top \mat{H}_b^\top$. The outer plan must be local with `missing="propagate"`, and the inner one may fill missing samples (with the outer plan's fill value) but not renormalise; plans that do not fold raise `PolicyError`, and their linear maps still compose in lineax as `outer.operator() @ inner.operator()`.

A plan on a grid target keeps the grid's shape, and `plan[window]` takes one slice per target axis and returns a `WindowedPlan` with the rows inside it, sources unchanged and `valid` sliced alongside. Each row is the parent's row, so every chip `plan[window](y)` equals the matching slice of `plan(y)` bit for bit: one plan on the destination grid, cut per chip, replaces warping each chip with a margin.

`plan.to_gaussx()` returns the indices and weights of a local plan in gaussx's row layout, with invalid rows zeroed, which is what `InterpolatedOperator(base_operator, interp_indices, interp_values)` takes for $\mat{W} \mat{K} \mat{W}^\top$; functax imports nothing from gaussx.

## 1-D series schemes

`CubicSpline`, `Pchip`, `Akima` and `Steffen` read one `Axis` (increasing or decreasing) with a cubic Hermite read-out: on cell $[x_i, x_{i+1}]$ of width $h_i$ at offset $t$,

$$
\hat{f} = h_{00}(t) y_i + h_{10}(t) h_i m_i + h_{01}(t) y_{i+1} + h_{11}(t) h_i m_{i+1},
$$

and they differ in the slopes $m$. `CubicSpline(bc)` solves one tridiagonal system for the $C^2$ spline (not-a-knot, natural, clamped, or periodic by Sherman-Morrison, as `scipy.interpolate.CubicSpline`); its fit is linear in $\vect{y}$, so it builds an ordinary `Plan` with an adjoint. `Pchip` (Fritsch-Carlson), `Akima` and `Steffen` preserve shape (PCHIP and Steffen never overshoot monotone data), and their slopes depend on $\vect{y}$ nonlinearly, so they build a `NonlinearPlan`: it reads like any plan, has no fixed operator or adjoint (`NonlinearSchemeError`), and `linearize(y)` returns the tangent-linear `Stencil` at $\vect{y}$, 4 taps per target (6 for Akima). `scheme.fit(axis, y)` returns a `SeriesInterpolant` whose fit runs in JAX, so gradients reach both $\vect{y}$ and the knots.

## Gaussian plans

A scheme with a probabilistic reading builds a `GaussianPlan`: its mean is the plan, $\hat{\vect{f}} = \mat{H} \mathcal{S}(\vect{y})$, and `variance()`, `covariance()` and `sample(y, key=…)` add the posterior covariance $\gmat{\Sigma}_*$ at the $M$ targets, held by a `PosteriorCovariance`. Other plans have no `variance` method, rather than one that raises.

On one axis, linear interpolation is the posterior mean of Brownian motion with variance rate $\sigma^2$ observed exactly at the knots, so `Linear(prior_variance_rate=σ²)` builds a `GaussianPlan` whose covariance is a `BrownianBridge`. A target at offset $t \in [0, 1]$ in cell $i$, of width $\Delta_i = \lvert x_{i+1} - x_i \rvert$, has variance

$$
\mathrm{Var}[f(x_*) \mid \vect{y}] = \sigma^2 \Delta_i t (1 - t),
$$

zero at the knots and $\sigma^2 \Delta_i / 4$ at mid-cell. Two targets with offsets $t_1 \le t_2$ in the same cell have covariance $\sigma^2 \Delta_i t_1 (1 - t_2)$, and targets in different cells are independent, so $\gmat{\Sigma}_*$ is block diagonal by cell and `sample` draws every cell's bridge at once. Targets outside the domain under `outside="nan"` have NaN variance. The bridge needs an `Axis` source and the policies under which the plan's mean is the bridge's mean: `outside` `"nan"` or `"raise"`, `extent="nodes"`, and any `missing` but `"renormalise"`.

!!! note "Which schemes have a variance"

    | Scheme | Variance from |
    | --- | --- |
    | Linear | the Brownian bridge between knots |
    | Cubic and smoothing splines | the integrated Wiener process prior |
    | RBF, kriging, GP | the kernel |
    | Variational fits | the posterior precision $\mat{H}^\top \mat{R}^{-1} \mat{H} + \lambda \mat{L}^\top \mat{L}$ |
    | Nearest, inverse distance, PCHIP, neural fields | none, without ensembles |

    Only the linear row exists today; the others arrive with their schemes.

## Curvilinear grids

A `CurvilinearGrid` (a swath) gives the coordinates $\vect{x}_{rc}$ of every node, so a target cannot be located axis by axis. Cell $(r, c)$ maps the unit square onto its four corners by the bilinear map

$$
\vect{x}(s, t) = (1-s)(1-t) \vect{x}_{r,c} + s(1-t) \vect{x}_{r+1,c} + (1-s) t \vect{x}_{r,c+1} + s t \vect{x}_{r+1,c+1},
$$

and `locate_curvilinear` solves $\vect{x}(s, t) = \vect{x}_*$ for the target $\vect{x}_* \in \mathbb{R}^2$. It starts each target in the cell whose centre is nearest (a brute-force search over the cells, chunked, so it traces), then runs four Newton steps from $(\frac{1}{2}, \frac{1}{2})$,

$$
(s, t) \leftarrow (s, t) - \mat{J}^{-1} \left(\vect{x}(s, t) - \vect{x}_*\right) \text{ with } \mat{J} = \frac{\partial \vect{x}}{\partial (s, t)} \in \mathbb{R}^{2 \times 2}.
$$

Where $s$ or $t$ leaves $[0, 1]$, the target moves $\lfloor s \rfloor$ (or $\lfloor t \rfloor$) cells and Newton runs again, up to three times. The index position $\vect{u} = (r + s, c + t)$ then plays the role of the index position on a regular grid, so `Linear`, `Nearest` and `Tensor` plan a swath with the same stencil and the same `Policies`: past the edge of the swath the edge cell's map is continued, and a target whose Newton steps do not converge, or whose walk ends outside an inner cell, is invalid under every `outside` rule. Coordinates are planar: a swath across the dateline needs continuous (unwrapped) longitudes, with the targets in the same branch.

## Caching plans

A plan is a pure function of its scheme, its policies and its two geometries, so a pipeline that reads the same points every step can build it once and keep it. Three pieces make that work without functax depending on any cache library; they follow pipekit's `get_config`, `serialize_weights` and `stable_json` shapes, so a pipekit operator can wrap a plan with no glue.

- `scheme.get_config()` and `Scheme.from_config(config)` round-trip a scheme through strict JSON (`json.dumps(config, allow_nan=False)` always succeeds). The config names the class under `"class"`, so `Tensor((Nearest(), Linear()))` is `{"class": "Tensor", "schemes": [{"class": "Nearest"}, {"class": "Linear"}]}`, and a new scheme is found by its class name with no registration. `Policies` has the same pair; strict JSON has no NaN, so a non-finite `fill_value` is written as `{"float": "nan"}`, `{"float": "inf"}` or `{"float": "-inf"}`.
- `plan_key(scheme, source, target, policies=…)` is the sha256 hex of the functax version, both configs, and each geometry's class, static fields (period, epoch, metric, shape) and arrays (dtype, shape and C-contiguous bytes). It is the same in every process and changes when any coordinate moves by one ULP.
- `plan.serialize()` writes the plan's array leaves to bytes with `eqx.tree_serialise_leaves`, and `Plan.deserialize(data, like=template)` reads them back into a template of the same structure. The template need not be computed: `eqx.filter_eval_shape(lambda: scheme.plan(source, target))` traces the build without running it.

```python
key = plan_key(scheme, grid, points)
template = eqx.filter_eval_shape(lambda: scheme.plan(grid, points))
if key in store:
    plan = Plan.deserialize(store[key], like=template)
else:
    plan = scheme.plan(grid, points)
    store[key] = plan.serialize()
```

All three run on the host and refuse traced inputs, so call them outside `jit`.

## Reference

::: functax.interp
