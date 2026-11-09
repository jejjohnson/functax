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

## Reference

::: functax.interp
