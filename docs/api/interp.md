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

`field.derivative(order=m)` and `plan.derivative(0, m)` read the closed-form derivative of the Hermite read-out. On cell $i$, $\mathrm{d}/\mathrm{d}x = h_i^{-1} \mathrm{d}/\mathrm{d}t$, so $\hat f'(x_k) = m_k$ at every knot, order 3 is piecewise constant, and orders 4 and above are zero. The derivative of a cubic spline plan is a `DerivativePlan` with an adjoint. That of a shape-preserving plan is a `NonlinearDerivativePlan`, which reads but refuses a fixed operator, as `NonlinearPlan` does.

On an `Axis(..., repeats="allow")`, a repeat $x_k = x_{k+1}$ is a cell of zero width where the read-out may jump. It splits the knots into pieces, and each piece gets the slopes its scheme gives it as an axis of its own, with the scheme's end rules at both of its ends. A piece of two knots is a line, a not-a-knot piece of three is SciPy's parabola, and `bc="periodic"` refuses repeats. Reads are right-continuous, so a target at a repeat reads its last knot, as `numpy.interp` does. `Nearest` and `Linear` read repeats the same way, and the other schemes raise a `GeometryError` that names the repeat. Under `outside="extrapolate"`, a repeat at an end leaves a piece of one knot, which continues as a constant.

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
    | Kriging (`Kriging`), and the GP posterior as simple kriging | the kernel |
    | Variational fits | the posterior precision $\mat{H}^\top \mat{R}^{-1} \mat{H} + \lambda \mat{L}^\top \mat{L}$ |
    | Nearest, inverse distance, PCHIP, neural fields | none, without ensembles |

    The linear and kriging rows exist today; the others arrive with their schemes.

## Kernels and RBF

The global schemes take a kernel $k(\vect{x}, \vect{x}')$ as an argument. `Kernel` is a protocol: anything called as `kernel(targets, sources)` with the $M \times D$ targets and the $N \times D$ sources that returns the $M \times N$ cross-kernel, as an array or as a lineax operator (which is materialised with `as_matrix()`), is a kernel. A plain function satisfies it, and so does another library's kernel object, so kernellib hands its kernels to functax without functax importing it. Distances, metrics and length scales are the kernel's own business.

functax ships two radial families, both `RadialKernel`s of the Euclidean distance $r = \lVert \vect{x} - \vect{x}' \rVert$ that round-trip through strict JSON inside a scheme's config. `Polyharmonic(k)` is $(-1)^m r^k$ for odd $k$ and $(-1)^m r^k \log r$ for even $k$, conditionally positive definite of order $m = \lceil k/2 \rceil$ (odd $k$) or $k/2 + 1$ (even $k$), so it needs a polynomial tail of degree at least $m - 1$ (`min_degree`); the signs are SciPy's (`"linear"`, `"thin_plate_spline"`, `"cubic"`, `"quintic"`), and $r^k \log r$ is guarded at $r = 0$ by the double `where`. `Wendland(smoothness, support)` is `wendland_c2` or `wendland_c4` of $r / \rho$, positive definite in up to three dimensions and zero beyond $\rho$.

`RBF(kernel, degree, smoothing, solver)` solves, with $\mat{K}$ on the $N$ sources, the $N \times P$ tail $\mat{P}$ of monomials up to `degree` (in coordinates scaled to the sources' bounding box, as SciPy does) and $\lambda$ = `smoothing`,

$$
\begin{pmatrix} \mat{K} + \lambda \mat{I} & \mat{P} \\ \mat{P}^\top & \mat{0} \end{pmatrix} \begin{pmatrix} \vect{c} \\ \vect{b} \end{pmatrix} = \begin{pmatrix} \vect{y} \\ \vect{0} \end{pmatrix} \text{ and } \hat{\vect{f}}(\mat{X}_*) = \mat{K}_* \vect{c} + \mat{P}_* \vect{b}.
$$

Its plan has a `KernelSolve` fit, which projects onto the null space $\mat{Q}_2$ of $\mat{P}^\top$ (from the complete QR $\mat{P} = \mat{Q} \mat{R}$) and factorises $\mat{Q}_2^\top (\mat{K} + \lambda \mat{I}) \mat{Q}_2$ once with the lineax solver (Cholesky, or LU when `degree` is below the kernel's `min_degree`), and a dense `KernelReadout` $\mat{H} = [\mat{K}_* \mat{P}_*]$, whose adjoint is $\mat{H}^\top \vect{r}$. The fit maps $N$ samples to $N + P$ coefficients, and the saddle matrix is symmetric, so its adjoint $\mathcal{S}^\top$ is the same solve restricted to the data block. The [gridding guide](../gridding/) walks through gridding stations with it.

`Kriging(kernel, drift, nugget)` reads the same system as a Gaussian process: $\mat{K}$ is a covariance, $\lambda = \tau^2$ the nugget and $\mat{P} = \mat{F}$ the drift (`None` for simple kriging, `"constant"` for ordinary, a degree or a callable $F(\vect{x})$ for universal). Its plan is a `GaussianPlan` whose `KernelCovariance` is $\gmat{\Sigma}_*(\vect{x}, \vect{x}') = k(\vect{x}, \vect{x}') - \vect{h}(\vect{x})^\top \mat{A}^{-1} \vect{h}(\vect{x}')$, with $\mat{A}$ the saddle matrix and $\vect{h}(\vect{x}) = [k(\mat{X}, \vect{x}); \vect{f}(\vect{x})]$ a row of the read-out, so its diagonal is the kriging variance

$$
\sigma^2(\vect{x}_*) = k_{**} - \vect{k}_*^\top \mat{K}^{-1} \vect{k}_* + \vect{u}^\top (\mat{F}^\top \mat{K}^{-1} \mat{F})^{-1} \vect{u} \text{ and } \vect{u} = \vect{f}_* - \mat{F}^\top \mat{K}^{-1} \vect{k}_*.
$$

It is the variance of the latent field, zero at a noiseless source; a negative part above $10^{-10} k_{**}$ raises, and the rest of the rounding is clipped. `sample` draws from the eigendecomposition of $\gmat{\Sigma}_*$, which is singular wherever a target sits on a noiseless source.

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

## Conservative remapping

`Conservative()` remaps cell averages between grids so that integrals are preserved: target cell $C_i$ reads every source cell $c_j$ it overlaps with weight $w_{ij} = \lvert c_j \cap C_i \rvert / \lvert C_i \rvert$, so $\sum_i \lvert C_i \rvert \hat{y}_i = \sum_j \lvert c_j \rvert y_j$ whenever the target cells cover the source cells, and an integer-factor coarsening is a block mean. On tensor grids (`Axis`, `RectilinearGrid`, north-up `AffineGrid`) the overlap is the product of the per-axis overlaps $\lvert c_j \cap C_i \rvert = \max(0, \min(e_{j+1}, E_{i+1}) - \max(e_j, E_i))$ of `overlap_weights_1d`, found by two `searchsorted` sweeps, so the stencil is a tensor product of bands with $S = \prod_d S_d$ taps. Cell edges are an axis's `edges` (CF bounds) when given, else the midpoints between nodes (`Axis.cell_edges`), and a periodic source axis reads targets modulo its period. Under `outside="nan"` a target cell not inside the source is invalid, `"extrapolate"` keeps $w_{ij}$ (ESMF's `destarea`), and `"clamp"` divides by the covered area instead; `missing="renormalise"` divides each row by its weight on valid samples, counting the area outside the source as missing, which is xrtoolz's `fracarea` and the coastal-mask mean. A `CurvilinearGrid` source or target, or `Conservative(sphere=True, unit=…)`, takes the polygon path: each node owns the quadrilateral whose corners are the means of the four nodes around it, candidate pairs come from a k-d tree on bounding circles (SciPy's `cKDTree`, the `functax[scipy]` extra), and each pair is clipped on the host by Sutherland–Hodgman against its convex target cell, with $S$ the most overlaps of any target and shorter rows padded with zero weights. On the sphere cells have great-circle edges, the clip runs in the gnomonic tangent plane of the target cell (where great circles are straight lines), and areas are spherical excesses from `spherical_polygon_area`. The [regridding guide](../regridding/) walks through both.

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

## Inverse distance

`InverseDistance(k, power, radius, floor)` plans scattered `Points` sources onto `Points` or a grid. For target $i$ with the distances $d_{ij}$ to its $Q = \min(k, N)$ nearest sources, power $p$ and floor $\varepsilon$, the weights are

$$
v_{ij} = \max(d_{ij}, \varepsilon)^{-p} \text{ and } w_{ij} = \frac{v_{ij}}{\sum_{j'} v_{ij'}},
$$

so the read-out is a `Stencil` of $Q$ taps and the plan has the identity fit, an adjoint and every policy of a local plan (`missing="renormalise"` drops a NaN source from each row it is in). The distance is the source's `Points.metric`: `"euclidean"` on the coordinates as given, or `"chord"` on $(\lambda, \phi)$ in the `unit` passed to `plan`, the straight line between unit vectors, so the search crosses the dateline. The search runs on the host with SciPy's `cKDTree` (the `functax[scipy]` extra) when the coordinates are concrete, and as a chunked brute force (`lax.map` over chunks of targets) when they are traced, so a plan built inside `jit` or `grad` traces; both searches give the same stencil, since the distances of the chosen neighbours are recomputed in JAX. A neighbour counts when $d < r$, and a target with none (beyond the radius, padding, or a non-finite coordinate) is invalid. The [gridding guide](../gridding/) grids stations onto a grid with it.

## Triangulation

`Delaunay()` triangulates scattered `Points` sources on the host with SciPy's Delaunay (the `functax[scipy]` extra), and `Delaunay(simplices=mesh)` takes a given mesh as it is, one row of $D + 1$ source indices per simplex. For a target $\vect{x}$ in simplex $\tau$ with inverse affine map $\mat{T}_\tau^{-1}$ and reference vertex $\vect{r}_\tau$ (the layout of `scipy.spatial.Delaunay.transform`), the barycentric weights are

$$
(b_1, \dots, b_D) = \mat{T}_\tau^{-1} (\vect{x} - \vect{r}_\tau) \text{ and } b_{D+1} = 1 - \sum_{k=1}^{D} b_k,
$$

at the simplex's vertices, so the read-out is a `Stencil` of $D + 1$ taps (three on a plane) that reproduces affine functions inside the hull and equals `LinearNDInterpolator`. A target outside the hull is invalid, with zero weights and its taps at vertex 0. A fresh triangulation is located by `find_simplex`; a given mesh by brute force over chunks of targets on the host, the first simplex whose barycentric coordinates are all at least $-10^{-12}$. The plan is built on the host from concrete coordinates; applying it traces.

## Aggregation

`Binning(statistic, weights)` runs sampling backwards: with $\mat{H}$ the sampling stencil from the $M$ cells of a grid to $N$ sample positions (one tap with `weights="nearest"`, $2^D$ with `"linear"`), the sums are $\mat{H}^\top \vect{y}$, the counts $\vect{n} = \mat{H}^\top \vect{1}$ and the means

$$
\bar{\vect{y}} = \mathrm{diag}(\vect{n})^{-1} \mat{H}^\top \vect{y},
$$

all through the stencil's scatter-add. Nearest weights bin each sample into the cell of its nearest node, with the cell boundaries half-way between nodes and half a cell past the end nodes, as `scipy.stats.binned_statistic_dd` does with those edges; linear weights give cloud-in-cell deposition, which conserves the total of the samples inside the hull of the nodes and is differentiable in the positions. Samples off the grid get zero weights and drop out of every cell; they are never clamped onto the edge.

The counts depend only on the positions, so for fixed positions the sum and the mean are linear in $\vect{y}$, and `plan()` returns a `BinningPlan` that is an ordinary plan from samples to cells: its read-out is a `TransposedStencil`, $\mathrm{diag}(\vect{s}) \mat{H}^\top$ with $s_j = 1$ for sums and $1 / n_j$ for means, its `adjoint` is $\mat{H} \mathrm{diag}(\vect{s})$, and an empty cell under the mean is an invalid row that reads `fill_value` (NaN by default) and has a zero row in `operator()`, just as a target outside the domain does for sampling. A count does not depend on the data at all, so under `statistic="count"` the plan returns $\vect{n}$ for every field, and its `operator`, `adjoint` and `linearize` raise `NonlinearSchemeError`; `plan.counts` holds $\vect{n}$ for every statistic.
## Spectral interpolation

`Fourier()` resamples $N$ samples of a periodic axis onto $M$ nodes of the same period and origin (both `Axis` objects with nodes $x_0 + jP/N$ and $x_0 + jP/M$). The samples define the trigonometric polynomial with their discrete Fourier coefficients $\hat{\vect{y}} = \mat{F}_N \vect{y}$, and the plan reads it on the new nodes by zero-padding the spectrum ($M > N$) or cropping it ($M < N$):

$$
\hat{\vect{f}} = \frac{M}{N} \mat{F}_M^{-1} \mat{S} \mat{F}_N \vect{y},
$$

with $\mat{F}_N$ the length-$N$ DFT and $\mat{S}$ $(M, N)$ the mode map that copies every wavenumber $|k| < \min(N, M)/2$. For even $\min(N, M)$ the unpaired Nyquist bin is halved into $\pm N/2$ when upsampling and the bins $\pm M/2$ are added into one when downsampling, as `scipy.signal.resample` does, which the plan equals to $10^{-12}$. A field whose spectrum lies in $|k| < \min(N, M)/2$ is resampled exactly, and upsampling then downsampling returns the samples. The read-out `FourierReadout` applies two FFTs, and its adjoint has the same form with the roles of $N$ and $M$ swapped, $\mat{H}^\top \vect{r} = \mathrm{Re} (\mat{F}_N^{-1} \mat{S}^\top \mat{F}_M \vect{r})$, checked by the dot-product test. The axis must be periodic or the field band-limited; at a jump the read-out rings (Gibbs).

`Chebyshev(n_coefficients=K)` reads samples on Chebyshev nodes at arbitrary points. The source is an `Axis` holding the nodes of `chebyshev_nodes` on an interval $[a, b]$ (Gauss–Lobatto by default, or `kind="gauss"`, in either order), and the samples $\vect{y}$ define the series of their first $K$ Chebyshev coefficients,

$$
\hat{f}(x) = \sum_{k=0}^{K-1} c_k T_k(\xi(x)) \text{ with } \vect{c} = \mat{P} \mat{C} \vect{y} \text{ and } \xi = (2x - a - b)/(b - a),
$$

with $\mat{C}$ the forward transform of `ChebyshevTransform1D` and $\mat{P}$ the truncation to $K$ modes; $K = N + 1$ on $N + 1$ Lobatto nodes interpolates the samples, and a field of degree below $K$ is reproduced exactly. The plan's fit is `ChebyshevFit` and its read-out the dense `KernelReadout` $T_{pk} = T_k(\xi_p)$, so the adjoint is $\mat{C}^\top \mat{P}^\top \mat{T}^\top \vect{r}$: on Lobatto nodes $\mat{C}$ is symmetric, and on Gauss nodes $\mat{C}^\top$ is $2/N$ times the synthesis of $\vect{g}$ with $g_0$ halved, both checked by the dot-product test. `scheme.fit(axis, y)` returns a `ChebyshevInterpolant`, which keeps $\vect{c}$ and sums the series at targets chosen inside the trace with `clenshaw`, Clenshaw's recurrence $b_k = c_k + 2\xi b_{k+1} - b_{k+2}$ from $k = K - 1$ down to 1 with $b_K = b_{K+1} = 0$ and $\hat{f} = c_0 + \xi b_1 - b_2$. `clenshaw` equals `numpy.polynomial.chebyshev.chebval` to $10^{-13}$ relative to $\sum_k |c_k|$. Targets off $[a, b]$ follow `outside`: `"nan"` and `"raise"` mark them, `"clamp"` reads the nearest end and `"extrapolate"` evaluates the polynomial.

## Reference

::: functax.interp
