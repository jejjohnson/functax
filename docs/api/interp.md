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

## Reference

::: functax.interp
