# Interpolation

Every plan reads a field as $\hat{\vect{f}}(\mat{X}_*) = \mat{H} \mathcal{S}(\vect{y})$: a fit $\mathcal{S}$ turns samples $\vect{y}$ into coefficients (the identity for local schemes), and a read-out $\mat{H}$ evaluates them at the targets. The adjoint $\mathcal{S}^\top \mat{H}^\top$ sends residuals back to the source.

## Stencils

A `Stencil` stores $\mat{H}$ row by row: row $k$ reads $S$ source values at flat indices $i_{ks}$ with weights $w_{ks}$, so

$$
\hat{f}_k = \sum_{s=1}^{S} w_{ks} c_{i_{ks}} \text{ and } (\mat{H}^\top \vect{r})_j = \sum_{k=1}^{M} \sum_{s=1}^{S} [i_{ks} = j] w_{ks} r_k.
$$

`apply` is a gather and a contraction, and `apply_adjoint` a scatter-add, so repeated indices in one row add. The hand-written adjoint is checked against `jax.linear_transpose` by the dot-product test, and `operator()` returns the same map as a lineax operator. Padding taps carry weight 0 and repeat an index their row already reads, so shapes stay static.

## Reference

::: functax.interp
