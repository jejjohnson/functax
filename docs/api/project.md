# Projection

A projection is a fit from data to coefficients, $\vect{c} = \mathcal{S}(\vect{y})$. Each fit here is linear in $\vect{y}$ and solves through lineax, so it traces under `jit`, `vmap` and `grad`, and its gradients come from lineax's implicit differentiation of the solve.

## Least squares and Galerkin

`least_squares` takes the basis already evaluated at the $M$ sites, $\gmat{\Phi}$ of shape $(M, J)$, so any basis of `functax.basis` or a user's feature map plugs in. It minimises $\lVert \gmat{\Phi} \vect{c} - \vect{y} \rVert^2 + \alpha \lVert \vect{c} \rVert^2$ by QR on $\gmat{\Phi}$ when $\alpha = 0$, and by Cholesky on $(\gmat{\Phi}^\top \gmat{\Phi} + \alpha \mat{I}) \vect{c} = \gmat{\Phi}^\top \vect{y}$ when $\alpha > 0$. `galerkin` projects with the weights $\vect{w}$ of a quadrature rule, $\vect{c} = (\gmat{\Phi}^\top \mat{W} \gmat{\Phi})^{-1} \gmat{\Phi}^\top \mat{W} \vect{y}$ with $\mat{W}$ the diagonal matrix of $\vect{w}$, the discrete $L^2$ projection onto the span of the basis.

## The variational fit

`Variational` solves the normal equations of a smoothing analysis with observation operator $\mat{H}$ $(M, N)$, observation precision $\mat{R}^{-1}$ $(M, M)$, roughness operator $\mat{L}$ $(G, N)$ and weight $\lambda \ge 0$:

$$
(\mat{H}^\top \mat{R}^{-1} \mat{H} + \lambda \mat{L}^\top \mat{L}) \vect{x} = \mat{H}^\top \mat{R}^{-1} \vect{y}.
$$

The three operators arrive as lineax operators (a plan's `operator()`, a diagonal precision, a difference operator from finitevolX or spectraldiffx), so functax imports none of them. The normal operator is applied matrix-free and tagged symmetric positive semi-definite for lineax's CG; the default solver is CG at $\sqrt{\epsilon}$ of the data dtype. The fit is a `Fit`, with the adjoint $\mathcal{S}^\top \vect{g} = \mat{R}^{-1} \mat{H} \mat{A}^{-1} \vect{g}$, so it sits in a `Plan` like any other.

## Gap fill

`fill_gaps` fills the holes of an image (the pixels where `mask` is false) from the observed ones, which stay fixed as Dirichlet values. With $\gmat{\Delta}$ the five-point Laplacian with mirror edges (the edge pixel is its own ghost neighbour), so that $\gmat{\Delta} = -\mat{D}^\top \mat{D}$ for the forward differences $\mat{D}$, `method="laplace"` gives $\gmat{\Delta} \vect{x} = 0$ in the holes and `method="biharmonic"` gives $\gmat{\Delta}^2 \vect{x} = 0$. These are the two named cases of the variational fit as the observation error goes to zero, with $\mat{L} = \mat{D}$ and $\mat{L} = \gmat{\Delta}$. In one dimension, the samples $[0, 1, ?, ?, 0]$ fill to $[2/3, 1/3]$ (Laplace) and $[1, 1/2]$ (biharmonic).

The Laplace fill equals xrtoolz's `fillnan_laplacian` with `boundary="reflect"` run to convergence: its red/black Gauss–Seidel sweeps have the same discrete equation as their fixed point, and a pinned test checks the two to $10^{-12}$ relative to the image scale.

::: functax.project
