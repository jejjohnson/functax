# Derivatives of fields

`functax.diffops` differentiates continuous fields by automatic differentiation, so the same operators apply to a network, an `Interpolant` or a basis expansion. Each operator takes a field of one point and returns a callable that evaluates the derivative at one point; batch it with `jax.vmap` and compile once at the top with `jax.jit`. Derivatives on grids belong to gridax, finitevolX and spectraldiffx.

For a field of $\vect{x} \in \mathbb{R}^D$, `gradient` returns $\nabla f$, `jacobian_field` the Jacobian $J_{kd} = \partial u_k / \partial x_d$ in forward mode, `divergence` $\nabla \cdot \vect{u}$, `curl2d` the planar vorticity $\partial_x u_y - \partial_y u_x$, and `laplacian` $\Delta f$ by Hessian-vector products, without forming the Hessian. The names follow geonnax, except that `grad` and `jacobian` became `gradient` and `jacobian_field`, so they do not shadow `jax.grad` and `jax.jacobian`.

On the unit sphere, the operators differentiate the radially constant extension $g(\vect{x}) = f(\vect{x} / \lVert \vect{x} \rVert)$: `surface_grad` returns $\nabla_S f = (\mat{I} - \vect{n} \vect{n}^\top) \nabla g$, `surface_grad_enu` its east and north components at a $(\lambda, \varphi)$ given in an explicit `unit`, and `laplace_beltrami` returns $\Delta_S f = \Delta g$ at $\lVert \vect{n} \rVert = 1$, whose eigenfunctions are the spherical harmonics, $\Delta_S Y_l^m = -l(l + 1) Y_l^m$. For a sphere of radius $R$, divide the surface gradient by $R$ and the Laplace–Beltrami operator by $R^2$.

::: functax.diffops
