# Bases

A basis supplies the functions $\phi_j(\vect{x}; \gvect{\theta})$ of the expansion $\hat{f}(\vect{x}) = \sum_j c_j \phi_j(\vect{x}; \gvect{\theta})$. Each family of bases is its own submodule of `functax.basis`, and the package itself re-exports none of their names, so every basis has one home. The parameters $\gvect{\theta}$ (centres, widths, frequencies) are ordinary arrays, so they may be learned.

## Feature maps

`functax.basis.features` evaluates feature maps of coordinates at given frequencies. For frequencies $\mat{W} \in \mathbb{R}^{D \times F}$ and a length scale $\ell$, `rff_forward` returns the random Fourier features

$$
\vect{\phi}(\vect{x}) = F^{-1/2} [\cos(\vect{x}^\top \mat{W} / \ell), \sin(\vect{x}^\top \mat{W} / \ell)],
$$

whose inner products $\vect{\phi}(\vect{x})^\top \vect{\phi}(\vect{x}')$ approximate the stationary kernel that $\mat{W}$ was drawn from. Drawing $\mat{W}$ is sampling, so it stays with the library that owns the kernel (kernellib or geonnax); no function here takes a PRNG key. `rff_cosine_forward` is the single-cosine form with random phases, `fourier_features` the cosines and sines at dyadic frequencies $2\pi 2^d$, `seasonal_features` those at the harmonics of given periods, and `gaussian_window_features` Gaussian windows in time. The names and signatures are geonnax's, and the outputs equal geonnax `be5537d` to $10^{-12}$ relative in float64.

::: functax.basis.features
