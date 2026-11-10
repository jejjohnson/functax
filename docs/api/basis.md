# Bases

A basis supplies the functions $\phi_j(\vect{x}; \gvect{\theta})$ of the expansion $\hat{f}(\vect{x}) = \sum_j c_j \phi_j(\vect{x}; \gvect{\theta})$. Each family of bases is its own submodule of `functax.basis`, and the package itself re-exports none of their names, so every basis has one home. The parameters $\gvect{\theta}$ (centres, widths, frequencies) are ordinary arrays, so they may be learned.

## Feature maps

`functax.basis.features` evaluates feature maps of coordinates at given frequencies. For frequencies $\mat{W} \in \mathbb{R}^{D \times F}$ and a length scale $\ell$, `rff_forward` returns the random Fourier features

$$
\vect{\phi}(\vect{x}) = F^{-1/2} [\cos(\vect{x}^\top \mat{W} / \ell), \sin(\vect{x}^\top \mat{W} / \ell)],
$$

whose inner products $\vect{\phi}(\vect{x})^\top \vect{\phi}(\vect{x}')$ approximate the stationary kernel that $\mat{W}$ was drawn from. Drawing $\mat{W}$ is sampling, so it stays with the library that owns the kernel (kernellib or geonnax); no function here takes a PRNG key. `rff_cosine_forward` is the single-cosine form with random phases, `fourier_features` the cosines and sines at dyadic frequencies $2\pi 2^d$, `seasonal_features` those at the harmonics of given periods, and `gaussian_window_features` Gaussian windows in time. The names and signatures are geonnax's, and the outputs equal geonnax `be5537d` to $10^{-12}$ relative in float64.

::: functax.basis.features

## Polynomials

`functax.basis.polynomial` evaluates the Legendre polynomials $P_l$ by their three-term recurrence and the associated Legendre functions $P_l^m$ by the fully normalised recurrences, carried as a mantissa and a binary exponent per order so they stay finite to high degree in float32 as well as float64. `associated_legendre` comes in the orthonormal, Schmidt and classical normalisations.

::: functax.basis.polynomial

## The sphere

`functax.basis.sphere` holds the bases on the unit sphere. `real_spherical_harmonics` returns the tesseral harmonics $Y_{lm}$, orthonormal under $\mathrm{d}\Omega$, each a polynomial in $(x, y, z)$, so they are differentiable at the poles. `vector_spherical_harmonics` returns the gradient harmonics $\vect{\Psi}_{lm} = \nabla_S Y_{lm}$ and the toroidal harmonics $\vect{\Phi}_{lm} = \hat{\vect{r}} \times \nabla_S Y_{lm}$, a Helmholtz basis for tangent fields. `needlet_basis` builds the needlet tight frame, localised in space and frequency, and `shell_basis` multiplies harmonics by Legendre or Chebyshev polynomials in radius for fields on a shell.

::: functax.basis.sphere

## Eigenbases

`functax.basis.eigen` holds the eigenbases of discrete operators. `cotangent_laplacian_eigpairs` solves $\mat{L} \vect{\phi} = \lambda \mat{M} \vect{\phi}$ for the cotangent Laplacian $\mat{L}$ of a triangle mesh and its lumped mass $\mat{M}$, so the eigenvectors are $\mat{M}$-orthonormal and converge to the surface's eigenfunctions under refinement ($\lambda \to \ell(\ell + 1)$ on the unit sphere).

::: functax.basis.eigen
