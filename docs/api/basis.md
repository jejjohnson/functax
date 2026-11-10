# Bases

A basis supplies the functions $\phi_j(\vect{x}; \gvect{\theta})$ of the expansion $\hat{f}(\vect{x}) = \sum_j c_j \phi_j(\vect{x}; \gvect{\theta})$. Each family of bases is its own submodule of `functax.basis`, and the package itself re-exports none of their names, so every basis has one home. The parameters $\gvect{\theta}$ (centres, widths, frequencies) are ordinary arrays, so they may be learned.

## Feature maps

`functax.basis.features` evaluates feature maps of coordinates at given frequencies. For frequencies $\mat{W} \in \mathbb{R}^{D \times F}$ and a length scale $\ell$, `rff_forward` returns the random Fourier features

$$
\vect{\phi}(\vect{x}) = F^{-1/2} [\cos(\vect{x}^\top \mat{W} / \ell), \sin(\vect{x}^\top \mat{W} / \ell)],
$$

whose inner products $\vect{\phi}(\vect{x})^\top \vect{\phi}(\vect{x}')$ approximate the stationary kernel that $\mat{W}$ was drawn from. Drawing $\mat{W}$ is sampling, so it stays with the library that owns the kernel (kernellib or geonnax); no function here takes a PRNG key. `rff_cosine_forward` is the single-cosine form with random phases, `fourier_features` the cosines and sines at dyadic frequencies $2\pi 2^d$, `seasonal_features` those at the harmonics of given periods, and `gaussian_window_features` Gaussian windows in time. The names and signatures are geonnax's, and the outputs equal geonnax `be5537d` to $10^{-12}$ relative in float64.

::: functax.basis.features

## Box Fourier and vector bases

`functax.basis.fourier` evaluates the Laplacian eigenfunctions of the box $[-L, L]^D$ under Dirichlet, Neumann or periodic boundaries, chosen per axis, with their eigenvalues; the $D$-dimensional functions are tensor products, flattened row-major. `functax.basis.vector` builds 2-D vector atoms from them: `divfree_basis` takes the skew gradient $\nabla^\perp \psi = (-\partial_y \psi, \partial_x \psi)$ of each stream function, so every atom is divergence-free, and `curlfree_basis` the gradient, so every atom is curl-free.

::: functax.basis.fourier

::: functax.basis.vector

## Local and multiscale bases

`functax.basis.local` holds bases whose columns are placed where the caller wants them. `rbf_basis` evaluates $\varphi_a(\vect{x}) = K(\lVert \vect{x} - \vect{c}_a \rVert / \ell_a)$ with a Gaussian or a compactly supported Wendland kernel, and `spherical_rbf_basis` does the same in geodesic distance on the sphere. `gabor_frame` multiplies a Gaussian envelope by an oscillation, radial or directional, and `gabor_frame_grid` lays out a dyadic, redundant frame over a box. `wavelet_basis_1d` and `wavelet_basis_2d` are the orthonormal, critically sampled counterpart: the periodic Haar, db2 or db4 DWT, with $\mat{\Phi}^\top \mat{\Phi} = \mat{I}$.

::: functax.basis.local

## Polynomials

`functax.basis.polynomial` evaluates the Legendre polynomials $P_l$ by their three-term recurrence and the associated Legendre functions $P_l^m$ by the fully normalised recurrences, carried as a mantissa and a binary exponent per order so they stay finite to high degree in float32 as well as float64. `associated_legendre` comes in the orthonormal, Schmidt and classical normalisations.

::: functax.basis.polynomial

## The sphere

`functax.basis.sphere` holds the bases on the unit sphere. `real_spherical_harmonics` returns the tesseral harmonics $Y_{lm}$, orthonormal under $\mathrm{d}\Omega$, each a polynomial in $(x, y, z)$, so they are differentiable at the poles. `vector_spherical_harmonics` returns the gradient harmonics $\vect{\Psi}_{lm} = \nabla_S Y_{lm}$ and the toroidal harmonics $\vect{\Phi}_{lm} = \hat{\vect{r}} \times \nabla_S Y_{lm}$, a Helmholtz basis for tangent fields. `needlet_basis` builds the needlet tight frame, localised in space and frequency, and `shell_basis` multiplies harmonics by Legendre or Chebyshev polynomials in radius for fields on a shell.

::: functax.basis.sphere

## Slepian functions

`functax.basis.slepian` builds the band-limited functions that concentrate the most energy in a region $R$ of the sphere: the eigenvectors of $D_{ij} = \int_R Y_i Y_j \mathrm{d}\Omega$, with the concentrations $\lambda_k \in [0, 1]$ as eigenvalues and the Shannon number $\operatorname{tr} D = (l_{\max} + 1)^2 A / 4\pi$ counting the well-concentrated ones. `slepian_cap_basis` handles a cap block by block in the azimuthal order and can be rotated to any centre, `slepian_polar_gap_basis` the band between two polar caps, and `slepian_region_basis` any region given by a quadrature over it, for instance a global grid masked by `spherical_polygon_mask`.

::: functax.basis.slepian

## Eigenbases

`functax.basis.eigen` holds the eigenbases of data and of discrete operators. `eof_basis` returns the empirical orthogonal functions of a (sample, grid) data matrix, its leading right singular vectors, with the singular values that set the variance of each mode. `graph_laplacian_eigpairs` returns the low-frequency eigenpairs of the combinatorial or normalised Laplacian of a graph. `cotangent_laplacian_eigpairs` solves $\mat{L} \vect{\phi} = \lambda \mat{M} \vect{\phi}$ for the cotangent Laplacian $\mat{L}$ of a triangle mesh and its lumped mass $\mat{M}$, so the eigenvectors are $\mat{M}$-orthonormal and converge to the surface's eigenfunctions under refinement ($\lambda \to \ell(\ell + 1)$ on the unit sphere).

::: functax.basis.eigen
