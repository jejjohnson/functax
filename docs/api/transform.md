# Transforms

A transform maps the coefficients of a field in one basis to another, $\vect{c}' = \mat{T} \vect{c}$.

## The spherical-harmonic transform

`SphericalHarmonicTransform` is a dense, differentiable matrix transform on a Gauss–Legendre grid of $n_{\text{lat}}$ latitudes and $n_{\text{lon}}$ equispaced longitudes from 0. Analysis computes $c_{lm} = \sum_q w_q y(\vect{x}_q) Y_l^m(\vect{x}_q)$ with the quadrature weights $w_q$, and synthesis $y(\vect{x}) = \sum_{l \le l_{\max}} \sum_m c_{lm} Y_l^m(\vect{x})$, both in the real harmonic layout of `real_spherical_harmonics`. With $n_{\text{lat}} > l_{\max}$ and $n_{\text{lon}} > 2 l_{\max}$, analysis after synthesis is exact for band-limited fields. The tables are built once in float64 and pass through `jax.lax.stop_gradient`, so gradients reach the field but never the tables.

spectraldiffx's complex transform $\hat{u}(l, m)$, an FFT in longitude followed by orthonormal associated Legendre functions in latitude on the same grid (rows north to south), holds the same information: $c_{l0} = \sqrt{2\pi} / n_{\text{lon}} \operatorname{Re} \hat{u}(l, 0)$, and for $m > 0$, $c_{lm} = 2\sqrt{\pi} / n_{\text{lon}} \operatorname{Re} \hat{u}(l, m)$ and $c_{l,-m} = -2\sqrt{\pi} / n_{\text{lon}} \operatorname{Im} \hat{u}(l, m)$. A test pins this to $10^{-10}$ against spectraldiffx `d68f8f6`.

## The non-uniform FFT

`nonuniform_rfft` computes the low Fourier modes of a field sampled at arbitrary points of the torus $[0, 1)^d$ as a quadrature of the continuous coefficient, $\hat{c}_{\vect{k}} = \sum_n w_n u_n e^{-2\pi i \vect{k} \cdot \vect{\xi}_n}$, so it converges as the points grow. `nonuniform_irfft` evaluates the truncated real Fourier series at arbitrary points, the scattered-point read-out of spectral interpolation. The spectrum is stored as the $2^{d-1}$ low corners of an rfft spectrum; on a uniform grid of $S$ nodes with $w_n = 1/S$ the coefficients equal `rfftn(u) / S`.

::: functax.transform
