# Transforms

A transform maps the coefficients of a field in one basis to another, $\vect{c}' = \mat{T} \vect{c}$.

## The spherical-harmonic transform

`SphericalHarmonicTransform` is a dense, differentiable matrix transform on a Gauss–Legendre grid of $n_{\text{lat}}$ latitudes and $n_{\text{lon}}$ equispaced longitudes from 0. Analysis computes $c_{lm} = \sum_q w_q y(\vect{x}_q) Y_l^m(\vect{x}_q)$ with the quadrature weights $w_q$, and synthesis $y(\vect{x}) = \sum_{l \le l_{\max}} \sum_m c_{lm} Y_l^m(\vect{x})$, both in the real harmonic layout of `real_spherical_harmonics`. With $n_{\text{lat}} > l_{\max}$ and $n_{\text{lon}} > 2 l_{\max}$, analysis after synthesis is exact for band-limited fields. The tables are built once in float64 and pass through `jax.lax.stop_gradient`, so gradients reach the field but never the tables.

spectraldiffx's complex transform $\hat{u}(l, m)$, an FFT in longitude followed by orthonormal associated Legendre functions in latitude on the same grid (rows north to south), holds the same information: $c_{l0} = \sqrt{2\pi} / n_{\text{lon}} \operatorname{Re} \hat{u}(l, 0)$, and for $m > 0$, $c_{lm} = 2\sqrt{\pi} / n_{\text{lon}} \operatorname{Re} \hat{u}(l, m)$ and $c_{l,-m} = -2\sqrt{\pi} / n_{\text{lon}} \operatorname{Im} \hat{u}(l, m)$. A test pins this to $10^{-10}$ against spectraldiffx `d68f8f6`.

## The spherical wavelet transform

`SphericalWaveletTransform` wraps a `SphericalHarmonicTransform` with $S$ dyadic band-pass windows $h_j(l)$, the scale-discretised wavelets or needlets of Marinucci et al. (2008) and Wiaux et al. (2008). With $x = \log_2 \max(l, 1)$, scale $j$ rises as $\sin(\pi \nu(x - j + 1) / 2)$ on the octave below $j$ and falls as $\cos(\pi \nu(x - j) / 2)$ on the octave above, with the Meyer polynomial $\nu(t) = t^4 (35 - 84 t + 70 t^2 - 20 t^3)$, so the squared windows partition unity, $\sum_j h_j(l)^2 = 1$ for every $l \le l_{\max}$. Analysis returns one full-resolution map per scale, $b_j = \mathrm{SHT}^{-1}(h_j \cdot \mathrm{SHT}(y))$, with shape (S, C, $n_{\text{lat}}$, $n_{\text{lon}}$), and synthesis applies the windows again and sums, $y = \mathrm{SHT}^{-1}(\sum_j h_j \cdot \mathrm{SHT}(b_j))$, so the round trip is exact to $10^{-12}$ in float64 on fields band-limited to $l_{\max}$. The default $S = \lceil \log_2 \max(l_{\max}, 2) \rceil + 1$ covers every octave; fewer scales leave the highest degrees out. The windows pass through `jax.lax.stop_gradient`, as the SHT tables do. The class is geonnax's, moved at `6e32651` with its signature and fields, and it equals geonnax there to $10^{-12}$; `SphericalWaveletConv`, which learns weights per scale, stays in geonnax.

## The non-uniform FFT

`nonuniform_rfft` computes the low Fourier modes of a field sampled at arbitrary points of the torus $[0, 1)^d$ as a quadrature of the continuous coefficient, $\hat{c}_{\vect{k}} = \sum_n w_n u_n e^{-2\pi i \vect{k} \cdot \vect{\xi}_n}$, so it converges as the points grow. `nonuniform_irfft` evaluates the truncated real Fourier series at arbitrary points, the scattered-point read-out of spectral interpolation. The spectrum is stored as the $2^{d-1}$ low corners of an rfft spectrum; on a uniform grid of $S$ nodes with $w_n = 1/S$ the coefficients equal `rfftn(u) / S`.

## Chebyshev transforms

`ChebyshevTransform1D` maps values $\vect{u}$ on the nodes of `chebyshev_nodes` to the true Chebyshev coefficients $\vect{a}$ of $u(x_j) = \sum_k a_k T_k(\xi_j)$, with $T_k(\xi) = \cos(k \arccos \xi)$ and $\xi_j$ the node mapped onto $[-1, 1]$, so a constant $c$ maps to $(c, 0, \ldots, 0)$ and $T_n$ to the unit vector $\vect{e}_n$. On the $N + 1$ Gauss–Lobatto nodes $\xi_j = \cos(j\pi/N)$ the forward transform is a type-I discrete cosine transform with the end coefficients halved,

$$
a_k = \frac{2}{N c_k} \sum_{j=0}^{N} \frac{u_j}{c_j} \cos(\pi j k / N),
$$

with $c_0 = c_N = 2$ and $c_k = 1$ otherwise, evaluated as a length-$2N$ real FFT of the even extension of $\vect{u}$. On the $N$ Gauss nodes $\xi_j = \cos((2j + 1)\pi/2N)$ it is the type-II transform $a_k = (2 / N c_k) \sum_j u_j \cos(\pi k (2j + 1)/2N)$ with $c_0 = 2$, a half-sample-shifted FFT of the zero-padded samples. The inverses synthesise $\sum_k a_k T_k(\xi_j)$, so forward then inverse returns the input to $10^{-12}$ relative (gate G4-roundtrip), and band-limited samples give back their coefficients. `ChebyshevTransform2D` is the tensor product on a $(N_y, N_x)$ grid, $x$ on the last axis: forward along $x$ then $y$, inverse along $y$ then $x$. Both act on the trailing axes and keep any leading batch dims and the input dtype.

The classes are spectraldiffx's, moved at `d68f8f6` with their method names (`to_spectral`, `from_spectral`, and `__call__(u, inverse=False)`) and their normalisation, and they equal spectraldiffx there to $10^{-13}$ on both node kinds. spectraldiffx's transforms wrapped a `ChebyshevGrid1D` or `ChebyshevGrid2D`; here they take the grid's degree and node kind (`"lobatto"` for its `"gauss-lobatto"`, or `"gauss"`), and the grids, derivatives, filters and solvers stay in spectraldiffx. The values follow the nodes in spectraldiffx's order, from the upper end of the interval down, while `clenshaw_curtis` orders the same Lobatto nodes ascending, so its rule pairs with the values reversed.

::: functax.transform
