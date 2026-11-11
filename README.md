<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img alt="functax" src="docs/assets/logo-light.svg" width="340">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/jejjohnson/functax/actions/workflows/ci.yml"><img alt="Tests" src="https://github.com/jejjohnson/functax/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/jejjohnson/functax/actions/workflows/lint.yml"><img alt="Lint" src="https://github.com/jejjohnson/functax/actions/workflows/lint.yml/badge.svg"></a>
  <a href="https://github.com/jejjohnson/functax/actions/workflows/typecheck.yml"><img alt="Type Check" src="https://github.com/jejjohnson/functax/actions/workflows/typecheck.yml/badge.svg"></a>
  <a href="https://codecov.io/gh/jejjohnson/functax"><img alt="codecov" src="https://codecov.io/gh/jejjohnson/functax/branch/main/graph/badge.svg"></a>
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-blue">
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-yellow.svg"></a>
</p>

<p align="center">
  <a href="https://jejjohnson.github.io/functax/"><b>Docs</b></a> ·
  <a href="https://jejjohnson.github.io/functax/reference/"><b>API</b></a> ·
  <a href="https://jejjohnson.github.io/functax/quickstart/"><b>Quickstart</b></a> ·
  <a href="https://jejjohnson.github.io/functax/methane/"><b>Example</b></a> ·
  <a href="#gallery"><b>Gallery</b></a>
</p>

**Function approximation for JAX: bases, interpolation plans with adjoints, quadrature and transforms.**

functax treats a field as a function, which is also where the name comes from (see [The name](#the-name)). A field $`x(\mathbf{s})`$ over the coordinates $`\mathbf{s} \in \Omega`$ is written as one expansion,

```math
x(\mathbf{s}) \approx \sum_{d=1}^{D_z} z_d \phi_d(\mathbf{s}; \boldsymbol{\theta})
```

where $`\phi_d`$ are the basis functions with parameters $`\boldsymbol{\theta}`$ and $`\mathbf{z} \in \mathbb{R}^{D_z}`$ are the coefficients. On a set of $`N`$ points $`\mathbf{s}_n`$ this becomes $`\mathbf{x} = \boldsymbol{\Phi} \mathbf{z}`$, with $`\mathbf{x} \in \mathbb{R}^N`$ and the basis matrix $`\boldsymbol{\Phi} \in \mathbb{R}^{N \times D_z}`$, $`[\boldsymbol{\Phi}]_{nd} = \phi_d(\mathbf{s}_n)`$. Almost everything in the library is one of five operations on this expansion. We interpolate it, differentiate it, integrate it, project data onto it and transform between its values and its coefficients. For interpolation, a plan gives us its read-out $`\mathbf{H}`$, the adjoint $`\mathbf{H}^\top`$ and its derivative. So the same object samples a field at the observations and sends the residuals back to the grid.

We keep functax small and functional. Building a plan or a rule may run on the host with NumPy, but applying it is pure JAX. So `jit`, `vmap` and `grad` go through every plan, rule and transform. We also leave models out of functax, so that it can sit at the bottom of the stack. A kernel comes in as a callable, a solver as a lineax operator and a difference operator as an argument.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hero-dark.svg">
    <img alt="functax is built around one expansion, a field x of the coordinates s equal to the sum over d of z_d phi_d of s, and five operations on it: interpolate, differentiate, integrate, project and transform, each in its own module, over a basis library and CRS-free geometry" src="docs/assets/hero-light.svg" width="100%">
  </picture>
</p>

## Installation

functax is not on PyPI yet, so install it from GitHub with [uv](https://docs.astral.sh/uv/):

```bash
uv add "functax[jax] @ git+https://github.com/jejjohnson/functax.git"
```

The base install, without `[jax]`, depends on NumPy alone and holds the affine raster kernels in `functax.geometry.affine`. We did this so that NumPy-only libraries such as geoarray can use them without pulling in JAX. Everything else needs the `jax` extra (JAX, Equinox, lineax, einx and jaxtyping). If you import a module that needs it without it, you get a `MissingExtraError` that names `functax[jax]`.

## What's inside

Each module links to its API page, and [`docs/api/capabilities.md`](docs/api/capabilities.md) lists every public name in one place.

| Module | Operation | Names |
|---|---|---|
| [`geometry`](https://jejjohnson.github.io/functax/reference/geometry/) | where the coordinates live | `Axis`, `AffineGrid`, `RectilinearGrid`, `CurvilinearGrid`, `Points`; sphere grids (`fibonacci_sphere`, `gauss_legendre_grid`, `latlon_grid`, `healpix_grid`, `icosphere`); frames (`lonlat_to_unit`, `lonlat_to_ecef`, `enu_basis`); the NumPy-only affine kernels in `geometry.affine` |
| [`basis`](https://jejjohnson.github.io/functax/reference/basis/) | the basis functions | `sphere` (real and vector spherical harmonics, needlets), `slepian` (caps, polar gaps, regions), `fourier`, `local` (RBF, Wendland, Gabor, wavelets), `eigen` (EOFs, graph and mesh Laplacians), `polynomial` (Legendre), `vector` (divergence-free and curl-free), `features` (random Fourier, seasonal) |
| [`interp`](https://jejjohnson.github.io/functax/reference/interp/) | interpolate | `Plan` with `adjoint` and `operator()`; `Nearest`, `Linear`, `Cubic`, `BSpline`, `Tensor`; `CubicSpline`, `Pchip`, `Akima`, `Steffen`; `InverseDistance`, `Delaunay`, `Binning`, `Conservative`; `RBF`, `Kriging`; `Fourier`, `Chebyshev` |
| [`diffops`](https://jejjohnson.github.io/functax/reference/diffops/) | differentiate | `gradient`, `divergence`, `curl2d`, `laplacian`, `jacobian_field`; on the sphere, `surface_grad`, `surface_grad_enu` and `laplace_beltrami` |
| [`quad`](https://jejjohnson.github.io/functax/reference/quad/) | integrate | `Rule`, `gauss_legendre`, `clenshaw_curtis`, `trapezoid`, `simpson`; `SphereRule` with `sphere_gauss_legendre`, `sphere_latlon`, `sphere_healpix` and `sphere_fibonacci`; `cell_area`; `polyline` |
| [`project`](https://jejjohnson.github.io/functax/reference/project/) | project | `least_squares`, `galerkin`, `Variational`, `fill_gaps` |
| [`transform`](https://jejjohnson.github.io/functax/reference/transform/) | transform | `SphericalHarmonicTransform`, `nonuniform_rfft`, `nonuniform_irfft`, `ChebyshevTransform1D`, `ChebyshevTransform2D` |

A quadrature rule is only data (the nodes and the weights), and [integrix](https://github.com/jejjohnson/integrix) does the summing, the scanning and the adapting. The [guides](https://jejjohnson.github.io/functax/) go through interpolation, gridding scattered data, regridding and quadrature in more detail.

## Where it sits

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/layers-dark.svg">
    <img alt="geonnax and integrix import functax today, and the stack's models, numerics, learning and data libraries switch to it in Wave 5; functax depends on NumPy alone, plus JAX, Equinox, lineax, einx and jaxtyping with the jax extra" src="docs/assets/layers-light.svg" width="100%">
  </picture>
</p>

functax sits at Core tier 0 of the [GeoStack](https://github.com/jejjohnson/geostack), which means it imports no other library of the stack and any of them may import it. Today [geonnax](https://github.com/jejjohnson/geonnax) builds its encoders and operators on functax's bases and transforms, and [integrix](https://github.com/jejjohnson/integrix) sums with its rules. The rest of the stack swaps its own samplers, bases and rules for functax in Wave 5 ([#13](https://github.com/jejjohnson/functax/issues/13)). We wrote down the design behind all of this on the GeoStack [functax design page](https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/functax.md).

## The name

functax is named after *functa*. In "From data to functa", Dupont et al. (2022) proposed that a data point can be treated as a function, stored as the parameters of a neural field. We can then do deep learning on those functions directly, and in their words, "we refer to the data as functa". Bauer et al. (2023) then arranged the latent representations spatially (*spatial functa*) and scaled the idea up to ImageNet classification and generation.

We really liked that way of thinking, and functax takes the first half of it. A field is a function, so we should be able to work with it as one, reading it anywhere, differentiating it, integrating it and projecting onto it. The neural-field half lives next door in [geonnax](https://github.com/jejjohnson/geonnax), whose coordinate networks are functa in the original sense.

- E. Dupont, H. Kim, S. M. A. Eslami, D. Rezende and D. Rosenbaum. From data to functa: your data point is a function and you can treat it like one. [arXiv:2201.12204](https://arxiv.org/abs/2201.12204), 2022.
- M. Bauer, E. Dupont, A. Brock, D. Rosenbaum, J. R. Schwarz and H. Kim. Spatial functa: scaling functa to ImageNet classification and generation. [arXiv:2302.03130](https://arxiv.org/abs/2302.03130), 2023.

## Gallery

These are some of the bases in `functax.basis`, drawn by [`docs/assets/readme/figures.py`](docs/assets/readme/figures.py).

The real spherical harmonics $`Y_\ell^m`$ are global functions on the sphere; here are six of the 25 with degree $`\ell \le 4`$ (`basis.sphere.real_spherical_harmonics`).

<p align="center"><img src="docs/assets/readme/sh_basis.png" alt="Six real spherical harmonics up to degree 4 on Mollweide maps" width="100%"></p>

Slepian functions are regional instead: these are for a 40° cap, ranked by their concentration $`\lambda`$, the share of their energy that sits inside the cap (`basis.slepian.slepian_cap_basis`). The first few live inside the cap, and past the Shannon number (about 52 here) they leak out of it.

<p align="center"><img src="docs/assets/readme/slepian.png" alt="Six Slepian functions for a 40 degree cap; concentration falls as the modes leak outside the dashed cap boundary" width="100%"></p>

Needlets are multiscale: here is one centre at four scales, each scale $`j`$ twice as fine as the last (`basis.sphere.needlet_basis`).

<p align="center"><img src="docs/assets/readme/needlets.png" alt="Four needlets centred near the same point at scales 0 to 3, each more localised than the last" width="100%"></p>

## Related projects

| Project | Role |
|---|---|
| [geonnax](https://github.com/jejjohnson/geonnax) | Networks and neural operators on the plane and the sphere, built on functax's bases |
| [integrix](https://github.com/jejjohnson/integrix) | Integration: sums, scans and adaptive quadrature over functax's rules |
| [gaussx](https://github.com/jejjohnson/gaussx) | Structured linear algebra, solvers and Gaussian primitives |
| [kernellib](https://github.com/jejjohnson/kernellib) | Kernels and scalable kernel methods |

## Development

```bash
git clone https://github.com/jejjohnson/functax.git
cd functax
uv run poe install     # every dependency group + the pre-commit hooks
uv run poe check       # lint, format check, typecheck, tests with the 90 % coverage gate
uv run poe docs        # the MyST prose and the Zensical API reference, assembled into public/
```

`uv run poe` lists every task. [`AGENTS.md`](AGENTS.md) holds the rules for contributors and coding agents, from the module map and the contracts to the tests that enforce them and the house style. [`docs/contributing.md`](docs/contributing.md) has the label taxonomy and the epic model. The icon and the diagrams come from [`docs/assets/render.py`](docs/assets/render.py) (`uv run --no-project python docs/assets/render.py`), and the figures from [`docs/assets/readme/figures.py`](docs/assets/readme/figures.py), which runs the code blocks of the quickstart and the [methane example](https://jejjohnson.github.io/functax/methane/) as written.

## License

MIT, see [LICENSE](LICENSE).
