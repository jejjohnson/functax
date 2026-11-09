# functax

[![Tests](https://github.com/jejjohnson/functax/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/functax/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/functax/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/functax/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/functax/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/functax/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/functax/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/functax/actions/workflows/pages.yml)
[![codecov](https://codecov.io/gh/jejjohnson/functax/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/functax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

Author: J. Emmanuel Johnson

Function approximation for JAX: bases, interpolation plans with adjoints, quadrature and transforms.

functax is built around one representation, an expansion f̂(x) = ∑ⱼ cⱼ φⱼ(x; θ) with coefficients c and basis functions φ, and five operations on it: interpolate, differentiate, integrate, project and transform. An interpolation plan exposes its read-out H, the adjoint Hᵀ and its derivative, so one object samples a field at observations and sends residuals back to the grid. It sits at Core tier 0 of the [GeoStack](https://github.com/jejjohnson/geostack): it imports only NumPy, JAX, Equinox, lineax, einx and jaxtyping, and every library above it may import it. It replaces the stack's hand-rolled bilinear samplers, takes over geonnax's bases and transforms, and replaces interpax and orthax outright.

## Status

functax is in Wave 0: the package, its CI and the CRS-free geometry module. The design is the GeoStack [functax design page](https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/functax.md), and the backlog is six wave epics under [geostack#129](https://github.com/jejjohnson/geostack/issues/129).

| Wave | Delivers | Epic |
|---|---|---|
| 0 | the package, `functax.geometry`, the SciPy oracle harness | [#8](https://github.com/jejjohnson/functax/issues/8) |
| 1 | bases, sphere geometry, derivatives and transforms moved in from geonnax | [#9](https://github.com/jejjohnson/functax/issues/9) |
| 2 | `Stencil`, `Plan` and the local interpolation schemes, with adjoints | [#10](https://github.com/jejjohnson/functax/issues/10) |
| 3 | global schemes (RBF, kriging, Delaunay, conservative remapping) and fits | [#11](https://github.com/jejjohnson/functax/issues/11) |
| 4 | quadrature rules, Chebyshev and Legendre machinery | [#12](https://github.com/jejjohnson/functax/issues/12) |
| 5 | the stack's libraries switch to functax | [#13](https://github.com/jejjohnson/functax/issues/13) |

## Installation

The base install depends on NumPy alone and holds the affine raster kernels, so NumPy-only consumers such as geoarray can use them without JAX. Everything else needs the `jax` extra. functax is not on PyPI; the stack pins it by git tag.

```bash
pip install "functax[jax] @ git+https://github.com/jejjohnson/functax.git"
```

Without the extra, importing a module that needs JAX raises `MissingExtraError` (also an `ImportError`), whose message names `functax[jax]`.

## Development

```bash
git clone https://github.com/jejjohnson/functax.git
cd functax
uv run poe install     # every dependency group + the pre-commit hooks
uv run poe check       # lint, format check, typecheck, tests with the 90 % coverage gate
uv run poe docs        # the MyST prose and the Zensical API reference, assembled into public/
```

`uv run poe` lists every task. [`AGENTS.md`](AGENTS.md) holds the rules for contributors and coding agents: the module map, the contracts, the tests that enforce them and the house style. [`docs/contributing.md`](docs/contributing.md) has the label taxonomy and the epic model.

## License

MIT; see [`LICENSE`](LICENSE).
