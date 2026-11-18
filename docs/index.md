# functax

> Function approximation for JAX: bases, interpolation plans with adjoints, quadrature and transforms.

functax is built around one representation, an expansion $\hat f(x) = \sum_j c_j \phi_j(x; \theta)$ with coefficients $c$ and basis functions $\phi$, and five operations on it: interpolate, differentiate, integrate, project and transform. An interpolation plan exposes its read-out $H$, the adjoint $H^\top$ and its derivative, so the same object samples a field at observations and sends residuals back to the grid.

```{figure} https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/functax_overview/functax_overview.png
:alt: functax in one picture: one expansion at the top; interpolate, differentiate, integrate, project and transform below it, each with its equation, module and main names; the basis and geometry modules underneath, and the base install and the two extras at the bottom
:width: 100%
:label: fig-overview

The expansion and the five operations on it, one module each. The schematic is kept in [geostack](https://github.com/jejjohnson/geostack/tree/main/docs/images/libraries/functax_overview) with the stack's other library schematics.
```

functax sits at Core tier 0 of the GeoStack: it imports only JAX, Equinox, lineax, einx and jaxtyping, and every library above it may import it. The design is in the GeoStack [functax design page](https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/functax.md), and the backlog is tracked in waves under [geostack#129](https://github.com/jejjohnson/geostack/issues/129).

:::{note} Status
functax is in Wave 0: the package, its CI and the CRS-free geometry module. The interpolation plans arrive in Wave 2.
:::

## Installation

The base install depends on NumPy alone and holds the affine raster kernels. Everything else needs the `jax` extra.

::::{tab-set}
:::{tab-item} With JAX

```bash
pip install "functax[jax] @ git+https://github.com/jejjohnson/functax.git"
```
:::
:::{tab-item} NumPy only

```bash
pip install "functax @ git+https://github.com/jejjohnson/functax.git"
```
:::
:::{tab-item} From source

```bash
git clone https://github.com/jejjohnson/functax.git
cd functax
uv run poe install
```
:::
::::

## Next steps

- [Quickstart](guide/quickstart.md): what each install gives you
- [API reference](https://jejjohnson.github.io/functax/reference/): every module and the capability index
