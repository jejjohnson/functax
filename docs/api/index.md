# API Reference

Everything listed in `functax.__all__` is importable from the top level, and each area of the library is a public module that re-exports its private implementation:

```python
import functax as fx

fx.FunctaxError  # the base of every functax error
fx.geometry  # CRS-free axes, grids, point sets and affine raster maths
```

The [capability index](capabilities.md) lists every public name on one page, with a one-line summary each, together with the public API of JAX, Equinox, lineax and einx.

## Modules

| Module | Contents |
|---|---|
| [Geometry](geometry.md) | CRS-free axes, grids, point sets, the sphere and affine raster maths (Wave 0); sphere grids, the icosphere and frames (Wave 1) |
| [Bases](basis.md) | One submodule per family of bases: `features` (Fourier features at given frequencies, seasonal and windowed features), `polynomial` (Legendre), `sphere` (scalar and vector harmonics, needlets, shell bases), `slepian` (caps, polar gaps, regions), `eigen` (the cotangent Laplacian eigenbasis) (Wave 1) |
| [Quadrature](quad.md) | `gauss_legendre` (Wave 1) |
| [Interpolation](interp.md) | `Stencil`, `locate`, `Policies`, `Nearest`, `Linear`, `Tensor`, `Plan` (Wave 2) |
| [Type aliases](typing.md) | `Shape` |
| [Exceptions](exceptions.md) | `FunctaxError`, `ValidationError`, `MissingExtraError` |

## Package overview

::: functax
    options:
      members: false
      show_root_heading: false
      show_source: false
