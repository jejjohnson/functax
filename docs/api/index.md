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
| [Geometry](geometry.md) | CRS-free axes, grids, point sets, the sphere and affine raster maths (Wave 0) |
| [Interpolation](interp.md) | `Stencil`, `locate`, `Policies`, `Nearest`, `Linear`, `Tensor`, `Plan` (Wave 2) |
| [Type aliases](typing.md) | `Shape` |
| [Exceptions](exceptions.md) | `FunctaxError`, `ValidationError`, `MissingExtraError` |

## Package overview

::: functax
    options:
      members: false
      show_root_heading: false
      show_source: false
