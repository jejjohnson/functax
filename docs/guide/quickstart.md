# Quickstart

## Two installs

functax has a NumPy-only base install and a `jax` extra. The base exists so that NumPy-only consumers, such as geoarray, can call functax's affine raster kernels without pulling in JAX.

| Install | Dependencies | Modules |
|---|---|---|
| `functax` | NumPy | `functax`, `functax.geometry.affine` |
| `functax[jax]` | NumPy, JAX, Equinox, lineax, einx, jaxtyping | every module |

Importing `functax` never imports JAX:

```python
import functax as fx

fx.__version__  # '0.0.1'
```

## Without the extra

A module that needs JAX checks for the extra on import. Without it, the import fails with a [`MissingExtraError`](xref:api#functax.MissingExtraError) that names the extra to install, rather than with a bare `No module named 'jax'`:

```text
MissingExtraError: functax.interp needs the functax[jax] extra, but 'jax' could
not be imported; install it with: pip install 'functax[jax]'
```

`MissingExtraError` is also an `ImportError`, so an existing `except ImportError` still catches it.
