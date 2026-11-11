"""Function approximation for JAX: bases, interpolation plans, quadrature, transforms.

functax is built around one representation, an expansion
f̂(x) = ∑ⱼ cⱼ φⱼ(x; θ) with coefficients c and basis functions φ, and five
operations on it: interpolate, differentiate, integrate, project and
transform. Each operation lives in its own public module, and each module
appears with the first feature that needs it.

| Module | What it gives you |
|---|---|
| `functax.geometry` | CRS-free axes, grids, point sets and affine raster maths |

The base install depends on NumPy alone. Every module except the affine raster
kernels in `functax.geometry.affine` needs the ``functax[jax]`` extra, which
brings in JAX, Equinox, lineax, einx and jaxtyping. Importing ``functax`` itself
never imports JAX.

Examples:
    >>> import functax
    >>> major, minor, patch = functax.__version__.split(".")[:3]
    >>> patch.isdigit()
    True
"""

from __future__ import annotations

from functax.exceptions import (
    FunctaxError,
    GeometryError,
    MissingExtraError,
    NonlinearSchemeError,
    OutsideDomainError,
    PolicyError,
    ValidationError,
)


__version__ = "0.0.9"

__all__ = [
    "FunctaxError",
    "GeometryError",
    "MissingExtraError",
    "NonlinearSchemeError",
    "OutsideDomainError",
    "PolicyError",
    "ValidationError",
    "__version__",
]
