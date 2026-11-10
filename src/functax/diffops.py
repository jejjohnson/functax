"""Derivatives of continuous fields, flat and on the sphere, by autodiff.

Each operator takes a field of one point and returns a callable that evaluates
the derivative at one point; batch with ``jax.vmap``. Derivatives on grids
belong to gridax, finitevolX and spectraldiffx. Needs the ``functax[jax]``
extra.
"""

from __future__ import annotations

from functax._src.diffops._fields import (
    curl2d,
    divergence,
    gradient,
    jacobian_field,
    laplacian,
)
from functax._src.diffops._sphere import (
    laplace_beltrami,
    surface_grad,
    surface_grad_enu,
)


__all__ = [
    "curl2d",
    "divergence",
    "gradient",
    "jacobian_field",
    "laplace_beltrami",
    "laplacian",
    "surface_grad",
    "surface_grad_enu",
]
