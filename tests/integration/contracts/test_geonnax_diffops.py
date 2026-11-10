"""functax.diffops (#22) against geonnax, pinned at geonnax be5537d as data: each
operator on fixed analytic fields, flat and on the sphere (geonnax's grad and
jacobian are functax's gradient and jacobian_field).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from _harness import assert_matches_oracle

from functax import diffops


GEONNAX = np.load(Path(__file__).parent / "data" / "geonnax_diffops.npz")
POINTS = jnp.asarray(GEONNAX["points"])


def _scalar(x: jax.Array) -> jax.Array:
    return jnp.sin(x[0]) * x[1] ** 2 + jnp.exp(0.3 * x[2]) * x[0]


def _vector(x: jax.Array) -> jax.Array:
    return jnp.array([x[0] * x[1], jnp.cos(x[2]) + x[1], x[0] ** 2 * x[2]])


def _plane(x: jax.Array) -> jax.Array:
    return jnp.array([jnp.sin(x[1]) * x[0], x[0] ** 3 - x[1]])


def _sphere_field(n: jax.Array) -> jax.Array:
    return n[0] * n[1] + jnp.sin(2 * n[2]) + n[0] ** 3


def test_euclidean_operators_equal_geonnax() -> None:
    cases = {
        "gradient": jax.vmap(diffops.gradient(_scalar))(POINTS),
        "jacobian": jax.vmap(diffops.jacobian_field(_vector))(POINTS),
        "divergence": jax.vmap(diffops.divergence(_vector))(POINTS),
        "curl2d": jax.vmap(diffops.curl2d(_plane))(jnp.asarray(GEONNAX["planar"])),
        "laplacian": jax.vmap(diffops.laplacian(_scalar))(POINTS),
    }
    for key, value in cases.items():
        assert_matches_oracle(value, GEONNAX[key], rtol=1e-12, atol=1e-14)


def test_sphere_operators_equal_geonnax() -> None:
    # geonnax's surface_grad_enu read degrees by default
    enu = diffops.surface_grad_enu(_sphere_field, unit="degrees")
    cases = {
        "surface_grad": jax.vmap(diffops.surface_grad(_sphere_field))(POINTS),
        "surface_grad_enu": jax.vmap(enu)(jnp.asarray(GEONNAX["lonlat"])),
        "laplace_beltrami": jax.vmap(diffops.laplace_beltrami(_sphere_field))(POINTS),
    }
    for key, value in cases.items():
        assert_matches_oracle(value, GEONNAX[key], rtol=1e-12, atol=1e-13)
