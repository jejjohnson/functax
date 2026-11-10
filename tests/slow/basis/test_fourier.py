"""Tests for `functax.basis.fourier`, copied from geonnax be5537d with functax imports.
The slow tier: orthonormality, eigen equations, boundary conditions and divergence
or curl by Jacobians, each a second or more.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from functax.basis.fourier import (
    fourier_basis,
    fourier_basis_1d,
    fourier_eigenvalues,
    fourier_eigenvalues_1d,
)


BOUNDARIES = ["dirichlet", "neumann", "periodic"]


L = 1.5


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_eigen_equation_and_sorted_eigenvalues(boundary):
    with jax.enable_x64(True):
        lam = fourier_eigenvalues_1d(7, L, boundary=boundary, dtype=jnp.float64)
        x = jnp.linspace(-L, L, 11)

        def column(t, m):
            return fourier_basis_1d(t[None], 7, L, boundary=boundary)[0, m]

        for m in range(7):
            second = jax.vmap(jax.grad(jax.grad(column), argnums=0), (0, None))(x, m)
            value = jax.vmap(column, (0, None))(x, m)
            np.testing.assert_allclose(-second, lam[m] * value, atol=1e-9)
    assert np.all(np.diff(np.asarray(lam)) >= 0)


def test_boundary_conditions_hold():
    with jax.enable_x64(True):
        ends = jnp.array([-L, L])

        def basis(t, boundary):
            return fourier_basis_1d(t, 6, L, boundary=boundary)

        np.testing.assert_allclose(basis(ends, "dirichlet"), 0.0, atol=1e-12)
        neumann = jax.vmap(
            lambda e: jax.jacfwd(lambda t: basis(t[None], "neumann")[0])(e)
        )(ends)
        np.testing.assert_allclose(neumann, 0.0, atol=1e-12)
        periodic = basis(ends, "periodic")
        np.testing.assert_allclose(periodic[0], periodic[1], atol=1e-12)
        grads = jax.vmap(
            lambda e: jax.jacfwd(lambda t: basis(t[None], "periodic")[0])(e)
        )(ends)
        np.testing.assert_allclose(grads[0], grads[1], atol=1e-12)


def test_mixed_boundaries_in_fourier_basis():
    xy = jax.random.uniform(jax.random.key(0), (20, 2), minval=-L, maxval=L)
    phi, lam = fourier_basis(xy, (4, 3), L, boundary=("periodic", "dirichlet"))
    px = fourier_basis_1d(xy[:, 0], 4, L, boundary="periodic")
    py = fourier_basis_1d(xy[:, 1], 3, L)
    np.testing.assert_allclose(
        phi, np.einsum("na,nb->nab", px, py).reshape(20, 12), atol=1e-6
    )
    np.testing.assert_allclose(
        lam,
        fourier_eigenvalues((4, 3), L, 2, boundary=("periodic", "dirichlet")),
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="boundary"):
        fourier_basis(xy, 3, L, boundary="robin")  # ty: ignore[invalid-argument-type]
