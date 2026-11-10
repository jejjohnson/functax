"""Tests for `functax.basis.vector`, copied from geonnax be5537d with functax imports,
plus a test of the boundary fix in `divfree_basis`. The slow tier: orthonormality,
eigen equations, boundary conditions and divergence or curl by Jacobians, each a
second or more.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from functax.basis.fourier import (
    fourier_basis,
    fourier_eigenvalues,
)
from functax.basis.vector import curlfree_basis, divfree_basis


BOUNDARIES = ["dirichlet", "neumann", "periodic"]


L = 1.5


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_curlfree_is_gradient_and_curl_free(boundary):
    with jax.enable_x64(True):
        xy = jax.random.uniform(
            jax.random.key(1), (8, 2), minval=-L, maxval=L, dtype=jnp.float64
        )
        phi, lam = curlfree_basis(xy, 3, L, boundary=boundary)

        def potential(p):
            return fourier_basis(p[None], 3, L, boundary=boundary)[0][0]

        grads = jax.vmap(jax.jacfwd(potential))(xy)  # (N, M, 2)
        np.testing.assert_allclose(np.asarray(phi), np.asarray(grads), atol=1e-10)

        def field(p, m):
            return curlfree_basis(p[None], 3, L, boundary=boundary)[0][0, m]

        for m in range(9):
            jac = jax.vmap(jax.jacfwd(field), (0, None))(xy, m)  # (N, 2, 2)
            np.testing.assert_allclose(jac[:, 1, 0] - jac[:, 0, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(
            lam, fourier_eigenvalues(3, L, 2, boundary=boundary, dtype=jnp.float64)
        )


@pytest.mark.parametrize("boundary", ["neumann", "periodic"])
def test_divfree_boundary_option_is_divergence_free(boundary):
    with jax.enable_x64(True):
        xy = jax.random.uniform(
            jax.random.key(2), (6, 2), minval=-L, maxval=L, dtype=jnp.float64
        )

        def field(p, m):
            return divfree_basis(p[None], 3, L, boundary=boundary)[0][0, m]

        for m in range(9):
            jac = jax.vmap(jax.jacfwd(field), (0, None))(xy, m)
            np.testing.assert_allclose(jac[:, 0, 0] + jac[:, 1, 1], 0.0, atol=1e-10)


class TestDivFreeBasis:
    def test_atoms_are_divergence_free(self):
        # d/dx u + d/dy v == 0 analytically for every atom, at arbitrary points.
        def atom_field(p, j):
            phi, _ = divfree_basis(p[None, :], 4, 1.5)
            return phi[0, j]  # (2,) velocity

        pts = jnp.array([[0.13, -0.21], [0.7, 0.4], [-0.9, 0.05]])
        for p in pts:
            for j in range(16):
                jac = jax.jacfwd(atom_field)(p, j)  # (2, 2): d u_i / d x_k
                div = jac[0, 0] + jac[1, 1]
                np.testing.assert_allclose(div, 0.0, atol=1e-5)

    def test_shapes(self):
        xy = jnp.zeros((7, 2))
        phi, lam = divfree_basis(xy, num_basis_per_dim=3, L=1.0)
        assert phi.shape == (7, 9, 2)  # M = 3 * 3, last axis = (u, v)
        assert lam.shape == (9,)

    def test_eigenvalues_match_fourier(self):

        _, lam = divfree_basis(jnp.zeros((2, 2)), num_basis_per_dim=3, L=2.0)
        expected = fourier_eigenvalues(3, 2.0, 2, dtype=lam.dtype)
        np.testing.assert_allclose(lam, expected, atol=1e-6)


@pytest.mark.parametrize("boundary", ["neumann", "periodic"])
def test_divfree_atoms_use_the_requested_boundary(boundary):
    # geonnax be5537d built the atoms from Dirichlet modes whatever the
    # boundary, and only the eigenvalues followed it; the atoms are the skew
    # gradients of the boundary's own stream functions
    xy = jnp.asarray(np.random.default_rng(0).uniform(-L, L, (6, 2)))

    def stream(point):
        return fourier_basis(point[None], 3, L, boundary=boundary)[0][0]

    gradient = jax.vmap(jax.jacfwd(stream))(xy)
    expected = jnp.stack([-gradient[..., 1], gradient[..., 0]], axis=-1)
    phi, lam = divfree_basis(xy, 3, L, boundary=boundary)
    np.testing.assert_allclose(phi, expected, atol=1e-12)
    np.testing.assert_allclose(
        lam,
        fourier_eigenvalues(3, L, 2, dtype=lam.dtype, boundary=boundary),
        atol=1e-12,
    )
