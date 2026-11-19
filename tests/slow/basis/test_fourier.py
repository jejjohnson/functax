"""Tests for `functax.basis.fourier`, copied from geonnax be5537d with functax imports.
The slow tier: orthonormality, the Dirichlet closed form, eigen equations, boundary
conditions and mixed boundaries, each a second or more.
"""

from __future__ import annotations

import math

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from hypothesis import given, settings

from functax.basis.fourier import (
    fourier_basis,
    fourier_basis_1d,
    fourier_eigenvalues,
    fourier_eigenvalues_1d,
)
from functax.quad import gauss_legendre


BOUNDARIES = ["dirichlet", "neumann", "periodic"]


L = 1.5


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_orthonormal_on_interval(boundary):
    nodes, weights = gauss_legendre(80)
    with jax.enable_x64(True):
        phi = np.asarray(
            fourier_basis_1d(jnp.asarray(L * nodes), 9, L, boundary=boundary)
        )
    gram = (phi * (L * weights)[:, None]).T @ phi
    np.testing.assert_allclose(gram, np.eye(9), atol=1e-12)


def test_dirichlet_default_unchanged():
    x = jnp.linspace(-L, L, 7)
    j = np.arange(1, 5)
    expected = np.sin(np.outer(np.asarray(x) + L, j) * math.pi / (2 * L)) / math.sqrt(L)
    np.testing.assert_allclose(
        np.asarray(fourier_basis_1d(x, 4, L)), expected, atol=1e-6
    )
    np.testing.assert_array_equal(
        np.asarray(fourier_basis_1d(x, 4, L)),
        np.asarray(fourier_basis_1d(x, 4, L, boundary="dirichlet")),
    )


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


# the parity of each column about x = 0: Dirichlet sin(jπ(x + L)/(2L)) and
# Neumann cos(jπ(x + L)/(2L)) alternate even, odd, …; periodic is the constant,
# then cos (even) and sin (odd) in turn
_PARITY = {
    "dirichlet": (-1.0) ** np.arange(6),
    "neumann": (-1.0) ** np.arange(6),
    "periodic": np.array([1.0, 1.0, -1.0, 1.0, -1.0, 1.0]),
}


_BASES_1D = {
    boundary: jax.jit(
        lambda x, boundary=boundary: fourier_basis_1d(x, 6, L, boundary=boundary)
    )
    for boundary in BOUNDARIES
}


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), boundary=st.sampled_from(BOUNDARIES))
def test_columns_are_even_or_odd_about_the_centre(seed: int, boundary: str) -> None:
    # φ_j(−x) = ±φ_j(x), the sign alternating with the column
    x = jax.random.uniform(jax.random.key(seed), (9,), minval=-L, maxval=L)
    basis = _BASES_1D[boundary]
    assert_matches_oracle(basis(-x), basis(x) * _PARITY[boundary], rtol=0.0, atol=1e-12)


def test_gradient_matches_finite_differences() -> None:
    # differentiated in the points x, one axis per boundary, so the
    # hand-written slopes of every boundary are checked in one compile
    xyz = jnp.array([[0.2, -0.7, 1.1], [-1.0, 0.9, -0.3], [0.6, 0.1, 0.4]])
    assert_grad_matches_fd(
        lambda p: fourier_basis(p, (3, 3, 2), L, boundary=tuple(BOUNDARIES))[0],
        xyz,
        rtol=1e-8,
    )


def test_basis_traces() -> None:
    # the periodic and Neumann axes take the hand-written value-and-slope path
    xy = jax.random.uniform(jax.random.key(2), (3, 4, 2), minval=-L, maxval=L)
    assert_traces(
        lambda p: fourier_basis(p, (3, 2), L, boundary=("periodic", "neumann")), xy
    )
