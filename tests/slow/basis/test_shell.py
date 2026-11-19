"""Tests for `shell_basis`, copied from geonnax be5537d with functax imports. The slow
tier: quadrature orthonormality, high-degree float32 checks, gradients and frame
reconstructions, each a second or more.
"""

from __future__ import annotations

import math

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from hypothesis import given, settings

from functax.basis.sphere import harmonic_degrees, real_spherical_harmonics, shell_basis
from functax.geometry import gauss_legendre_grid
from functax.quad import gauss_legendre


R0, R1 = 1.0, 1.8


def test_orthonormal_in_area_times_radius_measure():
    l_max, n_radial = 4, 5
    with jax.enable_x64(True):
        directions, w_dir = gauss_legendre_grid(l_max + 2)
        nodes, w_r = gauss_legendre(n_radial + 2)
        radii = 0.5 * (R1 - R0) * nodes + 0.5 * (R1 + R0)
        points = jnp.concatenate([directions * r for r in radii])
        weights = jnp.concatenate([w_dir * (0.5 * (R1 - R0) * w) for w in w_r])
        values, _ = shell_basis(points, l_max, n_radial, r_inner=R0, r_outer=R1)
    values = np.asarray(values)
    gram = (values * np.asarray(weights)[:, None]).T @ values
    np.testing.assert_allclose(gram, np.eye(values.shape[1]), atol=1e-10)


def test_single_radial_mode_is_scaled_harmonics_and_indices():
    with jax.enable_x64(True):
        directions, _ = gauss_legendre_grid(4)
        points = 1.3 * directions
        values, indices = shell_basis(points, 3, 1, r_inner=R0, r_outer=R1)
        expected = real_spherical_harmonics(directions, 3) / math.sqrt(R1 - R0)
    np.testing.assert_allclose(np.asarray(values), np.asarray(expected), atol=1e-12)
    assert indices.shape == (16, 2) and indices[-1].tolist() == [3, 0]


def test_chebyshev_radial_takes_its_end_values():
    points = jnp.array([[0.0, 0.0, R0], [0.0, 0.0, R1]])
    values, _ = shell_basis(points, 0, 3, r_inner=R0, r_outer=R1, radial="chebyshev")
    y00 = 1.0 / math.sqrt(4.0 * math.pi)
    np.testing.assert_allclose(
        np.asarray(values), y00 * np.array([[1, -1, 1], [1, 1, 1]]), atol=1e-6
    )


def _random_rotation(seed: int) -> np.ndarray:
    """A random proper rotation of ℝ³, from the QR factors of a Gaussian matrix."""
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def _shell_points(count: int, seed: int) -> jax.Array:
    """Random points with radii in the shell (R0, R1)."""
    keys = jax.random.split(jax.random.key(seed))
    gaussian = jax.random.normal(keys[0], (count, 3))
    radii = jax.random.uniform(keys[1], (count, 1), minval=R0 + 0.05, maxval=R1 - 0.05)
    return radii * gaussian / jnp.linalg.norm(gaussian, axis=-1, keepdims=True)


_L_MAX, _N_RADIAL = 3, 2


# compiled once per module, so each hypothesis example reuses the kernel
_SHELL = jax.jit(
    lambda xyz: shell_basis(xyz, _L_MAX, _N_RADIAL, r_inner=R0, r_outer=R1)[0]
)


# the (ℓ, k) group of each column, ℓ outermost and k fastest, one-hot
_GROUPS = np.array(
    [(ell, k) for ell in harmonic_degrees(_L_MAX) for k in range(_N_RADIAL)]
)


_ONEHOT = jnp.asarray(
    (_GROUPS[:, None, :] == np.unique(_GROUPS, axis=0)[None]).all(-1), dtype=float
)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_degree_kernels_are_rotation_invariant(seed: int) -> None:
    # ∑_m φ_ℓmk(x) φ_ℓmk(y) = (2ℓ + 1)/(4π) P_ℓ(x̂ · ŷ) R_k(‖x‖) R_k(‖y‖) by the
    # addition theorem: it depends on x̂ · ŷ and the radii only, so rotating
    # both points leaves it unchanged
    x = _shell_points(5, seed)
    y = _shell_points(5, seed + 1)
    rotation = jnp.asarray(_random_rotation(seed))

    def kernels(a: jax.Array, b: jax.Array) -> jax.Array:
        return (_SHELL(a) * _SHELL(b)) @ _ONEHOT

    assert_matches_oracle(
        kernels(x @ rotation.T, y @ rotation.T),
        kernels(x, y),
        rtol=0.0,
        atol=1e-12,
    )


def test_gradient_matches_finite_differences() -> None:
    # differentiated in the points: the radius, the direction and the
    # Chebyshev radial recurrence
    xyz = _shell_points(3, seed=5)
    assert_grad_matches_fd(
        lambda p: shell_basis(p, 2, 3, r_inner=R0, r_outer=R1, radial="chebyshev")[0],
        xyz,
        rtol=1e-8,
    )


def test_shell_basis_traces() -> None:
    xyz = jnp.stack([_shell_points(4, seed=6 + b) for b in range(3)])
    assert_traces(lambda p: shell_basis(p, 2, 2, r_inner=R0, r_outer=R1)[0], xyz)
