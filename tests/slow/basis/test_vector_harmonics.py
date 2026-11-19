"""Tests for `vector_spherical_harmonics`, copied from geonnax be5537d with functax
imports. The slow tier: quadrature orthonormality, high-degree float32 checks,
gradients and frame reconstructions, each a second or more.
"""

from __future__ import annotations

import math

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle, assert_traces
from hypothesis import given, settings

from functax.basis.sphere import (
    harmonic_degrees,
    real_spherical_harmonics,
    vector_spherical_harmonics,
)
from functax.geometry import enu_basis, fibonacci_sphere, gauss_legendre_grid


def _surface_divergence(field_fn, u):
    """∇_S · F at u, for F a tangent field given on R³ by `field_fn`."""
    jac = jax.jacfwd(field_fn)(u)  # (3, 3): ∂F_i/∂x_j
    projector = jnp.eye(3) - jnp.outer(u, u)
    return jnp.trace(projector @ jac)


def test_orthonormal_on_gauss_legendre_grid():
    l_max = 8
    with jax.enable_x64(True):
        points, weights = gauss_legendre_grid(l_max + 2)
        psi, phi, _ = vector_spherical_harmonics(points, l_max)
    fields = np.concatenate([np.asarray(psi), np.asarray(phi)], axis=1)  # (N, 2M, 3)
    gram = np.einsum("n,nid,njd->ij", np.asarray(weights), fields, fields)
    np.testing.assert_allclose(gram, np.eye(fields.shape[1]), atol=1e-10)


def test_tangent_and_finite_at_poles():
    poles = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    points = jnp.concatenate([fibonacci_sphere(50)[0], poles])
    psi, phi, _ = vector_spherical_harmonics(points, 5)
    assert jnp.isfinite(psi).all() and jnp.isfinite(phi).all()
    for field in (psi, phi):
        radial = jnp.einsum("nmd,nd->nm", field, points)
        assert float(jnp.max(jnp.abs(radial))) < 1e-5


@pytest.mark.parametrize("column", [0, 7])
def test_divergence_identities(column):
    l_max = 2
    degree = harmonic_degrees(l_max)[1:][column]

    def field(kind):
        def fn(x):
            u = x / jnp.linalg.norm(x)
            psi, phi, _ = vector_spherical_harmonics(u[None], l_max, normalized=False)
            return (psi if kind == "psi" else phi)[0, column]

        return fn

    with jax.enable_x64(True):
        points = fibonacci_sphere(20)[0]
        Y = real_spherical_harmonics(points, l_max)[:, 1 + column]
        # jitted once: an eager Jacobian per point costs about 2 s each
        divergence_psi = jax.jit(lambda u: _surface_divergence(field("psi"), u))
        divergence_phi = jax.jit(lambda u: _surface_divergence(field("phi"), u))
        for k, u in enumerate(points):
            div_psi = divergence_psi(u)
            div_phi = divergence_phi(u)
            assert abs(float(div_psi) + degree * (degree + 1) * float(Y[k])) < 1e-8
            assert abs(float(div_phi)) < 1e-8


def test_enu_output_matches_projection():
    points, _ = fibonacci_sphere(30)
    psi, _, _ = vector_spherical_harmonics(points, 3)
    psi_en, _, degrees = vector_spherical_harmonics(points, 3, output="enu")
    lonlat = jnp.stack(
        [jnp.arctan2(points[:, 1], points[:, 0]), jnp.arcsin(points[:, 2])], axis=-1
    )
    frame = enu_basis(lonlat, unit="radians")[:, :2, :]
    np.testing.assert_allclose(
        np.asarray(psi_en), np.einsum("nmd,ncd->nmc", psi, frame), atol=1e-5
    )
    assert psi_en.shape == (30, 15, 2) and degrees.shape == (15,)


def _random_units(count: int, seed: int) -> np.ndarray:
    gaussian = np.random.default_rng(seed).normal(size=(count, 3))
    return gaussian / np.linalg.norm(gaussian, axis=1, keepdims=True)


def _random_rotation(seed: int) -> np.ndarray:
    """A random proper rotation of ℝ³, from the QR factors of a Gaussian matrix."""
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def test_degree_one_and_the_zonal_degree_two_match_their_closed_forms() -> None:
    # Y_1m(r̂) = c a_m · r̂ with c = √(3/(4π)) and a_m a signed unit axis, so
    # Ψ_1m = c (a_m − (a_m · r̂) r̂)/√2; Y_20 = √(5/(16π)) (3z² − 1), so
    # Ψ_20 = √(5/(16π)) 6z (e_z − z r̂)/√6; and Φ = r̂ × Ψ throughout
    points = _random_units(6, seed=21)
    # jitted: one compile instead of an eager compile per operation
    psi, phi, _ = jax.jit(lambda u: vector_spherical_harmonics(u, 2))(
        jnp.asarray(points)
    )
    c = math.sqrt(3.0 / (4.0 * math.pi))
    # a_m from Y_1m at the coordinate axes; each is ±e_y, ±e_z, ±e_x in turn
    axes = np.asarray(real_spherical_harmonics(jnp.eye(3), 1))[:, 1:].T / c
    assert_matches_oracle(np.abs(axes), np.eye(3)[[1, 2, 0]], rtol=0.0, atol=1e-15)
    radial = points @ axes.T
    expected_1 = (
        c * (axes[None] - radial[..., None] * points[:, None, :]) / math.sqrt(2.0)
    )
    z = points[:, 2:]
    expected_20 = (
        math.sqrt(5.0 / (16.0 * math.pi))
        * 6.0
        * z
        * (np.array([0.0, 0.0, 1.0]) - z * points)
        / math.sqrt(6.0)
    )
    expected_psi = np.concatenate([expected_1, expected_20[:, None, :]], axis=1)
    expected_phi = np.cross(points[:, None, :], expected_psi)
    assert_matches_oracle(psi[:, [0, 1, 2, 5]], expected_psi, rtol=0.0, atol=1e-14)
    assert_matches_oracle(phi[:, [0, 1, 2, 5]], expected_phi, rtol=0.0, atol=1e-14)


_L_MAX = 3


# compiled once per module, so each hypothesis example reuses the kernel
_VSH = jax.jit(lambda u: vector_spherical_harmonics(u, _L_MAX)[:2])


# sums the columns of each degree ℓ = 1, …, ℓ_max
_DEGREE_ONEHOT = jnp.asarray(
    np.asarray(harmonic_degrees(_L_MAX)[1:])[:, None] == np.arange(1, _L_MAX + 1),
    dtype=float,
)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_degree_kernels_rotate_with_the_points(seed: int) -> None:
    # K_ℓ(u, v) = ∑_m Ψ_ℓm(u) Ψ_ℓm(v)ᵀ is rotation-equivariant,
    # K_ℓ(Ru, Rv) = R K_ℓ(u, v) Rᵀ, and likewise for Φ
    u = jnp.asarray(_random_units(4, seed))
    v = jnp.asarray(_random_units(4, seed + 1))
    rotation = jnp.asarray(_random_rotation(seed))

    def kernels(a: jax.Array, b: jax.Array) -> jax.Array:
        fields_a, fields_b = _VSH(a), _VSH(b)
        return jnp.stack(
            [
                jnp.einsum("nmi,nmj,ml->nlij", fa, fb, _DEGREE_ONEHOT)
                for fa, fb in zip(fields_a, fields_b, strict=True)
            ]
        )

    rotated = kernels(u @ rotation.T, v @ rotation.T)
    expected = jnp.einsum("ia,knlab,jb->knlij", rotation, kernels(u, v), rotation)
    assert_matches_oracle(rotated, expected, rtol=0.0, atol=1e-12)


def test_vector_harmonics_trace() -> None:
    # a batch of two: each eager call re-traces the inner Jacobian (~0.7 s)
    points = jnp.asarray(_random_units(8, seed=23).reshape(2, 4, 3))
    assert_traces(lambda u: vector_spherical_harmonics(u, 3), points)
