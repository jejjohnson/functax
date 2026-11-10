"""Tests for `functax.basis.local`, copied from geonnax be5537d with functax imports.
The slow tier: Gram matrices, gradients, rotation invariance and frame
reconstruction, each a second or more.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from functax.basis.local import (
    gabor_frame,
    gabor_frame_grid,
    rbf_basis,
    spherical_rbf_basis,
    wavelet_basis_1d,
    wendland_c2,
    wendland_c4,
)


KEY = jr.PRNGKey(0)


def test_rbf_gram_is_positive_semidefinite():
    pts = jr.uniform(KEY, (40, 2))
    centers = jr.uniform(jr.PRNGKey(1), (12, 2))
    phi = rbf_basis(pts, centers, jnp.full(12, 0.5), kernel="wendland_c2")
    eigmin = float(jnp.linalg.eigvalsh(phi.T @ phi).min())
    assert eigmin > -1e-6


def test_rbf_grad_finite_at_center():
    # An eval point coincident with a centre gives r = 0; the softened sqrt must
    # keep gradients finite (bare sqrt has an infinite derivative there).
    def loss(centers):
        phi = rbf_basis(
            jnp.zeros((1, 2)), centers, jnp.ones(centers.shape[0]), kernel="wendland_c2"
        )
        return jnp.sum(phi**2)

    grad = jax.grad(loss)(jnp.zeros((3, 2)))
    assert bool(jnp.isfinite(grad).all())


@pytest.mark.parametrize("kernel", ["wendland_c2", "wendland_c4"])
def test_rbf_narrow_atom_peaks_at_center(kernel):
    # A point on its centre must peak at 1 even for a width comparable to the
    # safe-sqrt softening — the value must not be shifted by an additive epsilon.
    width = 1e-6
    out = rbf_basis(
        jnp.zeros((1, 2)), jnp.zeros((1, 2)), jnp.array([width]), kernel=kernel
    )
    assert float(out[0, 0]) == pytest.approx(1.0)

    def loss(centers):
        return jnp.sum(
            rbf_basis(jnp.zeros((1, 2)), centers, jnp.array([width]), kernel=kernel)
            ** 2
        )

    assert bool(jnp.isfinite(jax.grad(loss)(jnp.zeros((1, 2)))).all())


def test_gabor_frame_reconstructs_smooth_field():
    grid = jnp.stack(
        jnp.meshgrid(jnp.linspace(0, 1, 16), jnp.linspace(0, 1, 16), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    bounds = jnp.array([[0.0, 1.0], [0.0, 1.0]])
    phi, *_ = gabor_frame_grid(
        grid, bounds, n_scales=3, base_scale=0.15, oversample=2.0
    )
    field = jnp.sin(2 * jnp.pi * grid[:, 0]) * jnp.cos(2 * jnp.pi * grid[:, 1])
    w, *_ = jnp.linalg.lstsq(phi, field, rcond=1e-6)
    rel = float(jnp.linalg.norm(phi @ w - field) / jnp.linalg.norm(field))
    assert rel < 0.05


class TestSphericalRBFBasis:
    def test_rotation_invariance(self):
        # A rigid rotation of both points and centres leaves the basis unchanged.
        key = jax.random.PRNGKey(4)
        x = jax.random.normal(key, (5, 3))
        x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
        c = jnp.eye(3)
        # a rotation about z by 0.7 rad
        a = 0.7
        rot = jnp.array(
            [[jnp.cos(a), -jnp.sin(a), 0.0], [jnp.sin(a), jnp.cos(a), 0.0], [0, 0, 1.0]]
        )
        base = spherical_rbf_basis(x, c, jnp.ones(3))
        rotated = spherical_rbf_basis(x @ rot.T, c @ rot.T, jnp.ones(3))
        np.testing.assert_allclose(base, rotated, atol=1e-5)

    def test_peak_at_centre(self):
        x = jnp.eye(3)  # three orthonormal points on the sphere
        phi = spherical_rbf_basis(x, x, jnp.ones(3))
        # geodesic distance to itself is 0 -> Gaussian peaks at 1 on the diagonal
        np.testing.assert_allclose(jnp.diag(phi), 1.0, atol=1e-6)

    @pytest.mark.parametrize("kernel", ["gaussian", "wendland_c2", "wendland_c4"])
    def test_finite_gradient_at_coincident_centre(self, kernel):
        # arccos has an infinite derivative at distance 0; the safe arccos must
        # keep gradients w.r.t. the evaluation point and the centre finite when
        # a point sits exactly on a centre.
        c = jnp.array([[0.0, 0.0, 1.0]])

        def value(x):
            return spherical_rbf_basis(x[None, :], c, jnp.array([0.5]), kernel=kernel)[
                0, 0
            ]

        g_point = jax.grad(value)(jnp.array([0.0, 0.0, 1.0]))  # x == centre
        assert bool(jnp.all(jnp.isfinite(g_point)))

        def value_c(centre):
            x = jnp.array([[0.0, 0.0, 1.0]])
            return spherical_rbf_basis(
                x, centre[None, :], jnp.array([0.5]), kernel=kernel
            )[0, 0]

        g_centre = jax.grad(value_c)(jnp.array([0.0, 0.0, 1.0]))
        assert bool(jnp.all(jnp.isfinite(g_centre)))

    @pytest.mark.parametrize("kernel", ["wendland_c2", "wendland_c4"])
    def test_wendland_compact_support_antipode(self, kernel):
        north = jnp.array([[0.0, 0.0, 1.0]])
        south = jnp.array([[0.0, 0.0, -1.0]])  # geodesic distance pi
        phi = spherical_rbf_basis(south, north, jnp.array([0.5]), kernel=kernel)
        assert float(phi[0, 0]) == 0.0


@pytest.mark.parametrize("kernel", [wendland_c2, wendland_c4])
def test_wendland_peak_and_compact_support(kernel):
    assert float(kernel(jnp.array(0.0))) == pytest.approx(1.0)
    # Exactly zero at and beyond the support radius r = 1.
    assert float(kernel(jnp.array(1.0))) == 0.0
    assert float(kernel(jnp.array(1.5))) == 0.0
    # Non-negative and decreasing on (0, 1).
    r = jnp.linspace(0.0, 1.0, 50)
    vals = kernel(r)
    assert bool((vals >= 0).all())
    assert bool((jnp.diff(vals) <= 1e-6).all())


def test_rbf_gaussian_matches_analytic():
    x = jnp.array([[0.5, 0.0]])
    out = rbf_basis(x, jnp.array([[0.0, 0.0]]), jnp.array([1.0]), kernel="gaussian")
    assert float(out[0, 0]) == pytest.approx(float(jnp.exp(-0.125)), abs=1e-6)


def test_rbf_wendland_is_locally_supported():
    centers = jnp.array([[0.0, 0.0], [5.0, 0.0]])
    out = rbf_basis(
        jnp.array([[0.0, 0.0]]), centers, jnp.array([1.0, 1.0]), kernel="wendland_c2"
    )
    assert float(out[0, 0]) == pytest.approx(1.0)  # on its own centre
    assert float(out[0, 1]) == 0.0  # 5 units away, width 1 -> exactly zero


def test_gabor_grad_finite_at_center():
    def loss(centers):
        phi = gabor_frame(
            jnp.zeros((1, 2)),
            centers,
            jnp.ones(centers.shape[0]),
            jnp.ones(centers.shape[0]),
        )
        return jnp.sum(phi**2)

    assert bool(jnp.isfinite(jax.grad(loss)(jnp.zeros((3, 2)))).all())


def test_gabor_frame_grid_is_overcomplete_with_geometry():
    grid = jnp.stack(
        jnp.meshgrid(jnp.linspace(0, 1, 12), jnp.linspace(0, 1, 12), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    bounds = jnp.array([[0.0, 1.0], [0.0, 1.0]])
    phi, centers, scales, wavenumbers = gabor_frame_grid(
        grid, bounds, n_scales=3, base_scale=0.15, oversample=2.0
    )
    n, m = phi.shape
    assert m == centers.shape[0] == scales.shape[0] == wavenumbers.shape[0]
    assert m > n  # overcomplete frame
    # Dyadic scales and matching wavenumbers k = 2 pi / L.
    assert len(np.unique(np.asarray(scales))) == 3
    assert jnp.allclose(wavenumbers, 2.0 * jnp.pi / scales)


def test_gabor_frame_grid_centers_stay_within_bounds():
    # base_scale 0.6 with oversample 1 gives spacing 0.6, which does not divide
    # the unit box; no generated centre may fall outside [lo, hi].
    bounds = jnp.array([[0.0, 1.0], [0.0, 1.0]])
    _, centers, _, _ = gabor_frame_grid(
        jnp.zeros((4, 2)), bounds, n_scales=2, base_scale=0.6, oversample=1.0
    )
    c = np.asarray(centers)
    assert bool((c[:, 0] >= 0.0).all() and (c[:, 0] <= 1.0).all())
    assert bool((c[:, 1] >= 0.0).all() and (c[:, 1] <= 1.0).all())


class TestWaveletBasis:
    @pytest.mark.parametrize("wavelet", ["haar", "db2", "db4"])
    @pytest.mark.parametrize("n", [8, 16, 32])
    def test_orthonormal_and_reconstructs(self, wavelet, n):
        phi = wavelet_basis_1d(n, wavelet=wavelet)
        assert phi.shape == (n, n)
        np.testing.assert_allclose(phi.T @ phi, jnp.eye(n), atol=1e-6)
        # synthesis(analysis(f)) == f for an orthonormal basis
        f = jax.random.normal(jax.random.PRNGKey(5), (n,))
        np.testing.assert_allclose(phi @ (phi.T @ f), f, atol=1e-5)


def test_gabor_radial_matches_analytic():
    out = gabor_frame(
        jnp.array([[1.0, 0.0]]),
        jnp.array([[0.0, 0.0]]),
        jnp.array([2.0]),
        jnp.array([3.0]),
    )
    want = float(jnp.exp(-0.5 * (1.0 / 2.0) ** 2) * jnp.cos(3.0))
    assert float(out[0, 0]) == pytest.approx(want, abs=1e-6)
