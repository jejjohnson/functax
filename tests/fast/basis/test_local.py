"""Tests for `functax.basis.local`, copied from geonnax be5537d with functax imports.
The fast tier: closed forms, supports, layouts and validation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from functax.basis.local import (
    gabor_frame,
    gabor_frame_grid,
    rbf_basis,
    spherical_rbf_basis,
    wavelet_basis_1d,
    wavelet_basis_2d,
    wendland_c2,
    wendland_c4,
)


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


def test_rbf_rejects_unknown_kernel():
    with pytest.raises(ValueError, match="'kernel' must be one of"):
        rbf_basis(jnp.zeros((2, 2)), jnp.zeros((1, 2)), jnp.ones(1), kernel="bogus")


def test_gabor_radial_matches_analytic():
    out = gabor_frame(
        jnp.array([[1.0, 0.0]]),
        jnp.array([[0.0, 0.0]]),
        jnp.array([2.0]),
        jnp.array([3.0]),
    )
    want = float(jnp.exp(-0.5 * (1.0 / 2.0) ** 2) * jnp.cos(3.0))
    assert float(out[0, 0]) == pytest.approx(want, abs=1e-6)


def test_gabor_directional_reduces_to_radial_when_aligned():
    x, c, L, k = (
        jnp.array([[1.0, 0.0]]),
        jnp.array([[0.0, 0.0]]),
        jnp.array([2.0]),
        jnp.array([3.0]),
    )
    radial = gabor_frame(x, c, L, k)
    directional = gabor_frame(x, c, L, k, orientations=jnp.array([[1.0, 0.0]]))
    assert float(directional[0, 0]) == pytest.approx(float(radial[0, 0]), abs=1e-6)


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


def test_gabor_frame_grid_rejects_bad_args():
    grid, bounds = jnp.zeros((4, 2)), jnp.array([[0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="'n_scales' must be"):
        gabor_frame_grid(grid, bounds, n_scales=0, base_scale=0.1)
    with pytest.raises(ValueError, match="'base_scale' must be"):
        gabor_frame_grid(grid, bounds, n_scales=2, base_scale=0.0)
    with pytest.raises(ValueError, match="lo < hi"):
        gabor_frame_grid(
            grid, jnp.array([[1.0, 0.0], [0.0, 1.0]]), n_scales=2, base_scale=0.1
        )


class TestSphericalRBFBasis:
    def test_peak_at_centre(self):
        x = jnp.eye(3)  # three orthonormal points on the sphere
        phi = spherical_rbf_basis(x, x, jnp.ones(3))
        # geodesic distance to itself is 0 -> Gaussian peaks at 1 on the diagonal
        np.testing.assert_allclose(jnp.diag(phi), 1.0, atol=1e-6)

    @pytest.mark.parametrize("kernel", ["wendland_c2", "wendland_c4"])
    def test_wendland_compact_support_antipode(self, kernel):
        north = jnp.array([[0.0, 0.0, 1.0]])
        south = jnp.array([[0.0, 0.0, -1.0]])  # geodesic distance pi
        phi = spherical_rbf_basis(south, north, jnp.array([0.5]), kernel=kernel)
        assert float(phi[0, 0]) == 0.0

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

    def test_rejects_unknown_kernel(self):
        with pytest.raises(ValueError):
            spherical_rbf_basis(jnp.eye(3), jnp.eye(3), jnp.ones(3), kernel="nope")


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

    def test_levels_capped_and_partial(self):
        # one level is still orthonormal; more levels than log2(n) is capped.
        phi1 = wavelet_basis_1d(16, wavelet="haar", levels=1)
        np.testing.assert_allclose(phi1.T @ phi1, jnp.eye(16), atol=1e-6)
        full = wavelet_basis_1d(16, wavelet="haar")
        capped = wavelet_basis_1d(16, wavelet="haar", levels=99)
        np.testing.assert_allclose(full, capped, atol=1e-6)

    def test_2d_orthonormal(self):
        phi = wavelet_basis_2d(4, 8, wavelet="db2")
        assert phi.shape == (32, 32)
        np.testing.assert_allclose(phi.T @ phi, jnp.eye(32), atol=1e-6)

    def test_rejects_non_power_of_two_and_unknown(self):
        with pytest.raises(ValueError):
            wavelet_basis_1d(12, wavelet="haar")
        with pytest.raises(ValueError):
            wavelet_basis_1d(16, wavelet="db8")


def test_remaining_validation() -> None:
    from functax import ValidationError

    x = jnp.zeros((2, 2))
    with pytest.raises(ValidationError, match="'oversample' must be positive"):
        gabor_frame_grid(
            x,
            jnp.array([[0.0, 1.0], [0.0, 1.0]]),
            n_scales=1,
            base_scale=0.5,
            oversample=0.0,
        )
    with pytest.raises(ValidationError, match=r"'bounds' must have shape \(d, 2\)"):
        # unwrapped: the type hook would reject the shape first
        import inspect

        inspect.unwrap(gabor_frame_grid)(
            x, jnp.zeros((2, 3)), n_scales=1, base_scale=0.5
        )
    with pytest.raises(ValidationError, match="'wavelet' must be one of"):
        wavelet_basis_1d(8, wavelet="sym4")
    with pytest.raises(ValidationError, match="'levels' must be at least 1"):
        wavelet_basis_1d(8, levels=0)
