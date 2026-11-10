"""Tests for `functax.basis.local`, copied from geonnax be5537d with functax imports.
The fast tier: closed forms, supports, layouts and validation.
"""

from __future__ import annotations

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
)


def test_rbf_rejects_unknown_kernel():
    with pytest.raises(ValueError, match="'kernel' must be one of"):
        rbf_basis(jnp.zeros((2, 2)), jnp.zeros((1, 2)), jnp.ones(1), kernel="bogus")


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
    def test_rejects_unknown_kernel(self):
        with pytest.raises(ValueError):
            spherical_rbf_basis(jnp.eye(3), jnp.eye(3), jnp.ones(3), kernel="nope")


class TestWaveletBasis:
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
