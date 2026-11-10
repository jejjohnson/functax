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
    gabor_frame_grid,
    rbf_basis,
    spherical_rbf_basis,
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
