"""Tests for `functax.basis.local`, copied from geonnax be5537d with functax imports.
The slow tier: Gram matrices, gradients, rotation invariance and frame
reconstruction, each a second or more.
"""

from __future__ import annotations

import math

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from hypothesis import given, settings

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


KEY = jr.PRNGKey(0)


def test_rbf_gram_is_positive_semidefinite():
    pts = jr.uniform(KEY, (40, 2))
    centers = jr.uniform(jr.PRNGKey(1), (12, 2))
    phi = rbf_basis(pts, centers, jnp.full(12, 0.5), kernel="wendland_c2")
    eigmin = float(jnp.linalg.eigvalsh(phi.T @ phi).min())
    assert eigmin > -1e-6


@pytest.mark.parametrize("kernel", ["wendland_c2", "wendland_c4"])
def test_rbf_narrow_atom_peaks_at_center(kernel):
    # A point on its centre must peak at 1 even for a width comparable to the
    # safe-sqrt softening — the value must not be shifted by an additive epsilon.
    width = 1e-6
    out = rbf_basis(
        jnp.zeros((1, 2)), jnp.zeros((1, 2)), jnp.array([width]), kernel=kernel
    )
    assert float(out[0, 0]) == pytest.approx(1.0, rel=1e-12)

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
    assert float(kernel(jnp.array(0.0))) == pytest.approx(1.0, rel=1e-12)
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
    assert float(out[0, 0]) == pytest.approx(1.0, rel=1e-12)  # on its own centre
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
    assert jnp.allclose(wavenumbers, 2.0 * jnp.pi / scales, rtol=1e-12, atol=0.0)


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


def _random_rotation(seed: int) -> np.ndarray:
    """A random proper rotation of ℝ³, from the QR factors of a Gaussian matrix."""
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def _unit(points: jax.Array) -> jax.Array:
    return points / jnp.linalg.norm(points, axis=-1, keepdims=True)


# compiled once per module, so each hypothesis example reuses the kernels
_RBF = {
    kernel: jax.jit(lambda x, c, w, kernel=kernel: rbf_basis(x, c, w, kernel=kernel))
    for kernel in ("gaussian", "wendland_c2", "wendland_c4")
}


_GABOR = jax.jit(gabor_frame)


@settings(max_examples=20, deadline=None)
@given(
    seed=st.integers(0, 2**31 - 1),
    kernel=st.sampled_from(["gaussian", "wendland_c2", "wendland_c4"]),
)
def test_rbf_basis_is_invariant_under_rigid_motions(seed: int, kernel: str) -> None:
    # φ_a(Qx + t) with centres Qc_a + t is φ_a(x): only ‖x − c_a‖ enters
    keys = jr.split(jr.key(seed), 4)
    x = jr.normal(keys[0], (6, 3))
    centres = jr.normal(keys[1], (4, 3))
    widths = jr.uniform(keys[2], (4,), minval=0.5, maxval=3.0)
    shift = jr.normal(keys[3], (3,))
    rotation = jnp.asarray(_random_rotation(seed))
    moved = _RBF[kernel](x @ rotation.T + shift, centres @ rotation.T + shift, widths)
    assert_matches_oracle(moved, _RBF[kernel](x, centres, widths), rtol=0.0, atol=1e-12)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_gabor_frame_is_equivariant_under_rigid_motions(seed: int) -> None:
    # moving the points, the centres and the orientations by the same rigid
    # motion leaves every atom unchanged, radial or directional
    keys = jr.split(jr.key(seed), 6)
    x = jr.normal(keys[0], (6, 3))
    centres = jr.normal(keys[1], (4, 3))
    scales = jr.uniform(keys[2], (4,), minval=0.5, maxval=2.0)
    wavenumbers = jr.uniform(keys[3], (4,), minval=0.5, maxval=4.0)
    orientations = jr.normal(keys[4], (4, 3))
    shift = jr.normal(keys[5], (3,))
    rotation = jnp.asarray(_random_rotation(seed))
    moved_x = x @ rotation.T + shift
    moved_centres = centres @ rotation.T + shift
    for direction in (None, orientations):
        moved = _GABOR(
            moved_x,
            moved_centres,
            scales,
            wavenumbers,
            orientations=None if direction is None else direction @ rotation.T,
        )
        expected = _GABOR(x, centres, scales, wavenumbers, orientations=direction)
        assert_matches_oracle(moved, expected, rtol=0.0, atol=1e-12)


_POINTS = jnp.array([[0.3, -0.2, 0.5], [-0.4, 0.6, 0.1], [0.9, 0.2, -0.3]])


_CENTRES = jnp.array([[0.1, 0.1, 0.2], [-0.5, 0.3, 0.4]])


@pytest.mark.parametrize("kernel", ["gaussian", "wendland_c2", "wendland_c4"])
def test_rbf_gradient_matches_finite_differences(kernel: str) -> None:
    # differentiated in the points, the centres and the widths together; every
    # scaled distance is between 0.1 and 0.9, off the centre and the support edge
    widths = jnp.array([1.4, 1.9])
    assert_grad_matches_fd(
        lambda args: rbf_basis(*args, kernel=kernel),
        (_POINTS, _CENTRES, widths),
        rtol=1e-8,
    )


def test_spherical_rbf_gradient_matches_finite_differences() -> None:
    # differentiated in the points, the centres and the widths, at points off
    # every centre (the arccos is singular there)
    points = _unit(_POINTS)
    centres = _unit(_CENTRES)
    # the Wendland profiles' own gradients are checked on the Euclidean basis
    assert_grad_matches_fd(
        lambda args: spherical_rbf_basis(*args),
        (points, centres, jnp.array([1.5, 2.5])),
        rtol=1e-7,
    )


def test_gabor_gradient_matches_finite_differences() -> None:
    # differentiated in the points, centres, scales, wavenumbers and (for the
    # directional atoms) orientations, at points off every centre
    scales = jnp.array([0.8, 1.3])
    wavenumbers = jnp.array([2.0, 3.5])
    orientations = jnp.array([[1.0, 0.5, -0.2], [0.3, -1.0, 0.4]])
    assert_grad_matches_fd(
        lambda args: gabor_frame(*args), (_POINTS, _CENTRES, scales, wavenumbers)
    )
    assert_grad_matches_fd(
        lambda args: gabor_frame(*args[:4], orientations=args[4]),
        (_POINTS, _CENTRES, scales, wavenumbers, orientations),
    )


@pytest.mark.parametrize("kernel", [wendland_c2, wendland_c4])
def test_wendland_gradient_matches_finite_differences(kernel) -> None:
    # inside the support, off r = 0 and r = 1
    assert_grad_matches_fd(kernel, jnp.array([0.05, 0.3, 0.6, 0.95]), rtol=1e-8)


def test_local_bases_trace() -> None:
    # batches of point sets against fixed atoms
    points = jr.normal(jr.key(7), (3, 5, 3))
    widths = jnp.array([1.4, 1.9])
    orientations = jnp.array([[1.0, 0.5, -0.2], [0.3, -1.0, 0.4]])

    def bases(x: jax.Array) -> tuple[jax.Array, ...]:
        return (
            rbf_basis(x, _CENTRES, widths),
            rbf_basis(x, _CENTRES, widths, kernel="wendland_c2"),
            spherical_rbf_basis(
                _unit(x), _unit(_CENTRES), widths, kernel="wendland_c4"
            ),
            gabor_frame(x, _CENTRES, widths, widths, orientations=orientations),
        )

    assert_traces(bases, points)


def test_gabor_frame_grid_matches_its_lattice() -> None:
    # bounds [0, 1] × [0, 2] with L₀ = ½ and two scales: spacing ½ gives 3 × 5
    # centres, spacing 1 gives 2 × 3, x outermost (meshgrid "ij")
    bounds = jnp.array([[0.0, 1.0], [0.0, 2.0]])
    x = jnp.array([[0.3, 0.4], [0.9, 1.7], [0.0, 0.0]])
    atoms, centres, scales, wavenumbers = gabor_frame_grid(
        x, bounds, n_scales=2, base_scale=0.5
    )
    lattice = [
        np.stack(
            np.meshgrid(
                np.arange(0.0, 1.01, step), np.arange(0.0, 2.01, step), indexing="ij"
            ),
            -1,
        ).reshape(-1, 2)
        for step in (0.5, 1.0)
    ]
    expected_centres = np.concatenate(lattice)
    expected_scales = np.repeat([0.5, 1.0], [15, 6])
    assert_matches_oracle(centres, expected_centres, rtol=1e-12)
    assert_matches_oracle(scales, expected_scales, rtol=1e-12)
    assert_matches_oracle(wavenumbers, 2.0 * np.pi / expected_scales, rtol=1e-12)
    # φ_a(x) = exp(−r²/(2L_a²)) cos(2π r/L_a), r = ‖x − c_a‖
    r = np.linalg.norm(np.asarray(x)[:, None, :] - expected_centres[None], axis=-1)
    expected = np.exp(-(r**2) / (2.0 * expected_scales**2)) * np.cos(
        2.0 * np.pi * r / expected_scales
    )
    assert_matches_oracle(atoms, expected, rtol=1e-12, atol=1e-15)


_S = 1.0 / math.sqrt(2.0)


def test_haar_basis_matches_the_haar_matrix() -> None:
    # Φ = Wᵀ, the rows of W the scaling function, the coarse wavelet and the
    # two fine wavelets; one level keeps the two fine approximations instead
    full = np.array(
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5, -0.5],
            [_S, -_S, 0.0, 0.0],
            [0.0, 0.0, _S, -_S],
        ]
    )
    one_level = np.array(
        [
            [_S, _S, 0.0, 0.0],
            [0.0, 0.0, _S, _S],
            [_S, -_S, 0.0, 0.0],
            [0.0, 0.0, _S, -_S],
        ]
    )
    assert_matches_oracle(wavelet_basis_1d(4), full.T, rtol=1e-12, atol=1e-15)
    assert_matches_oracle(
        wavelet_basis_1d(4, levels=1), one_level.T, rtol=1e-12, atol=1e-15
    )
    # the 2-D basis is the Kronecker product, rows iy · nx + ix
    haar_2 = np.array([[_S, _S], [_S, -_S]]).T
    assert_matches_oracle(
        wavelet_basis_2d(2, 4), np.kron(haar_2, full.T), rtol=1e-12, atol=1e-15
    )


def test_db2_level_matches_the_closed_form_filter() -> None:
    # h = (1 − √3, 3 − √3, 3 + √3, 1 + √3)/(4√2), g_k = (−1)ᵏ h_{3−k}; one
    # periodic level reads row n of W from x_{(2n + k) mod 8}
    root3 = math.sqrt(3.0)
    h = np.array([1.0 - root3, 3.0 - root3, 3.0 + root3, 1.0 + root3]) / (
        4.0 * math.sqrt(2.0)
    )
    g = (-1.0) ** np.arange(4) * h[::-1]
    analysis = np.zeros((8, 8))
    for n in range(4):
        for k in range(4):
            analysis[n, (2 * n + k) % 8] += h[k]
            analysis[4 + n, (2 * n + k) % 8] += g[k]
    # the taps are PyWavelets' printed decimals, which differ from the closed
    # form by up to 3.4e-13 (measured), so the tolerance is absolute at 1e-12
    assert_matches_oracle(
        wavelet_basis_1d(8, wavelet="db2", levels=1),
        analysis.T,
        rtol=0.0,
        atol=1e-12,
    )
