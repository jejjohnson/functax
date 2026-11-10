"""Tests for `functax.basis.slepian`, the region tests copied from geonnax be5537d with
functax imports and the cap tests written for functax. The slow tier:
concentrations, eigenproblems and frames, each a second or more.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from functax.basis.slepian import (
    slepian_cap_basis,
    slepian_polar_gap_basis,
    slepian_region_basis,
)
from functax.geometry import fibonacci_sphere
from functax.quad import gauss_legendre


def _band_quadrature(z_lo: float, z_hi: float, n_z: int, n_lon: int):
    """Exact quadrature over {z_lo < z < z_hi}: Gauss–Legendre in z × uniform lon."""
    nodes, weights = gauss_legendre(n_z)
    z = 0.5 * (z_hi - z_lo) * nodes + 0.5 * (z_hi + z_lo)
    w_z = 0.5 * (z_hi - z_lo) * weights
    lon = 2.0 * math.pi * np.arange(n_lon) / n_lon
    zz, ll = np.meshgrid(z, lon, indexing="ij")
    r = np.sqrt(1.0 - zz**2)
    points = np.stack([r * np.cos(ll), r * np.sin(ll), zz], axis=-1).reshape(-1, 3)
    weights = np.repeat(w_z * 2.0 * math.pi / n_lon, n_lon)
    return jnp.asarray(points), jnp.asarray(weights)


def test_region_basis_reproduces_cap_concentrations():
    l_max, cap = 6, 0.5
    with jax.enable_x64(True):
        points, weights = _band_quadrature(math.cos(cap), 1.0, l_max + 2, 4 * l_max)
        region = slepian_region_basis(l_max, points, weights, n_modes=49)
        reference = slepian_cap_basis(l_max, cap)
    np.testing.assert_allclose(
        np.asarray(region.eigenvalues), np.asarray(reference.eigenvalues), atol=1e-6
    )


def test_shannon_number_default_and_concentration_range():
    l_max, cap = 8, 0.6
    with jax.enable_x64(True):
        points, weights = _band_quadrature(math.cos(cap), 1.0, l_max + 2, 4 * l_max)
        full = slepian_region_basis(l_max, points, weights, n_modes=(l_max + 1) ** 2)
        default = slepian_region_basis(l_max, points, weights)
    shannon = (l_max + 1) ** 2 * (1.0 - math.cos(cap)) / 2.0
    assert abs(float(np.sum(np.asarray(full.eigenvalues))) - shannon) < 1e-8
    assert default.num_modes == round(shannon)
    vals = np.asarray(full.eigenvalues)
    assert vals.min() >= 0.0 and vals.max() <= 1.0
    assert np.all(np.diff(vals) <= 1e-12)  # sorted by decreasing concentration


def test_polar_gap_matches_band_quadrature_and_is_symmetric():
    l_max, gap = 6, 0.3
    with jax.enable_x64(True):
        basis = slepian_polar_gap_basis(l_max, gap, n_modes=(l_max + 1) ** 2)
        c = math.cos(gap)
        points, weights = _band_quadrature(-c, c, l_max + 2, 4 * l_max)
        region = slepian_region_basis(l_max, points, weights, n_modes=(l_max + 1) ** 2)
        x = fibonacci_sphere(64)[0]
        flipped = x * jnp.array([1.0, 1.0, -1.0])
        g, g_flipped = basis.evaluate(x), basis.evaluate(flipped)
    np.testing.assert_allclose(
        np.asarray(basis.eigenvalues), np.asarray(region.eigenvalues), atol=1e-10
    )
    # Concentrations come in near-degenerate pairs, so compare per-mode
    # magnitudes only for modes whose parity is well defined.
    np.testing.assert_allclose(
        np.sort(np.abs(np.asarray(g)), axis=0),
        np.sort(np.abs(np.asarray(g_flipped)), axis=0),
        atol=1e-6,  # eigenvectors of near-degenerate pairs mix at ~1e-8
    )


def test_cap_concentration_matrix_is_symmetric_with_the_shannon_trace():
    from functax.basis.slepian import shannon_number, slepian_concentration_matrix

    l_max, cap = 5, 0.6
    matrix = np.asarray(slepian_concentration_matrix(l_max, cap))
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-14)
    # tr D = (l_max + 1)² A/(4π), with the cap area A = 2π(1 − cos Θ)
    area = 2 * math.pi * (1 - math.cos(cap))
    assert math.isclose(
        np.trace(matrix), float(shannon_number(l_max, area)), rel_tol=1e-12
    )
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert eigenvalues.min() > -1e-12 and eigenvalues.max() < 1 + 1e-12


def test_cap_eigh_per_m_matches_the_full_eigenproblem():
    from functax.basis.slepian import (
        slepian_cap_eigh_per_m,
        slepian_concentration_matrix,
    )

    l_max, cap = 4, 0.8
    matrix = np.asarray(slepian_concentration_matrix(l_max, cap))
    values, coeffs = map(np.asarray, slepian_cap_eigh_per_m(l_max, cap))
    np.testing.assert_allclose(
        values, np.sort(np.linalg.eigvalsh(matrix))[::-1].clip(0, 1), atol=1e-12
    )
    # D C = C diag(λ) and the coefficients are orthonormal
    np.testing.assert_allclose(matrix @ coeffs, coeffs * values, atol=1e-12)
    np.testing.assert_allclose(coeffs.T @ coeffs, np.eye(len(values)), atol=1e-12)


def test_cap_modes_concentrate_in_the_cap():
    # λ_k = ∫_cap g_k² / ∫_S² g_k², checked by an exact band quadrature
    l_max, cap = 6, 0.5
    basis = slepian_cap_basis(l_max, cap, n_modes=5)
    inside, w_inside = _band_quadrature(math.cos(cap), 1.0, l_max + 2, 4 * l_max)
    whole, w_whole = _band_quadrature(-1.0, 1.0, l_max + 2, 4 * l_max)
    energy_in = np.asarray(w_inside) @ np.asarray(basis.evaluate(inside)) ** 2
    energy_all = np.asarray(w_whole) @ np.asarray(basis.evaluate(whole)) ** 2
    np.testing.assert_allclose(
        energy_in / energy_all, np.asarray(basis.eigenvalues), atol=1e-12
    )


def test_rotation_uses_the_frame_of_rotation_to_pole():
    from functax.geometry import lonlat_to_unit, rotation_to_pole

    centre = jnp.array([0.3, 0.5])
    basis = slepian_cap_basis(4, 0.4, n_modes=3, eig_threshold=0.0).rotate_to(centre)
    rng = np.random.default_rng(0)
    lonlat = np.stack([rng.uniform(-180, 180, 5), rng.uniform(-80, 80, 5)], axis=-1)
    xyz = lonlat_to_unit(jnp.asarray(lonlat), unit="degrees")
    rotation = rotation_to_pole(centre, unit="radians")
    np.testing.assert_allclose(
        np.asarray(basis.centred_coordinates(xyz)),
        np.asarray(xyz) @ np.asarray(rotation).T,
        atol=1e-12,
    )


def test_cap_validation():
    from functax import ValidationError

    with pytest.raises(ValidationError, match="'cap_radius' must lie in"):
        slepian_cap_basis(3, 4.0)
    with pytest.raises(ValidationError, match="'n_modes' must be at least 1"):
        slepian_cap_basis(3, 0.5, n_modes=0)
    with pytest.raises(ValidationError, match="keeps only"):
        slepian_cap_basis(3, 0.2, n_modes=10, eig_threshold=0.9)
    import inspect

    from functax.basis.slepian import SlepianCapBasis

    # unwrapped: the type hook would reject the shape first
    rotate_to = inspect.unwrap(SlepianCapBasis.rotate_to)
    with pytest.raises(ValidationError, match="'lonlat_centre' must have shape"):
        rotate_to(slepian_cap_basis(3, 0.5), jnp.zeros(3))
