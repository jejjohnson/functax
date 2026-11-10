"""Tests for the sphere grids of `functax.geometry` that need the spherical harmonics,
copied from geonnax be5537d with functax imports. The slow tier: quadrature
exactness and convergence, each a second or more.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from functax.basis.sphere import real_spherical_harmonics
from functax.geometry import fibonacci_sphere, gauss_legendre_grid, healpix_grid


def test_gauss_legendre_grid_is_exact_for_harmonic_products() -> None:
    # exact for l + l′ ≤ 17 with n_lon = 18 > l + l′
    points, weights = gauss_legendre_grid(9)
    harmonics = np.asarray(real_spherical_harmonics(points, 8))
    gram = (harmonics * np.asarray(weights)[:, None]).T @ harmonics
    np.testing.assert_allclose(gram, np.eye(harmonics.shape[1]), atol=1e-12)


def test_fibonacci_quadrature_error_is_small() -> None:
    # Fibonacci quadrature of a smooth, low-degree integrand has an error of
    # roughly O(1/n) (Hardin et al. 2016); 5e-3 at n = 10⁴ is a loose bound
    points, weights = fibonacci_sphere(10_000)
    harmonics = np.asarray(real_spherical_harmonics(points, 4))
    # ∫ Y_lm for l ≥ 1
    means = (np.asarray(weights)[:, None] * harmonics).sum(axis=0)[1:]
    assert np.max(np.abs(means)) < 5e-3


def _healpix_harmonic_integrals(
    nside: int, l_max: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    ls = np.concatenate([[ell] * (2 * ell + 1) for ell in range(l_max + 1)])
    ms = np.concatenate([np.arange(-ell, ell + 1) for ell in range(l_max + 1)])
    # the 4-fold azimuthal and north-south symmetry of the HEALPix centres
    forbidden = ((ls + ms) % 2 == 1) | (ms % 4 != 0)
    points, weights = healpix_grid(nside)
    integral = np.asarray(real_spherical_harmonics(points, l_max).T @ weights)
    return integral, forbidden


def test_healpix_harmonic_quadrature_symmetry() -> None:
    # HEALPix is not an exact rule, but its symmetry makes ∫ Y_lm vanish exactly
    # when m ≢ 0 (mod 4) or l + m is odd
    integral, forbidden = _healpix_harmonic_integrals(4)
    # x64 round-off on a sum of 192 terms of size O(1)
    np.testing.assert_allclose(integral[forbidden], 0.0, atol=1e-12)
    np.testing.assert_allclose(integral[0], math.sqrt(4 * math.pi), rtol=1e-12)


def test_healpix_harmonic_quadrature_converges() -> None:
    # the remaining low-degree terms decay as O(nside⁻²): a ratio of 4 per
    # doubling (observed 4.0 to 4.6 for nside 2 to 32)
    errors = []
    for nside in (2, 4, 8, 16):
        integral, forbidden = _healpix_harmonic_integrals(nside)
        errors.append(np.abs(integral[~forbidden][1:]).max())
    assert all(a / b > 3.5 for a, b in itertools.pairwise(errors)), errors
