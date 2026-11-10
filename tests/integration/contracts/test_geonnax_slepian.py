"""functax's Slepian bases (#19) against geonnax, pinned at geonnax be5537d as data.

Eigenvectors are compared through the projector onto each cluster of eigenvalues:
the ±m blocks of a cap or a polar gap share their eigenvalues, and many
concentrations sit within 1e-9 of 1, so eigh may pick any orthonormal basis of
such a cluster, and a vector separated by a gap δ moves by about ε/δ.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from _harness import assert_matches_oracle

from functax.basis.slepian import (
    shannon_number,
    slepian_cap_basis,
    slepian_cap_eigh_per_m,
    slepian_concentration_matrix,
    slepian_polar_gap_basis,
    slepian_region_basis,
    spherical_polygon_mask,
)
from functax.geometry import gauss_legendre_grid


GEONNAX = np.load(Path(__file__).parent / "data" / "geonnax_slepian.npz")
POINTS = jnp.asarray(GEONNAX["points"])


def _assert_same_eigenvectors(
    values: np.ndarray, actual: np.ndarray, expected: np.ndarray
) -> None:
    """Compare the eigenvectors through the projector onto each eigenvalue cluster."""
    start = 0
    while start < len(values):
        stop = start + 1
        # a gap of 1e-6 bounds the eigenvector sensitivity ε/δ near 1e-10
        while stop < len(values) and abs(values[stop] - values[stop - 1]) < 1e-6:
            stop += 1
        a = actual[:, start:stop]
        e = expected[:, start:stop]
        assert_matches_oracle(a @ a.T, e @ e.T, rtol=0, atol=1e-9)
        start = stop


def test_shannon_number_and_concentration_matrix_equal_geonnax() -> None:
    assert_matches_oracle(shannon_number(5, 1.3), GEONNAX["shannon"], rtol=1e-14)
    assert_matches_oracle(
        slepian_concentration_matrix(5, 0.6),
        GEONNAX["concentration"],
        rtol=1e-12,
        atol=1e-14,
    )


def test_cap_eigenproblem_equals_geonnax() -> None:
    values, coeffs = map(np.asarray, slepian_cap_eigh_per_m(5, 0.6))
    assert_matches_oracle(values, GEONNAX["cap_values"], rtol=1e-10, atol=1e-13)
    _assert_same_eigenvectors(values, coeffs, GEONNAX["cap_coeffs"])


def test_rotated_cap_basis_spans_what_geonnax_spans() -> None:
    basis = slepian_cap_basis(5, 0.6, n_modes=6, lonlat_centre=jnp.array([0.4, -0.3]))
    values = np.asarray(basis.eigenvalues)
    assert_matches_oracle(values, GEONNAX["cap_eigenvalues"], rtol=1e-10, atol=1e-13)
    _assert_same_eigenvectors(
        values, np.asarray(basis.evaluate(POINTS)), GEONNAX["cap_eval"]
    )


def test_polar_gap_basis_equals_geonnax() -> None:
    gap = slepian_polar_gap_basis(6, 0.25)
    values = np.asarray(gap.eigenvalues)
    assert_matches_oracle(values, GEONNAX["gap_eigenvalues"], rtol=1e-10, atol=1e-13)
    _assert_same_eigenvectors(values, np.asarray(gap.coeffs), GEONNAX["gap_coeffs"])


def test_region_basis_equals_geonnax() -> None:
    grid, weights = gauss_legendre_grid(14)
    north = np.asarray(grid[:, 2]) > 0.3
    region = slepian_region_basis(5, grid[north], weights[north])
    values = np.asarray(region.eigenvalues)
    assert_matches_oracle(values, GEONNAX["region_eigenvalues"], rtol=1e-10, atol=1e-13)
    _assert_same_eigenvectors(
        values, np.asarray(region.coeffs), GEONNAX["region_coeffs"]
    )


def test_polygon_mask_equals_geonnax() -> None:
    mask = spherical_polygon_mask(
        POINTS, jnp.asarray(GEONNAX["polygon"]), unit="degrees"
    )
    assert_matches_oracle(mask, GEONNAX["mask"], rtol=0)
