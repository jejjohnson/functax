"""functax's box Fourier and vector bases (#19) against geonnax, pinned at geonnax
be5537d as data. `divfree_basis` is compared under the Dirichlet boundary only:
under the others geonnax built Dirichlet atoms with the other eigenvalues, which
functax fixes.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax.basis.fourier import (
    fourier_basis,
    fourier_basis_1d,
    fourier_eigenvalues,
    fourier_eigenvalues_1d,
)
from functax.basis.vector import curlfree_basis, divfree_basis


GEONNAX = np.load(Path(__file__).parent / "data" / "geonnax_box_bases.npz")
X1 = jnp.asarray(GEONNAX["x1"])
XY = jnp.asarray(GEONNAX["xy"])


@pytest.mark.parametrize("boundary", ["dirichlet", "neumann", "periodic"])
def test_one_dimensional_bases_equal_geonnax(boundary: str) -> None:
    values = fourier_basis_1d(X1, 6, 1.5, boundary=boundary)
    assert_matches_oracle(
        values, GEONNAX[f"basis_1d_{boundary}"], rtol=1e-12, atol=1e-14
    )
    eigenvalues = fourier_eigenvalues_1d(6, 1.5, dtype=jnp.float64, boundary=boundary)
    assert_matches_oracle(
        eigenvalues, GEONNAX[f"eigenvalues_1d_{boundary}"], rtol=1e-12
    )


def test_tensor_bases_equal_geonnax() -> None:
    phi, lam = fourier_basis(XY, (3, 4), (1.5, 2.0), boundary=("periodic", "dirichlet"))
    assert_matches_oracle(phi, GEONNAX["basis_mixed"], rtol=1e-12, atol=1e-14)
    assert_matches_oracle(lam, GEONNAX["basis_mixed_lam"], rtol=1e-12)
    eigenvalues = fourier_eigenvalues(
        (2, 3, 2), (1.0, 2.0, 0.5), 3, dtype=jnp.float64, boundary="neumann"
    )
    assert_matches_oracle(eigenvalues, GEONNAX["eigenvalues_3d"], rtol=1e-12)


@pytest.mark.parametrize("boundary", ["dirichlet", "neumann", "periodic"])
def test_curlfree_basis_equals_geonnax(boundary: str) -> None:
    phi, lam = curlfree_basis(XY, (3, 4), (1.5, 2.0), boundary=boundary)
    assert_matches_oracle(phi, GEONNAX[f"curlfree_{boundary}"], rtol=1e-12, atol=1e-14)
    assert_matches_oracle(lam, GEONNAX[f"curlfree_{boundary}_lam"], rtol=1e-12)


def test_divfree_basis_equals_geonnax_under_dirichlet() -> None:
    phi, lam = divfree_basis(XY, (3, 4), (1.5, 2.0))
    assert_matches_oracle(phi, GEONNAX["divfree"], rtol=1e-12, atol=1e-14)
    assert_matches_oracle(lam, GEONNAX["divfree_lam"], rtol=1e-12)
