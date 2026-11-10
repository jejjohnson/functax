"""functax's polynomial and sphere bases (#19) against geonnax, pinned at geonnax
be5537d as data.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax.basis.polynomial import associated_legendre, legendre_polynomials
from functax.basis.sphere import (
    needlet_basis,
    needlet_window,
    real_spherical_harmonics,
    shell_basis,
    vector_spherical_harmonics,
)


GEONNAX = np.load(Path(__file__).parent / "data" / "geonnax_sphere_bases.npz")
T = jnp.asarray(GEONNAX["t"])
POINTS = jnp.asarray(GEONNAX["points"])


def test_legendre_polynomials_equal_geonnax() -> None:
    assert_matches_oracle(
        legendre_polynomials(T, 12),
        GEONNAX["legendre_polynomials"],
        rtol=1e-12,
        atol=1e-15,
    )


@pytest.mark.parametrize("normalization", ["orthonormal", "schmidt", "none"])
def test_associated_legendre_equals_geonnax(normalization: str) -> None:
    values = associated_legendre(T, 10, normalization=normalization)
    expected = GEONNAX[f"associated_legendre_{normalization}"]
    assert_matches_oracle(
        values, expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max()
    )


@pytest.mark.parametrize(("l_max", "points"), [(8, slice(None)), (40, slice(-6, None))])
def test_real_spherical_harmonics_equal_geonnax(l_max: int, points: slice) -> None:
    values = real_spherical_harmonics(POINTS[points], l_max)
    assert_matches_oracle(
        values, GEONNAX[f"real_spherical_harmonics_{l_max}"], rtol=1e-12, atol=1e-13
    )


def test_vector_spherical_harmonics_equal_geonnax() -> None:
    psi, phi, degrees = vector_spherical_harmonics(POINTS, 4)
    assert_matches_oracle(psi, GEONNAX["vsh_psi"], rtol=1e-12, atol=1e-13)
    assert_matches_oracle(phi, GEONNAX["vsh_phi"], rtol=1e-12, atol=1e-13)
    assert_matches_oracle(degrees, GEONNAX["vsh_degrees"], rtol=0)
    psi, phi, _ = vector_spherical_harmonics(
        POINTS[:-3], 3, output="enu", normalized=False
    )
    assert_matches_oracle(psi, GEONNAX["vsh_enu_psi"], rtol=1e-12, atol=1e-13)
    assert_matches_oracle(phi, GEONNAX["vsh_enu_phi"], rtol=1e-12, atol=1e-13)


def test_needlets_equal_geonnax() -> None:
    window = needlet_window(1.7)(jnp.asarray(GEONNAX["xi"]))
    assert_matches_oracle(window, GEONNAX["needlet_window"], rtol=1e-12, atol=1e-15)
    values, levels, centres = needlet_basis(POINTS, j_max=1)
    assert_matches_oracle(values, GEONNAX["needlet_values"], rtol=1e-12, atol=1e-13)
    assert_matches_oracle(levels, GEONNAX["needlet_levels"], rtol=0)
    assert_matches_oracle(centres, GEONNAX["needlet_centres"], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("radial", ["legendre", "chebyshev"])
def test_shell_basis_equals_geonnax(radial: str) -> None:
    values, indices = shell_basis(
        jnp.asarray(GEONNAX["shell_points"]),
        3,
        4,
        r_inner=1.0,
        r_outer=1.8,
        radial=radial,
    )
    assert_matches_oracle(values, GEONNAX[f"shell_{radial}"], rtol=1e-12, atol=1e-13)
    assert_matches_oracle(indices, GEONNAX[f"shell_{radial}_indices"], rtol=0)
