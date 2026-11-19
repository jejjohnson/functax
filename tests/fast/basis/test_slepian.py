"""Tests for `functax.basis.slepian`, the region tests copied from geonnax be5537d with
functax imports and the cap tests written for functax. The fast tier: validation,
each check reached with the smallest input (a degree of 0, or the unwrapped method
on no basis at all).
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import pytest

from functax import ValidationError
from functax.basis.slepian import (
    SlepianCapBasis,
    slepian_cap_basis,
    slepian_concentration_matrix,
    slepian_polar_gap_basis,
    slepian_region_basis,
    spherical_polygon_mask,
)


def test_validation():
    with pytest.raises(ValueError, match="gap_colatitude"):
        slepian_polar_gap_basis(3, 2.0)
    with pytest.raises(ValueError, match="polygon"):
        spherical_polygon_mask(jnp.zeros((1, 3)), jnp.zeros((2, 2)), unit="degrees")
    # unwrapped: the type hook would reject the mismatched lengths first
    with pytest.raises(ValueError, match="region_weights"):
        inspect.unwrap(slepian_region_basis)(2, jnp.zeros((3, 3)), jnp.zeros(2))


def test_cap_basis_rejects_a_radius_outside_the_sphere() -> None:
    with pytest.raises(ValidationError, match="'cap_radius' must lie in"):
        slepian_cap_basis(0, 4.0)


def test_cap_basis_rejects_zero_modes() -> None:
    with pytest.raises(ValidationError, match="'n_modes' must be at least 1"):
        slepian_cap_basis(0, 0.5, n_modes=0)


def test_rotate_to_rejects_a_centre_of_the_wrong_shape() -> None:
    # the shape check runs before the basis is read, so no basis is built
    rotate_to = inspect.unwrap(SlepianCapBasis.rotate_to)
    with pytest.raises(ValidationError, match="'lonlat_centre' must have shape"):
        rotate_to(None, jnp.zeros(3))


def test_centred_coordinates_rejects_points_of_the_wrong_shape() -> None:
    centred_coordinates = inspect.unwrap(SlepianCapBasis.centred_coordinates)
    with pytest.raises(ValidationError, match=r"'unit_xyz' must have shape \(N, 3\)"):
        centred_coordinates(None, jnp.zeros((2, 2)))


def test_concentration_matrix_rejects_a_negative_quadrature_size() -> None:
    with pytest.raises(ValidationError, match="'num_quadrature' must be at least 1"):
        slepian_concentration_matrix(0, 0.5, num_quadrature=-1)
