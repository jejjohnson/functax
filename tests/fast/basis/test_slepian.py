"""Tests for `functax.basis.slepian`, the region tests copied from geonnax be5537d with
functax imports and the cap tests written for functax. The fast tier: the polygon
test and validation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from functax.basis.slepian import (
    slepian_polar_gap_basis,
    slepian_region_basis,
    spherical_polygon_mask,
)
from functax.geometry import fibonacci_sphere


def _brute_force_inside(lonlat_deg, lon_range, lat_range):
    lon, lat = lonlat_deg[:, 0], lonlat_deg[:, 1]
    lon_lo, lon_hi = lon_range
    in_lon = (
        (lon >= lon_lo) & (lon <= lon_hi)
        if lon_lo < lon_hi
        else (lon >= lon_lo) | (lon <= lon_hi)
    )
    return in_lon & (lat >= lat_range[0]) & (lat <= lat_range[1])


@pytest.mark.parametrize(
    ("lon_range", "lat_range"),
    [((-20.0, 30.0), (-10.0, 25.0)), ((170.0, -165.0), (5.0, 20.0))],
)
def test_polygon_mask_on_rectangles(lon_range, lat_range):
    # Dense vertices along the parallels make the great-circle edges follow
    # them closely; test points well away from the boundary.
    lo, hi = lon_range
    span = (hi - lo) % 360.0
    south = [[lo + span * t, lat_range[0]] for t in np.linspace(0, 1, 200)]
    north = [[lo + span * t, lat_range[1]] for t in np.linspace(1, 0, 200)]
    polygon = np.array(south + north)
    polygon[:, 0] = (polygon[:, 0] + 180.0) % 360.0 - 180.0
    points_lonlat, _ = fibonacci_sphere(4000, output="lonlat", unit="degrees")
    points_xyz, _ = fibonacci_sphere(4000)
    mask = np.asarray(
        spherical_polygon_mask(points_xyz, jnp.asarray(polygon), unit="degrees")
    )
    expected = _brute_force_inside(np.asarray(points_lonlat), lon_range, lat_range)
    lat = np.asarray(points_lonlat)[:, 1]
    lon = np.asarray(points_lonlat)[:, 0]
    margin = 1.5
    near_edge = (
        (np.abs(lat - lat_range[0]) < margin)
        | (np.abs(lat - lat_range[1]) < margin)
        | (np.abs(((lon - lo + 180) % 360) - 180) < margin)
        | (np.abs(((lon - hi + 180) % 360) - 180) < margin)
    )
    assert expected.sum() > 20
    np.testing.assert_array_equal(mask[~near_edge], expected[~near_edge])


def test_validation():
    with pytest.raises(ValueError, match="gap_colatitude"):
        slepian_polar_gap_basis(3, 2.0)
    with pytest.raises(ValueError, match="polygon"):
        spherical_polygon_mask(jnp.zeros((1, 3)), jnp.zeros((2, 2)), unit="degrees")
    # unwrapped: the type hook would reject the mismatched lengths first
    import inspect

    with pytest.raises(ValueError, match="region_weights"):
        inspect.unwrap(slepian_region_basis)(2, jnp.zeros((3, 3)), jnp.zeros(2))


def test_remaining_validation() -> None:
    import inspect

    from functax import ValidationError
    from functax.basis.slepian import (
        SlepianCapBasis,
        slepian_cap_basis,
        slepian_concentration_matrix,
        slepian_polar_gap_basis,
    )

    with pytest.raises(ValidationError, match="'num_quadrature' must be at least 1"):
        slepian_concentration_matrix(2, 0.5, num_quadrature=-1)
    with pytest.raises(ValidationError, match="'lonlat_centre' must have shape"):
        inspect.unwrap(slepian_cap_basis)(2, 0.5, lonlat_centre=jnp.zeros(3))
    basis = slepian_cap_basis(2, 0.5)
    with pytest.raises(ValidationError, match=r"'unit_xyz' must have shape \(N, 3\)"):
        inspect.unwrap(SlepianCapBasis.centred_coordinates)(basis, jnp.zeros((2, 2)))
    with pytest.raises(ValidationError, match="'n_modes' must be at least 1"):
        slepian_polar_gap_basis(3, 0.3, n_modes=0)


def test_integer_polygon_vertices_are_accepted() -> None:
    import inspect

    square = jnp.array([[-10, -10], [10, -10], [10, 10], [-10, 10]])
    points = jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    mask = inspect.unwrap(spherical_polygon_mask)(points, square, unit="degrees")
    assert mask.tolist() == [True, False]
