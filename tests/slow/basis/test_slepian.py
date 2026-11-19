"""Tests for `functax.basis.slepian`, the region tests copied from geonnax be5537d with
functax imports and the cap tests written for functax. The slow tier:
concentrations, eigenproblems and frames, each a second or more.
"""

from __future__ import annotations

import functools
import inspect
import math

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from hypothesis import given, settings

from functax import ValidationError
from functax.basis.slepian import (
    slepian_cap_basis,
    slepian_concentration_matrix,
    slepian_polar_gap_basis,
    slepian_region_basis,
    spherical_polygon_mask,
)
from functax.basis.sphere import harmonic_degrees
from functax.geometry import fibonacci_sphere
from functax.quad import gauss_legendre


def _brute_force_inside(lonlat_deg, lon_range, lat_range):
    lon, lat = lonlat_deg[:, 0], lonlat_deg[:, 1]
    lon_lo, lon_hi = lon_range
    in_lon = (
        (lon >= lon_lo) & (lon <= lon_hi)
        if lon_lo < lon_hi
        else (lon >= lon_lo) | (lon <= lon_hi)
    )
    return in_lon & (lat >= lat_range[0]) & (lat <= lat_range[1])


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


def test_cap_basis_rejects_a_threshold_that_keeps_too_few_modes():
    # the check needs the concentrations, so it runs after the eigenproblem;
    # degree 0 is the smallest one (the other argument checks are fast tests)
    with pytest.raises(ValidationError, match="keeps only"):
        slepian_cap_basis(0, 0.2, n_modes=1, eig_threshold=0.9)


def test_cap_basis_rejects_a_lonlat_centre_of_the_wrong_shape():
    # the centre is checked after the eigenproblem too; degree 0 again, and
    # unwrapped, as the type hook would reject the shape first
    with pytest.raises(ValidationError, match="'lonlat_centre' must have shape"):
        inspect.unwrap(slepian_cap_basis)(0, 0.5, lonlat_centre=jnp.zeros(3))


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


def test_polar_gap_basis_rejects_zero_modes() -> None:
    # the check runs after the concentration matrix is built; degree 0 is the
    # smallest one (the other argument checks are fast tests)
    with pytest.raises(ValidationError, match="'n_modes' must be at least 1"):
        slepian_polar_gap_basis(0, 0.3, n_modes=0)


def test_integer_polygon_vertices_are_accepted() -> None:
    square = jnp.array([[-10, -10], [10, -10], [10, 10], [-10, 10]])
    points = jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    mask = inspect.unwrap(spherical_polygon_mask)(points, square, unit="degrees")
    assert mask.tolist() == [True, False]


# degree 3 throughout: each Slepian build is an eigenproblem per order, seconds
# eagerly, so the properties below reuse a few cached bases and jitted kernels
_L_MAX = 3


def _random_units(count: int, seed: int) -> np.ndarray:
    gaussian = np.random.default_rng(seed).normal(size=(count, 3))
    return gaussian / np.linalg.norm(gaussian, axis=1, keepdims=True)


def _random_rotation(seed: int) -> np.ndarray:
    """A random proper rotation of ℝ³, from the QR factors of a Gaussian matrix."""
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def _rotation_from_pole(lon: float, lat: float) -> np.ndarray:
    """A proper rotation R with R e_z the unit vector at (lon, lat), in radians."""
    centre = np.array(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]
    )
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north = np.cross(centre, east)
    return np.stack([east, north, centre], axis=1)


# (−1)^ℓ per harmonic: the parity of Y_ℓm under z → −z within each order m
_REFLECTION = jnp.asarray((-1.0) ** np.asarray(harmonic_degrees(_L_MAX)))


_CONCENTRATION = jax.jit(lambda radius: slepian_concentration_matrix(_L_MAX, radius))


@settings(max_examples=20, deadline=None)
@given(radius=st.floats(0.05, math.pi - 0.05))
def test_a_cap_and_its_complement_partition_the_identity(radius: float) -> None:
    # the cap θ < Θ and the reflected cap θ < π − Θ cover the sphere, so
    # D(Θ) + S D(π − Θ) S = I with S = diag((−1)^ℓ), the cap rule being exact
    cap = _CONCENTRATION(radius)
    complement = _REFLECTION[:, None] * _CONCENTRATION(math.pi - radius) * _REFLECTION
    assert_matches_oracle(
        cap + complement, jnp.eye((_L_MAX + 1) ** 2), rtol=0.0, atol=1e-13
    )


def test_concentration_gradient_matches_finite_differences() -> None:
    # differentiated in the cap radius Θ, inside (0, π)
    assert_grad_matches_fd(
        lambda radius: slepian_concentration_matrix(1, radius), jnp.asarray(0.7)
    )


@functools.cache
def _cap_basis() -> object:
    """Every mode of the degree-3 cap of radius 0.6 at the north pole."""
    # jitted: one compile instead of an eager compile per operation
    build = jax.jit(lambda radius: slepian_cap_basis(_L_MAX, radius, n_modes=16))
    return build(0.6)


@jax.jit
def _weighted_kernel(basis: object, x: jax.Array, y: jax.Array) -> jax.Array:
    """K(x, y) = ∑_k λ_k g_k(x) g_k(y) over every mode."""
    return (basis.evaluate(x) * basis.eigenvalues) @ basis.evaluate(y).T


@settings(max_examples=20, deadline=None)
@given(
    seed=st.integers(0, 2**31 - 1),
    lon=st.floats(-math.pi, math.pi),
    lat=st.floats(-1.5, 1.5),
)
def test_a_moved_cap_kernel_is_the_rotated_kernel(
    seed: int, lon: float, lat: float
) -> None:
    # K(x, y) = ∑_k λ_k g_k(x) g_k(y) = ∫_cap of the band-limited kernels, which
    # depends on the cap alone (not the modes' frame), so the cap moved to c
    # has K_c(Rx, Ry) = K(x, y) for any rotation R taking the pole to c
    basis = _cap_basis()
    moved = basis.rotate_to(jnp.array([lon, lat]))
    rotation = jnp.asarray(_rotation_from_pole(lon, lat))
    x = jnp.asarray(_random_units(5, seed))
    y = jnp.asarray(_random_units(5, seed + 1))
    assert_matches_oracle(
        _weighted_kernel(moved, x @ rotation.T, y @ rotation.T),
        _weighted_kernel(basis, x, y),
        rtol=0.0,
        atol=1e-13,
    )


@functools.cache
def _region_nodes() -> tuple[jax.Array, jax.Array]:
    """40 random nodes with random positive weights: a generic discrete region."""
    rng = np.random.default_rng(31)
    return jnp.asarray(_random_units(40, seed=30)), jnp.asarray(
        rng.uniform(0.01, 0.1, 40)
    )


# compiled once per module, so each hypothesis example reuses the kernel
_region_concentrations = jax.jit(
    lambda nodes, weights: (
        slepian_region_basis(
            _L_MAX, nodes, weights, n_modes=(_L_MAX + 1) ** 2
        ).eigenvalues
    )
)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_region_concentrations_are_rotation_invariant(seed: int) -> None:
    # D = Yᵀ W Y and Y(Rx) = Y(x) Qᵀ with Q orthogonal on each degree, so
    # rotating the nodes keeps the spectrum of D
    nodes, weights = _region_nodes()
    rotation = jnp.asarray(_random_rotation(seed))
    assert_matches_oracle(
        _region_concentrations(nodes @ rotation.T, weights),
        _region_concentrations(nodes, weights),
        rtol=0.0,
        atol=1e-13,
    )


def test_mode_gradients_match_finite_differences() -> None:
    # evaluate differentiated in the points, for a moved cap (a region's modes
    # are the same harmonics times fixed coefficients, without the rotation)
    points = jnp.asarray(_random_units(3, seed=32))
    moved = _cap_basis().rotate_to(jnp.array([0.4, 0.3]))
    assert_grad_matches_fd(moved.evaluate, points, rtol=1e-8)


def test_modes_trace() -> None:
    points = jnp.asarray(_random_units(12, seed=33).reshape(3, 4, 3))
    moved = _cap_basis().rotate_to(jnp.array([0.4, 0.3]))
    nodes, weights = _region_nodes()
    region = jax.jit(lambda n, w: slepian_region_basis(_L_MAX, n, w, n_modes=4))(
        nodes, weights
    )
    assert_traces(lambda x: (moved.evaluate(x), region.evaluate(x)), points)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_polygon_mask_on_rotated_octants(seed: int) -> None:
    # the octant x, y, z > 0 is the spherical triangle e_x → e_y → e_z
    # (counter-clockwise from outside, its edges great circles); rotated by R,
    # a point p is inside exactly when every coordinate of Rᵀp is positive
    rotation = _random_rotation(seed)
    corners = np.eye(3) @ rotation.T
    polygon = np.stack(
        [np.arctan2(corners[:, 1], corners[:, 0]), np.arcsin(corners[:, 2])], axis=-1
    )
    points = _random_units(200, seed + 1)
    local = points @ rotation
    expected = (local > 0).all(axis=1)
    clear = np.abs(local).min(axis=1) > 1e-9
    mask = spherical_polygon_mask(
        jnp.asarray(points), jnp.asarray(polygon), unit="radians"
    )
    assert_matches_oracle(np.asarray(mask)[clear], expected[clear], rtol=0.0)
