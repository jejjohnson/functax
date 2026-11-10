"""Tests for the sphere grids of `functax.geometry`. The fast tier: weights, layouts,
the HEALPix orders against healpy's pixel table, and validation, each well under
a second.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax import GeometryError
from functax.geometry import (
    fibonacci_sphere,
    gauss_legendre_grid,
    healpix_grid,
    latlon_grid,
    lonlat_to_unit,
)


GRIDS: dict[str, Callable[..., tuple[jax.Array, jax.Array]]] = {
    "fibonacci": lambda **kw: fibonacci_sphere(500, **kw),
    "gauss_legendre": lambda **kw: gauss_legendre_grid(12, **kw),
    "latlon": lambda **kw: latlon_grid(18, 36, **kw),
    "healpix": lambda **kw: healpix_grid(4, **kw),
}


@pytest.mark.parametrize("name", GRIDS)
def test_weights_sum_to_four_pi_on_unit_vectors(name: str) -> None:
    # gate of #20: each grid's weights sum to 4π to 1e-12
    points, weights = GRIDS[name]()
    assert abs(float(weights.sum()) - 4.0 * math.pi) < 1e-12
    assert_matches_oracle(
        np.linalg.norm(points, axis=1), np.ones(len(points)), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("name", GRIDS)
def test_lonlat_and_xyz_agree(name: str) -> None:
    xyz, _ = GRIDS[name]()
    radians, _ = GRIDS[name](output="lonlat", unit="radians")
    degrees, _ = GRIDS[name](output="lonlat", unit="degrees")
    assert_matches_oracle(
        lonlat_to_unit(radians, unit="radians"), xyz, rtol=0, atol=1e-12
    )
    assert_matches_oracle(np.radians(degrees), radians, rtol=0, atol=1e-12)


def test_latlon_first_cell_centre() -> None:
    lonlat, _ = latlon_grid(18, 36, output="lonlat", unit="degrees")
    assert_matches_oracle(lonlat[0], [-175.0, -85.0], rtol=1e-12)


def test_outputs_are_jax_arrays() -> None:
    points, weights = fibonacci_sphere(10)
    assert isinstance(points, jax.Array)
    assert isinstance(weights, jax.Array)


class TestValidation:
    @pytest.mark.parametrize("name", GRIDS)
    def test_lonlat_output_needs_a_unit(self, name: str) -> None:
        with pytest.raises(
            GeometryError, match="'unit' is required with output='lonlat'"
        ):
            GRIDS[name](output="lonlat")

    def test_rejects_an_unknown_output_or_unit(self) -> None:
        with pytest.raises(GeometryError, match="'output' must be one of"):
            gauss_legendre_grid(4, output="polar")
        with pytest.raises(GeometryError, match="'unit' must be one of"):
            latlon_grid(4, 8, output="lonlat", unit="turns")

    @pytest.mark.parametrize(
        ("call", "match"),
        [
            (lambda: fibonacci_sphere(0), "'n' must be an integer of at least 1"),
            (lambda: gauss_legendre_grid(0), "'n_lat' must be"),
            (lambda: gauss_legendre_grid(4, 0), "'n_lon' must be"),
            (lambda: latlon_grid(0, 4), "'n_lat' must be"),
            (lambda: healpix_grid(0), "'nside' must be an integer"),
            (lambda: healpix_grid(3), "power of two"),
        ],
    )
    def test_rejects_bad_counts(self, call: Callable[[], object], match: str) -> None:
        with pytest.raises(GeometryError, match=match):
            call()

    def test_healpix_counts_are_integers(self) -> None:
        import inspect

        raw = inspect.unwrap(healpix_grid)
        with pytest.raises(GeometryError, match="integer"):
            raw(2.0)
        with pytest.raises(GeometryError, match="integer"):
            raw(True)
        assert healpix_grid(3, nest=False)[0].shape == (108, 3)
        assert raw(np.int64(4))[0].shape == (192, 3)


# --- HEALPix against healpy -------------------------------------------------

# Reference values from healpy 1.20.1 (``healpy.pix2ang``, ``healpy.nest2ring``).
# The ring order at nside = 2: z = sin φ per ring with its pixel count, and the
# longitudes in units of π/8.
Z2_RINGS = [(11 / 12, 4), (2 / 3, 8), (1 / 3, 8), (0.0, 8), (-1 / 3, 8),
            (-2 / 3, 8), (-11 / 12, 4)]  # fmt: skip
PHI2_RING_EIGHTHS = [2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10,
                     12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14,
                     1, 3, 5, 7, 9, 11, 13, 15, 2, 6, 10, 14]  # fmt: skip
# healpy.nest2ring(4, arange(192)): the ring index of each nested pixel
NEST2RING_4 = [
    74, 58, 57, 42, 43, 27, 26, 14, 41, 25, 24, 12, 13, 5, 4, 0,
    78, 62, 61, 46, 47, 31, 30, 17, 45, 29, 28, 15, 16, 7, 6, 1,
    82, 66, 65, 50, 51, 35, 34, 20, 49, 33, 32, 18, 19, 9, 8, 2,
    86, 70, 69, 54, 55, 39, 38, 23, 53, 37, 36, 21, 22, 11, 10, 3,
    136, 120, 135, 104, 105, 89, 88, 73, 119, 103, 102, 87, 72, 56, 71, 40,
    140, 124, 123, 108, 109, 93, 92, 77, 107, 91, 90, 75, 76, 60, 59, 44,
    144, 128, 127, 112, 113, 97, 96, 81, 111, 95, 94, 79, 80, 64, 63, 48,
    148, 132, 131, 116, 117, 101, 100, 85, 115, 99, 98, 83, 84, 68, 67, 52,
    188, 181, 180, 169, 170, 155, 154, 139, 168, 153, 152, 137, 138, 122, 121, 106,
    189, 183, 182, 172, 173, 159, 158, 143, 171, 157, 156, 141, 142, 126, 125, 110,
    190, 185, 184, 175, 176, 163, 162, 147, 174, 161, 160, 145, 146, 130, 129, 114,
    191, 187, 186, 178, 179, 167, 166, 151, 177, 165, 164, 149, 150, 134, 133, 118,
]  # fmt: skip


class TestHealpix:
    def test_ring_centres_match_healpy(self) -> None:
        lonlat, weights = healpix_grid(2, nest=False, output="lonlat", unit="radians")
        z = np.concatenate([[value] * count for value, count in Z2_RINGS])
        lon = np.asarray(PHI2_RING_EIGHTHS) * math.pi / 8.0
        wrapped = np.mod(lon + math.pi, 2.0 * math.pi) - math.pi
        assert_matches_oracle(np.sin(lonlat[:, 1]), z, rtol=0, atol=1e-12)
        assert_matches_oracle(lonlat[:, 0], wrapped, rtol=0, atol=1e-12)
        assert_matches_oracle(weights, np.full(48, 4.0 * math.pi / 48), rtol=1e-14)

    def test_nested_is_the_ring_order_permuted_like_healpy(self) -> None:
        ring, _ = healpix_grid(4, nest=False)
        nested, _ = healpix_grid(4, nest=True)
        assert_matches_oracle(
            nested, np.asarray(ring)[np.asarray(NEST2RING_4)], rtol=0, atol=1e-12
        )

    @pytest.mark.parametrize("nside", [1, 2, 4, 8])
    def test_nested_children_surround_their_parent(self, nside: int) -> None:
        # nested pixel p at nside holds pixels 4p to 4p + 3 at 2 nside, so the
        # mean of the four children lies within a quarter of a parent pixel
        # width √(4π/N) of the parent centre (observed under 0.06 for nside ≤ 8)
        parent, _ = healpix_grid(nside)
        child, _ = healpix_grid(2 * nside)
        mean = np.asarray(child).reshape(-1, 4, 3).mean(1)
        mean /= np.linalg.norm(mean, axis=1, keepdims=True)
        cos_separation = np.sum(mean * np.asarray(parent), axis=1)
        width = math.sqrt(4 * math.pi / parent.shape[0])
        assert np.all(cos_separation > np.cos(0.25 * width))

    @pytest.mark.parametrize("nside", [1, 2, 3, 4, 5])
    def test_count_and_distinct_centres(self, nside: int) -> None:
        points, _ = healpix_grid(nside, nest=False)
        assert points.shape == (12 * nside**2, 3)
        distinct = np.unique(np.round(np.asarray(points), 9), axis=0)
        assert distinct.shape[0] == len(points)


def test_float32_without_x64_stays_float32() -> None:
    with jax.enable_x64(False):
        points, weights = gauss_legendre_grid(4)
    assert points.dtype == jnp.float32
    assert weights.dtype == jnp.float32
