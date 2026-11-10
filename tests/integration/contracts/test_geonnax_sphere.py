"""functax's sphere geometry (#20) against geonnax, pinned at geonnax be5537d as
data: the frames with WGS84 and the sphere passed explicitly, the sphere grids,
the icosphere and its vertex areas, the cotangent eigenbasis and Gauss–Legendre.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax.basis.eigen import cotangent_laplacian_eigpairs
from functax.geometry import (
    cartesian_to_tangent,
    ecef_to_lonlat,
    enu_basis,
    fibonacci_sphere,
    gauss_legendre_grid,
    healpix_grid,
    icosphere,
    latlon_grid,
    local_tangent_plane,
    lonlat_to_ecef,
    mesh_vertex_areas,
    rotation_to_pole,
    tangent_to_cartesian,
)
from functax.quad import gauss_legendre


GEONNAX = np.load(Path(__file__).parent / "data" / "geonnax_sphere_geometry.npz")
# geonnax's built-in ellipsoids, passed explicitly: functax holds no Earth constant
WGS84 = {"semi_major": 6378137.0, "flattening": 1.0 / 298.257223563}
SPHERE = {"semi_major": 6371008.8, "flattening": 0.0}
LONLAT = jnp.asarray(GEONNAX["lonlat"])


def _array(key: str) -> jnp.ndarray:
    return jnp.asarray(GEONNAX[key])


class TestFrames:
    @pytest.mark.parametrize(
        ("ellipsoid", "name"), [(WGS84, "wgs84"), (SPHERE, "sphere")]
    )
    def test_ecef_round_trip_equals_geonnax(
        self, ellipsoid: dict[str, float], name: str
    ) -> None:
        xyz = lonlat_to_ecef(LONLAT, _array("height"), unit="degrees", **ellipsoid)
        assert_matches_oracle(xyz, GEONNAX[f"ecef_{name}"], rtol=1e-12, atol=1e-9)
        lonlat, height = ecef_to_lonlat(
            _array(f"ecef_{name}"), unit="degrees", **ellipsoid
        )
        assert_matches_oracle(lonlat, GEONNAX[f"lonlat_{name}"], rtol=1e-12, atol=1e-12)
        assert_matches_oracle(height, GEONNAX[f"height_{name}"], rtol=1e-12, atol=1e-8)

    def test_tangent_frames_equal_geonnax(self) -> None:
        assert_matches_oracle(
            enu_basis(LONLAT, unit="degrees"), GEONNAX["enu"], rtol=1e-12, atol=1e-15
        )
        assert_matches_oracle(
            tangent_to_cartesian(LONLAT, _array("uv"), unit="degrees"),
            GEONNAX["tangent_to_cartesian"],
            rtol=1e-12,
            atol=1e-15,
        )
        assert_matches_oracle(
            cartesian_to_tangent(LONLAT, _array("vec"), unit="degrees"),
            GEONNAX["cartesian_to_tangent"],
            rtol=1e-12,
            atol=1e-15,
        )
        assert_matches_oracle(
            rotation_to_pole(_array("origin"), unit="degrees"),
            GEONNAX["rotation_to_pole"],
            rtol=1e-12,
            atol=1e-15,
        )

    def test_tangent_planes_equal_geonnax_at_its_default_radius(self) -> None:
        # geonnax's default radius was the mean Earth radius in km
        plane = local_tangent_plane(
            LONLAT, _array("origin"), radius=6371.0088, unit="degrees"
        )
        assert_matches_oracle(
            plane, GEONNAX["azimuthal_equidistant"], rtol=1e-12, atol=1e-9
        )
        gnomonic = local_tangent_plane(
            _array("near"),
            jnp.zeros(2),
            radius=6371.0088,
            projection="gnomonic",
            unit="degrees",
        )
        assert_matches_oracle(gnomonic, GEONNAX["gnomonic"], rtol=1e-12, atol=1e-9)


GRIDS = {
    "fibonacci": lambda **k: fibonacci_sphere(50, **k),
    "gauss_legendre": lambda **k: gauss_legendre_grid(6, 10, **k),
    "latlon": lambda **k: latlon_grid(6, 12, **k),
    "healpix_nest": lambda **k: healpix_grid(4, nest=True, **k),
    "healpix_ring": lambda **k: healpix_grid(3, nest=False, **k),
}


@pytest.mark.parametrize("name", GRIDS)
def test_sphere_grids_equal_geonnax(name: str) -> None:
    points, weights = GRIDS[name]()
    lonlat, _ = GRIDS[name](output="lonlat", unit="degrees")
    assert_matches_oracle(points, GEONNAX[f"grid_{name}_xyz"], rtol=1e-12, atol=1e-15)
    assert_matches_oracle(weights, GEONNAX[f"grid_{name}_weights"], rtol=1e-12)
    assert_matches_oracle(
        lonlat, GEONNAX[f"grid_{name}_lonlat_degrees"], rtol=1e-12, atol=1e-12
    )


class TestMesh:
    def test_icosphere_and_areas_equal_geonnax(self) -> None:
        vertices, triangles = icosphere(2, radius=1.5)
        assert_matches_oracle(
            vertices, GEONNAX["icosphere_vertices"], rtol=1e-12, atol=1e-15
        )
        assert_matches_oracle(triangles, GEONNAX["icosphere_triangles"], rtol=0)
        areas = mesh_vertex_areas(vertices, triangles)
        assert_matches_oracle(areas, GEONNAX["mesh_vertex_areas"], rtol=1e-12)

    def test_cotangent_eigenbasis_equals_geonnax(self) -> None:
        vertices, triangles = icosphere(2)
        eigvals, eigvecs, areas = cotangent_laplacian_eigpairs(vertices, triangles, 9)
        assert_matches_oracle(eigvals, GEONNAX["eigvals"], rtol=1e-10, atol=1e-12)
        assert_matches_oracle(areas, GEONNAX["eig_areas"], rtol=1e-12)
        # eigenvectors inside a degenerate cluster are not unique; the
        # M-orthogonal projection onto whole clusters (degrees 0 to 2) is
        fields = np.asarray(GEONNAX["eig_fields"])
        basis = np.asarray(eigvecs)
        projected = basis @ (basis.T @ (np.asarray(areas)[:, None] * fields))
        assert_matches_oracle(
            projected, GEONNAX["eig_projected"], rtol=1e-10, atol=1e-12
        )


def test_gauss_legendre_equals_geonnax() -> None:
    nodes, weights = gauss_legendre(9)
    assert_matches_oracle(
        nodes, GEONNAX["gauss_legendre_nodes"], rtol=1e-14, atol=1e-16
    )
    assert_matches_oracle(weights, GEONNAX["gauss_legendre_weights"], rtol=1e-14)
