"""Tests for the icosphere and the lumped vertex areas of `functax.geometry`. The
fast tier: counts, orientation, areas and validation, each well under a second.
"""

from __future__ import annotations

import inspect
import math
from collections import Counter
from itertools import pairwise

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax import GeometryError
from functax.geometry import icosphere, mesh_vertex_areas


def _edges(triangles: np.ndarray) -> Counter[tuple[int, int]]:
    return Counter(
        (min(int(a), int(b)), max(int(a), int(b)))
        for triangle in triangles
        for a, b in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        )
    )


class TestIcosphere:
    @pytest.mark.parametrize("level", range(5))
    def test_counts_and_a_closed_manifold(self, level: int) -> None:
        vertices, triangles = map(np.asarray, icosphere(level))
        n_vertices, n_triangles = 10 * 4**level + 2, 20 * 4**level
        assert vertices.shape == (n_vertices, 3)
        assert triangles.shape == (n_triangles, 3)
        edges = _edges(triangles)
        # every edge borders exactly two triangles, and V − E + T = 2
        assert set(edges.values()) == {2}
        assert len(edges) == 30 * 4**level
        assert n_vertices - len(edges) + n_triangles == 2

    def test_radius_and_outward_orientation(self) -> None:
        vertices, triangles = map(np.asarray, icosphere(3, radius=2.5))
        assert_matches_oracle(
            np.linalg.norm(vertices, axis=1), np.full(len(vertices), 2.5), rtol=1e-12
        )
        a, b, c = (vertices[triangles[:, k]] for k in range(3))
        normals = np.cross(b - a, c - a)
        assert np.all(np.sum(normals * (a + b + c), axis=1) > 0)

    def test_area_grows_toward_the_sphere(self) -> None:
        areas = []
        for level in range(6):
            vertices, triangles = map(np.asarray, icosphere(level))
            a, b, c = (vertices[triangles[:, k]] for k in range(3))
            areas.append(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
        assert all(x < y for x, y in pairwise(areas))
        assert 0 < 4.0 * math.pi - areas[-1] < 1e-2

    def test_rejects_a_negative_level(self) -> None:
        with pytest.raises(GeometryError, match="'level' must be an integer"):
            icosphere(-1)

    def test_rejects_a_non_integer_level(self) -> None:
        with pytest.raises(GeometryError, match="'level' must be an integer"):
            inspect.unwrap(icosphere)(1.5)


class TestVertexAreas:
    @pytest.mark.parametrize("radius", [1.0, 2.0])
    def test_sum_to_the_surface_area(self, radius: float) -> None:
        vertices, triangles = icosphere(3, radius=radius)
        areas = mesh_vertex_areas(vertices, triangles)
        assert float(areas.sum()) == pytest.approx(4 * math.pi * radius**2, rel=0.01)
        assert bool(jnp.all(areas > 0))

    def test_do_not_depend_on_orientation(self) -> None:
        vertices, triangles = icosphere(1)
        flipped = jnp.asarray(np.asarray(triangles)[:, ::-1])
        assert_matches_oracle(
            mesh_vertex_areas(vertices, flipped),
            mesh_vertex_areas(vertices, triangles),
            rtol=1e-14,
        )

    def test_a_single_triangle_gives_a_third_to_each_corner(self) -> None:
        vertices = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        triangles = jnp.array([[0, 1, 2]])
        assert_matches_oracle(
            mesh_vertex_areas(vertices, triangles), [1.0, 1.0, 1.0], rtol=1e-14
        )

    def test_rejects_out_of_range_indices_and_zero_areas(self) -> None:
        vertices = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
        with pytest.raises(GeometryError, match="'triangles' indices must lie"):
            mesh_vertex_areas(vertices, jnp.array([[1, 2, 4]]))
        with pytest.raises(GeometryError, match="zero area"):
            mesh_vertex_areas(vertices, jnp.array([[0, 0, 1]]))

    def test_rejects_bad_shapes_behind_the_type_hook(self) -> None:
        raw = inspect.unwrap(mesh_vertex_areas)
        with pytest.raises(GeometryError, match="'vertices' must have shape"):
            raw(np.zeros((4, 2)), np.array([[0, 1, 2]]))
        with pytest.raises(GeometryError, match="integer indices"):
            raw(np.eye(3), np.array([[0.0, 1.0, 2.0]]))
