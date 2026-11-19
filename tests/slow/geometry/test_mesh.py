"""Tests for the icosphere and the vertex areas of `functax.geometry`. The slow tier:
the flat-triangle area of every level up to 5 grows towards the sphere's 4π, and
the vertex areas scale with a similarity on random meshes.
"""

from __future__ import annotations

import math
from itertools import pairwise

import hypothesis.strategies as st
import numpy as np
from _harness import assert_matches_oracle
from hypothesis import given, settings

from functax.geometry import icosphere, mesh_vertex_areas


class TestIcosphere:
    def test_area_grows_toward_the_sphere(self) -> None:
        areas = []
        for level in range(6):
            vertices, triangles = map(np.asarray, icosphere(level))
            a, b, c = (vertices[triangles[:, k]] for k in range(3))
            areas.append(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
        assert all(x < y for x, y in pairwise(areas))
        assert 0 < 4.0 * math.pi - areas[-1] < 1e-2


class TestVertexAreas:
    @settings(max_examples=20, deadline=None)
    @given(seed=st.integers(0, 2**31 - 1), scale=st.floats(0.1, 10.0))
    def test_scale_with_a_similarity_and_sum_to_the_mesh_area(
        self, seed: int, scale: float
    ) -> None:
        # a(s Q x + t) = s² a(x) for an orthogonal Q, and ∑ a_v = ∑ |t|
        rng = np.random.default_rng(seed)
        vertices, triangles = map(np.asarray, icosphere(1))
        vertices = vertices + 0.1 * rng.normal(size=vertices.shape)
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        moved = scale * vertices @ (q * np.sign(np.diag(r))).T + rng.normal(size=3)
        areas = np.asarray(mesh_vertex_areas(vertices, triangles))
        a, b, c = (vertices[triangles[:, k]] for k in range(3))
        total = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
        assert_matches_oracle(areas.sum(), total, rtol=1e-13)
        assert_matches_oracle(
            mesh_vertex_areas(moved, triangles), scale**2 * areas, rtol=1e-12
        )
