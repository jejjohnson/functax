"""Tests for the icosphere of `functax.geometry`. The slow tier: the flat-triangle
area of every level up to 5 grows towards the sphere's 4π.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np

from functax.geometry import icosphere


class TestIcosphere:
    def test_area_grows_toward_the_sphere(self) -> None:
        areas = []
        for level in range(6):
            vertices, triangles = map(np.asarray, icosphere(level))
            a, b, c = (vertices[triangles[:, k]] for k in range(3))
            areas.append(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
        assert all(x < y for x, y in pairwise(areas))
        assert 0 < 4.0 * math.pi - areas[-1] < 1e-2
