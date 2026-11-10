"""Tests for the cotangent Laplacian eigenbasis of `functax.basis.eigen`. The fast
tier: the icosphere spectrum, mass-orthonormality, an independent assembly, the
Neumann square and validation, each well under a second.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax import GeometryError, ValidationError
from functax.basis.eigen import cotangent_laplacian_eigpairs
from functax.geometry import icosphere, mesh_vertex_areas


def _square_mesh(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The unit square [0, 1]² at z = 0: an n × n grid of 2n² triangles."""
    side = np.linspace(0.0, 1.0, n + 1)
    x, y = np.meshgrid(side, side, indexing="ij")
    vertices = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    index = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b = index[:-1, :-1].ravel(), index[1:, :-1].ravel()
    c, d = index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    triangles = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return vertices, triangles


def _reference_laplacian(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """An independent loop assembly of the cotangent L, from geonnax's prototype."""
    points, corners = np.asarray(vertices, dtype=float), np.asarray(triangles)
    laplacian = np.zeros((len(points), len(points)))
    for triangle in corners:
        for k in range(3):
            p, q, r = triangle[k], triangle[(k + 1) % 3], triangle[(k + 2) % 3]
            e1, e2 = points[q] - points[p], points[r] - points[p]
            cot = e1 @ e2 / np.linalg.norm(np.cross(e1, e2))
            laplacian[q, r] -= 0.5 * cot
            laplacian[r, q] -= 0.5 * cot
    np.fill_diagonal(laplacian, -laplacian.sum(1))
    return laplacian


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    triangles = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return vertices, triangles


@pytest.fixture(scope="module")
def sphere3() -> tuple[np.ndarray, ...]:
    vertices, triangles = icosphere(3)
    eigvals, eigvecs, areas = cotangent_laplacian_eigpairs(vertices, triangles, 16)
    return tuple(map(np.asarray, (vertices, triangles, eigvals, eigvecs, areas)))


class TestIcosphereSpectrum:
    def test_approximates_l_l_plus_1_ascending(
        self, sphere3: tuple[np.ndarray, ...]
    ) -> None:
        eigvals = sphere3[2]
        expected = np.repeat([0.0, 2.0, 6.0, 12.0], [1, 3, 5, 7])
        assert abs(eigvals[0]) < 1e-10
        assert_matches_oracle(eigvals[1:], expected[1:], rtol=0.02)
        assert np.all(np.diff(eigvals) >= -1e-12)

    def test_multiplicities_are_2l_plus_1(
        self, sphere3: tuple[np.ndarray, ...]
    ) -> None:
        eigvals = sphere3[2]
        gaps = np.flatnonzero(np.diff(eigvals) > 0.1 * np.maximum(eigvals[1:], 1.0))
        sizes = np.diff(np.concatenate([[0], gaps + 1, [eigvals.size]]))
        assert sizes.tolist() == [1, 3, 5, 7]

    def test_eigvecs_are_mass_orthonormal(
        self, sphere3: tuple[np.ndarray, ...]
    ) -> None:
        eigvecs, areas = sphere3[3], sphere3[4]
        gram = eigvecs.T @ (areas[:, None] * eigvecs)
        assert_matches_oracle(gram, np.eye(16), rtol=0, atol=1e-10)

    def test_the_constant_is_in_the_null_space(
        self, sphere3: tuple[np.ndarray, ...]
    ) -> None:
        eigvecs, areas = sphere3[3], sphere3[4]
        expected = np.full(len(areas), 1.0 / math.sqrt(areas.sum()))
        assert_matches_oracle(np.abs(eigvecs[:, 0]), expected, rtol=1e-10)

    def test_areas_are_the_vertex_areas(self, sphere3: tuple[np.ndarray, ...]) -> None:
        vertices, triangles, *_, areas = sphere3
        assert_matches_oracle(mesh_vertex_areas(vertices, triangles), areas, rtol=1e-14)


def test_solves_the_generalised_problem_of_an_independent_assembly() -> None:
    vertices, triangles = icosphere(2)
    eigvals, eigvecs, areas = map(
        np.asarray, cotangent_laplacian_eigpairs(vertices, triangles, 10)
    )
    laplacian = _reference_laplacian(np.asarray(vertices), np.asarray(triangles))
    # L φ = λ M φ
    residual = laplacian @ eigvecs - areas[:, None] * eigvecs * eigvals[None, :]
    assert np.abs(residual).max() < 1e-10
    assert np.abs(laplacian @ np.ones(len(laplacian))).max() < 1e-10


def test_the_flat_square_has_the_neumann_spectrum() -> None:
    vertices, triangles = _square_mesh(20)
    eigvals, _, areas = cotangent_laplacian_eigpairs(vertices, triangles, 6)
    expected = np.pi**2 * np.array([0, 1, 1, 2, 4, 4], dtype=float)
    assert float(areas.sum()) == pytest.approx(1.0, rel=1e-12)
    assert abs(float(eigvals[0])) < 1e-10
    assert_matches_oracle(np.asarray(eigvals)[1:], expected[1:], rtol=0.05)


def test_a_tetrahedron_runs() -> None:
    vertices, triangles = _tetrahedron()
    eigvals, eigvecs, _ = cotangent_laplacian_eigpairs(vertices, triangles, 4)
    assert eigvals.shape == (4,)
    assert eigvecs.shape == (4, 4)
    assert abs(float(eigvals[0])) < 1e-10


@pytest.mark.parametrize(
    ("vertices", "triangles", "num_basis", "error", "match"),
    [
        (
            np.zeros((4, 2)),
            _tetrahedron()[1],
            2,
            GeometryError,
            "'vertices' must have shape",
        ),
        (
            _tetrahedron()[0],
            _tetrahedron()[1][:, :2],
            2,
            GeometryError,
            "'triangles' must have shape",
        ),
        (
            _tetrahedron()[0],
            np.zeros((0, 3), dtype=int),
            2,
            GeometryError,
            "'triangles' must have shape",
        ),
        (
            _tetrahedron()[0],
            _tetrahedron()[1].astype(float),
            2,
            GeometryError,
            "integer",
        ),
        (
            _tetrahedron()[0],
            _tetrahedron()[1] + 1,
            2,
            GeometryError,
            "indices must lie",
        ),
        (
            _tetrahedron()[0],
            _tetrahedron()[1] - 1,
            2,
            GeometryError,
            "indices must lie",
        ),
        (
            np.where(np.eye(4, 3) > 0, np.nan, 0.0),
            _tetrahedron()[1],
            2,
            GeometryError,
            "finite",
        ),
        (
            _tetrahedron()[0],
            np.array([[0, 1, 1], [0, 1, 2]]),
            2,
            GeometryError,
            "zero area",
        ),
        (
            np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0.0]]),
            np.array([[0, 1, 2]]),
            1,
            GeometryError,
            "zero area",
        ),
        (_tetrahedron()[0], _tetrahedron()[1], 0, ValidationError, "'num_basis'"),
        (_tetrahedron()[0], _tetrahedron()[1], 5, ValidationError, "'num_basis'"),
        (
            np.vstack([_tetrahedron()[0], [[5.0, 5.0, 5.0]]]),
            _tetrahedron()[1],
            2,
            GeometryError,
            "no triangle",
        ),
    ],
)
def test_validation(
    vertices: np.ndarray,
    triangles: np.ndarray,
    num_basis: int,
    error: type[Exception],
    match: str,
) -> None:
    # unwrapped: the type hook would reject the malformed shapes before the checks
    with pytest.raises(error, match=match):
        inspect.unwrap(cotangent_laplacian_eigpairs)(vertices, triangles, num_basis)


def test_rejects_a_non_integer_num_basis() -> None:
    vertices, triangles = _tetrahedron()
    with pytest.raises(ValidationError, match="'num_basis' must be an integer"):
        inspect.unwrap(cotangent_laplacian_eigpairs)(vertices, triangles, 2.5)
