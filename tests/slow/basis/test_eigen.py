"""Tests for `functax.basis.eigen`. The slow tier, a second or more each: geonnax's
EOF tests, copied from be5537d with functax imports (mostly compiling the SVD), and
the cotangent Laplacian on a tetrahedron, on the level-3 icosphere, against an
independent assembly and on the flat square.
"""

from __future__ import annotations

import math

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from hypothesis import given, settings

from functax.basis.eigen import (
    cotangent_laplacian_eigpairs,
    eof_basis,
)
from functax.geometry import icosphere, mesh_vertex_areas


@pytest.fixture(scope="module")
def sphere3() -> tuple[np.ndarray, ...]:
    vertices, triangles = icosphere(3)
    eigvals, eigvecs, areas = cotangent_laplacian_eigpairs(vertices, triangles, 16)
    return tuple(map(np.asarray, (vertices, triangles, eigvals, eigvecs, areas)))


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


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    triangles = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
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


class TestEOFBasis:
    def test_shapes_and_orthonormal_columns(self):
        X = jax.random.normal(jax.random.PRNGKey(0), (20, 8))
        phi, s = eof_basis(X, n_modes=5)
        assert phi.shape == (8, 5)
        assert s.shape == (5,)
        np.testing.assert_allclose(phi.T @ phi, jnp.eye(5), atol=1e-5)

    def test_singular_values_descending(self):
        X = jax.random.normal(jax.random.PRNGKey(1), (30, 10))
        _, s = eof_basis(X, n_modes=10)
        assert bool(jnp.all(jnp.diff(s) <= 1e-6))

    def test_reconstructs_full_rank_anomalies(self):
        # Keeping all min(T, N) modes reconstructs the centred data exactly.
        X = jax.random.normal(jax.random.PRNGKey(2), (12, 6))
        phi, _ = eof_basis(X, n_modes=6)
        xc = X - X.mean(axis=0, keepdims=True)
        np.testing.assert_allclose(xc @ phi @ phi.T, xc, atol=1e-4)

    def test_recovers_known_low_rank_pattern(self):
        # Data built from a single spatial pattern -> first EOF spans it.
        pattern = jnp.array([1.0, -1.0, 2.0, 0.5])
        pattern = pattern / jnp.linalg.norm(pattern)
        amps = jax.random.normal(jax.random.PRNGKey(3), (40,))
        X = amps[:, None] * pattern[None, :]
        phi, _ = eof_basis(X, n_modes=1, center=False)
        # the leading EOF is the pattern up to sign
        cos = jnp.abs(phi[:, 0] @ pattern)
        np.testing.assert_allclose(cos, 1.0, atol=1e-5)


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


def test_a_tetrahedron_runs() -> None:
    vertices, triangles = _tetrahedron()
    eigvals, eigvecs, _ = cotangent_laplacian_eigpairs(vertices, triangles, 4)
    assert eigvals.shape == (4,)
    assert eigvecs.shape == (4, 4)
    assert abs(float(eigvals[0])) < 1e-10


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


def _eof_subspace(data: jax.Array) -> tuple[jax.Array, jax.Array]:
    """ΦΦᵀ and σ of the three leading EOFs: free of the sign of each pattern."""
    patterns, singular_values = eof_basis(data, 3)
    return patterns @ patterns.T, singular_values


# compiled once per module, so each hypothesis example reuses the kernel
_EOF_SUBSPACE = jax.jit(_eof_subspace)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_eofs_ignore_the_mean_and_the_sample_order(seed: int) -> None:
    # centring removes x̄, so adding a constant pattern to every sample, or
    # reordering the samples, leaves the EOF subspace and the σ unchanged
    keys = jax.random.split(jax.random.key(seed), 3)
    data = jax.random.normal(keys[0], (12, 6)) * jnp.arange(6.0, 0.0, -1.0)
    offset = 5.0 * jax.random.normal(keys[1], (6,))
    order = jax.random.permutation(keys[2], 12)
    projector, singular_values = _EOF_SUBSPACE(data)
    moved_projector, moved_values = _EOF_SUBSPACE(data[order] + offset)
    assert_matches_oracle(moved_projector, projector, rtol=0.0, atol=1e-10)
    assert_matches_oracle(moved_values, singular_values, rtol=1e-10)


def test_eof_gradient_matches_finite_differences() -> None:
    # differentiated in the data, through ΦΦᵀ and σ; the column scales keep the
    # leading singular values apart, so the SVD is differentiable there
    data = jax.random.normal(jax.random.key(5), (8, 4)) * jnp.array(
        [4.0, 2.5, 1.5, 1.0]
    )
    assert_grad_matches_fd(_eof_subspace, data, rtol=1e-7)


def test_eof_traces() -> None:
    data = jax.random.normal(jax.random.key(6), (3, 8, 4)) * jnp.array(
        [4.0, 2.5, 1.5, 1.0]
    )
    assert_traces(_eof_subspace, data, rtol=1e-10)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), scale=st.floats(0.2, 5.0))
def test_mesh_spectrum_follows_similarity_transforms(seed: int, scale: float) -> None:
    # an orthogonal map and a shift leave the cotangent spectrum unchanged; a
    # scale s divides the eigenvalues by s² and multiplies the vertex areas by s²
    vertices, triangles = icosphere(1)
    rng = np.random.default_rng(seed)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    moved = scale * np.asarray(vertices) @ orthogonal.T + rng.normal(size=3)
    eigvals, _, areas = cotangent_laplacian_eigpairs(vertices, triangles, 9)
    moved_eigvals, _, moved_areas = cotangent_laplacian_eigpairs(moved, triangles, 9)
    assert_matches_oracle(
        moved_eigvals * scale**2, eigvals, rtol=0.0, atol=1e-10 * float(eigvals[-1])
    )
    assert_matches_oracle(moved_areas, scale**2 * areas, rtol=1e-12)
