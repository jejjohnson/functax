"""Tests for `functax.basis.eigen`. The fast tier, each well under a second: the
validation of the cotangent Laplacian; the validation of geonnax's
EOFs, copied from be5537d with functax imports; and graph-Laplacian tests written for
functax (geonnax had none) against the closed-form spectra of the path and the cycle.
"""

from __future__ import annotations

import inspect
import math

import hypothesis.strategies as st
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle
from hypothesis import given, settings

from functax import GeometryError, ValidationError
from functax.basis.eigen import (
    cotangent_laplacian_eigpairs,
    eof_basis,
    graph_laplacian_eigpairs,
)


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    triangles = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return vertices, triangles


def _path(n: int) -> np.ndarray:
    adjacency = np.zeros((n, n))
    index = np.arange(n - 1)
    adjacency[index, index + 1] = adjacency[index + 1, index] = 1.0
    return adjacency


def _cycle(n: int) -> np.ndarray:
    adjacency = _path(n)
    adjacency[0, -1] = adjacency[-1, 0] = 1.0
    return adjacency


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


class TestEOFBasis:
    def test_rejects_bad_args(self):
        import inspect

        # unwrapped: the type hook would reject the 3-D input first
        with pytest.raises(ValueError):
            inspect.unwrap(eof_basis)(jnp.ones((4, 3, 2)), n_modes=1)
        with pytest.raises(ValueError):
            eof_basis(jnp.ones((4, 3)), n_modes=5)  # > min(T, N)


def _random_adjacency(seed: int, n: int = 8) -> np.ndarray:
    """A symmetric weighted adjacency, each edge present with probability ½."""
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.1, 2.0, (n, n)) * (rng.uniform(size=(n, n)) < 0.5)
    weights = np.triu(weights, 1)
    return weights + weights.T


class TestGraphLaplacian:
    def test_path_combinatorial_spectrum(self):
        # D − A of the n-node path: 2 − 2 cos(kπ/n), k = 0, …, n − 1
        n = 9
        values, _ = graph_laplacian_eigpairs(_path(n), n, normalized=False)
        expected = 2 - 2 * np.cos(np.pi * np.arange(n) / n)
        np.testing.assert_allclose(values, expected, atol=1e-12)

    def test_path_normalised_spectrum(self):
        # I − D^{−½} A D^{−½} of the n-node path: 1 − cos(kπ/(n − 1))
        n = 9
        values, _ = graph_laplacian_eigpairs(_path(n), n)
        expected = 1 - np.cos(np.pi * np.arange(n) / (n - 1))
        np.testing.assert_allclose(values, expected, atol=1e-12)

    def test_cycle_spectrum_and_orthonormal_vectors(self):
        # D − A of the n-cycle: 2 − 2 cos(2πk/n), each k ≠ 0, n/2 twice
        n = 10
        values, vectors = graph_laplacian_eigpairs(_cycle(n), n, normalized=False)
        expected = np.sort(2 - 2 * np.cos(2 * np.pi * np.arange(n) / n))
        np.testing.assert_allclose(values, expected, atol=1e-12)
        # the Gram in NumPy: a JAX matmul would compile a kernel for this shape
        vectors = np.asarray(vectors)
        np.testing.assert_allclose(vectors.T @ vectors, np.eye(n), atol=1e-12)
        # the constant spans the null space
        np.testing.assert_allclose(np.abs(vectors[:, 0]), 1 / math.sqrt(n), atol=1e-12)

    def test_symmetrises_and_ignores_self_loops(self):
        adjacency = _path(5)
        adjacency[0, 1] = 3.0
        adjacency[2, 2] = 7.0
        symmetric = 0.5 * (adjacency + adjacency.T)
        np.fill_diagonal(symmetric, 0.0)
        values, _ = graph_laplacian_eigpairs(adjacency, 5, normalized=False)
        laplacian = np.diag(symmetric.sum(1)) - symmetric
        np.testing.assert_allclose(
            values, np.linalg.eigvalsh(laplacian).clip(0), atol=1e-12
        )

    def test_returns_the_lowest_modes(self):
        values, vectors = graph_laplacian_eigpairs(_cycle(8), 3)
        assert values.shape == (3,) and vectors.shape == (8, 3)
        assert bool(np.all(np.diff(np.asarray(values)) >= 0))

    @settings(max_examples=20, deadline=None)
    @given(seed=st.integers(0, 2**31 - 1), normalized=st.booleans())
    def test_spectrum_is_invariant_under_relabelling(
        self, seed: int, normalized: bool
    ) -> None:
        # λ(PAPᵀ) = λ(A), and the eigenpairs solve L v = λ v for L assembled here
        adjacency = _random_adjacency(seed)
        order = np.random.default_rng(seed + 1).permutation(8)
        eigvals, eigvecs = map(
            np.asarray, graph_laplacian_eigpairs(adjacency, 8, normalized=normalized)
        )
        relabelled, _ = graph_laplacian_eigpairs(
            adjacency[np.ix_(order, order)], 8, normalized=normalized
        )
        assert_matches_oracle(relabelled, eigvals, rtol=0.0, atol=1e-12)
        degree = adjacency.sum(axis=1)
        if normalized:
            # L = I − D^{−½} A D^{−½}, with d^{−½} = 0 for an isolated vertex
            inverse_sqrt = np.zeros(8)
            inverse_sqrt[degree > 0] = degree[degree > 0] ** -0.5
            laplacian = np.eye(8) - inverse_sqrt[:, None] * adjacency * inverse_sqrt
        else:
            # L = D − A
            laplacian = np.diag(degree) - adjacency
        assert_matches_oracle(
            laplacian @ eigvecs, eigvecs * eigvals, rtol=0.0, atol=1e-12
        )

    @pytest.mark.parametrize(
        ("adjacency", "num_basis", "match"),
        [
            (np.zeros((3, 4)), 1, "'adjacency' must be square"),
            (-_path(3), 1, "non-negative"),
            (_path(3), 0, "'num_basis' must be an integer in"),
            (_path(3), 4, "'num_basis' must be an integer in"),
            (_path(3), 1.5, "'num_basis' must be an integer in"),
        ],
    )
    def test_validation(self, adjacency, num_basis, match):
        import inspect

        # unwrapped: the type hook would reject the shape or the float first
        with pytest.raises(ValidationError, match=match):
            inspect.unwrap(graph_laplacian_eigpairs)(adjacency, num_basis)
