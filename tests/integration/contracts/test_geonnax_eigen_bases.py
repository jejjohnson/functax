"""functax's EOF and graph-Laplacian bases (#19) against geonnax, pinned at geonnax
be5537d as data. Singular and eigenvectors are compared up to the sign of each
column, the one freedom an SVD or eigh leaves with distinct values.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax.basis.eigen import eof_basis, graph_laplacian_eigpairs


GEONNAX = np.load(Path(__file__).parent / "data" / "geonnax_eigen_bases.npz")


def _align_signs(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Flip each column of ``actual`` to agree in sign with ``expected``."""
    signs = np.sign(np.sum(actual * expected, axis=0))
    return actual * signs


@pytest.mark.parametrize("center", [True, False])
def test_eof_basis_equals_geonnax(center: bool) -> None:
    phi, singular_values = eof_basis(jnp.asarray(GEONNAX["data"]), 5, center=center)
    assert_matches_oracle(singular_values, GEONNAX[f"eof_{center}_s"], rtol=1e-12)
    expected = GEONNAX[f"eof_{center}"]
    aligned = _align_signs(np.asarray(phi), expected)
    assert_matches_oracle(aligned, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("normalized", [True, False])
def test_graph_laplacian_eigpairs_equal_geonnax(normalized: bool) -> None:
    values, vectors = graph_laplacian_eigpairs(
        GEONNAX["adjacency"], 12, normalized=normalized
    )
    assert_matches_oracle(
        values, GEONNAX[f"graph_{normalized}"], rtol=1e-12, atol=1e-14
    )
    expected = GEONNAX[f"graph_{normalized}_vecs"]
    aligned = _align_signs(np.asarray(vectors), expected)
    assert_matches_oracle(aligned, expected, rtol=0, atol=1e-10)
