"""functax's Chebyshev nodes and transforms (#46) against spectraldiffx, pinned at
spectraldiffx d68f8f6 as data. The generator loads ``_src/chebyshev/grid.py`` and
``transforms.py`` from a ``git archive`` export (without the package
``__init__``) and records ``ChebyshevGrid1D.x`` for N = 1, 2, 3, 8, 16, 33 on
[−1, 1] and [−2.5, 2.5], and ``ChebyshevTransform1D/2D.to_spectral`` and
``from_spectral`` of standard normal (3, n) batches and 2-D grids, for both node
types.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax.geometry import chebyshev_nodes
from functax.transform import ChebyshevTransform1D, ChebyshevTransform2D


SPECTRALDIFFX = np.load(Path(__file__).parent / "data" / "spectraldiffx_chebyshev.npz")
KINDS = ["lobatto", "gauss"]
DEGREES = [1, 2, 3, 8, 16, 33]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("half_length", [1.0, 2.5])
def test_nodes_equal_spectraldiffx(kind: str, degree: int, half_length: float) -> None:
    # the same order, from +L down to −L
    nodes = chebyshev_nodes(degree, kind=kind, interval=(-half_length, half_length))
    expected = SPECTRALDIFFX[f"nodes_{kind}_{degree}_{half_length}"]
    assert_matches_oracle(nodes, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("degree", DEGREES)
def test_transform_1d_equals_spectraldiffx(kind: str, degree: int) -> None:
    transform = ChebyshevTransform1D(degree, kind=kind)
    forward = transform.to_spectral(
        jnp.asarray(SPECTRALDIFFX[f"values_{kind}_{degree}"])
    )
    expected = SPECTRALDIFFX[f"forward_{kind}_{degree}"]
    assert_matches_oracle(forward, expected, rtol=1e-13, atol=1e-13)
    coefficients = jnp.asarray(SPECTRALDIFFX[f"coefficients_{kind}_{degree}"])
    inverse = transform.from_spectral(coefficients)
    expected = SPECTRALDIFFX[f"inverse_{kind}_{degree}"]
    assert_matches_oracle(inverse, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(("nx", "ny"), [(8, 5), (1, 3)])
def test_transform_2d_equals_spectraldiffx(kind: str, nx: int, ny: int) -> None:
    transform = ChebyshevTransform2D(nx, ny, kind=kind)
    key = f"{kind}_{nx}_{ny}"
    forward = transform.to_spectral(jnp.asarray(SPECTRALDIFFX[f"values_2d_{key}"]))
    expected = SPECTRALDIFFX[f"forward_2d_{key}"]
    assert_matches_oracle(forward, expected, rtol=1e-13, atol=1e-13)
    coefficients = jnp.asarray(SPECTRALDIFFX[f"coefficients_2d_{key}"])
    inverse = transform.from_spectral(coefficients)
    expected = SPECTRALDIFFX[f"inverse_2d_{key}"]
    assert_matches_oracle(inverse, expected, rtol=1e-13, atol=1e-13)
