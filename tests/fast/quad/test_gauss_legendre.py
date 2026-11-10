"""Tests for `functax.quad.gauss_legendre`. The fast tier: exactness at the highest
degree and validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from functax import ValidationError
from functax.quad import gauss_legendre


@pytest.mark.parametrize("n", [1, 4, 9])
def test_is_exact_up_to_degree_2n_minus_1(n: int) -> None:
    nodes, weights = gauss_legendre(n)
    for degree in (2 * n - 2, 2 * n - 1):
        # ∫₋₁¹ tᵈ dt = (1 − (−1)^(d+1)) / (d + 1)
        exact = (1.0 - (-1.0) ** (degree + 1)) / (degree + 1)
        assert abs(np.sum(weights * nodes**degree) - exact) < 1e-13


def test_nodes_ascend_inside_the_interval() -> None:
    nodes, _ = gauss_legendre(7)
    assert np.all(np.diff(nodes) > 0)
    assert np.all(np.abs(nodes) < 1)


@pytest.mark.parametrize("n", [0, -2, True])
def test_rejects_a_bad_count(n: int) -> None:
    with pytest.raises(ValidationError, match="'n' must be an integer of at least 1"):
        gauss_legendre(n)


def test_rejects_a_non_integer_count() -> None:
    import inspect

    with pytest.raises(ValidationError, match="'n' must be an integer"):
        inspect.unwrap(gauss_legendre)(2.5)
