"""functax's Clenshaw–Curtis rule (#44) against spectraldiffx's
``clenshaw_curtis_weights``, pinned at spectraldiffx d68f8f6 as data. spectraldiffx
orders its Gauss–Lobatto nodes from +1 down to −1, functax from −1 up, so the
reference is read reversed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax.quad import clenshaw_curtis


SPECTRALDIFFX = np.load(
    Path(__file__).parent / "data" / "spectraldiffx_clenshaw_curtis.npz"
)


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5, 8, 16, 32, 33])
def test_weights_equal_spectraldiffx(degree: int) -> None:
    rule = clenshaw_curtis(degree + 1)
    expected_weights = SPECTRALDIFFX[f"weights_{degree}"][::-1]
    expected_nodes = SPECTRALDIFFX[f"nodes_{degree}"][::-1]
    assert_matches_oracle(rule.weights, expected_weights, rtol=1e-14, atol=1e-15)
    assert_matches_oracle(rule.nodes, expected_nodes, rtol=0.0, atol=1e-15)
