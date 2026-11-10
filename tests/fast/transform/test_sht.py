"""Tests for `SphericalHarmonicTransform`, copied from geonnax be5537d with functax
imports, plus the float64 round-trip gate of #23, the grid layout and the stopped
table gradients. The fast tier: layouts and validation.
"""

from __future__ import annotations

import pytest

from functax.transform import SphericalHarmonicTransform


def test_sht_rejects_negative_l_max():
    with pytest.raises(ValueError, match="'l_max' must be"):
        SphericalHarmonicTransform.init(8, 16, -1)


def test_sht_rejects_under_resolved_grid():
    with pytest.raises(ValueError, match="'n_lat' must exceed l_max"):
        SphericalHarmonicTransform.init(8, 64, 12)
    with pytest.raises(ValueError, match="'n_lon' must exceed 2"):
        SphericalHarmonicTransform.init(32, 16, 12)
