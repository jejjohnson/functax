"""Tests for `SphericalWaveletTransform`, copied from geonnax 6e32651 with functax
imports, plus the partition-of-unity gate of #111 and validation. The fast tier:
the windows and validation.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from functax._src.transform._spherical_wavelet import _partition_filters
from functax.exceptions import ValidationError
from functax.transform import SphericalWaveletTransform


@pytest.mark.parametrize("l_max", [0, 1, 2, 8, 12, 16, 31, 32])
def test_squared_windows_partition_unity(l_max):
    # gate of #111: ∑ⱼ hⱼ(l)² = 1 for every l ≤ L, which makes the round trip exact
    filters = _partition_filters(l_max)
    np.testing.assert_allclose((filters**2).sum(axis=0), 1.0, rtol=0, atol=1e-12)
    assert (filters >= 0).all()


@pytest.mark.parametrize("l_max", [0, 1, 2, 8, 12])
def test_extra_scales_keep_the_partition(l_max):
    # scales past the default fall above l_max, so their windows are zero
    default = _partition_filters(l_max)
    extra = _partition_filters(l_max, default.shape[0] + 2)
    np.testing.assert_array_equal(extra[: default.shape[0]], default)
    np.testing.assert_array_equal(extra[default.shape[0] :], 0.0)


def test_default_scales_cover_every_octave():
    # S = ⌈log₂ max(L, 2)⌉ + 1
    for l_max, n_scales in [(0, 2), (2, 2), (4, 3), (8, 4), (12, 5), (32, 6)]:
        assert _partition_filters(l_max).shape == (n_scales, l_max + 1)


@pytest.mark.parametrize("n_scales", [0, -1, 2.0, True, "3"])
def test_rejects_bad_n_scales(n_scales):
    # the check runs before the transform is read, so a stand-in for it will do,
    # and unwrapping skips the jaxtyping hook, which rejects a float first
    sht = SimpleNamespace(l_max=4)
    init = inspect.unwrap(SphericalWaveletTransform.init)
    with pytest.raises(ValidationError, match="'n_scales' must be"):
        init(SphericalWaveletTransform, sht, n_scales=n_scales)
