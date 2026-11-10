"""Tests for the non-uniform FFTs of `functax.transform`, copied from geonnax be5537d
with functax imports. The fast tier: layouts and validation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from functax._src.transform._nufft import _corner_freqs, _corner_slices
from functax.transform import nonuniform_irfft, nonuniform_rfft


@pytest.mark.parametrize("n_modes", [(4,), (3, 2), (2, 3, 2)])
def test_corner_freqs_match_corner_slices(n_modes):
    """`_corner_freqs[c]` are exactly the wavenumbers `_corner_slices[c]` selects."""
    shape = (*(2 * m + 2 for m in n_modes[:-1]), n_modes[-1] + 1)
    per_axis = [np.fft.fftfreq(s, 1.0 / s).astype(int) for s in shape[:-1]]
    per_axis.append(np.arange(shape[-1]))
    wavenumbers = np.stack(np.meshgrid(*per_axis, indexing="ij"), axis=-1)
    slices = _corner_slices(n_modes)
    freqs = _corner_freqs(n_modes)
    assert len(freqs) == len(slices) == 2 ** (len(n_modes) - 1)
    for sl, fr in zip(slices, freqs, strict=True):
        expected = np.stack(np.meshgrid(*fr, indexing="ij"), axis=-1)
        np.testing.assert_array_equal(wavenumbers[sl], expected)


def test_nonuniform_rfft_validation():
    import inspect

    # unwrapped: the type hook would reject the mismatched shapes first
    raw_rfft = inspect.unwrap(nonuniform_rfft)
    xi = jnp.zeros((10, 2))
    u = jnp.zeros((1, 10))
    w = jnp.zeros((10,))
    with pytest.raises(ValueError, match="'n_modes' must not be empty"):
        raw_rfft(u, xi, w, ())
    with pytest.raises(ValueError, match="at least 1"):
        raw_rfft(u, xi, w, (0, 2))
    with pytest.raises(ValueError, match="'xi' must have shape"):
        raw_rfft(u, xi, w, (2, 2, 2))
    with pytest.raises(ValueError, match="'u' must have shape"):
        raw_rfft(jnp.zeros((1, 9)), xi, w, (2, 2))
    with pytest.raises(ValueError, match="'weights' must have shape"):
        raw_rfft(u, xi, jnp.zeros((9,)), (2, 2))


def test_nonuniform_irfft_validation():
    xi = jnp.zeros((10, 2))
    block = jnp.zeros((1, 2, 2), dtype=jnp.complex64)
    with pytest.raises(ValueError, match="'coeffs' must hold"):
        nonuniform_irfft([block], xi, (2, 2))
    with pytest.raises(ValueError, match="corner block must have shape"):
        nonuniform_irfft([block, jnp.zeros((1, 3, 2), jnp.complex64)], xi, (2, 2))
    with pytest.raises(ValueError, match="'xi' must have shape"):
        nonuniform_irfft([block, block], jnp.zeros((10, 3)), (2, 2))
