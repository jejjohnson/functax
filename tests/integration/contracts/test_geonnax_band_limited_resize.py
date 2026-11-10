"""`Fourier` (#43) against geonnax ``band_limited_resize``, pinned as data.

The fixture holds random 1-D fields resampled by ``band_limited_resize`` from
``src/geonnax/layers/_antialias.py`` at geonnax ``be5537d`` (loaded by path), up
and down, odd and even sizes. The two agree on every wavenumber below
min(N, M)/2. They differ by design at the unpaired Nyquist bin of an even
min(N, M): geonnax drops it, while `Fourier` splits or folds it as
``scipy.signal.resample`` does (#43). So with an odd min(N, M) the outputs are
compared directly, and with an even one after removing the input's bins at
±min(N, M)/2. The generator is ``gen_geonnax_band_limited_resize.py`` in the
Wave 3 scratch directory.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax.geometry import Axis
from functax.interp import Fourier


DATA = np.load(Path(__file__).parent / "data" / "geonnax_band_limited_resize.npz")
SIZES = sorted(
    tuple(int(size) for size in key.split("_")[1:])
    for key in DATA
    if key.startswith("input_")
)


def _without_nyquist(y: np.ndarray, m: int) -> np.ndarray:
    """y with its DFT bins ±m/2 removed (nothing to remove for odd m)."""
    if m % 2:
        return y
    n_source = y.shape[-1]
    spectrum = np.fft.fft(y, axis=-1)
    spectrum[..., m // 2] = 0.0
    spectrum[..., n_source - m // 2] = 0.0
    return np.fft.ifft(spectrum, axis=-1).real


@pytest.mark.parametrize(("n_source", "n_target"), SIZES)
def test_fourier_equals_geonnax_band_limited_resize(
    n_source: int, n_target: int
) -> None:
    y = DATA[f"input_{n_source}_{n_target}"]
    expected = DATA[f"output_{n_source}_{n_target}"]
    source = Axis(jnp.arange(n_source) / n_source, period=1.0)
    target = Axis(jnp.arange(n_target) / n_target, period=1.0)
    plan = Fourier().plan(source, target)
    band_limited = _without_nyquist(y, min(n_source, n_target))
    actual = plan(jnp.asarray(band_limited))
    scale = np.abs(expected).max()
    assert_matches_oracle(actual, expected, rtol=1e-12, atol=1e-13 * scale)
