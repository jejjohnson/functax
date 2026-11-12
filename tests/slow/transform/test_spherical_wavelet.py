"""Tests for `SphericalWaveletTransform`, copied from geonnax 6e32651 with functax
imports, plus the float64 round-trip and trace gates of #111. The slow tier:
windows and shapes, round trips, band localisation, the jit, vmap and grad
checks and the stopped window gradients.
"""

from __future__ import annotations

import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces

from functax.transform import SphericalHarmonicTransform, SphericalWaveletTransform


KEY = jr.PRNGKey(0)


def _band_limited_field(sht, channels, key=KEY):
    coeffs = jr.normal(key, (channels, (sht.l_max + 1) ** 2))
    return sht.inverse(coeffs)


def test_round_trip_is_exact_to_1e_12_in_float64():
    # gate of #111: synthesis ∘ analysis is the identity on a band-limited field
    sht = SphericalHarmonicTransform.init(16, 32, 12)
    swt = SphericalWaveletTransform.init(sht)
    field = _band_limited_field(sht, 3)
    bands = swt.forward(field)
    assert bands.shape == (swt.n_scales, 3, 16, 32)
    assert_matches_oracle(swt.inverse(bands), field, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("grid", "extra"),
    [((2, 3, 0), 0), ((3, 5, 1), 0), ((4, 7, 2), 0), ((4, 7, 2), 2), ((9, 17, 7), 2)],
)
def test_round_trip_at_low_band_limits_and_extra_scales(grid, extra):
    sht = SphericalHarmonicTransform.init(*grid)
    default = SphericalWaveletTransform.init(sht).n_scales
    swt = SphericalWaveletTransform.init(sht, n_scales=default + extra)
    field = _band_limited_field(sht, 2)
    assert_matches_oracle(
        swt.inverse(swt.forward(field)), field, rtol=1e-12, atol=1e-12
    )


def test_bands_sum_to_the_field_only_through_the_windows():
    # the maps bⱼ hold hⱼ(l) c_lm, so ∑ⱼ bⱼ is not y but ∑ⱼ hⱼ(l) c_lm; the
    # inverse applies hⱼ a second time, which ∑ⱼ hⱼ² = 1 makes exact
    sht = SphericalHarmonicTransform.init(16, 32, 12)
    swt = SphericalWaveletTransform.init(sht)
    field = _band_limited_field(sht, 1)
    bands = swt.forward(field)
    assert float(jnp.abs(jnp.sum(bands, axis=0) - field).max()) > 1e-3


def test_scales_localise_distinct_degrees():
    # power at two well-separated degrees lights up at least two scales
    sht = SphericalHarmonicTransform.init(16, 32, 12)
    swt = SphericalWaveletTransform.init(sht)
    degrees = np.asarray(swt.degree_index)
    coeffs = np.zeros((1, degrees.shape[0]))
    # x = log₂ 1 = 0, the coarsest scale, and x = log₂ 10 ≈ 3.3, a fine one
    coeffs[0, degrees == 1] = 1.0
    coeffs[0, degrees == 10] = 1.0
    bands = swt.forward(sht.inverse(jnp.asarray(coeffs)))
    energy = jnp.sum(bands**2, axis=(1, 2, 3))
    active = int(jnp.sum(energy / energy.sum() > 0.01))
    assert active >= 2, (active, energy)


def test_windows_are_carried_in_the_default_dtype():
    sht = SphericalHarmonicTransform.init(5, 9, 4)
    swt = SphericalWaveletTransform.init(sht)
    assert swt.filters.dtype == jnp.zeros(()).dtype
    assert swt.filters.shape == (swt.n_scales, sht.l_max + 1)
    np.testing.assert_allclose(jnp.sum(swt.filters**2, axis=0), 1.0, atol=1e-12)


def test_shapes():
    sht = SphericalHarmonicTransform.init(5, 9, 4)
    swt = SphericalWaveletTransform.init(sht)
    bands = swt.forward(jnp.ones((2, 5, 9)))
    assert bands.shape == (swt.n_scales, 2, 5, 9)
    assert swt.inverse(bands).shape == (2, 5, 9)


def test_n_scales_override():
    sht = SphericalHarmonicTransform.init(5, 9, 4)
    swt = SphericalWaveletTransform.init(sht, n_scales=2)
    assert swt.n_scales == 2
    # a truncated partition no longer reconstructs exactly, but shapes hold
    assert swt.forward(jnp.ones((2, 5, 9))).shape == (2, 2, 5, 9)
    # a NumPy integer counts, as it did in geonnax (unwrapped past the jaxtyping
    # hook, which holds `n_scales` to its `int` annotation)
    init = inspect.unwrap(SphericalWaveletTransform.init)
    assert init(SphericalWaveletTransform, sht, n_scales=np.int64(3)).n_scales == 3


def test_forward_and_inverse_trace():
    sht = SphericalHarmonicTransform.init(9, 17, 7)
    swt = SphericalWaveletTransform.init(sht)
    fields = jr.normal(jr.PRNGKey(1), (3, 2, 9, 17))
    # a lambda, as a bound method of a module with array leaves is not hashable
    assert_traces(lambda y: swt.forward(y), fields)
    bands = jr.normal(jr.PRNGKey(2), (3, swt.n_scales, 2, 9, 17))
    assert_traces(lambda b: swt.inverse(b), bands)


def test_gradient_in_the_field_matches_finite_differences():
    sht = SphericalHarmonicTransform.init(5, 9, 4)
    swt = SphericalWaveletTransform.init(sht)
    field = jr.normal(jr.PRNGKey(3), (1, 5, 9))
    assert_grad_matches_fd(lambda y: swt.inverse(swt.forward(y) ** 2), field)


def test_gradient_traces():
    # jit and vmap of the gradient agree with an eager loop of gradients
    sht = SphericalHarmonicTransform.init(5, 9, 4)
    swt = SphericalWaveletTransform.init(sht)
    fields = jr.normal(jr.PRNGKey(4), (3, 1, 5, 9))
    weights = jr.normal(jr.PRNGKey(5), (swt.n_scales, 1, 5, 9))
    assert_traces(jax.grad(lambda y: jnp.sum(weights * swt.forward(y) ** 2)), fields)


def test_windows_are_frozen():
    sht = SphericalHarmonicTransform.init(12, 24, 8)
    swt = SphericalWaveletTransform.init(sht)
    field = jnp.ones((2, 12, 24))

    def loss(transform, y):
        return jnp.mean(transform.forward(y) ** 2)

    grads = eqx.filter_grad(loss)(swt, field)
    assert float(jnp.abs(grads.filters).max()) == 0.0
    # gradients reach the field
    field_grad = jax.grad(lambda y: loss(swt, y))(field)
    assert float(jnp.abs(field_grad).max()) > 0.0
