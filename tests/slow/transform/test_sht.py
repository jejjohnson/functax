"""Tests for `SphericalHarmonicTransform`, copied from geonnax be5537d with functax
imports, plus the float64 round-trip gate of #23, the grid layout and the stopped
table gradients. The slow tier: round trips, grid bridges, convergence and the jit,
vmap and grad check, each a second or more (mostly building the tables).
"""

from __future__ import annotations

import functools

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from _harness import assert_matches_oracle
from hypothesis import given, settings

from functax.basis.sphere import real_spherical_harmonics
from functax.transform import SphericalHarmonicTransform


KEY = jr.PRNGKey(0)


def test_sht_tables_are_built_in_float64():
    # #122: the tables are constants, so `init` evaluates the harmonics in float64
    # and casts once. analy @ synth = I is then exact to float32 rounding of the
    # tables: measured 6.2e-8 at l_max = 31 (1.6e-6 when the harmonics were
    # evaluated in float32); 5e-7 sits between the two, so the float32 build fails.
    # l_max = 31 keeps the fast tier light: 2048 x 1024 tables, a 1024² Gram.
    l_max = 31
    sht = SphericalHarmonicTransform.init(l_max + 1, 2 * l_max + 2, l_max)
    assert sht.synth.dtype == jnp.zeros(()).dtype  # default float, cast once
    with jax.enable_x64(True):
        gram = jnp.asarray(sht.analy, jnp.float64) @ jnp.asarray(sht.synth, jnp.float64)
        err = jnp.abs(gram - jnp.eye(gram.shape[0])).max()
    assert float(err) < 5e-7


def test_round_trip_is_exact_to_1e_12_in_float64():
    # gate G4-roundtrip of #23, on a random band-limited field
    l_max, n_lat, n_lon = 10, 11, 21
    sht = SphericalHarmonicTransform.init(n_lat, n_lon, l_max)
    coeffs = jr.normal(jr.PRNGKey(3), (2, (l_max + 1) ** 2))
    recovered = sht.forward(sht.inverse(coeffs))
    np.testing.assert_allclose(recovered, coeffs, rtol=0, atol=1e-12)


def test_synthesis_is_the_harmonics_at_the_grid_points():
    # column j of the synthesis is Y_j on the Gauss–Legendre nodes, sines of
    # latitude ascending and longitudes 2πk/n_lon from 0
    from functax.geometry import lonlat_to_unit
    from functax.quad import gauss_legendre

    n_lat, n_lon, l_max = 5, 9, 4
    sht = SphericalHarmonicTransform.init(n_lat, n_lon, l_max)
    nodes, _ = gauss_legendre(n_lat)
    lat, lon = np.meshgrid(
        np.arcsin(nodes), 2 * np.pi * np.arange(n_lon) / n_lon, indexing="ij"
    )
    points = lonlat_to_unit(
        jnp.asarray(np.stack([lon.ravel(), lat.ravel()], -1)), unit="radians"
    )
    np.testing.assert_allclose(
        sht.synth, real_spherical_harmonics(points, l_max), atol=1e-13
    )


def test_gradients_flow_to_the_field_not_the_tables():
    sht = SphericalHarmonicTransform.init(5, 9, 3)
    field = jr.normal(KEY, (1, 5, 9))
    grad_field = jax.grad(lambda f: jnp.sum(sht.forward(f) ** 2))(field)
    assert bool(jnp.any(grad_field != 0))
    grad_tables = jax.grad(lambda s: jnp.sum(s.forward(field) ** 2))(sht)
    assert bool(jnp.all(grad_tables.analy == 0))


@functools.cache
def _small_sht() -> SphericalHarmonicTransform:
    return SphericalHarmonicTransform.init(7, 13, 6)


@jax.jit
def _degree_power_of_synthesis(
    sht: SphericalHarmonicTransform, coeffs: jax.Array, steps: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """The power per degree of a band-limited field and of it rolled in longitude."""
    field = sht.inverse(coeffs)
    rolled = jnp.roll(field, steps, axis=-1)
    onehot = jnp.asarray(sht.degrees)[:, None] == jnp.arange(sht.l_max + 1)
    return (
        sht.forward(field) ** 2 @ onehot,
        sht.forward(rolled) ** 2 @ onehot,
    )


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), steps=st.integers(-12, 12))
def test_power_per_degree_is_invariant_under_longitude_rotation(
    seed: int, steps: int
) -> None:
    # rolling the grid by k longitudes rotates the field about the pole by
    # 2πk/n_lon, which mixes the orders ±m within each degree and keeps
    # ∑_m c_ℓm² per degree
    sht = _small_sht()
    coeffs = jr.normal(jr.key(seed), (2, (sht.l_max + 1) ** 2))
    power, rolled_power = _degree_power_of_synthesis(sht, coeffs, jnp.asarray(steps))
    assert_matches_oracle(rolled_power, power, rtol=1e-12, atol=1e-13)
