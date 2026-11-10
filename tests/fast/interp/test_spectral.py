"""Tests for `Fourier` and `FourierReadout`: ``scipy.signal.resample`` and the
dot-product test on one up- and one downsampling, exactness on band-limited fields,
and validation. The sweeps over odd and even sizes are in tests/slow.
"""

from __future__ import annotations

import inspect
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.signal
from _harness import assert_adjoint, assert_matches_oracle, assert_transpose_matches

from functax import GeometryError, PolicyError, ValidationError
from functax.geometry import Axis
from functax.interp import Fourier, FourierReadout, Policies, Scheme


PERIOD = 2.0 * np.pi


def _axis(n_nodes: int, *, period: float = PERIOD, origin: float = 0.0) -> Axis:
    return Axis(origin + jnp.arange(n_nodes) * period / n_nodes, period=period)


@pytest.mark.parametrize(("n_source", "n_target"), [(15, 40), (16, 10)])
def test_equals_scipy_signal_resample(n_source: int, n_target: int) -> None:
    # gate of #43: scipy.signal.resample to 1e-12 for odd and even N, up and down
    y = np.random.default_rng(n_source * n_target).normal(size=(3, n_source))
    plan = Fourier().plan(_axis(n_source), _axis(n_target))
    expected = scipy.signal.resample(y, n_target, axis=-1)
    scale = np.abs(expected).max()
    assert_matches_oracle(
        plan(jnp.asarray(y)), expected, rtol=1e-12, atol=1e-13 * scale
    )


@pytest.mark.parametrize(("n_source", "n_target"), [(16, 40), (15, 7), (16, 10)])
def test_exact_on_band_limited_fields(n_source: int, n_target: int) -> None:
    # every wavenumber |k| < min(N, M)/2 is carried exactly
    k_max = (min(n_source, n_target) - 1) // 2
    rng = np.random.default_rng(1)
    amplitudes = rng.normal(size=(2, k_max + 1))

    def field(x: jax.Array) -> jax.Array:
        k = jnp.arange(k_max + 1)
        phase = jnp.outer(x, k)
        return jnp.cos(phase) @ amplitudes[0] + jnp.sin(phase) @ amplitudes[1]

    source, target = _axis(n_source), _axis(n_target)
    plan = Fourier().plan(source, target)
    assert_matches_oracle(
        plan(field(source.values)), field(target.values), rtol=1e-12, atol=1e-13
    )


@pytest.mark.parametrize("n_source", [15, 16])
def test_upsampling_then_downsampling_returns_the_samples(n_source: int) -> None:
    y = jnp.asarray(np.random.default_rng(2).normal(size=n_source))
    up = Fourier().plan(_axis(n_source), _axis(40))
    down = Fourier().plan(_axis(40), _axis(n_source))
    assert_matches_oracle(down(up(y)), y, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(("n_source", "n_target"), [(16, 40), (15, 8)])
def test_adjoint_passes_the_dot_product_test(n_source: int, n_target: int) -> None:
    plan = Fourier().plan(_axis(n_source), _axis(n_target))
    assert_adjoint(plan, plan.adjoint, n_source, n_target, key=jax.random.key(n_source))


def test_adjoint_equals_the_linear_transpose() -> None:
    read = FourierReadout(16, 25)
    assert_transpose_matches(read.apply, read.apply_adjoint, jnp.zeros(16))


def test_operator_is_the_read_out() -> None:
    plan = Fourier().plan(_axis(16), _axis(24))
    y = jnp.asarray(np.random.default_rng(3).normal(size=16))
    assert_matches_oracle(plan.operator().mv(y), plan(y), rtol=1e-12, atol=1e-14)


def test_equal_sizes_are_the_identity() -> None:
    y = jnp.asarray(np.random.default_rng(4).normal(size=12))
    plan = Fourier().plan(_axis(12), _axis(12))
    assert_matches_oracle(plan(y), y, rtol=1e-12, atol=1e-14)


def test_float32_is_kept() -> None:
    source = Axis(jnp.arange(8, dtype=jnp.float32) / 8, period=1.0)
    target = Axis(jnp.arange(12, dtype=jnp.float32) / 12, period=1.0)
    plan = Fourier().plan(source, target)
    assert plan(jnp.ones(8, dtype=jnp.float32)).dtype == jnp.float32
    assert plan.adjoint(jnp.ones(12, dtype=jnp.float32)).dtype == jnp.float32


def test_missing_fill_replaces_non_finite_samples() -> None:
    plan = Fourier().plan(
        _axis(8), _axis(16), policies=Policies(missing="fill", fill_value=0.0)
    )
    y = jnp.zeros(8).at[3].set(jnp.nan)
    assert bool(jnp.all(plan(y) == 0.0))


def test_config_round_trips_through_json() -> None:
    config = json.loads(json.dumps(Fourier().get_config(), allow_nan=False))
    assert Scheme.from_config(config) == Fourier()


@pytest.mark.parametrize(
    ("source", "target", "match"),
    [
        (Axis(jnp.arange(8.0)), _axis(16), "periodic"),
        (_axis(8), _axis(16, period=1.0), "period"),
        (_axis(8), _axis(16, origin=0.1), "first node"),
        (Axis(jnp.arange(8) / 10, period=1.0), _axis(16), "nodes"),
        (Axis(jnp.arange(8.0)[::-1] / 8, period=1.0), _axis(16), "nodes"),
        (jnp.arange(8.0), _axis(16), "Axis"),
    ],
)
def test_plan_validates_the_axes(source, target, match) -> None:
    with pytest.raises(GeometryError, match=match):
        Fourier().plan(source, target)


def test_renormalise_has_no_fourier_plan() -> None:
    with pytest.raises(PolicyError, match="renormalise"):
        Fourier().plan(_axis(8), _axis(16), policies=Policies(missing="renormalise"))


@pytest.mark.parametrize(("n_source", "n_target"), [(0, 4), (4, 0)])
def test_readout_validates_its_sizes(n_source, n_target) -> None:
    with pytest.raises(ValidationError, match="n_"):
        FourierReadout(n_source, n_target)


def test_apply_checks_the_last_axis() -> None:
    read = FourierReadout(8, 16)
    with pytest.raises(ValidationError, match="'coefficients'"):
        inspect.unwrap(FourierReadout.apply)(read, jnp.ones(5))
    with pytest.raises(ValidationError, match="'residual'"):
        inspect.unwrap(FourierReadout.apply_adjoint)(read, jnp.ones(5))
