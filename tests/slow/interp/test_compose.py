"""Tests for `compose` and `ComposedPlan`: H = H_outer H_inner as one stencil. The slow
tier: oracle sweeps, dot-product tests across policies, gradient and trace checks,
each a second or more (mostly einx compiling new shapes).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import (
    assert_adjoint,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
    assert_transpose_matches,
)

from functax import PolicyError
from functax.geometry import Axis, Points
from functax.interp import (
    ComposedPlan,
    Linear,
    Plan,
    Policies,
    Stencil,
    compose,
)


N_SOURCE = 11


N_INTERMEDIATE = 9


N_TARGET = 6


def _random_stencil(
    n_target: int, n_source: int, n_taps: int, seed: int, *, dtype: type = jnp.float64
) -> Stencil:
    """A random stencil whose rows read three neighbouring sources, so taps repeat."""
    key_indices, key_weights = jax.random.split(jax.random.key(seed))
    offsets = jax.random.randint(key_indices, (n_target, n_taps), 0, 3)
    starts = jnp.arange(n_target) * (n_source - 3) // max(n_target - 1, 1)
    indices = offsets + starts[:, None]
    weights = jax.random.normal(key_weights, (n_target, n_taps), dtype=dtype)
    return Stencil(indices, weights, n_source=n_source)


def _pair(seed: int) -> tuple[Stencil, Stencil]:
    """An (M, P) outer and a (P, N) inner stencil."""
    outer = _random_stencil(N_TARGET, N_INTERMEDIATE, 3, seed)
    inner = _random_stencil(N_INTERMEDIATE, N_SOURCE, 4, seed + 100)
    return outer, inner


def _spectral_and_srf() -> tuple[Plan, Plan]:
    """Spectral interpolation onto 12 band centres, then a 7-channel read-out."""
    rng = np.random.default_rng(20)
    fine = Axis(jnp.asarray(np.cumsum(rng.uniform(0.5, 1.5, size=20))))
    centres = np.sort(rng.uniform(float(fine.values[0]), float(fine.values[-1]), 12))
    spectral = Linear().plan(fine, Points(jnp.asarray(centres)[:, None]))
    channels = np.sort(rng.uniform(centres[0], centres[-1], 7))
    srf = Linear().plan(
        Axis(jnp.asarray(centres)), Points(jnp.asarray(channels)[:, None])
    )
    return srf, spectral


class TestCompose:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_equals_the_dense_product(self, seed: int) -> None:
        outer, inner = _pair(seed)
        product = outer @ inner
        assert product.n_taps == 3 * 4
        assert product.n_source == N_SOURCE and product.n_target == N_TARGET
        expected = outer.to_dense() @ inner.to_dense()
        assert_matches_oracle(product.to_dense(), expected, rtol=1e-14, atol=1e-14)

    @pytest.mark.parametrize("seed", [3, 4])
    def test_dot_product_and_transpose(self, seed: int) -> None:
        product = compose(*_pair(seed))
        assert_adjoint(
            product.apply,
            product.apply_adjoint,
            N_SOURCE,
            N_TARGET,
            key=jax.random.key(seed),
        )
        assert_transpose_matches(
            product.apply, product.apply_adjoint, jnp.zeros(N_SOURCE)
        )

    def test_an_invalid_inner_row_invalidates_the_rows_reading_it(self) -> None:
        inner = Stencil(
            jnp.array([[0, 1], [1, 2], [2, 3]]),
            jnp.full((3, 2), 0.5),
            n_source=4,
            valid=jnp.array([True, False, True]),
        )
        outer = Stencil(
            jnp.array([[0, 0], [0, 1], [2, 2]]),
            jnp.array([[1.0, 0.0], [1.0, 0.0], [0.5, 0.5]]),
            n_source=3,
        )
        product = outer @ inner
        # row 1 reads inner row 1 at weight 0, and is still invalid
        assert product.valid.tolist() == [True, False, True]
        assert bool(jnp.all(product.weights[1] == 0))
        assert product.to_dense()[2].tolist() == [0.0, 0.0, 0.5, 0.5]

    def test_float32_stays_float32(self) -> None:
        outer = _random_stencil(N_TARGET, N_INTERMEDIATE, 2, 8, dtype=jnp.float32)
        inner = _random_stencil(N_INTERMEDIATE, N_SOURCE, 2, 9, dtype=jnp.float32)
        assert compose(outer, inner).weights.dtype == jnp.float32
        assert compose(outer, inner, merge=True).weights.dtype == jnp.float32

    def test_traces(self) -> None:
        outer, inner = _pair(10)
        weights = jax.random.normal(jax.random.key(11), (3, N_INTERMEDIATE, 4))
        coefficients = jax.random.normal(jax.random.key(12), (N_SOURCE,))

        def read(inner_weights: jax.Array) -> jax.Array:
            rows = Stencil(inner.indices, inner_weights, n_source=N_SOURCE)
            return (outer @ rows).apply(coefficients)

        assert_traces(read, weights)


class TestComposedPlan:
    def test_equals_the_plans_in_turn(self) -> None:
        srf, spectral = _spectral_and_srf()
        instrument = srf @ spectral
        assert isinstance(instrument, ComposedPlan)
        radiance = jax.random.normal(jax.random.key(21), (3, 20))
        assert_matches_oracle(
            instrument(radiance), srf(spectral(radiance)), rtol=1e-12, atol=1e-14
        )
        dense = srf.operator().as_matrix() @ spectral.operator().as_matrix()
        assert_matches_oracle(
            instrument.operator().as_matrix(), dense, rtol=1e-14, atol=1e-14
        )

    def test_dot_product_and_transpose(self) -> None:
        srf, spectral = _spectral_and_srf()
        instrument = srf @ spectral
        operator = instrument.operator()
        assert_adjoint(
            lambda y: operator.mv(y), instrument.adjoint, 20, 7, key=jax.random.key(22)
        )
        assert_transpose_matches(
            lambda y: operator.mv(y), instrument.adjoint, jnp.zeros(20)
        )
        residual = jax.random.normal(jax.random.key(23), (7,))
        assert_matches_oracle(
            instrument.adjoint(residual),
            spectral.adjoint(srf.adjoint(residual)),
            rtol=1e-12,
            atol=1e-14,
        )

    def test_an_invalid_intermediate_row_propagates(self) -> None:
        inner = Linear().plan(
            Axis(jnp.arange(5.0)), Points(jnp.array([[0.5], [7.0], [2.5], [3.5]]))
        )
        outer = Linear().plan(
            Axis(jnp.arange(4.0)),
            Points(jnp.array([[2.25], [1.5], [0.0]])),
            policies=Policies(fill_value=-1.0),
        )
        composed = outer @ inner
        assert inner.valid.tolist() == [True, False, True, True]
        # 0.0 reads node 1 (the invalid one) at weight 0, and is invalid too
        assert composed.valid.tolist() == [True, False, False]
        data = jnp.arange(5.0)
        assert composed(data).tolist() == [2.75, -1.0, -1.0]
        assert composed.policies.fill_value == -1.0
        # on the rows it marks valid, it equals the plans in turn
        assert composed(data)[0] == outer(inner(data))[0]

    def test_the_inner_missing_policy_acts_on_the_samples(self) -> None:
        inner = Linear().plan(
            Axis(jnp.arange(5.0)),
            Points(jnp.array([[0.5], [1.5], [3.5]])),
            policies=Policies(missing="fill", fill_value=0.0),
        )
        outer = Linear().plan(
            Axis(jnp.arange(3.0)),
            Points(jnp.array([[0.5], [1.5]])),
            policies=Policies(fill_value=0.0),
        )
        data = jnp.array([1.0, jnp.nan, 3.0, 4.0, 5.0])
        composed = outer @ inner
        assert composed.policies.missing == "fill"
        assert_matches_oracle(
            composed(data), outer(inner(data)), rtol=1e-12, atol=1e-14
        )
        with pytest.raises(PolicyError, match="must share the outer plan's fill"):
            Linear().plan(Axis(jnp.arange(3.0)), Points(jnp.zeros((1, 1)))) @ inner

    def test_merge(self) -> None:
        # ten intermediate nodes on three source cells: neighbours share taps
        inner = Linear().plan(
            Axis(jnp.arange(4.0)), Points(jnp.linspace(0.1, 2.9, 10)[:, None])
        )
        outer = Linear().plan(
            Axis(jnp.arange(10.0)), Points(jnp.array([[0.5], [3.3], [7.8]]))
        )
        merged = ComposedPlan(outer, inner, merge=True)
        assert merged.read.n_taps < (outer @ inner).read.n_taps
        data = jax.random.normal(jax.random.key(26), (4,))
        assert_matches_oracle(
            merged(data), (outer @ inner)(data), rtol=1e-12, atol=1e-14
        )

    def test_grad_in_the_target_positions(self) -> None:
        radiance = jax.random.normal(jax.random.key(28), (20,))
        # channels away from the band centres, where the weights have kinks
        channels = jnp.asarray(np.linspace(3.0, 9.0, 5))[:, None]
        centres = Axis(jnp.asarray(np.linspace(1.0, 12.0, 12)))
        fine = Axis(jnp.asarray(np.linspace(0.0, 13.0, 20)))
        spectral = Linear().plan(fine, Points(centres.values[:, None]))

        def read(positions: jax.Array) -> jax.Array:
            return (Linear().plan(centres, Points(positions)) @ spectral)(radiance)

        assert_grad_matches_fd(read, channels + 0.1)

    def test_composed_inside_the_trace(self) -> None:
        srf, spectral = _spectral_and_srf()
        radiance = jax.random.normal(jax.random.key(29), (4, 20))
        assert_traces(lambda y: (srf @ spectral)(y), radiance)
