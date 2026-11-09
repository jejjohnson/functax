"""Tests for `compose` and `ComposedPlan`: H = H_outer H_inner as one stencil. The fast
tier: behaviour, validation and small closed-form checks, each well under a second.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import (
    assert_matches_oracle,
)

from functax import PolicyError, ValidationError
from functax.geometry import Axis, Points
from functax.interp import (
    Fit,
    Identity,
    Linear,
    Plan,
    Policies,
    Readout,
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
    def test_merge_sums_repeated_indices(self) -> None:
        outer, inner = _pair(5)
        product = compose(outer, inner)
        merged = compose(outer, inner, merge=True)
        # rows read neighbours, so merging leaves far fewer than S_outer · S_inner
        assert merged.n_taps < product.n_taps
        assert_matches_oracle(
            merged.to_dense(), product.to_dense(), rtol=1e-14, atol=1e-14
        )
        for row_indices, row_weights in zip(
            np.asarray(merged.indices), np.asarray(merged.weights), strict=True
        ):
            # each index once, plus padding at weight 0
            carried = row_indices[row_weights != 0]
            assert carried.size == np.unique(carried).size

    def test_merge_needs_concrete_stencils(self) -> None:
        outer, inner = _pair(6)
        with pytest.raises(ValidationError, match="'merge' needs concrete stencils"):
            jax.jit(lambda o, i: compose(o, i, merge=True).weights)(outer, inner)

    def test_rejects_mismatched_sizes(self) -> None:
        outer, inner = _pair(7)
        with pytest.raises(ValidationError, match="'outer' reads 11 sources"):
            compose(inner, outer)


class TestComposedPlan:
    def test_rejects_plans_that_do_not_fold(self) -> None:
        srf, spectral = _spectral_and_srf()

        class Doubled(Fit):
            def apply(self, data):  # type: ignore[no-untyped-def]
                return 2.0 * data

            def apply_adjoint(self, coefficients):  # type: ignore[no-untyped-def]
                return 2.0 * coefficients

        class Dense(Readout):
            matrix: jax.Array
            valid: jax.Array

            def apply(self, coefficients):  # type: ignore[no-untyped-def]
                return coefficients @ self.matrix.T

            def apply_adjoint(self, residual):  # type: ignore[no-untyped-def]
                return residual @ self.matrix

            def operator(self):  # type: ignore[no-untyped-def]
                return None

        with pytest.raises(PolicyError, match="the outer plan of a composition"):
            Plan(Doubled(), srf.read) @ spectral
        with pytest.raises(PolicyError, match="needs missing='propagate'"):
            Plan(Identity(), srf.read, Policies(missing="fill")) @ spectral
        with pytest.raises(PolicyError, match="cannot use missing='renormalise'"):
            srf @ Plan(Identity(), spectral.read, Policies(missing="renormalise"))
        dense = Dense(jnp.ones((12, 20)), jnp.ones(12, dtype=bool))
        with pytest.raises(PolicyError, match="needs a stencil read-out"):
            srf @ Plan(Identity(), dense)
        with pytest.raises(ValidationError, match="'outer' reads 12 sources"):
            srf @ srf

    def test_an_inner_fit_is_kept(self) -> None:
        srf, spectral = _spectral_and_srf()

        class Doubled(Fit):
            def apply(self, data):  # type: ignore[no-untyped-def]
                return 2.0 * data

            def apply_adjoint(self, coefficients):  # type: ignore[no-untyped-def]
                return 2.0 * coefficients

        inner = Plan(Doubled(), spectral.read)
        radiance = jax.random.normal(jax.random.key(30), (20,))
        assert_matches_oracle(
            (srf @ inner)(radiance), srf(inner(radiance)), rtol=1e-12, atol=1e-14
        )
