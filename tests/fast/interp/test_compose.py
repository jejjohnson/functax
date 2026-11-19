"""Tests for `compose` and `ComposedPlan`: H = H_outer H_inner as one stencil. The fast
tier: behaviour, validation and small closed-form checks, each well under a second.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

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


def _plans() -> tuple[Plan, Plan]:
    """A 4 → 3 inner and a 3 → 2 outer linear plan, the smallest that fold."""
    inner = Linear().plan(
        Axis(jnp.arange(4.0)), Points(jnp.array([[0.5], [1.5], [2.5]]))
    )
    outer = Linear().plan(Axis(jnp.arange(3.0)), Points(jnp.array([[0.5], [1.5]])))
    return outer, inner


def _pair(seed: int) -> tuple[Stencil, Stencil]:
    """An (M, P) outer and a (P, N) inner stencil."""
    outer = _random_stencil(N_TARGET, N_INTERMEDIATE, 3, seed)
    inner = _random_stencil(N_INTERMEDIATE, N_SOURCE, 4, seed + 100)
    return outer, inner


class TestCompose:
    def test_rejects_mismatched_sizes(self) -> None:
        outer, inner = _pair(7)
        with pytest.raises(ValidationError, match="'outer' reads 11 sources"):
            compose(inner, outer)

    def test_merge_needs_concrete_stencils(self) -> None:
        outer, inner = _pair(6)
        with pytest.raises(ValidationError, match="'merge' needs concrete stencils"):
            jax.jit(lambda o, i: compose(o, i, merge=True).weights)(outer, inner)


class TestComposedPlan:
    def test_the_inner_fill_must_be_the_outer_plans(self) -> None:
        outer, _ = _plans()
        filled = Linear().plan(
            Axis(jnp.arange(4.0)),
            Points(jnp.array([[0.5], [1.5], [2.5]])),
            policies=Policies(missing="fill", fill_value=0.0),
        )
        with pytest.raises(PolicyError, match="must share the outer plan's fill"):
            outer @ filled

    def test_rejects_plans_that_do_not_fold(self) -> None:
        outer, inner = _plans()

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
            Plan(Doubled(), outer.read) @ inner
        with pytest.raises(PolicyError, match="needs missing='propagate'"):
            Plan(Identity(), outer.read, Policies(missing="fill")) @ inner
        with pytest.raises(PolicyError, match="cannot use missing='renormalise'"):
            outer @ Plan(Identity(), inner.read, Policies(missing="renormalise"))
        dense = Dense(jnp.ones((3, 4)), jnp.ones(3, dtype=bool))
        with pytest.raises(PolicyError, match="needs a stencil read-out"):
            outer @ Plan(Identity(), dense)
        with pytest.raises(ValidationError, match="'outer' reads 3 sources"):
            outer @ outer
