"""Tests for `compose` and `ComposedPlan`: H = H_outer H_inner as one stencil. The fast
tier: behaviour, validation and small closed-form checks, each well under a second.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from functax import ValidationError
from functax.interp import (
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


class TestCompose:
    def test_rejects_mismatched_sizes(self) -> None:
        outer, inner = _pair(7)
        with pytest.raises(ValidationError, match="'outer' reads 11 sources"):
            compose(inner, outer)
