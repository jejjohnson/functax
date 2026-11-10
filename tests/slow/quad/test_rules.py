"""Tests for `functax.quad`'s trapezoid and Simpson rules (#44). The slow tier:
gradients in the nodes and jit/vmap invariance, which compile per shape.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from _harness import assert_grad_matches_fd, assert_traces

from functax.quad import simpson, trapezoid


def _nodes() -> jax.Array:
    rng = np.random.default_rng(0)
    return jnp.asarray(np.sort(rng.uniform(0.0, 3.0, 7)))


def test_trapezoid_gradient_in_the_nodes() -> None:
    values = jnp.cos(_nodes())
    assert_grad_matches_fd(lambda x: trapezoid(x).weights @ values, _nodes())


def test_simpson_gradient_in_the_nodes() -> None:
    values = jnp.cos(_nodes())
    assert_grad_matches_fd(lambda x: simpson(x).weights @ values, _nodes())


def test_jit_and_vmap_agree_with_a_loop() -> None:
    rng = np.random.default_rng(1)
    batch = jnp.asarray(np.sort(rng.uniform(0.0, 3.0, (3, 6)), axis=1))
    assert_traces(lambda x: trapezoid(x).weights, batch)
    assert_traces(lambda x: simpson(x).weights, batch)
