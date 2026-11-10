"""Tests for the `Kernel` protocol, `Polyharmonic` and `Wendland`. The slow tier:
sweeps, oracle and dot-product checks, gradient and trace checks, anything over
0.3 s on CI (mostly einx and XLA compiling new shapes).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from functax.interp import Polyharmonic, Wendland


class TestPolyharmonic:
    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_gradient_at_zero_distance_is_finite(self, order: int) -> None:
        kernel = Polyharmonic(order)
        gradient = jax.grad(lambda r: kernel.profile(r))(jnp.array(0.0))
        assert bool(jnp.isfinite(gradient))

    def test_kernel_gradient_at_coincident_points_is_finite(self) -> None:
        kernel = Polyharmonic(2)
        sources = jnp.array([[0.0, 0.0], [1.0, 0.5]])

        def total(points: jax.Array) -> jax.Array:
            return jnp.sum(kernel(points, sources))

        gradient = jax.grad(total)(sources)
        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert kernel(sources, sources)[0, 0] == 0.0


class TestWendland:
    def test_gradient_at_zero_distance_is_finite(self) -> None:
        kernel = Wendland(2)
        points = jnp.array([[0.3, 0.2]])
        gradient = jax.grad(lambda p: jnp.sum(kernel(p, points)))(points)
        assert bool(jnp.all(jnp.isfinite(gradient)))
