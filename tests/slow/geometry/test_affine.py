"""Tests for the affine raster kernels in `functax.geometry.affine`. The slow tier:
oracle sweeps, dot-product tests across policies, gradient and trace checks, each a
second or more (mostly einx compiling new shapes).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from _harness import assert_grad_matches_fd, assert_traces

from functax.geometry import to_pixel


def _random_transforms(count: int, seed: int) -> np.ndarray:
    """Random rotated, sheared and scaled transforms with |det A| ≥ 0.1."""
    rng = np.random.default_rng(seed)
    transforms = []
    while len(transforms) < count:
        a, b, d, e = rng.uniform(-30.0, 30.0, size=4)
        if abs(a * e - b * d) < 0.1:
            continue
        c, f = rng.uniform(-1e6, 1e6, size=2)
        transforms.append([a, b, c, d, e, f])
    return np.array(transforms)


class TestTraces:
    def test_to_pixel_under_jit_and_vmap(self) -> None:
        transforms = jnp.asarray(_random_transforms(4, seed=6))
        points = jnp.asarray(
            np.random.default_rng(7).uniform(-1e3, 1e3, size=(4, 5, 2))
        )
        assert_traces(to_pixel, transforms, points)

    def test_grad_in_the_points_and_the_transform(self) -> None:
        # rotated and sheared, with det A = −3.15, so central differences
        # with ε = 1e-6 are accurate to about 1e-10
        transform = jnp.asarray([2.0, 0.5, 10.0, -0.3, -1.5, 20.0])
        xy = jnp.asarray([[12.0, -7.0], [3.0, 4.0]])
        assert_grad_matches_fd(lambda p: to_pixel(transform, p), xy)
        assert_grad_matches_fd(lambda t: to_pixel(t, xy), transform)
