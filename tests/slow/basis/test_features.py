"""Tests for the feature maps of `functax.basis.features`. The slow tier: the
random-feature kernel estimate at 10⁴ features, gradient and trace checks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_traces

from functax.basis.features import (
    fourier_features,
    gaussian_window_features,
    rff_cosine_forward,
    rff_forward,
    seasonal_features,
)


W = jnp.array([[1.0, -0.5, 2.0], [0.3, 1.0, 0.5]])
PHASES = jnp.array([0.1, 2.0, 4.0])


class TestKernelEstimate:
    def test_rff_inner_products_approach_the_squared_exponential(self) -> None:
        n_features = 10_000
        lengthscale = 0.9
        frequencies = jax.random.normal(jax.random.key(0), (3, n_features))
        points = jax.random.normal(jax.random.key(1), (6, 3))
        features = jax.vmap(
            lambda point: rff_forward(frequencies, lengthscale, n_features, point)
        )(points)
        estimate = features @ features.T
        # k(x, x′) = exp(−‖x − x′‖² / (2ℓ²))
        difference = np.asarray(points)[:, None, :] - np.asarray(points)[None, :, :]
        kernel = np.exp(-(difference**2).sum(-1) / (2 * lengthscale**2))
        # each entry is a mean of F terms cos(·), each of variance ≤ 1/2, so its
        # standard error is ≤ √(1/(2F)) ≈ 0.0071; 0.05 is seven standard errors
        np.testing.assert_allclose(np.asarray(estimate), kernel, atol=0.05)


class TestGradients:
    def test_rff_forward(self) -> None:
        assert_grad_matches_fd(
            lambda x: rff_forward(W, 0.8, 3, x), jnp.array([0.4, -1.1])
        )

    def test_rff_cosine_forward(self) -> None:
        assert_grad_matches_fd(
            lambda x: rff_cosine_forward(W, PHASES, 0.8, 3, x), jnp.array([0.4, -1.1])
        )

    def test_fourier_features(self) -> None:
        assert_grad_matches_fd(
            lambda x: fourier_features(x, 3, rescale=True), jnp.array([0.1, 0.35])
        )

    def test_seasonal_features(self) -> None:
        assert_grad_matches_fd(
            lambda t: seasonal_features(t, (7.0, 30.0), (2, 1)), jnp.array([0.4, 3.5])
        )

    def test_gaussian_window_features_in_the_centres(self) -> None:
        t = jnp.linspace(0.0, 10.0, 5)
        assert_grad_matches_fd(
            lambda centres: gaussian_window_features(t, centres, jnp.array([2.0, 3.0])),
            jnp.array([2.5, 7.0]),
        )


class TestTraces:
    @pytest.mark.parametrize(
        "fn",
        [
            lambda x: rff_forward(W, 0.8, 3, x),
            lambda x: rff_cosine_forward(W, PHASES, 0.8, 3, x),
        ],
        ids=["rff_forward", "rff_cosine_forward"],
    )
    def test_random_features_over_points(self, fn: object) -> None:
        points = jax.random.normal(jax.random.key(2), (4, 2))
        assert_traces(fn, points)

    @pytest.mark.parametrize(
        "fn",
        [
            lambda x: fourier_features(x, 3, rescale=True),
            lambda x: seasonal_features(x, (7.0, 30.0), (2, 1), rescale=True),
            lambda x: gaussian_window_features(
                x, jnp.array([2.0, 8.0]), jnp.array([1.0, 1.5])
            ),
        ],
        ids=["fourier_features", "seasonal_features", "gaussian_window_features"],
    )
    def test_coordinate_features_over_batches(self, fn: object) -> None:
        batches = jax.random.uniform(jax.random.key(3), (3, 5), maxval=10.0)
        assert_traces(fn, batches)
