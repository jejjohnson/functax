"""Tests for the `Kernel` protocol, `Polyharmonic` and `Wendland`. The slow tier:
finite gradients at zero distance, conditional positive definiteness and rigid-motion
invariance of the Gram matrix, gradient and trace checks, each compiling its own
shapes.
"""

from __future__ import annotations

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from hypothesis import given, settings

from functax.interp import Polyharmonic, Wendland


class TestPolyharmonic:
    @pytest.mark.parametrize("order", [1, 2])
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


# kernels with their tail degree m − 1 (−1 for none): conditionally positive
# definite of order m, so cᵀ K c > 0 for every c ⟂ the polynomials of that degree
KERNELS = [
    Polyharmonic(1),
    Polyharmonic(2),
    Polyharmonic(3),
    Polyharmonic(5),
    Wendland(2, support=0.8),
    Wendland(4, support=0.8),
]


KERNEL_IDS = ["linear", "thin-plate", "cubic", "quintic", "wendland-c2", "wendland-c4"]


def _random_points(seed: int, count: int) -> np.ndarray:
    """``count`` points uniform in the unit square, a fixed shape for every seed."""
    return np.random.default_rng(seed).uniform(0.0, 1.0, (count, 2))


def _monomials(points: np.ndarray, degree: int) -> np.ndarray:
    """P, the monomials xᵃ yᵇ with a + b ≤ degree at each point, (N, q)."""
    x, y = points.T
    return np.column_stack(
        [
            x ** (total - b) * y**b
            for total in range(degree + 1)
            for b in range(total + 1)
        ]
    )


def _gap(targets: jax.Array, sources: jax.Array) -> jax.Array:
    """The smallest distance between a target and a source."""
    return jnp.min(jnp.linalg.norm(targets[:, None] - sources[None], axis=-1))


class TestProperties:
    @pytest.mark.parametrize("kernel", KERNELS, ids=KERNEL_IDS)
    @settings(max_examples=15, deadline=None)
    @given(seed=st.integers(0, 2**31 - 1))
    def test_gram_is_positive_on_the_tail_null_space(self, kernel, seed: int) -> None:
        # Zᵀ K_XX Z ⪰ 0 with the columns of Z spanning {c : Pᵀ c = 0}; the sign
        # (−1)ᵐ of a polyharmonic kernel is what makes it so, and a Wendland
        # kernel (no tail, Z = I) is positive definite in 2-D
        points = _random_points(seed, 12)
        gram = np.asarray(kernel(jnp.asarray(points), jnp.asarray(points)))
        if kernel.min_degree < 0:
            null_space = np.eye(12)
        else:
            tail = _monomials(points, kernel.min_degree)
            q_factor, _ = np.linalg.qr(tail, mode="complete")
            null_space = q_factor[:, tail.shape[1] :]
        projected = null_space.T @ gram @ null_space
        eigenvalues = np.linalg.eigvalsh(0.5 * (projected + projected.T))
        # rounding of a symmetric eigensolve: ε N ‖K‖ ≈ 1e-14 ‖K‖
        floor = -1e-12 * float(np.max(np.abs(eigenvalues)))
        assert float(np.min(eigenvalues)) >= floor

    @pytest.mark.parametrize("kernel", KERNELS, ids=KERNEL_IDS)
    @settings(max_examples=15, deadline=None)
    @given(seed=st.integers(0, 2**31 - 1), angle=st.floats(0.0, 2.0 * np.pi))
    def test_gram_is_symmetric_and_invariant_under_rigid_motions(
        self, kernel, seed: int, angle: float
    ) -> None:
        # K(Qx + t, Qx′ + t) = K(x, x′) = K(x′, x) for a rotation Q and shift t
        points = _random_points(seed, 10)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        shift = np.random.default_rng(seed + 1).uniform(-5.0, 5.0, 2)
        moved = points @ rotation.T + shift
        gram = np.asarray(kernel(jnp.asarray(points), jnp.asarray(points)))
        moved_gram = np.asarray(kernel(jnp.asarray(moved), jnp.asarray(moved)))
        # entries cross zero (r² log r at r = 1, Wendland at r = ρ): an absolute
        # floor at the rounding of the distances, 1e-12 of the largest entry
        atol = 1e-12 * float(np.max(np.abs(gram)))
        assert_matches_oracle(gram, gram.T, rtol=0.0, atol=atol)
        assert_matches_oracle(moved_gram, gram, rtol=1e-12, atol=atol)


class TestTransforms:
    @pytest.mark.parametrize("kernel", KERNELS, ids=KERNEL_IDS)
    def test_gradient_in_the_targets_matches_finite_differences(self, kernel) -> None:
        # ∂K_*X / ∂x_* at targets at least 0.05 from every source: r = 0 is a
        # kink of −r and a log singularity of the second derivative of r² log r,
        # so the central difference is well conditioned only off it
        sources = jnp.asarray(_random_points(0, 6))
        targets = sources[:4] + jnp.array([0.07, -0.05])
        assert float(_gap(targets, sources)) > 0.05
        assert_grad_matches_fd(lambda xy: kernel(xy, sources), targets)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Wendland.support is annotated `float`, so the suite's jaxtyping hook "
            "rejects the tracer of jax.jacrev at support=0.6 (TypeCheckError in "
            "Wendland.__init__), though __check_init__ documents a traced support"
        ),
    )
    def test_gradient_in_the_wendland_support_matches_finite_differences(self) -> None:
        # ∂K / ∂ρ, the continuous parameter, with distances on both sides of ρ
        sources = jnp.asarray(_random_points(1, 6))
        targets = jnp.asarray(_random_points(2, 5))
        assert_grad_matches_fd(
            lambda rho: Wendland(4, support=rho)(targets, sources), jnp.asarray(0.6)
        )

    @pytest.mark.parametrize("kernel", KERNELS, ids=KERNEL_IDS)
    def test_jit_and_vmap_agree_with_a_loop(self, kernel) -> None:
        targets = jnp.asarray(_random_points(3, 15).reshape(3, 5, 2))
        sources = jnp.asarray(_random_points(4, 21).reshape(3, 7, 2))
        assert_traces(kernel, targets, sources)
