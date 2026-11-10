"""Tests for the `Kernel` protocol, `Polyharmonic` and `Wendland`. The fast tier:
closed forms, gradients at r = 0, validation and configs.
"""

from __future__ import annotations

import inspect
import json

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax import ValidationError
from functax._src.interp._kernels import kernel_matrix
from functax.basis.local import wendland_c2, wendland_c4
from functax.interp import Kernel, Polyharmonic, RadialKernel, Wendland


DISTANCES = jnp.array([0.0, 0.25, 1.0, 1.7, 3.0])


class TestPolyharmonic:
    @pytest.mark.parametrize(
        ("order", "expected"),
        [
            # SciPy's RBFInterpolator kernels: linear, thin-plate, cubic, quintic
            (1, lambda r: -r),
            (2, lambda r: np.where(r > 0, r**2 * np.log(np.where(r > 0, r, 1)), 0)),
            (3, lambda r: r**3),
            (4, lambda r: -np.where(r > 0, r**4 * np.log(np.where(r > 0, r, 1)), 0)),
            (5, lambda r: -(r**5)),
        ],
        ids=["linear", "thin-plate", "cubic", "r4-log", "quintic"],
    )
    def test_profile_has_scipys_sign(self, order: int, expected) -> None:
        r = np.asarray(DISTANCES)
        assert_matches_oracle(
            Polyharmonic(order).profile(DISTANCES), expected(r), rtol=1e-15
        )

    @pytest.mark.parametrize(
        ("order", "min_degree"), [(1, 0), (2, 1), (3, 1), (4, 2), (5, 2)]
    )
    def test_min_degree_follows_the_order_of_positive_definiteness(
        self, order: int, min_degree: int
    ) -> None:
        assert Polyharmonic(order).min_degree == min_degree

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

    @pytest.mark.parametrize("order", [0, -1, True])
    def test_rejects_a_bad_order(self, order: int) -> None:
        with pytest.raises(ValidationError, match="'order'"):
            Polyharmonic(order)


class TestWendland:
    @pytest.mark.parametrize(
        ("smoothness", "profile"), [(2, wendland_c2), (4, wendland_c4)]
    )
    def test_profile_is_the_basis_wendland_scaled_by_support(
        self, smoothness: int, profile
    ) -> None:
        kernel = Wendland(smoothness, support=2.0)
        assert_matches_oracle(
            kernel.profile(DISTANCES), profile(DISTANCES / 2.0), rtol=1e-15
        )
        assert kernel.profile(jnp.array(2.0)) == 0.0
        assert kernel.min_degree == -1

    def test_gradient_at_zero_distance_is_finite(self) -> None:
        kernel = Wendland(2)
        points = jnp.array([[0.3, 0.2]])
        gradient = jax.grad(lambda p: jnp.sum(kernel(p, points)))(points)
        assert bool(jnp.all(jnp.isfinite(gradient)))

    @pytest.mark.parametrize("smoothness", [0, 3, 6])
    def test_rejects_a_bad_smoothness(self, smoothness: int) -> None:
        with pytest.raises(ValidationError, match="'smoothness'"):
            Wendland(smoothness)

    @pytest.mark.parametrize("support", [0.0, -1.0, float("inf"), float("nan")])
    def test_rejects_a_bad_support(self, support: float) -> None:
        with pytest.raises(ValidationError, match="'support'"):
            Wendland(2, support=support)


class TestProtocolAndConfig:
    def test_plain_function_and_operator_kernel_are_kernels(self) -> None:
        def gaussian(targets: jax.Array, sources: jax.Array) -> jax.Array:
            return jnp.exp(-jnp.sum((targets[:, None] - sources[None]) ** 2, -1))

        def operator_kernel(targets: jax.Array, sources: jax.Array):
            return lx.MatrixLinearOperator(gaussian(targets, sources))

        assert isinstance(gaussian, Kernel)
        assert isinstance(Polyharmonic(), Kernel)
        points = jnp.array([[0.0, 0.0], [1.0, 2.0], [0.5, 0.5]])
        assert_matches_oracle(
            kernel_matrix(operator_kernel, points, points[:2]),
            kernel_matrix(gaussian, points, points[:2]),
            rtol=0.0,
        )

    def test_kernel_matrix_rejects_the_wrong_shape(self) -> None:
        def transposed(targets: jax.Array, sources: jax.Array) -> jax.Array:
            return jnp.zeros((sources.shape[0], targets.shape[0]))

        kernel_matrix_unchecked = inspect.unwrap(kernel_matrix)
        with pytest.raises(ValidationError, match="'kernel'"):
            kernel_matrix_unchecked(transposed, jnp.zeros((3, 2)), jnp.zeros((2, 2)))

    @pytest.mark.parametrize(
        "kernel", [Polyharmonic(3), Wendland(4, support=0.5)], ids=["poly", "wend"]
    )
    def test_config_round_trips_through_strict_json(self, kernel) -> None:
        text = json.dumps(kernel.get_config(), allow_nan=False)
        assert RadialKernel.from_config(json.loads(text)) == kernel
