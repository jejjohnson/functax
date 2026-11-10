"""Tests for `RBF`, `KernelSolve` and `KernelReadout`. The slow tier: the SciPy
oracle sweep, the dot-product test, gradients and trace checks, each a second
or more (mostly einx compiling new shapes).
"""

from __future__ import annotations

import inspect
import warnings

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
import scipy.interpolate
from _harness import (
    assert_adjoint,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
    assert_transpose_matches,
)

from functax import ValidationError
from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import (
    RBF,
    KernelSolve,
    Policies,
    Polyharmonic,
    Wendland,
)


def _scattered(seed: int, n_source: int = 30, n_target: int = 20):
    """Random sources in the unit square, a smooth field on them, random targets."""
    rng = np.random.default_rng(seed)
    sources = rng.uniform(size=(n_source, 2))
    targets = rng.uniform(-0.1, 1.1, size=(n_target, 2))
    data = np.sin(3.0 * sources[:, 0]) + sources[:, 1] ** 2
    return sources, targets, data


def _gaussian(targets: jax.Array, sources: jax.Array) -> jax.Array:
    """A user kernel: k(x, x′) = exp(−‖x − x′‖² / 0.1)."""
    difference = targets[:, None, :] - sources[None, :, :]
    return jnp.exp(-jnp.sum(difference**2, axis=-1) / 0.1)


class TestOracle:
    @pytest.mark.parametrize("smoothing", [0.0, 1e-2], ids=["exact", "smoothed"])
    @pytest.mark.parametrize("degree", [0, 1])
    @pytest.mark.parametrize(
        ("name", "order"), [("thin_plate_spline", 2), ("cubic", 3)]
    )
    def test_equals_scipy_rbf_interpolator(
        self, name: str, order: int, degree: int, smoothing: float
    ) -> None:
        sources, targets, data = _scattered(seed=order + 2 * degree)
        plan = RBF(Polyharmonic(order), degree=degree, smoothing=smoothing).plan(
            Points(jnp.asarray(sources)), Points(jnp.asarray(targets))
        )
        with warnings.catch_warnings():
            # degree 0 is below SciPy's minimum for both kernels, and it warns
            warnings.simplefilter("ignore", UserWarning)
            expected = scipy.interpolate.RBFInterpolator(
                sources, data, kernel=name, degree=degree, smoothing=smoothing
            )(targets)
        # 1e-9 relative to the field's scale (the issue's tolerance for a solve;
        # measured about 1e-14)
        scale = float(np.max(np.abs(expected)))
        assert_matches_oracle(
            plan(jnp.asarray(data)), expected, rtol=1e-9, atol=1e-9 * scale
        )


class TestAdjoint:
    @pytest.mark.parametrize(
        ("scheme", "rtol"),
        [
            (RBF(Polyharmonic(2)), 1e-12),
            # cubic with a constant tail is indefinite on the null space, so it
            # is solved by LU; cond(M) ≈ 2e5 there, and the forward and the
            # transposed LU solves differ by about cond(M) · ε ≈ 4e-11
            (RBF(Polyharmonic(3), degree=0), 1e-10),
            (RBF(Wendland(2)), 1e-12),
        ],
        ids=["tail", "lu", "no-tail"],
    )
    def test_dot_product(self, scheme: RBF, rtol: float) -> None:
        sources, targets, _ = _scattered(seed=6)
        plan = scheme.plan(Points(jnp.asarray(sources)), Points(jnp.asarray(targets)))
        assert_adjoint(plan, plan.adjoint, 30, 20, key=jax.random.key(0), rtol=rtol)
        assert_transpose_matches(plan, plan.adjoint, jnp.zeros(30), rtol=rtol)

    def test_operator_is_the_plan(self) -> None:
        sources, targets, data = _scattered(seed=7)
        plan = RBF(Polyharmonic(2)).plan(
            Points(jnp.asarray(sources)), Points(jnp.asarray(targets))
        )
        operator = plan.operator()
        assert operator.in_structure().shape == (30,)
        assert_matches_oracle(
            operator.mv(jnp.asarray(data)), plan(jnp.asarray(data)), rtol=1e-12
        )

    def test_gradient_with_respect_to_the_data_is_finite_and_linear(self) -> None:
        sources, targets, data = _scattered(seed=8)
        plan = RBF(Polyharmonic(2)).plan(
            Points(jnp.asarray(sources)), Points(jnp.asarray(targets))
        )
        weights = jnp.linspace(-1.0, 1.0, 20)

        def loss(values: jax.Array) -> jax.Array:
            return jnp.vdot(weights, plan(values))

        gradient = jax.grad(loss)(jnp.asarray(data))
        assert bool(jnp.all(jnp.isfinite(gradient)))
        # the loss is linear in y, so its gradient is Hᵀ w
        assert_matches_oracle(gradient, plan.adjoint(weights), rtol=1e-10)


class TestShapes:
    def test_callable_kernel_equals_the_dense_solve(self) -> None:
        sources, targets, data = _scattered(seed=4, n_source=12)
        plan = RBF(_gaussian, smoothing=1e-3).plan(
            Points(jnp.asarray(sources)), Points(jnp.asarray(targets))
        )
        gram = np.asarray(_gaussian(jnp.asarray(sources), jnp.asarray(sources)))
        weights = np.linalg.solve(gram + 1e-3 * np.eye(12), data)
        cross = np.asarray(_gaussian(jnp.asarray(targets), jnp.asarray(sources)))
        assert_matches_oracle(plan(jnp.asarray(data)), cross @ weights, rtol=1e-10)

    def test_batches_over_leading_dims(self) -> None:
        sources, targets, data = _scattered(seed=5)
        plan = RBF(Polyharmonic(3)).plan(
            Points(jnp.asarray(sources)), Points(jnp.asarray(targets))
        )
        batch = jnp.stack([jnp.asarray(data), 2.0 * jnp.asarray(data)])
        values = plan(batch)
        assert values.shape == (2, 20)
        assert_matches_oracle(values[1], 2.0 * values[0], rtol=1e-12)

    def test_grid_target_keeps_its_shape(self) -> None:
        sources, _, data = _scattered(seed=9)
        grid = RectilinearGrid(
            (Axis(jnp.linspace(0.0, 1.0, 3)), Axis(jnp.linspace(0.0, 1.0, 4)))
        )
        plan = RBF(Polyharmonic(2)).plan(Points(jnp.asarray(sources)), grid)
        assert plan.grid_shape == (3, 4)
        assert plan(jnp.asarray(data)).shape == (12,)

    def test_float32_stays_float32(self) -> None:
        sources, targets, data = _scattered(seed=13)
        plan = RBF(Wendland(2, support=2.0)).plan(
            Points(jnp.asarray(sources, jnp.float32)),
            Points(jnp.asarray(targets, jnp.float32)),
        )
        assert plan(jnp.asarray(data, jnp.float32)).dtype == jnp.float32


class TestTransforms:
    def test_jit_and_vmap_agree_with_a_loop(self) -> None:
        sources, targets, data = _scattered(seed=14)
        plan = RBF(Polyharmonic(2)).plan(
            Points(jnp.asarray(sources)), Points(jnp.asarray(targets))
        )
        batch = jnp.stack([jnp.asarray(data) * scale for scale in (1.0, -0.5, 2.0)])
        assert_traces(lambda values: plan(values), batch)

    def test_gradient_in_the_data_matches_finite_differences(self) -> None:
        sources, targets, data = _scattered(seed=15, n_source=12, n_target=5)
        plan = RBF(Polyharmonic(3), smoothing=1e-2).plan(
            Points(jnp.asarray(sources)), Points(jnp.asarray(targets))
        )
        assert_grad_matches_fd(plan, jnp.asarray(data))


class TestInterpolation:
    @pytest.mark.parametrize(
        "kernel",
        [Polyharmonic(2), Polyharmonic(3), Wendland(4, support=1.5), _gaussian],
        ids=["thin-plate", "cubic", "wendland", "callable"],
    )
    def test_passes_through_the_samples_without_smoothing(self, kernel) -> None:
        sources, _, data = _scattered(seed=1)
        points = Points(jnp.asarray(sources))
        plan = RBF(kernel).plan(points, points)
        assert_matches_oracle(plan(jnp.asarray(data)), data, rtol=1e-10, atol=1e-10)

    def test_degree_one_tail_reproduces_an_affine_field(self) -> None:
        sources, targets, _ = _scattered(seed=2)
        affine = 1.0 - 2.0 * sources[:, 0] + 0.5 * sources[:, 1]
        plan = RBF(Polyharmonic(2), smoothing=0.3).plan(
            Points(jnp.asarray(sources)), Points(jnp.asarray(targets))
        )
        expected = 1.0 - 2.0 * targets[:, 0] + 0.5 * targets[:, 1]
        assert_matches_oracle(
            plan(jnp.asarray(affine)), expected, rtol=1e-10, atol=1e-10
        )

    def test_operator_kernel_equals_the_array_kernel(self) -> None:
        sources, targets, data = _scattered(seed=3)

        def operator_kernel(targets: jax.Array, sources: jax.Array):
            return lx.MatrixLinearOperator(_gaussian(targets, sources))

        source, target = Points(jnp.asarray(sources)), Points(jnp.asarray(targets))
        expected = RBF(_gaussian).plan(source, target)(jnp.asarray(data))
        actual = RBF(operator_kernel).plan(source, target)(jnp.asarray(data))
        assert_matches_oracle(actual, expected, rtol=1e-12)


class TestTargetsAndPolicies:
    def test_padding_targets_read_the_fill_value(self) -> None:
        sources, targets, data = _scattered(seed=10)
        valid = jnp.arange(20) % 4 != 0
        padded = jnp.where(valid[:, None], jnp.asarray(targets), jnp.nan)
        plan = RBF(Polyharmonic(2)).plan(
            Points(jnp.asarray(sources)), Points(padded, valid=valid)
        )
        values = plan(jnp.asarray(data))
        assert bool(jnp.all(jnp.isnan(values) == ~valid))
        assert bool(jnp.all(jnp.isfinite(plan.adjoint(jnp.ones(20)))))

    def test_missing_fill_fills_before_the_fit(self) -> None:
        sources, targets, data = _scattered(seed=11)
        source, target = Points(jnp.asarray(sources)), Points(jnp.asarray(targets))
        gappy = jnp.asarray(data).at[3].set(jnp.nan)
        filled = jnp.asarray(data).at[3].set(0.0)
        policies = Policies(missing="fill", fill_value=0.0)
        plan = RBF(Polyharmonic(2)).plan(source, target, policies=policies)
        expected = RBF(Polyharmonic(2)).plan(source, target)(filled)
        assert_matches_oracle(plan(gappy), expected, rtol=1e-12)


class TestValidationAndConfig:
    def test_rejects_too_many_tail_terms(self) -> None:
        points = Points(jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
        with pytest.raises(ValidationError, match="'terms'"):
            RBF(Polyharmonic(2), degree=1).plan(points, points)

    def test_kernel_solve_checks_the_right_hand_side(self) -> None:
        fit = KernelSolve(jnp.eye(3), jnp.ones((3, 1)))
        assert fit.data_size(4) == 3
        with pytest.raises(ValidationError, match="'rhs'"):
            inspect.unwrap(fit.solve)(fit, jnp.ones(3))
