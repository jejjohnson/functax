"""Tests for `RBF`, `KernelSolve` and `KernelReadout`. The fast tier: exact
interpolation, kernels given as callables and lineax operators, policies,
validation and configs.
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

from functax import GeometryError, PolicyError, ValidationError
from functax.geometry import Axis, Points
from functax.interp import (
    RBF,
    KernelReadout,
    KernelSolve,
    Policies,
    Polyharmonic,
    Scheme,
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

    def test_rejects_renormalise(self) -> None:
        sources, targets, _ = _scattered(seed=12)
        with pytest.raises(PolicyError, match="renormalise"):
            RBF(Polyharmonic(2)).plan(
                Points(jnp.asarray(sources)),
                Points(jnp.asarray(targets)),
                policies=Policies(missing="renormalise"),
            )

    def test_rejects_a_grid_source_and_padded_sources(self) -> None:
        axis = Axis(jnp.arange(4.0))
        with pytest.raises(GeometryError, match="'source'"):
            RBF(Polyharmonic(2)).plan(axis, Points(jnp.zeros((1, 1))))
        padded = Points(jnp.zeros((3, 1)), valid=jnp.array([True, True, False]))
        with pytest.raises(GeometryError, match="'valid'"):
            RBF(Polyharmonic(2)).plan(padded, Points(jnp.zeros((1, 1))))


class TestValidationAndConfig:
    @pytest.mark.parametrize(
        ("arguments", "field"),
        [
            ({"degree": -2}, "'degree'"),
            ({"smoothing": -1.0}, "'smoothing'"),
            ({"smoothing": float("inf")}, "'smoothing'"),
        ],
    )
    def test_rejects_bad_fields(self, arguments: dict, field: str) -> None:
        with pytest.raises(ValidationError, match=field):
            RBF(Polyharmonic(2), **arguments)

    def test_rejects_too_many_tail_terms(self) -> None:
        points = Points(jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
        with pytest.raises(ValidationError, match="'terms'"):
            RBF(Polyharmonic(2), degree=1).plan(points, points)

    def test_kernel_solve_validates_its_inputs(self) -> None:
        build = inspect.unwrap(KernelSolve.__init__)
        fit = object.__new__(KernelSolve)
        with pytest.raises(ValidationError, match="'gram'"):
            build(fit, jnp.zeros((2, 3)))
        with pytest.raises(ValidationError, match="'terms'"):
            build(fit, jnp.eye(3), jnp.ones((2, 1)))

    def test_kernel_solve_checks_the_right_hand_side(self) -> None:
        fit = KernelSolve(jnp.eye(3), jnp.ones((3, 1)))
        assert fit.data_size(4) == 3
        with pytest.raises(ValidationError, match="'rhs'"):
            inspect.unwrap(fit.solve)(fit, jnp.ones(3))

    def test_readout_checks_its_inputs(self) -> None:
        build = inspect.unwrap(KernelReadout.__init__)
        read = object.__new__(KernelReadout)
        with pytest.raises(ValidationError, match="'matrix'"):
            build(read, jnp.zeros(3))
        with pytest.raises(ValidationError, match="'valid'"):
            build(read, jnp.zeros((2, 3)), jnp.ones(3, dtype=bool))
        read = KernelReadout(jnp.ones((2, 3)))
        with pytest.raises(ValidationError, match="'coefficients'"):
            inspect.unwrap(read.apply)(read, jnp.ones(2))
        with pytest.raises(ValidationError, match="'residual'"):
            inspect.unwrap(read.apply_adjoint)(read, jnp.ones(3))

    @pytest.mark.parametrize(
        "scheme",
        [
            RBF(Polyharmonic(2)),
            RBF(Wendland(4, support=0.5), degree=0, smoothing=0.1),
            RBF(Polyharmonic(3), solver=lx.LU()),
        ],
        ids=["default", "wendland", "solver"],
    )
    def test_config_round_trips_through_strict_json(self, scheme: RBF) -> None:
        text = json.dumps(scheme.get_config(), allow_nan=False)
        assert Scheme.from_config(json.loads(text)) == scheme

    def test_callable_kernel_has_no_config(self) -> None:
        with pytest.raises(ValidationError, match="'kernel'"):
            RBF(_gaussian).get_config()
