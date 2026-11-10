"""Slow tests for `Fourier`: ``scipy.signal.resample`` and the dot-product test
over odd and even N and M, up and down; jit and vmap against an eager loop; and
the gradient in the samples against finite differences. For `Chebyshev`: the
dot-product test over degrees and truncations, jit and vmap of the plan and the
interpolant, and gradients in the samples and the target positions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.signal
from _harness import (
    assert_adjoint,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)

from functax.geometry import Axis, Points, chebyshev_nodes
from functax.interp import Chebyshev, Fourier


def _plan(n_source: int, n_target: int):
    source = Axis(jnp.arange(n_source) / n_source, period=1.0)
    target = Axis(jnp.arange(n_target) / n_target, period=1.0)
    return Fourier().plan(source, target)


# odd and even N and M, upsampling and downsampling
SIZES = [(15, 40), (16, 40), (15, 41), (16, 24), (40, 15), (40, 16), (41, 16), (9, 8)]


@pytest.mark.parametrize(("n_source", "n_target"), SIZES)
def test_equals_scipy_signal_resample(n_source: int, n_target: int) -> None:
    # gate of #43: scipy.signal.resample to 1e-12 for odd and even N
    y = np.random.default_rng(n_source * n_target).normal(size=(3, n_source))
    expected = scipy.signal.resample(y, n_target, axis=-1)
    scale = np.abs(expected).max()
    actual = _plan(n_source, n_target)(jnp.asarray(y))
    assert_matches_oracle(actual, expected, rtol=1e-12, atol=1e-13 * scale)


@pytest.mark.parametrize(("n_source", "n_target"), SIZES)
def test_adjoint_passes_the_dot_product_test(n_source: int, n_target: int) -> None:
    plan = _plan(n_source, n_target)
    key = jax.random.key(n_source)
    assert_adjoint(plan, plan.adjoint, n_source, n_target, key=key)


@pytest.mark.parametrize(("n_source", "n_target"), [(16, 40), (15, 8)])
def test_jit_and_vmap_agree_with_a_loop(n_source: int, n_target: int) -> None:
    plan = _plan(n_source, n_target)
    batch = jnp.asarray(np.random.default_rng(0).normal(size=(3, n_source)))
    assert_traces(lambda y: plan(y), batch)
    residuals = jnp.asarray(np.random.default_rng(1).normal(size=(3, n_target)))
    assert_traces(lambda r: plan.adjoint(r), residuals)


def test_gradient_in_the_samples_matches_finite_differences() -> None:
    plan = _plan(16, 24)
    y = jnp.asarray(np.random.default_rng(2).normal(size=16))
    assert_grad_matches_fd(lambda data: plan(data), y)


# --- Chebyshev (#43) ---------------------------------------------------------


def _chebyshev_axis(degree: int, kind: str) -> Axis:
    return Axis(chebyshev_nodes(degree, kind=kind, interval=(0.0, 2.0)))


@pytest.mark.parametrize(
    ("kind", "degree", "n_coefficients"),
    [
        ("lobatto", 1, 1),
        ("lobatto", 2, 2),
        ("gauss", 2, 2),
        *((kind, 7, 3) for kind in ("lobatto", "gauss")),
        *((kind, 16, 16) for kind in ("lobatto", "gauss")),
        *((kind, 33, 20) for kind in ("lobatto", "gauss")),
    ],
)
def test_chebyshev_plan_passes_the_dot_product_test(
    kind: str, degree: int, n_coefficients: int
) -> None:
    axis = _chebyshev_axis(degree, kind)
    n_nodes = axis.values.shape[0]
    n_coefficients = min(n_coefficients, n_nodes)
    targets = Points(jnp.asarray(np.linspace(0.0, 2.0, 11))[:, None])
    plan = Chebyshev(n_coefficients=n_coefficients, kind=kind).plan(axis, targets)
    key = jax.random.key(degree)
    assert_adjoint(plan, plan.adjoint, n_nodes, 11, key=key)


@pytest.mark.parametrize("kind", ["lobatto", "gauss"])
def test_chebyshev_plan_and_interpolant_trace(kind: str) -> None:
    axis = _chebyshev_axis(9, kind)
    n_nodes = axis.values.shape[0]
    targets = jnp.asarray(np.linspace(0.1, 1.9, 5))[:, None]
    scheme = Chebyshev(n_coefficients=8, kind=kind)
    plan = scheme.plan(axis, Points(targets))
    batch = jnp.asarray(np.random.default_rng(0).normal(size=(3, n_nodes)))
    assert_traces(lambda y: plan(y), batch)
    residuals = jnp.asarray(np.random.default_rng(1).normal(size=(3, 5)))
    assert_traces(lambda r: plan.adjoint(r), residuals)
    field = scheme.fit(axis, batch[0])
    moving = jnp.asarray(np.random.default_rng(2).uniform(0.0, 2.0, (3, 5, 1)))
    assert_traces(lambda points: field(points), moving)
    assert_traces(lambda y: scheme.fit(axis, y)(targets), batch)


def test_chebyshev_gradients_match_finite_differences() -> None:
    axis = _chebyshev_axis(10, "lobatto")
    y = jnp.asarray(np.random.default_rng(3).normal(size=11))
    targets = jnp.asarray([[0.3], [1.1], [1.7]])
    scheme = Chebyshev(n_coefficients=11)
    assert_grad_matches_fd(lambda data: scheme.fit(axis, data)(targets), y)
    field = scheme.fit(axis, y)
    assert_grad_matches_fd(lambda points: field(points), targets)
