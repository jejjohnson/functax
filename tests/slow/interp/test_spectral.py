"""Slow tests for `Fourier`: ``scipy.signal.resample`` and the dot-product test
over odd and even N and M, up and down; jit and vmap against an eager loop; and
the gradient in the samples against finite differences. For `Chebyshev`: the
dot-product test over degrees and truncations, jit and vmap of the plan and the
interpolant, and gradients in the samples and the target positions.
"""

from __future__ import annotations

import inspect

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
    assert_transpose_matches,
)

from functax import OutsideDomainError, ValidationError
from functax.geometry import Axis, Points, chebyshev_nodes
from functax.interp import (
    Chebyshev,
    Fourier,
    FourierReadout,
    Policies,
    clenshaw,
)


PERIOD = 2.0 * np.pi


def _axis(n_nodes: int, *, period: float = PERIOD, origin: float = 0.0) -> Axis:
    return Axis(origin + jnp.arange(n_nodes) * period / n_nodes, period=period)


CHEBVAL = np.polynomial.chebyshev.chebval


KINDS = ["lobatto", "gauss"]


def _chebyshev_axis(degree: int, kind: str, *, ascending: bool = False) -> Axis:
    nodes = chebyshev_nodes(degree, kind=kind, interval=(-1.0, 3.0))
    return Axis(nodes[::-1] if ascending else nodes)


def _band_limited(axis: Axis, n_modes: int, seed: int) -> tuple[np.ndarray, jax.Array]:
    """Coefficients of degree < n_modes on [−1, 3], and their samples on the axis."""
    coefficients = np.random.default_rng(seed).standard_normal((2, n_modes))
    reference = (np.asarray(axis.values) - 1.0) / 2.0
    return coefficients, jnp.asarray(CHEBVAL(reference, coefficients.T))


def _plan(n_source: int, n_target: int):
    source = Axis(jnp.arange(n_source) / n_source, period=1.0)
    target = Axis(jnp.arange(n_target) / n_target, period=1.0)
    return Fourier().plan(source, target)


SIZES = [(15, 40), (16, 40), (15, 41), (16, 24), (40, 15), (40, 16), (41, 16), (9, 8)]


def _chebyshev_axis_on_0_2(degree: int, kind: str) -> Axis:
    return Axis(chebyshev_nodes(degree, kind=kind, interval=(0.0, 2.0)))


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
    axis = _chebyshev_axis_on_0_2(degree, kind)
    n_nodes = axis.values.shape[0]
    n_coefficients = min(n_coefficients, n_nodes)
    targets = Points(jnp.asarray(np.linspace(0.0, 2.0, 11))[:, None])
    plan = Chebyshev(n_coefficients=n_coefficients, kind=kind).plan(axis, targets)
    key = jax.random.key(degree)
    assert_adjoint(plan, plan.adjoint, n_nodes, 11, key=key)


@pytest.mark.parametrize("kind", ["lobatto", "gauss"])
def test_chebyshev_plan_and_interpolant_trace(kind: str) -> None:
    axis = _chebyshev_axis_on_0_2(9, kind)
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
    axis = _chebyshev_axis_on_0_2(10, "lobatto")
    y = jnp.asarray(np.random.default_rng(3).normal(size=11))
    targets = jnp.asarray([[0.3], [1.1], [1.7]])
    scheme = Chebyshev(n_coefficients=11)
    assert_grad_matches_fd(lambda data: scheme.fit(axis, data)(targets), y)
    field = scheme.fit(axis, y)
    assert_grad_matches_fd(lambda points: field(points), targets)


@pytest.mark.parametrize(("n_source", "n_target"), [(15, 40), (16, 10)])
def test_equals_scipy_signal_resample_on_a_wider_period(
    n_source: int, n_target: int
) -> None:
    # gate of #43: scipy.signal.resample to 1e-12 for odd and even N, up and down
    y = np.random.default_rng(n_source * n_target).normal(size=(3, n_source))
    plan = Fourier().plan(_axis(n_source), _axis(n_target))
    expected = scipy.signal.resample(y, n_target, axis=-1)
    scale = np.abs(expected).max()
    assert_matches_oracle(
        plan(jnp.asarray(y)), expected, rtol=1e-12, atol=1e-13 * scale
    )


@pytest.mark.parametrize(("n_source", "n_target"), [(16, 40), (15, 7), (16, 10)])
def test_exact_on_band_limited_fields(n_source: int, n_target: int) -> None:
    # every wavenumber |k| < min(N, M)/2 is carried exactly
    k_max = (min(n_source, n_target) - 1) // 2
    rng = np.random.default_rng(1)
    amplitudes = rng.normal(size=(2, k_max + 1))

    def field(x: jax.Array) -> jax.Array:
        k = jnp.arange(k_max + 1)
        phase = jnp.outer(x, k)
        return jnp.cos(phase) @ amplitudes[0] + jnp.sin(phase) @ amplitudes[1]

    source, target = _axis(n_source), _axis(n_target)
    plan = Fourier().plan(source, target)
    assert_matches_oracle(
        plan(field(source.values)), field(target.values), rtol=1e-12, atol=1e-13
    )


@pytest.mark.parametrize("n_source", [15, 16])
def test_upsampling_then_downsampling_returns_the_samples(n_source: int) -> None:
    y = jnp.asarray(np.random.default_rng(2).normal(size=n_source))
    up = Fourier().plan(_axis(n_source), _axis(40))
    down = Fourier().plan(_axis(40), _axis(n_source))
    assert_matches_oracle(down(up(y)), y, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(("n_source", "n_target"), [(16, 40), (15, 8)])
def test_adjoint_passes_the_dot_product_test_on_a_wider_period(
    n_source: int, n_target: int
) -> None:
    plan = Fourier().plan(_axis(n_source), _axis(n_target))
    assert_adjoint(plan, plan.adjoint, n_source, n_target, key=jax.random.key(n_source))


def test_adjoint_equals_the_linear_transpose() -> None:
    read = FourierReadout(16, 25)
    assert_transpose_matches(read.apply, read.apply_adjoint, jnp.zeros(16))


def test_operator_is_the_read_out() -> None:
    plan = Fourier().plan(_axis(16), _axis(24))
    y = jnp.asarray(np.random.default_rng(3).normal(size=16))
    assert_matches_oracle(plan.operator().mv(y), plan(y), rtol=1e-12, atol=1e-14)


def test_equal_sizes_are_the_identity() -> None:
    y = jnp.asarray(np.random.default_rng(4).normal(size=12))
    plan = Fourier().plan(_axis(12), _axis(12))
    assert_matches_oracle(plan(y), y, rtol=1e-12, atol=1e-14)


def test_float32_is_kept() -> None:
    source = Axis(jnp.arange(8, dtype=jnp.float32) / 8, period=1.0)
    target = Axis(jnp.arange(12, dtype=jnp.float32) / 12, period=1.0)
    plan = Fourier().plan(source, target)
    assert plan(jnp.ones(8, dtype=jnp.float32)).dtype == jnp.float32
    assert plan.adjoint(jnp.ones(12, dtype=jnp.float32)).dtype == jnp.float32


def test_clenshaw_shapes_dtypes_and_validation() -> None:
    assert clenshaw(jnp.zeros((2, 3)), jnp.array([2.0])).tolist() == [[2.0] * 3] * 2
    x32 = jnp.linspace(-1.0, 1.0, 4, dtype=jnp.float32)
    assert clenshaw(x32, jnp.ones(3, dtype=jnp.float32)).dtype == jnp.float32
    unwrapped = inspect.unwrap(clenshaw)
    with pytest.raises(ValidationError, match="'coefficients' must be 1-D"):
        unwrapped(jnp.zeros(3), jnp.ones((2, 2)))
    with pytest.raises(ValidationError, match="'coefficients' must be 1-D"):
        unwrapped(jnp.zeros(3), jnp.ones(0))
    with pytest.raises(ValidationError, match="'interval' must have a < b"):
        clenshaw(jnp.zeros(3), jnp.ones(2), interval=(1.0, 1.0))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("ascending", [False, True])
@pytest.mark.parametrize(
    ("degree", "n_coefficients", "n_modes"), [(12, 13, 13), (12, 6, 6)]
)
def test_exact_on_band_limited_samples(
    kind: str, ascending: bool, degree: int, n_coefficients: int, n_modes: int
) -> None:
    # gate of #43: samples of a series of degree < K come back exactly, from the
    # plan's dense read-out and from the interpolant's Clenshaw sum
    axis = _chebyshev_axis(degree, kind, ascending=ascending)
    n_modes = min(n_modes, axis.values.shape[0])
    coefficients, samples = _band_limited(axis, n_modes, seed=degree)
    x = np.random.default_rng(1).uniform(-1.0, 3.0, 9)
    expected = CHEBVAL((x - 1.0) / 2.0, coefficients.T)
    scheme = Chebyshev(n_coefficients=min(n_coefficients, n_modes), kind=kind)
    targets = Points(jnp.asarray(x)[:, None])
    scale = float(np.max(np.abs(expected)))
    assert_matches_oracle(
        scheme.plan(axis, targets)(samples), expected, rtol=0.0, atol=1e-12 * scale
    )
    field = scheme.fit(axis, samples)
    assert_matches_oracle(field(targets), expected, rtol=0.0, atol=1e-12 * scale)
    assert_matches_oracle(
        field.coefficients[..., :n_modes], coefficients, rtol=0.0, atol=1e-12 * 3
    )


@pytest.mark.parametrize("kind", KINDS)
def test_interpolant_equals_the_plan(kind: str) -> None:
    axis = _chebyshev_axis(9, kind, ascending=True)
    y = jnp.asarray(np.random.default_rng(4).standard_normal((3, axis.values.shape[0])))
    targets = Points(jnp.array([[-1.0], [0.2], [2.9], [3.0]]))
    scheme = Chebyshev(n_coefficients=7, kind=kind)
    field = scheme.fit(axis, y)
    assert_matches_oracle(field(targets), scheme.plan(axis, targets)(y), rtol=1e-12)
    assert_matches_oracle(field.plan(targets)(y), field(targets.xy), rtol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("ascending", [False, True])
def test_plan_passes_the_dot_product_test(kind: str, ascending: bool) -> None:
    # Testing list of #43: ⟨Hy, r⟩ = ⟨y, Hᵀr⟩, H = T P C, with a truncation
    axis = _chebyshev_axis(10, kind, ascending=ascending)
    n_nodes = axis.values.shape[0]
    targets = Points(jnp.asarray(np.linspace(-1.0, 3.0, 7))[:, None])
    plan = Chebyshev(n_coefficients=8, kind=kind).plan(axis, targets)
    assert_adjoint(plan, plan.adjoint, n_nodes, 7, key=jax.random.key(n_nodes))
    assert_transpose_matches(plan, plan.adjoint, jnp.ones(n_nodes))
    # the fit on its own: Cᵀ Pᵀ against the linear transpose of P C
    fit = plan.fit
    assert_adjoint(fit.apply, fit.apply_adjoint, n_nodes, 8, key=jax.random.key(1))
    assert_transpose_matches(fit.apply, fit.apply_adjoint, jnp.ones(n_nodes))


def test_outside_rules() -> None:
    axis = _chebyshev_axis(8, "lobatto")
    y = (axis.values - 1.0) ** 2
    coordinates = jnp.array([[-2.0], [1.0], [4.0]])
    targets = Points(coordinates)
    scheme = Chebyshev(n_coefficients=9)
    nan = scheme.plan(axis, targets)(y)
    assert bool(jnp.isnan(nan[0])) and bool(jnp.isnan(nan[2]))
    clamp = scheme.plan(axis, targets, policies=Policies(outside="clamp"))(y)
    assert jnp.allclose(clamp, jnp.array([4.0, 0.0, 4.0]))
    extrapolate = Policies(outside="extrapolate")
    assert jnp.allclose(
        scheme.plan(axis, targets, policies=extrapolate)(y), jnp.array([9.0, 0.0, 9.0])
    )
    assert jnp.allclose(
        scheme.fit(axis, y, policies=extrapolate)(coordinates),
        jnp.array([9.0, 0.0, 9.0]),
    )
    with pytest.raises(OutsideDomainError, match="2 of 3 targets"):
        scheme.plan(axis, targets, policies=Policies(outside="raise"))
    filled = scheme.fit(axis, y, policies=Policies(fill_value=-9.0))(coordinates)
    assert filled.tolist()[0] == -9.0


def test_missing_fill_and_float32() -> None:
    axis = _chebyshev_axis(6, "gauss")
    y = jnp.ones(6).at[2].set(jnp.nan)
    targets = Points(jnp.array([[0.0], [2.0]]))
    policies = Policies(missing="fill", fill_value=1.0)
    scheme = Chebyshev(n_coefficients=6, kind="gauss")
    assert jnp.allclose(scheme.plan(axis, targets, policies=policies)(y), 1.0)
    assert jnp.allclose(scheme.fit(axis, y, policies=policies)(targets), 1.0)
    axis32 = Axis(chebyshev_nodes(6).astype(jnp.float32))
    y32 = jnp.ones(7, dtype=jnp.float32)
    field = Chebyshev(n_coefficients=7).fit(axis32, y32)
    assert field(jnp.array([[0.5]], dtype=jnp.float32)).dtype == jnp.float32


def test_float32_ends_are_inside() -> None:
    # an end node in float32 rounds to |ξ| slightly above 1; within 8ε it is on
    # the end, not outside
    axis = Axis(chebyshev_nodes(9, interval=(0.1, 0.7)).astype(jnp.float32))
    y = jnp.ones(10, dtype=jnp.float32)
    field = Chebyshev(n_coefficients=10).fit(axis, y)
    ends = jnp.array([[0.1], [0.7]], dtype=jnp.float32)
    assert bool(jnp.all(jnp.isfinite(field(ends))))


def test_clamp_keeps_the_derivative_at_the_ends() -> None:
    # f = x² on [1, 3]: f′ = 2 and 6 at the ends, 0 beyond them under clamp
    axis = Axis(chebyshev_nodes(6, interval=(1.0, 3.0)))
    field = Chebyshev(n_coefficients=7).fit(
        axis, axis.values**2, policies=Policies(outside="clamp")
    )
    gradient = jax.grad(lambda x: field(x).sum())(jnp.array([[1.0], [3.0], [4.0]]))
    assert jnp.allclose(gradient[:, 0], jnp.array([2.0, 6.0, 0.0]))


def test_missing_fill_replaces_non_finite_samples() -> None:
    plan = Fourier().plan(
        _axis(8), _axis(16), policies=Policies(missing="fill", fill_value=0.0)
    )
    y = jnp.zeros(8).at[3].set(jnp.nan)
    assert bool(jnp.all(plan(y) == 0.0))
