"""Tests for `Fourier` and `FourierReadout`: ``scipy.signal.resample`` and the
dot-product test on one up- and one downsampling, exactness on band-limited fields,
and validation. The sweeps over odd and even sizes are in tests/slow. Then
`clenshaw`, `Chebyshev`, `ChebyshevFit` and `ChebyshevInterpolant`: ``chebval``,
exactness on band-limited samples, the dot-product test, the outside rules and
validation.
"""

from __future__ import annotations

import inspect
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.signal
from _harness import assert_adjoint, assert_matches_oracle, assert_transpose_matches

from functax import GeometryError, OutsideDomainError, PolicyError, ValidationError
from functax.geometry import Axis, Points, chebyshev_nodes
from functax.interp import (
    Chebyshev,
    ChebyshevFit,
    Fourier,
    FourierReadout,
    Policies,
    Scheme,
    clenshaw,
)
from functax.transform import ChebyshevTransform1D


PERIOD = 2.0 * np.pi


def _axis(n_nodes: int, *, period: float = PERIOD, origin: float = 0.0) -> Axis:
    return Axis(origin + jnp.arange(n_nodes) * period / n_nodes, period=period)


@pytest.mark.parametrize(("n_source", "n_target"), [(15, 40), (16, 10)])
def test_equals_scipy_signal_resample(n_source: int, n_target: int) -> None:
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
def test_adjoint_passes_the_dot_product_test(n_source: int, n_target: int) -> None:
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


def test_missing_fill_replaces_non_finite_samples() -> None:
    plan = Fourier().plan(
        _axis(8), _axis(16), policies=Policies(missing="fill", fill_value=0.0)
    )
    y = jnp.zeros(8).at[3].set(jnp.nan)
    assert bool(jnp.all(plan(y) == 0.0))


def test_config_round_trips_through_json() -> None:
    config = json.loads(json.dumps(Fourier().get_config(), allow_nan=False))
    assert Scheme.from_config(config) == Fourier()


@pytest.mark.parametrize(
    ("source", "target", "match"),
    [
        (Axis(jnp.arange(8.0)), _axis(16), "periodic"),
        (_axis(8), _axis(16, period=1.0), "period"),
        (_axis(8), _axis(16, origin=0.1), "first node"),
        (Axis(jnp.arange(8) / 10, period=1.0), _axis(16), "nodes"),
        (Axis(jnp.arange(8.0)[::-1] / 8, period=1.0), _axis(16), "nodes"),
        (jnp.arange(8.0), _axis(16), "Axis"),
    ],
)
def test_plan_validates_the_axes(source, target, match) -> None:
    with pytest.raises(GeometryError, match=match):
        Fourier().plan(source, target)


def test_renormalise_has_no_fourier_plan() -> None:
    with pytest.raises(PolicyError, match="renormalise"):
        Fourier().plan(_axis(8), _axis(16), policies=Policies(missing="renormalise"))


@pytest.mark.parametrize(("n_source", "n_target"), [(0, 4), (4, 0)])
def test_readout_validates_its_sizes(n_source, n_target) -> None:
    with pytest.raises(ValidationError, match="n_"):
        FourierReadout(n_source, n_target)


def test_apply_checks_the_last_axis() -> None:
    read = FourierReadout(8, 16)
    with pytest.raises(ValidationError, match="'coefficients'"):
        inspect.unwrap(FourierReadout.apply)(read, jnp.ones(5))
    with pytest.raises(ValidationError, match="'residual'"):
        inspect.unwrap(FourierReadout.apply_adjoint)(read, jnp.ones(5))


# --- Chebyshev (#43) ---------------------------------------------------------

CHEBVAL = np.polynomial.chebyshev.chebval
KINDS = ["lobatto", "gauss"]


@pytest.mark.parametrize("n_coefficients", [1, 2, 3, 10, 50, 200])
def test_clenshaw_equals_numpy_chebval(n_coefficients: int) -> None:
    # gate of #43: 1e-13 relative to ∑ₖ |aₖ|, the bound on |f| since |Tₖ| ≤ 1;
    # a pointwise relative error is meaningless where the series crosses zero
    rng = np.random.default_rng(n_coefficients)
    coefficients = rng.standard_normal(n_coefficients)
    x = np.concatenate([np.linspace(-1.0, 1.0, 401), [-1.0, 1.0, 0.0]])
    scale = float(np.sum(np.abs(coefficients)))
    values = clenshaw(jnp.asarray(x), jnp.asarray(coefficients))
    expected = CHEBVAL(x, coefficients)
    assert_matches_oracle(values, expected, rtol=0.0, atol=1e-13 * scale)
    # an interval maps affinely: x ∈ [−3, 5] reads ξ = (2x − 2)/8
    mapped = clenshaw(
        jnp.asarray(1.0 + 4.0 * x), jnp.asarray(coefficients), interval=(-3.0, 5.0)
    )
    assert_matches_oracle(mapped, expected, rtol=0.0, atol=1e-13 * scale)


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


def _chebyshev_axis(degree: int, kind: str, *, ascending: bool = False) -> Axis:
    nodes = chebyshev_nodes(degree, kind=kind, interval=(-1.0, 3.0))
    return Axis(nodes[::-1] if ascending else nodes)


def _band_limited(axis: Axis, n_modes: int, seed: int) -> tuple[np.ndarray, jax.Array]:
    """Coefficients of degree < n_modes on [−1, 3], and their samples on the axis."""
    coefficients = np.random.default_rng(seed).standard_normal((2, n_modes))
    reference = (np.asarray(axis.values) - 1.0) / 2.0
    return coefficients, jnp.asarray(CHEBVAL(reference, coefficients.T))


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


def test_read_out_is_the_dense_chebyshev_table() -> None:
    # T_pk = Tₖ(ξ_p)
    axis = _chebyshev_axis(6, "lobatto")
    x = np.array([-1.0, 0.0, 2.5])
    plan = Chebyshev(n_coefficients=5).plan(axis, Points(jnp.asarray(x)[:, None]))
    reference = (x - 1.0) / 2.0
    expected = np.polynomial.chebyshev.chebvander(reference, 4)
    assert_matches_oracle(plan.read.matrix, expected, rtol=0.0, atol=1e-14)


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


def test_chebyshev_config_round_trips_through_json() -> None:
    scheme = Chebyshev(n_coefficients=12, kind="gauss")
    config = json.loads(json.dumps(scheme.get_config(), allow_nan=False))
    assert Scheme.from_config(config) == scheme


@pytest.mark.parametrize(
    ("source", "match"),
    [
        (Axis(jnp.linspace(-1.0, 1.0, 9)), "lobatto Chebyshev nodes"),
        (Axis(chebyshev_nodes(8)[::-1], period=10.0), "periodic"),
        (chebyshev_nodes(8), "Axis"),
    ],
)
def test_chebyshev_validates_the_source(source, match) -> None:
    targets = Points(jnp.array([[0.0]]))
    with pytest.raises(GeometryError, match=match):
        Chebyshev(n_coefficients=2).plan(source, targets)


def test_chebyshev_validates_its_arguments() -> None:
    axis = _chebyshev_axis(4, "lobatto")
    with pytest.raises(ValidationError, match="'n_coefficients' must be an integer"):
        Chebyshev(n_coefficients=0)
    with pytest.raises(ValidationError, match="'kind' must be one of"):
        Chebyshev(n_coefficients=3, kind="gauss-lobatto")
    with pytest.raises(ValidationError, match="'n_coefficients' must be at most the 5"):
        Chebyshev(n_coefficients=6).fit(axis, jnp.ones(5))
    with pytest.raises(PolicyError, match="renormalise"):
        Chebyshev(n_coefficients=3).plan(
            axis, Points(jnp.array([[0.0]])), policies=Policies(missing="renormalise")
        )
    field = Chebyshev(n_coefficients=3).fit(axis, jnp.ones(5))
    with pytest.raises(GeometryError, match=r"\(M, 1\)"):
        inspect.unwrap(type(field).__call__)(field, jnp.ones((2, 2)))
    fit = ChebyshevFit(ChebyshevTransform1D(4), 3)
    with pytest.raises(ValidationError, match="'data'"):
        inspect.unwrap(ChebyshevFit.apply)(fit, jnp.ones(4))
    with pytest.raises(ValidationError, match="'coefficients'"):
        inspect.unwrap(ChebyshevFit.apply_adjoint)(fit, jnp.ones(4))


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
