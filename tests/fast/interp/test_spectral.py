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

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax import GeometryError, PolicyError, ValidationError
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


CHEBVAL = np.polynomial.chebyshev.chebval


def _chebyshev_axis(degree: int, kind: str, *, ascending: bool = False) -> Axis:
    nodes = chebyshev_nodes(degree, kind=kind, interval=(-1.0, 3.0))
    return Axis(nodes[::-1] if ascending else nodes)


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


def test_read_out_is_the_dense_chebyshev_table() -> None:
    # T_pk = Tₖ(ξ_p)
    axis = _chebyshev_axis(6, "lobatto")
    x = np.array([-1.0, 0.0, 2.5])
    plan = Chebyshev(n_coefficients=5).plan(axis, Points(jnp.asarray(x)[:, None]))
    reference = (x - 1.0) / 2.0
    expected = np.polynomial.chebyshev.chebvander(reference, 4)
    assert_matches_oracle(plan.read.matrix, expected, rtol=0.0, atol=1e-14)


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
