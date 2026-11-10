"""Tests for `functax.basis.polynomial`, copied from geonnax be5537d (and, for
`legendre_series`, kernellib c2d1fce) with functax imports. The fast tier: closed
forms and validation, each well under a second.
"""

from __future__ import annotations

import inspect
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special as sp

from functax.basis.polynomial import (
    associated_legendre,
    associated_legendre_indices,
    legendre_polynomials,
    legendre_series,
)
from functax.exceptions import ValidationError
from functax.quad import gauss_legendre


T = np.linspace(-1.0, 1.0, 41)


def test_legendre_polynomials_match_scipy():
    with jax.enable_x64(True):
        P = np.asarray(legendre_polynomials(jnp.asarray(T), 200))
    ref = np.stack([sp.eval_legendre(ell, T) for ell in range(201)], axis=-1)
    assert P.shape == (41, 201)
    np.testing.assert_allclose(P, ref, atol=1e-12)


def test_legendre_polynomials_shape_and_l_max_zero():
    t = jnp.zeros((2, 3))
    assert legendre_polynomials(t, 4).shape == (2, 3, 5)
    assert legendre_polynomials(t, 0).shape == (2, 3, 1)
    with pytest.raises(ValueError, match="l_max"):
        legendre_polynomials(t, -1)


def test_indices_order_and_count():
    idx = associated_legendre_indices(3)
    assert len(idx) == 10
    assert idx[:4] == ((0, 0), (1, 0), (1, 1), (2, 0))


def test_schmidt_matches_definition():
    l_max = 20
    with jax.enable_x64(True):
        S = np.asarray(
            associated_legendre(jnp.asarray(T), l_max, normalization="schmidt")
        )
    ref = np.stack(
        [
            math.sqrt(
                (2 - (m == 0)) * math.factorial(ell - m) / math.factorial(ell + m)
            )
            * (-1.0) ** m
            * sp.lpmv(m, ell, T)
            for ell, m in associated_legendre_indices(l_max)
        ],
        axis=-1,
    )
    np.testing.assert_allclose(S, ref, atol=1e-12)


def test_ldexp_helper_exact_at_zero_and_out_of_range():
    # #122: the exponent carry scales by powers of two with `_ldexp`, which
    # must have derivative 2^e at x = 0 (jnp.ldexp gives 1) and must not form
    # 0 · inf when e exceeds float32's range (exact poles at l_max ≈ 370).
    from functax._src.basis._legendre import _ldexp

    zero = jnp.float32(0.0)
    assert float(_ldexp(zero, jnp.int32(400))) == 0.0
    assert float(_ldexp(jnp.float32(0.75), jnp.int32(-400))) == 0.0
    assert float(_ldexp(jnp.float32(0.75), jnp.int32(5))) == 24.0
    grad = jax.grad(lambda x: _ldexp(x, jnp.int32(3)))(zero)
    assert float(grad) == 8.0


def test_invalid_normalization_raises():
    with pytest.raises(ValueError, match="normalization"):
        associated_legendre(jnp.zeros(3), 2, normalization="bad")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("n", [1, 4, 9])
def test_gauss_legendre_exactness(n):
    nodes, weights = gauss_legendre(n)
    for degree in (2 * n - 2, 2 * n - 1):
        exact = (1.0 - (-1.0) ** (degree + 1)) / (degree + 1)  # ∫_{-1}^{1} t^d dt
        assert abs(np.sum(weights * nodes**degree) - exact) < 1e-13
    with pytest.raises(ValueError, match="'n' must"):
        gauss_legendre(0)


@pytest.mark.parametrize("l_max", [True, 2.0])
def test_degrees_must_be_integers(l_max: object) -> None:
    import inspect

    from functax import ValidationError

    with pytest.raises(ValidationError, match="'l_max' must be an integer"):
        inspect.unwrap(legendre_polynomials)(jnp.zeros(3), l_max)


def test_unknown_options_are_rejected() -> None:
    from functax import ValidationError
    from functax.basis.sphere import shell_basis, vector_spherical_harmonics

    points = jnp.array([[0.0, 0.0, 1.5]])
    with pytest.raises(ValidationError, match="'radial' must be one of"):
        shell_basis(points, 1, 2, r_inner=1.0, r_outer=2.0, radial="bessel")
    with pytest.raises(ValidationError, match="'output' must be one of"):
        vector_spherical_harmonics(jnp.array([[0.0, 0.0, 1.0]]), 1, output="polar")


# --- legendre_series (kernellib c2d1fce) --------------------------------------


def test_legendre_series_matches_scipy() -> None:
    # kernellib's test: L = 200 against scipy's P_l, summed term by term
    degree = 200
    rng = np.random.default_rng(0)
    coeffs = rng.standard_normal(degree + 1)
    x = np.concatenate([np.linspace(-1.0, 1.0, 401), [-1.0, 1.0, 0.0]])
    expected = sum(c * sp.eval_legendre(ell, x) for ell, c in enumerate(coeffs))
    got = np.asarray(legendre_series(jnp.asarray(x), jnp.asarray(coeffs)))
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 10, 50, 100, 200])
def test_legendre_series_equals_numpy_legval(degree: int) -> None:
    # gate of #47: 1e-13 relative to ∑ₗ |cₗ|, the bound on |f| since |Pₗ| ≤ 1;
    # the two recurrences round differently, so a pointwise relative error is
    # meaningless where the series crosses zero
    rng = np.random.default_rng(degree)
    coeffs = rng.standard_normal(degree + 1)
    x = np.concatenate([np.linspace(-1.0, 1.0, 401), [-1.0, 1.0, 0.0]])
    got = legendre_series(jnp.asarray(x), jnp.asarray(coeffs))
    expected = np.polynomial.legendre.legval(x, coeffs)
    scale = float(np.sum(np.abs(coeffs)))
    np.testing.assert_allclose(np.asarray(got), expected, rtol=0.0, atol=1e-13 * scale)


def test_legendre_series_low_degrees_and_shapes() -> None:
    x = jnp.array([[-0.3, 0.2], [0.9, 1.0]])
    assert legendre_series(x, jnp.array([2.0])).tolist() == [[2.0, 2.0], [2.0, 2.0]]
    np.testing.assert_allclose(legendre_series(x, jnp.array([1.0, 3.0])), 1.0 + 3 * x)
    # rounding outside [-1, 1] is clipped: P_l(1) = 1
    assert float(legendre_series(jnp.array(1.0 + 1e-9), jnp.ones(5))) == 5.0


@pytest.mark.parametrize("shape", [(2, 2), (0,)])
def test_legendre_series_coefficients_must_be_one_non_empty_vector(
    shape: tuple[int, ...],
) -> None:
    # unwrapped, so the module's own check runs rather than the type hook
    with pytest.raises(ValidationError, match="'coeffs' must be 1-D and non-empty"):
        inspect.unwrap(legendre_series)(jnp.zeros(3), jnp.ones(shape))


def test_legendre_series_keeps_float32() -> None:
    x = jnp.linspace(-1.0, 1.0, 5, dtype=jnp.float32)
    assert legendre_series(x, jnp.ones(4, dtype=jnp.float32)).dtype == jnp.float32


@pytest.mark.parametrize("end", [-1.0, 1.0])
def test_legendre_series_gradient_is_finite_and_exact_at_the_ends(end: float) -> None:
    # P_l′(±1) = (±1)^{l+1} l(l + 1)/2, so f′(±1) = ∑ₗ cₗ (±1)^{l+1} l(l + 1)/2;
    # the clip passes the gradient straight through, so it is not halved there
    degree = 30
    coeffs = jnp.asarray(np.random.default_rng(1).standard_normal(degree + 1))
    gradient = jax.grad(lambda x: legendre_series(x, coeffs))(jnp.asarray(end))
    ells = np.arange(degree + 1)
    expected = np.sum(np.asarray(coeffs) * end ** (ells + 1) * ells * (ells + 1) / 2)
    assert bool(jnp.isfinite(gradient))
    np.testing.assert_allclose(float(gradient), expected, rtol=1e-12)
