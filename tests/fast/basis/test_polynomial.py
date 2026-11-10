"""Tests for `functax.basis.polynomial`, copied from geonnax be5537d (and, for
`legendre_series`, kernellib c2d1fce) with functax imports. The fast tier: closed
forms and validation, each well under a second.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from functax.basis.polynomial import (
    associated_legendre,
    associated_legendre_indices,
    legendre_polynomials,
    legendre_series,
)
from functax.exceptions import ValidationError
from functax.quad import gauss_legendre


def test_indices_order_and_count():
    idx = associated_legendre_indices(3)
    assert len(idx) == 10
    assert idx[:4] == ((0, 0), (1, 0), (1, 1), (2, 0))


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


@pytest.mark.parametrize("shape", [(2, 2), (0,)])
def test_legendre_series_coefficients_must_be_one_non_empty_vector(
    shape: tuple[int, ...],
) -> None:
    # unwrapped, so the module's own check runs rather than the type hook
    with pytest.raises(ValidationError, match="'coeffs' must be 1-D and non-empty"):
        inspect.unwrap(legendre_series)(jnp.zeros(3), jnp.ones(shape))
