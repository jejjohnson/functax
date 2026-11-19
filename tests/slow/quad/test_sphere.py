"""Tests for the sphere rules in `functax.quad`. The slow tier: every rule integrates
each real spherical harmonic Y_l^m with l up to its stated degree to 1e-12, and
misses one at the next degree (each new node count compiles the harmonics).
"""

from __future__ import annotations

import itertools
import math

import hypothesis.strategies as st
import numpy as np
import pytest
from _harness import assert_matches_oracle
from hypothesis import given, settings

from functax.basis.sphere import real_spherical_harmonics
from functax.quad import (
    SphereRule,
    sphere_fibonacci,
    sphere_gauss_legendre,
    sphere_healpix,
    sphere_latlon,
)


def _harmonic_integrals(rule: SphereRule, l_max: int) -> np.ndarray:
    """∑_q w_q Y_l^m(u_q) − √(4π) δ_l0, for every (l, m) with l ≤ l_max."""
    harmonics = real_spherical_harmonics(rule.nodes, l_max)
    integrals = np.array(rule.weights @ harmonics)
    # ∫ Y_0^0 dΩ = 4π / √(4π) = √(4π); every other Y_l^m integrates to 0
    integrals[0] -= math.sqrt(4.0 * math.pi)
    return integrals


def _monomial_moment(a: int, b: int, c: int) -> float:
    """∫_{S²} xᵃ yᵇ zᶜ dΩ, which is 2 Γ(α) Γ(β) Γ(γ) / Γ(α + β + γ) with α = (a + 1)/2,
    β = (b + 1)/2 and γ = (c + 1)/2 when a, b and c are even, and 0 otherwise.
    """
    if a % 2 or b % 2 or c % 2:
        return 0.0
    alpha, beta, gamma = (a + 1) / 2, (b + 1) / 2, (c + 1) / 2
    log_moment = (
        math.lgamma(alpha)
        + math.lgamma(beta)
        + math.lgamma(gamma)
        - math.lgamma(alpha + beta + gamma)
    )
    return 2.0 * math.exp(log_moment)


def _assert_exact_for_monomials(rule: SphereRule, radius: float) -> None:
    """∑_q w_q xᵃ yᵇ zᶜ = R² ∫_{S²} xᵃ yᵇ zᶜ dΩ for every a + b + c ≤ L."""
    nodes, weights = np.asarray(rule.nodes), np.asarray(rule.weights)
    exponents = [
        powers
        for powers in itertools.product(range(rule.degree + 1), repeat=3)
        if sum(powers) <= rule.degree
    ]
    estimates = np.array(
        [weights @ np.prod(nodes**powers, axis=1) for powers in exponents]
    )
    expected = radius**2 * np.array([_monomial_moment(*powers) for powers in exponents])
    assert_matches_oracle(estimates, expected, rtol=0, atol=1e-12 * radius**2)


RULES = [
    pytest.param(lambda: sphere_gauss_legendre(32, 64), id="gauss_legendre-32"),
    pytest.param(lambda: sphere_gauss_legendre(1), id="gauss_legendre-1"),
    pytest.param(lambda: sphere_gauss_legendre(4), id="gauss_legendre-4"),
    pytest.param(lambda: sphere_gauss_legendre(8, 9), id="gauss_legendre-8x9"),
    pytest.param(lambda: sphere_gauss_legendre(5, 3), id="gauss_legendre-5x3"),
    pytest.param(lambda: sphere_gauss_legendre(16, 32), id="gauss_legendre-16"),
    pytest.param(lambda: sphere_latlon(18, 36), id="latlon-18x36"),
    pytest.param(lambda: sphere_latlon(3, 1), id="latlon-3x1"),
    pytest.param(lambda: sphere_healpix(1), id="healpix-1"),
    pytest.param(lambda: sphere_healpix(4, nest=False), id="healpix-4-ring"),
    pytest.param(lambda: sphere_fibonacci(100), id="fibonacci-100"),
]


@pytest.mark.parametrize("make", RULES)
def test_integrates_every_harmonic_up_to_its_degree(make) -> None:
    rule = make()
    integrals = _harmonic_integrals(rule, rule.degree)
    assert_matches_oracle(integrals, np.zeros_like(integrals), rtol=0, atol=1e-12)


@pytest.mark.parametrize("make", RULES)
def test_the_stated_degree_is_sharp(make) -> None:
    rule = make()
    assert np.max(np.abs(_harmonic_integrals(rule, rule.degree + 1))) > 1e-4


@settings(max_examples=8, deadline=None)
@given(
    n_lat=st.integers(1, 8),
    n_lon=st.integers(1, 16),
    nside=st.integers(1, 6),
    n=st.integers(1, 500),
    radius=st.floats(0.1, 10.0),
)
def test_every_size_is_exact_for_the_monomials_of_its_degree(
    n_lat: int, n_lon: int, nside: int, n: int, radius: float
) -> None:
    # exactness to the stated degree at sizes and radii the sweep above misses;
    # the degree-0 moment is the sum of the weights, 4πR²
    for rule in (
        sphere_gauss_legendre(n_lat, n_lon, radius=radius),
        sphere_latlon(n_lat, n_lon, radius=radius),
        sphere_healpix(nside, nest=False, radius=radius),
        sphere_fibonacci(n, radius=radius),
    ):
        _assert_exact_for_monomials(rule, radius)
