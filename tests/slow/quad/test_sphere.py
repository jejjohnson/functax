"""Tests for the sphere rules in `functax.quad`. The slow tier: every rule integrates
each real spherical harmonic Y_l^m with l up to its stated degree to 1e-12, and
misses one at the next degree (each new node count compiles the harmonics).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

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
    assert np.max(np.abs(_harmonic_integrals(rule, rule.degree))) < 1e-12


@pytest.mark.parametrize("make", RULES)
def test_the_stated_degree_is_sharp(make) -> None:
    rule = make()
    assert np.max(np.abs(_harmonic_integrals(rule, rule.degree + 1))) > 1e-4
