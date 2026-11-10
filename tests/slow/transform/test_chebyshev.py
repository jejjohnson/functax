"""Tests for `ChebyshevTransform1D` and `ChebyshevTransform2D`, copied from
spectraldiffx d68f8f6 with functax imports, plus the round-trip gate G4-roundtrip
of #46 and the gradient and trace checks. The slow tier: each a second or more,
mostly the eager FFTs compiling for every new shape.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces

from functax.geometry import chebyshev_nodes
from functax.transform import ChebyshevTransform1D, ChebyshevTransform2D


KINDS = ["lobatto", "gauss"]


def test_transform1d_roundtrip_gauss() -> None:
    x = chebyshev_nodes(24, kind="gauss", interval=(-2.0, 2.0))
    transform = ChebyshevTransform1D(24, kind="gauss")
    u = jnp.cos(jnp.pi * x / 2.0)
    a = transform.to_spectral(u)
    assert jnp.allclose(transform.from_spectral(a), u, atol=1e-12)


def test_transform2d_roundtrip() -> None:
    x, y = chebyshev_nodes(16), chebyshev_nodes(16)
    transform = ChebyshevTransform2D(16, 16)
    u = jnp.outer(jnp.cos(jnp.pi * y), jnp.sin(jnp.pi * x))
    a = transform.to_spectral(u)
    assert jnp.allclose(transform.from_spectral(a), u, atol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_random_roundtrip(kind: str) -> None:
    transform = ChebyshevTransform1D(16, kind=kind)
    u = jax.random.normal(jax.random.key(42), (transform.n_nodes,))
    assert jnp.allclose(transform.from_spectral(transform.to_spectral(u)), u)
    grid = ChebyshevTransform2D(8, 8, kind=kind)
    field = jax.random.normal(jax.random.key(0), grid.shape)
    assert jnp.allclose(grid.from_spectral(grid.to_spectral(field)), field)


def test_both_kinds_agree_on_the_convention() -> None:
    # T₀ + 2T₁ − ½T₂
    expected = jnp.array([1.0, 2.0, -0.5])
    for kind in KINDS:
        x = chebyshev_nodes(10, kind=kind)
        u = 1.0 + 2.0 * x - 0.5 * (2 * x**2 - 1)
        a = ChebyshevTransform1D(10, kind=kind).to_spectral(u)
        assert jnp.allclose(a[:3], expected, atol=1e-13), kind
        assert jnp.allclose(a[3:], 0.0, atol=1e-13), kind


@pytest.mark.parametrize("kind", KINDS)
def test_2d_tensor_mode(kind: str) -> None:
    # T₂(x) T₃(y) maps to the unit coefficient at [ky = 3, kx = 2]
    transform = ChebyshevTransform2D(8, 9, kind=kind)
    x, y = chebyshev_nodes(8, kind=kind), chebyshev_nodes(9, kind=kind)
    u = jnp.outer(4 * y**3 - 3 * y, 2 * x**2 - 1)
    a = transform.to_spectral(u)
    assert jnp.allclose(a, jnp.zeros_like(a).at[3, 2].set(1.0), atol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("degree", [1, 2, 7, 32])
def test_g4_roundtrip_1d(kind: str, degree: int) -> None:
    # gate G4-roundtrip: u = ∑ₖ aₖ Tₖ at the nodes, band-limited to the modes
    # the nodes resolve, comes back from forward then inverse to 1e-12 relative,
    # and forward recovers its coefficients
    transform = ChebyshevTransform1D(degree, kind=kind)
    nodes = chebyshev_nodes(degree, kind=kind, interval=(-3.0, 5.0))
    reference = (2.0 * nodes - 2.0) / 8.0
    rng = np.random.default_rng(degree)
    coefficients = rng.standard_normal((2, transform.n_nodes))
    u = jnp.asarray(
        np.polynomial.chebyshev.chebval(np.asarray(reference), coefficients.T)
    )
    forward = transform.to_spectral(u)
    scale = float(np.max(np.abs(coefficients)))
    assert_matches_oracle(forward, coefficients, rtol=0.0, atol=1e-12 * scale)
    roundtrip = transform.from_spectral(forward)
    scale = float(jnp.max(jnp.abs(u)))
    assert_matches_oracle(roundtrip, u, rtol=0.0, atol=1e-12 * scale)


@pytest.mark.parametrize("kind", KINDS)
def test_g4_roundtrip_2d(kind: str) -> None:
    transform = ChebyshevTransform2D(7, 5, kind=kind)
    x, y = chebyshev_nodes(7, kind=kind), chebyshev_nodes(5, kind=kind)
    rng = np.random.default_rng(1)
    coefficients = rng.standard_normal(transform.shape)
    # u(yᵢ, xⱼ) = ∑ₗ ∑ₖ a_{lk} Tₗ(yᵢ) Tₖ(xⱼ)
    chebgrid2d = np.polynomial.chebyshev.chebgrid2d
    u = jnp.asarray(chebgrid2d(np.asarray(y), np.asarray(x), coefficients))
    forward = transform.to_spectral(u)
    assert_matches_oracle(forward, coefficients, rtol=0.0, atol=1e-12 * 3.0)
    roundtrip = transform.from_spectral(forward)
    scale = float(jnp.max(jnp.abs(u)))
    assert_matches_oracle(roundtrip, u, rtol=0.0, atol=1e-12 * scale)


@pytest.mark.parametrize("kind", KINDS)
def test_batches_and_dtypes(kind: str) -> None:
    transform = ChebyshevTransform1D(5, kind=kind)
    u = jax.random.normal(jax.random.key(3), (2, 3, transform.n_nodes))
    a = transform(u)
    assert a.shape == u.shape
    rows = jnp.stack([transform.to_spectral(u[0, i]) for i in range(3)])
    assert jnp.allclose(a[0], rows, atol=1e-14)
    assert jnp.allclose(transform(a, inverse=True), u)
    assert transform(u.astype(jnp.float32)).dtype == jnp.float32
    # integers become float32
    assert transform(jnp.arange(transform.n_nodes)).dtype == jnp.float32
    grid = ChebyshevTransform2D(3, 2, kind=kind)
    field = jnp.ones((4, *grid.shape), dtype=jnp.float32)
    assert grid(field).dtype == jnp.float32
    assert grid(grid(field), inverse=True).shape == (4, *grid.shape)


@pytest.mark.parametrize("kind", KINDS)
def test_transforms_trace(kind: str) -> None:
    transform = ChebyshevTransform1D(6, kind=kind)
    u = jax.random.normal(jax.random.key(5), (3, transform.n_nodes))
    assert_traces(transform.to_spectral, u)
    assert_traces(transform.from_spectral, u)
    grid = ChebyshevTransform2D(4, 3, kind=kind)
    field = jax.random.normal(jax.random.key(6), (3, *grid.shape))
    assert_traces(grid.to_spectral, field)
    assert_traces(grid.from_spectral, field)


@pytest.mark.parametrize("kind", KINDS)
def test_transform_gradients_match_finite_differences(kind: str) -> None:
    transform = ChebyshevTransform1D(5, kind=kind)
    u = jax.random.normal(jax.random.key(7), (transform.n_nodes,))
    assert_grad_matches_fd(transform.to_spectral, u)
    assert_grad_matches_fd(transform.from_spectral, u)
