"""Tests for `ChebyshevTransform1D` and `ChebyshevTransform2D`, copied from
spectraldiffx d68f8f6 (``test_chebyshev_new_features.py``, ``test_chebyshev_grid.py``
and ``test_chebyshev_extensions.py``) with functax imports, plus the testing list
of #46. The fast tier: closed forms, unit vectors and validation.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import pytest
from _harness import assert_matches_oracle

from functax import ValidationError
from functax.geometry import chebyshev_nodes
from functax.transform import ChebyshevTransform1D, ChebyshevTransform2D


KINDS = ["lobatto", "gauss"]


def _chebyshev(k: int, x: jax.Array) -> jax.Array:
    """Tₖ(x) = cos(k arccos x), for x in [−1, 1]."""
    return jnp.cos(k * jnp.arccos(x))


# --- spectraldiffx's transform tests -------------------------------------------


def test_transform1d_roundtrip_lobatto() -> None:
    x = chebyshev_nodes(32)
    transform = ChebyshevTransform1D(32)
    u = jnp.sin(jnp.pi * x) + 0.25 * jnp.cos(3 * jnp.pi * x)
    a = transform.to_spectral(u)
    assert jnp.allclose(transform.from_spectral(a), u, atol=1e-12)


def test_lobatto_constant_is_the_unit_mode() -> None:
    a = ChebyshevTransform1D(12).to_spectral(3.0 * jnp.ones(13))
    assert jnp.allclose(a, jnp.zeros(13).at[0].set(3.0), atol=1e-13)


def test_gauss_transform_preserves_float64() -> None:
    transform = ChebyshevTransform1D(8, kind="gauss")
    u = _chebyshev(3, chebyshev_nodes(8, kind="gauss"))
    a = transform.to_spectral(u)
    assert a.dtype == jnp.float64
    assert jnp.abs(a[3] - 1.0) < 1e-13
    assert transform.from_spectral(a).dtype == jnp.float64


# --- the testing list of #46 ---------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 23])
def test_transform_of_t_k_is_the_kth_unit_vector(kind: str, k: int) -> None:
    # also spectraldiffx's test, there for Lobatto with N = 24 (k = 24, the end
    # mode, is checked below, as it exists on the Lobatto nodes only)
    transform = ChebyshevTransform1D(24, kind=kind)
    a = transform.to_spectral(_chebyshev(k, chebyshev_nodes(24, kind=kind)))
    unit = jnp.eye(transform.n_nodes)[k]
    assert_matches_oracle(a, unit, rtol=0.0, atol=1e-13)


def test_transform_of_the_end_mode_on_lobatto_nodes() -> None:
    a = ChebyshevTransform1D(24).to_spectral(_chebyshev(24, chebyshev_nodes(24)))
    assert_matches_oracle(a, jnp.eye(25)[24], rtol=0.0, atol=1e-13)


# --- behaviour ---------------------------------------------------------------


def test_shapes_and_sizes() -> None:
    assert ChebyshevTransform1D(4).n_nodes == 5
    assert ChebyshevTransform1D(4, kind="gauss").n_nodes == 4
    assert ChebyshevTransform2D(4, 2).shape == (3, 5)
    assert ChebyshevTransform2D(4, 2, kind="gauss").shape == (2, 4)


@pytest.mark.parametrize("n", [0, -2, True])
def test_degrees_must_be_positive_integers(n: object) -> None:
    with pytest.raises(ValidationError, match="'n' must be an integer of at least 1"):
        ChebyshevTransform1D(n)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValidationError, match="'ny' must be an integer"):
        ChebyshevTransform2D(3, n)  # ty: ignore[invalid-argument-type]


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValidationError, match="'kind' must be one of"):
        ChebyshevTransform1D(4, kind="gauss-lobatto")


def test_values_must_fit_the_nodes() -> None:
    transform = ChebyshevTransform1D(4)
    to_spectral = inspect.unwrap(ChebyshevTransform1D.to_spectral)
    from_spectral = inspect.unwrap(ChebyshevTransform1D.from_spectral)
    with pytest.raises(
        ValidationError, match=r"'u' must have trailing axes of shape \(5,\)"
    ):
        to_spectral(transform, jnp.ones(4))
    with pytest.raises(ValidationError, match="'a' must have trailing axes"):
        from_spectral(transform, jnp.ones(()))
    with pytest.raises(ValidationError, match="'u' must be real"):
        to_spectral(transform, jnp.ones(5, dtype=jnp.complex128))
    grid = ChebyshevTransform2D(2, 3)
    with pytest.raises(
        ValidationError, match=r"'u' must have trailing axes of shape \(4, 3\)"
    ):
        inspect.unwrap(ChebyshevTransform2D.to_spectral)(grid, jnp.ones((3, 4)))
