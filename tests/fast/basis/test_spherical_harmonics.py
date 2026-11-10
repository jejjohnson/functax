"""Tests for `real_spherical_harmonics` and `harmonic_degrees`, copied from geonnax
be5537d with functax imports. The fast tier: closed forms and validation, each well
under a second.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from functax.basis.sphere import real_spherical_harmonics


def _unit_points(n: int, seed: int = 0) -> np.ndarray:
    """``n`` random unit vectors plus both poles and an equator point."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    special = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    return np.concatenate([pts, special], axis=0)


def test_trace_size_independent_of_degree():
    x = jnp.asarray(_unit_points(4), dtype=jnp.float32)

    def n_eqns(l_max):
        return len(jax.make_jaxpr(lambda u: real_spherical_harmonics(u, l_max))(x).eqns)

    assert n_eqns(10) == n_eqns(60)


def test_input_validation():
    with pytest.raises(ValueError, match="l_max"):
        real_spherical_harmonics(jnp.zeros((2, 3)), -1)
    # unwrapped: the type hook would reject the shape before the function's check
    import inspect

    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        inspect.unwrap(real_spherical_harmonics)(jnp.zeros((2, 2)), 2)
