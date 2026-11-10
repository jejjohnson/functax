"""Tests for `functax.diffops`, copied from geonnax be5537d with functax imports, the
renames grad → gradient and jacobian → jacobian_field, explicit units, and a sine
MLP (`eqx.nn.MLP`) in place of geonnax's SIREN network as the test field. The fast
tier: closed forms, shapes and validation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from functax import diffops


def test_divergence_and_curl_shape_errors():
    with pytest.raises(ValueError, match="divergence needs"):
        diffops.divergence(lambda x: x[:1])(jnp.zeros(2))
    with pytest.raises(ValueError, match="curl2d needs"):
        diffops.curl2d(lambda x: x)(jnp.zeros(3))
    with pytest.raises(ValueError, match="single point"):
        diffops.gradient(jnp.sum)(jnp.zeros((4, 2)))


def test_integer_points_are_promoted():
    np.testing.assert_allclose(
        diffops.gradient(lambda x: jnp.sum(x**2))(jnp.array([1, 2])),
        jnp.array([2.0, 4.0]),
    )


def test_sphere_operators_validate_input():
    with pytest.raises(ValueError, match="3-D vector"):
        diffops.surface_grad(jnp.sum)(jnp.zeros(2))
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        diffops.surface_grad_enu(jnp.sum, unit="degrees")(jnp.zeros(3))
