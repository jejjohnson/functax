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


def test_gradient_and_jacobian_closed_form():
    x = jnp.array([0.3, -1.2])
    np.testing.assert_allclose(
        diffops.gradient(lambda x: x[0] ** 2 * x[1])(x),
        jnp.array([2 * 0.3 * -1.2, 0.3**2]),
        rtol=1e-6,
    )
    J = diffops.jacobian_field(lambda x: jnp.array([jnp.sin(x[0]), x[0] * x[1], x[1]]))(
        x
    )
    assert J.shape == (3, 2)
    np.testing.assert_allclose(
        J, jnp.array([[jnp.cos(0.3), 0.0], [-1.2, 0.3], [0.0, 1.0]]), rtol=1e-6
    )


@pytest.mark.parametrize("d", [1, 2, 3])
def test_laplacian_of_plane_wave(d):
    # Δ sin(2π k·x) = -4π²|k|² sin(2π k·x), exactly.
    k = jnp.arange(1.0, d + 1.0)
    x = jnp.linspace(0.1, 0.4, d)
    f = lambda x: jnp.sin(2 * jnp.pi * jnp.vdot(k, x))
    expected = -4 * jnp.pi**2 * jnp.sum(k**2) * f(x)
    # float32: the Hessian diagonal is O(4π²|k|²) ~ 550 for d = 3; rtol 1e-4.
    np.testing.assert_allclose(diffops.laplacian(f)(x), expected, rtol=1e-4)


def test_divergence_and_curl_shape_errors():
    with pytest.raises(ValueError, match="divergence needs"):
        diffops.divergence(lambda x: x[:1])(jnp.zeros(2))
    with pytest.raises(ValueError, match="curl2d needs"):
        diffops.curl2d(lambda x: x)(jnp.zeros(3))
    with pytest.raises(ValueError, match="single point"):
        diffops.gradient(jnp.sum)(jnp.zeros((4, 2)))


def test_sphere_operators_use_the_input_direction():
    # Off the unit sphere the operators act at n / |n|, not at a rescaled field.
    lb = diffops.laplace_beltrami(lambda n: n[0])
    sg = diffops.surface_grad(lambda n: n[2])
    n = jnp.array([0.6, 0.0, 0.8])
    for r in (0.5, 1.01, 2.0):
        np.testing.assert_allclose(lb(r * n), lb(n), rtol=1e-5)
        np.testing.assert_allclose(sg(r * n), sg(n), atol=1e-6)


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


def test_divergence_of_the_gradient_is_the_laplacian():
    # gate of #22, on a polynomial field
    def field(x):
        return x[0] ** 3 * x[1] - 2 * x[1] ** 2 * x[2] + x[0] * x[2] ** 4

    point = jnp.array([0.4, -1.3, 0.7])
    divergence_of_gradient = diffops.divergence(diffops.gradient(field))(point)
    np.testing.assert_allclose(
        divergence_of_gradient, diffops.laplacian(field)(point), rtol=1e-12
    )
