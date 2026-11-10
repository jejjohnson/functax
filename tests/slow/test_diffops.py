"""Tests for `functax.diffops`, copied from geonnax be5537d with functax imports, the
renames grad → gradient and jacobian → jacobian_field, explicit units, and a sine
MLP (`eqx.nn.MLP`) in place of geonnax's SIREN network as the test field. The slow
tier: spherical-harmonic eigenvalues, network fields, and the jit, vmap and grad
check, each a second or more.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from functax import diffops
from functax.basis.sphere import real_spherical_harmonics
from functax.basis.vector import curlfree_basis, divfree_basis
from functax.geometry import enu_basis, lonlat_to_unit


def _sine_mlp(
    in_size: int, width: int, out_size: int, *, depth: int, key: jax.Array
) -> eqx.nn.MLP:
    """A small sine-activated network, the smooth test field geonnax took from SIREN."""
    return eqx.nn.MLP(in_size, out_size, width, depth, activation=jnp.sin, key=key)


KEY = jr.PRNGKey(0)


def _unit_points(n: int, key: jax.Array) -> jax.Array:
    """Isotropic random unit vectors."""
    v = jr.normal(key, (n, 3))
    return v / jnp.linalg.norm(v, axis=-1, keepdims=True)


def test_laplacian_matches_dense_hessian_trace():
    net = _sine_mlp(3, 32, 1, depth=3, key=KEY)
    f = lambda x: net(x)[0]
    x = jnp.array([0.1, -0.2, 0.3])
    np.testing.assert_allclose(
        # Same second derivatives by two float32 routes (HVP sum vs dense trace).
        diffops.laplacian(f)(x),
        jnp.trace(jax.hessian(f)(x)),
        rtol=1e-5,
        atol=1e-4,
    )


def test_divfree_and_curlfree_columns():
    # Divergence-free atoms have ∇·u = 0; curl-free atoms have curl u = 0.
    xy = jnp.array([0.31, 0.62])

    def col(fn, j):
        return lambda p: fn(p[None, :], 3, 1.0)[0][0, j, :]

    # Analytically zero; float32 round-off on Jacobian entries of O(10) is ~1e-6.
    for j in range(4):
        u = col(divfree_basis, j)
        assert abs(float(diffops.divergence(u)(xy))) < 1e-5
        w = col(curlfree_basis, j)
        assert abs(float(diffops.curl2d(w)(xy))) < 1e-5


@pytest.mark.parametrize("l_max", [4, 10])
def test_laplace_beltrami_spherical_harmonic_eigenvalues(l_max):
    # Δ_S Y_lm = -l(l+1) Y_lm for every real spherical harmonic.
    pts = _unit_points(16, KEY)
    m = (l_max + 1) ** 2
    degrees = jnp.concatenate(
        [jnp.full((2 * l + 1,), l, dtype=jnp.float64) for l in range(l_max + 1)]
    )

    # One trace for all harmonics: select Y_j with a one-hot weight vector.
    def lb_j(w, n):
        f = lambda n: real_spherical_harmonics(n[None, :], l_max)[0] @ w
        return diffops.laplace_beltrami(f)(n)

    eye = jnp.eye(m)
    lb = jax.jit(jax.vmap(jax.vmap(lb_j, (None, 0)), (0, None)))(eye, pts)  # (M, N)
    y = real_spherical_harmonics(pts, l_max).T  # (M, N)
    lam = -degrees * (degrees + 1)
    # gate of #22: 1e-10 in float64 (geonnax allowed 1e-5·|λ| for float32)
    atol = 1e-10 * jnp.maximum(1.0, -lam)[:, None]
    assert bool(jnp.all(jnp.abs(lb - lam[:, None] * y) <= atol))


def test_laplace_beltrami_general_extension_identity():
    # For any smooth extension f, Δ_S f = Δf − nᵀ∇²f n − 2 nᵀ∇f on |n| = 1.
    net = _sine_mlp(3, 32, 1, depth=3, key=jr.PRNGKey(1))
    f = lambda x: net(x)[0]
    for n in _unit_points(5, jr.PRNGKey(2)):
        H = jax.hessian(f)(n)
        rhs = jnp.trace(H) - n @ H @ n - 2 * n @ jax.grad(f)(n)
        np.testing.assert_allclose(
            # float32: both sides from the same second derivatives; measured ≤ 3e-7.
            diffops.laplace_beltrami(f)(n),
            rhs,
            rtol=1e-5,
            atol=1e-5,
        )


def test_surface_grad_is_tangent_and_matches_projection():
    net = _sine_mlp(3, 32, 1, depth=3, key=jr.PRNGKey(3))
    f = lambda x: net(x)[0]
    pts = _unit_points(32, jr.PRNGKey(4))
    sg = jax.vmap(diffops.surface_grad(f))(pts)
    np.testing.assert_allclose(jnp.sum(sg * pts, axis=-1), 0.0, atol=1e-5)
    # Equals the tangential projection of the Euclidean gradient of f itself.
    eg = jax.vmap(jax.grad(f))(pts)
    proj = eg - pts * jnp.sum(eg * pts, axis=-1, keepdims=True)
    # float32: |∇f| ~ 10 for this SIREN, so compare relative to that scale.
    np.testing.assert_allclose(sg, proj, rtol=1e-5, atol=1e-4)


def test_surface_grad_enu_of_latitude_and_longitude_fields():
    lonlat = jnp.array([[10.0, 20.0], [-120.0, -45.0], [75.0, 60.0]])
    rad = jnp.deg2rad(lonlat)
    # f = sin(lat): ∂/∂north = cos(lat), no east component.
    d_lat = jax.vmap(diffops.surface_grad_enu(lambda n: n[2], unit="degrees"))(lonlat)
    np.testing.assert_allclose(d_lat[:, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(d_lat[:, 1], jnp.cos(rad[:, 1]), atol=1e-6)
    # f = x = cos(lat)cos(lon): east = -sin(lon), north = -sin(lat)cos(lon).
    d_x = jax.vmap(diffops.surface_grad_enu(lambda n: n[0], unit="degrees"))(lonlat)
    np.testing.assert_allclose(d_x[:, 0], -jnp.sin(rad[:, 0]), atol=1e-6)
    np.testing.assert_allclose(
        d_x[:, 1], -jnp.sin(rad[:, 1]) * jnp.cos(rad[:, 0]), atol=1e-6
    )


def test_surface_grad_enu_radians_matches_degrees():
    f = lambda n: n[0] * n[1] + n[2] ** 2
    ll = jnp.array([33.0, -12.0])
    np.testing.assert_allclose(
        diffops.surface_grad_enu(f, unit="radians")(jnp.deg2rad(ll)),
        diffops.surface_grad_enu(f, unit="degrees")(ll),
        atol=1e-6,
    )
    # Consistent with the Cartesian surface gradient through enu_basis.
    n = lonlat_to_unit(ll[None], unit="degrees")[0]
    frame = enu_basis(ll[None], unit="degrees")[0]
    np.testing.assert_allclose(
        diffops.surface_grad_enu(f, unit="degrees")(ll),
        frame[:2] @ diffops.surface_grad(f)(n),
        atol=1e-6,
    )


def test_sphere_operators_are_finite_at_the_origin():
    # Padded rows of zeros must not poison gradients through a masked mean.
    net = _sine_mlp(3, 8, 1, depth=2, key=KEY)

    def loss(net):
        f = lambda n: net(n)[0]
        pts = jnp.stack([jnp.array([0.0, 0.0, 1.0]), jnp.zeros(3)])
        lb = jax.vmap(diffops.laplace_beltrami(f))(pts)
        return jnp.sum(jnp.where(jnp.array([True, False]), lb, 0.0))

    grads = eqx.filter_grad(loss)(net)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    assert bool(
        jnp.all(jnp.isfinite(diffops.surface_grad(lambda n: n[0])(jnp.zeros(3))))
    )


def test_operators_jit_vmap_and_differentiate():
    net = _sine_mlp(3, 16, 1, depth=2, key=jr.PRNGKey(5))
    pts = _unit_points(64, jr.PRNGKey(6))

    # filter_jit: the MLP carries its activation function as a static field
    @eqx.filter_jit
    def residual(params_net, pts):
        f = lambda n: params_net(n)[0]
        return jnp.mean(jax.vmap(diffops.laplace_beltrami(f))(pts) ** 2)

    val = residual(net, pts)
    assert jnp.isfinite(val)
    grads = eqx.filter_grad(residual)(net, pts)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_laplacian_of_plane_wave(d):
    # Δ sin(2π k·x) = -4π²|k|² sin(2π k·x), exactly.
    k = jnp.arange(1.0, d + 1.0)
    x = jnp.linspace(0.1, 0.4, d)
    f = lambda x: jnp.sin(2 * jnp.pi * jnp.vdot(k, x))
    expected = -4 * jnp.pi**2 * jnp.sum(k**2) * f(x)
    # float32: the Hessian diagonal is O(4π²|k|²) ~ 550 for d = 3; rtol 1e-4.
    np.testing.assert_allclose(diffops.laplacian(f)(x), expected, rtol=1e-4)


def test_sphere_operators_use_the_input_direction():
    # Off the unit sphere the operators act at n / |n|, not at a rescaled field.
    lb = diffops.laplace_beltrami(lambda n: n[0])
    sg = diffops.surface_grad(lambda n: n[2])
    n = jnp.array([0.6, 0.0, 0.8])
    for r in (0.5, 1.01, 2.0):
        np.testing.assert_allclose(lb(r * n), lb(n), rtol=1e-5)
        np.testing.assert_allclose(sg(r * n), sg(n), atol=1e-6)


def test_divergence_of_the_gradient_is_the_laplacian():
    # gate of #22, on a polynomial field
    def field(x):
        return x[0] ** 3 * x[1] - 2 * x[1] ** 2 * x[2] + x[0] * x[2] ** 4

    point = jnp.array([0.4, -1.3, 0.7])
    divergence_of_gradient = diffops.divergence(diffops.gradient(field))(point)
    np.testing.assert_allclose(
        divergence_of_gradient, diffops.laplacian(field)(point), rtol=1e-12
    )


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
