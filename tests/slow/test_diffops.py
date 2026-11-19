"""Tests for `functax.diffops`, copied from geonnax be5537d with functax imports, the
renames grad → gradient and jacobian → jacobian_field, explicit units, and a sine
MLP (`eqx.nn.MLP`) in place of geonnax's SIREN network as the test field. The slow
tier: spherical-harmonic eigenvalues, network fields, and the jit, vmap and grad
check, each a second or more.
"""

from __future__ import annotations

import equinox as eqx
import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from hypothesis import given, settings

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
        assert_matches_oracle(diffops.divergence(u)(xy), 0.0, rtol=0, atol=1e-5)
        w = col(curlfree_basis, j)
        assert_matches_oracle(diffops.curl2d(w)(xy), 0.0, rtol=0, atol=1e-5)


@pytest.mark.parametrize("l_max", [10])
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
    # gate of #22: 1e-10 · max(1, |λ|) in float64 (geonnax allowed 1e-5·|λ| for
    # float32); dividing both sides by max(1, |λ|) makes that one absolute bound
    scale = jnp.maximum(1.0, -lam)[:, None]
    assert_matches_oracle(lb / scale, lam[:, None] * y / scale, rtol=0, atol=1e-10)


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


def _waves(params: tuple[jax.Array, jax.Array, jax.Array]):  # type: ignore[no-untyped-def]
    """f(x) = ∑ₖ aₖ sin(wₖ · x + φₖ), a smooth field with known derivatives."""
    amplitude, wavenumber, phase = params

    def field(x: jax.Array) -> jax.Array:
        return jnp.sum(amplitude * jnp.sin(wavenumber @ x + phase))

    return field


def _random_waves(seed: int, dim: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Four waves with amplitudes, wavenumbers and phases drawn from the seed."""
    rng = np.random.default_rng(seed)
    return (
        jnp.asarray(rng.normal(size=4)),
        jnp.asarray(rng.normal(size=(4, dim))),
        jnp.asarray(rng.uniform(0.0, 2.0 * np.pi, 4)),
    )


def _random_rotation(seed: int) -> jax.Array:
    """A random proper rotation, from the QR factors of a Gaussian matrix."""
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return jnp.asarray(q if np.linalg.det(q) > 0 else -q)


@jax.jit
def _plane_identities(params, x):  # type: ignore[no-untyped-def]
    """curl ∇f, ∇·∇f, Δf, ∇·(∇f)^⊥ and curl (∇f)^⊥ at x, with (u, v)^⊥ = (−v, u)."""
    f = _waves(params)
    grad = diffops.gradient(f)

    def perp(p: jax.Array) -> jax.Array:
        u = grad(p)
        return jnp.stack([-u[1], u[0]])

    return jnp.stack(
        [
            diffops.curl2d(grad)(x),
            diffops.divergence(grad)(x),
            diffops.laplacian(f)(x),
            diffops.divergence(perp)(x),
            diffops.curl2d(perp)(x),
        ]
    )


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_plane_operators_satisfy_the_vector_identities(seed: int) -> None:
    # curl ∇f = 0, ∇·(∇f)^⊥ = 0, and ∇·∇f = curl (∇f)^⊥ = Δf = −∑ aₖ|wₖ|² sin(…)
    params = _random_waves(seed, 2)
    x = jnp.asarray(np.random.default_rng(seed).uniform(-2.0, 2.0, 2))
    curl_grad, div_grad, lap, div_perp, curl_perp = _plane_identities(params, x)
    amplitude, wavenumber, phase = params
    exact = -jnp.sum(
        amplitude * jnp.sum(wavenumber**2, axis=1) * jnp.sin(wavenumber @ x + phase)
    )
    scale = float(jnp.sum(jnp.abs(amplitude) * jnp.sum(wavenumber**2, axis=1)))
    for value in (curl_grad, div_perp):
        assert_matches_oracle(value, 0.0, rtol=0, atol=1e-14 * scale)
    for value in (div_grad, lap, curl_perp):
        assert_matches_oracle(value, exact, rtol=0, atol=1e-14 * scale)


@jax.jit
def _jacobian_checks(params, x):  # type: ignore[no-untyped-def]
    """The Jacobian of ∇f, and the Jacobian and divergence of a vector field u."""
    f = _waves(params)

    def u(p: jax.Array) -> jax.Array:
        return jnp.stack([f(p), f(p[::-1]), f(jnp.roll(p, 1)) * p[0]])

    hessian = diffops.jacobian_field(diffops.gradient(f))(x)
    return hessian, diffops.jacobian_field(u)(x), diffops.divergence(u)(x)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_jacobian_field_is_symmetric_on_a_gradient_with_the_divergence_as_trace(
    seed: int,
) -> None:
    # J(∇f) = ∇²f is symmetric, and tr J(u) = ∇·u
    params = _random_waves(seed, 3)
    x = jnp.asarray(np.random.default_rng(seed).uniform(-2.0, 2.0, 3))
    hessian, jacobian, divergence = _jacobian_checks(params, x)
    scale = float(jnp.max(jnp.abs(hessian)))
    assert_matches_oracle(hessian, hessian.T, rtol=0, atol=1e-14 * scale)
    assert_matches_oracle(jnp.trace(jacobian), divergence, rtol=1e-13, atol=1e-14)


@jax.jit
def _sphere_rotation_checks(params, rotation, n):  # type: ignore[no-untyped-def]
    """∇_S and Δ_S of f at n, and of f ∘ Qᵀ at Q n."""
    f = _waves(params)

    def rotated(m: jax.Array) -> jax.Array:
        return f(rotation.T @ m)

    return (
        diffops.surface_grad(f)(n),
        diffops.surface_grad(rotated)(rotation @ n),
        diffops.laplace_beltrami(f)(n),
        diffops.laplace_beltrami(rotated)(rotation @ n),
    )


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_sphere_operators_commute_with_rotations(seed: int) -> None:
    # ∇_S(f ∘ Qᵀ)(Q n) = Q ∇_S f(n) and Δ_S(f ∘ Qᵀ)(Q n) = Δ_S f(n)
    params = _random_waves(seed, 3)
    rotation = _random_rotation(seed)
    n = _unit_points(1, jr.PRNGKey(seed))[0]
    grad, grad_rotated, lap, lap_rotated = _sphere_rotation_checks(params, rotation, n)
    scale = float(jnp.max(jnp.abs(grad))) + 1.0
    assert_matches_oracle(grad_rotated, rotation @ grad, rtol=0, atol=1e-13 * scale)
    assert_matches_oracle(lap_rotated, lap, rtol=1e-12, atol=1e-13 * (abs(lap) + 1))


@jax.jit
def _enu_check(params, lonlat):  # type: ignore[no-untyped-def]
    """surface_grad_enu at (λ, φ), and the east/north rows times surface_grad."""
    f = _waves(params)
    n = lonlat_to_unit(lonlat[None], unit="radians")[0]
    frame = enu_basis(lonlat[None], unit="radians")[0]
    return diffops.surface_grad_enu(f, unit="radians")(lonlat), frame[:2] @ (
        diffops.surface_grad(f)(n)
    )


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_surface_grad_enu_is_the_surface_gradient_in_the_local_frame(
    seed: int,
) -> None:
    # (e_eastᵀ ∇_S f, e_northᵀ ∇_S f) at any (λ, φ) off the poles
    rng = np.random.default_rng(seed)
    lonlat = jnp.asarray([rng.uniform(-np.pi, np.pi), rng.uniform(-1.4, 1.4)])
    enu, projected = _enu_check(_random_waves(seed, 3), lonlat)
    scale = float(jnp.max(jnp.abs(projected))) + 1.0
    assert_matches_oracle(enu, projected, rtol=0, atol=1e-14 * scale)


def test_curl2d_of_a_rotational_field() -> None:
    # u = (−y eˣ, x cos y): curl u = ∂ₓ(x cos y) − ∂ᵧ(−y eˣ) = cos y + eˣ
    def u(p: jax.Array) -> jax.Array:
        return jnp.stack([-p[1] * jnp.exp(p[0]), p[0] * jnp.cos(p[1])])

    points = jnp.array([[0.3, -1.2], [-0.7, 0.4], [1.5, 2.0]])
    curl = jax.vmap(diffops.curl2d(u))(points)
    expected = jnp.cos(points[:, 1]) + jnp.exp(points[:, 0])
    assert_matches_oracle(curl, expected, rtol=1e-14)


_FLAT_FIELD = _waves(_random_waves(7, 3))


def _flat_vector(p: jax.Array) -> jax.Array:
    return jnp.stack([_FLAT_FIELD(p), _FLAT_FIELD(p[::-1]), p[0] * _FLAT_FIELD(p)])


def _plane_vector(p: jax.Array) -> jax.Array:
    return jnp.stack([jnp.sin(p[0] * p[1]), jnp.cos(p[0] - 2.0 * p[1])])


OPERATORS = {
    "gradient": (diffops.gradient(_FLAT_FIELD), 3),
    "jacobian_field": (diffops.jacobian_field(_flat_vector), 3),
    "divergence": (diffops.divergence(_flat_vector), 3),
    "curl2d": (diffops.curl2d(_plane_vector), 2),
    "laplacian": (diffops.laplacian(_FLAT_FIELD), 3),
    "surface_grad": (diffops.surface_grad(_FLAT_FIELD), 3),
    "surface_grad_enu": (diffops.surface_grad_enu(_FLAT_FIELD, unit="radians"), 2),
    "laplace_beltrami": (diffops.laplace_beltrami(_FLAT_FIELD), 3),
}


@pytest.mark.parametrize("name", OPERATORS)
def test_operator_traces_and_its_gradient_matches_finite_differences(
    name: str,
) -> None:
    # differentiated in the evaluation point (off the origin and the poles), so
    # the check reaches one derivative past the operator's own
    operator, dim = OPERATORS[name]
    points = jnp.asarray(np.random.default_rng(8).uniform(0.2, 1.2, (4, dim)))
    assert_traces(operator, points)
    assert_grad_matches_fd(operator, points[0])
