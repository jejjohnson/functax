"""Tests for `functax.quad.polyline`. The slow tier: the rule traces (eager, `jit`
and `vmap` agree) and differentiates in the vertices (`jacrev` against central
differences), so a transect can move under an optimiser.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_traces

from functax.quad import polyline


def _flux(vertices: jnp.ndarray, rule: str) -> jnp.ndarray:
    """Φ = ∑ₖ Δsₖ c(xₖ) (u(xₖ) · nₖ) for a smooth c and a rotating wind u."""
    transect = polyline(vertices, 3, rule=rule)
    x, y = transect.nodes[:, 0], transect.nodes[:, 1]
    concentration = jnp.exp(-(x**2) - 0.5 * y**2)
    wind = jnp.stack([1.0 + 0.3 * y, -0.2 * x], axis=-1)
    normal_wind = jnp.sum(wind * transect.normals, axis=-1)
    return jnp.sum(transect.weights * concentration * normal_wind)


@pytest.mark.parametrize("rule", ["midpoint", "gauss_legendre"])
def test_flux_gradient_in_the_vertices_matches_finite_differences(rule: str) -> None:
    vertices = jnp.array([[-1.0, -0.5], [0.2, 0.1], [0.8, 1.2]])
    assert_grad_matches_fd(lambda v: _flux(v, rule), vertices)


@pytest.mark.parametrize("rule", ["midpoint", "gauss_legendre"])
def test_rule_traces_under_jit_and_vmap(rule: str) -> None:
    rng = np.random.default_rng(0)
    batch = jnp.asarray(rng.normal(size=(4, 5, 2)))

    def fields(vertices: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        transect = polyline(vertices, 3, rule=rule)
        return transect.nodes, transect.weights, transect.normals

    assert_traces(fields, batch)
