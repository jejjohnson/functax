"""Moving particles: an interpolated wind read inside a 1,000-step lax.scan traces once
with no host callback, and gradients reach the initial positions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from _harness import assert_matches_oracle

from functax.geometry import Axis, RectilinearGrid
from functax.interp import (
    BSpline,
    Linear,
    Policies,
)


GRID = RectilinearGrid(
    (Axis(jnp.linspace(0.0, 5.0, 8)), Axis(jnp.linspace(-1.0, 2.0, 9)))
)


FIELD = jax.random.normal(jax.random.key(0), (72,))


def _targets(count: int, seed: int) -> jax.Array:
    rng = np.random.default_rng(seed)
    return jnp.asarray(
        np.column_stack([rng.uniform(-0.5, 5.5, count), rng.uniform(-1.3, 2.3, count)])
    )


class TestAdvection:
    def test_a_long_scan_traces_once_and_has_no_callbacks(self) -> None:
        # 10⁴ particles advected for 1,000 steps by a steady 2-D wind
        wind = Linear().fit(
            GRID,
            jnp.stack([jnp.ones(72), 0.5 * jnp.ones(72)]),
            policies=Policies(outside="clamp"),
        )
        traces = []

        def step(positions: jax.Array, _: None) -> tuple[jax.Array, None]:
            traces.append(1)
            # u(x): (2, M), one row per wind component
            velocity = wind(positions)
            return positions + 1e-3 * jnp.stack(
                [velocity[0], velocity[1]], axis=-1
            ), None

        def run(initial: jax.Array) -> jax.Array:
            final, _ = jax.lax.scan(step, initial, None, length=1_000)
            return final

        compiled = jax.jit(run)
        initial = _targets(10_000, seed=6)
        final = compiled(initial).block_until_ready()
        compiled(initial + 0.01).block_until_ready()
        assert len(traces) == 1
        # a steady wind (1, ½) read everywhere (clamped outside): 1,000 steps
        # of 10⁻³ move every particle by exactly (1, ½)
        assert_matches_oracle(
            final, initial + jnp.array([1.0, 0.5]), rtol=1e-9, atol=1e-9
        )
        jaxpr = str(jax.make_jaxpr(run)(initial))
        assert "callback" not in jaxpr

    def test_gradient_of_the_final_positions_is_finite(self) -> None:
        wind = BSpline(3).fit(GRID, jnp.stack([jnp.sin(FIELD), jnp.cos(FIELD)]))

        def advect(initial: jax.Array) -> jax.Array:
            def step(positions: jax.Array, _: None) -> tuple[jax.Array, None]:
                velocity = wind(positions)
                return positions + 1e-2 * jnp.stack(
                    [velocity[0], velocity[1]], axis=-1
                ), None

            final, _ = jax.lax.scan(step, initial, None, length=20)
            return jnp.sum(final)

        gradient = jax.jit(jax.grad(advect))(_targets(16, seed=7) * 0.5 + 1.0)
        assert bool(jnp.all(jnp.isfinite(gradient)))
