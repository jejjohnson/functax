"""Conservative remapping on tensor grids (#40): conservation, block means, adjoint.

Over a second each, mostly einx compiling each new shape in an eager call.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import (
    assert_adjoint,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
    assert_transpose_matches,
)

from functax.geometry import Axis, RectilinearGrid
from functax.interp import Conservative, Policies


def random_edges(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    inner = np.sort(rng.uniform(lo, hi, n - 1))
    return np.concatenate([[lo], inner, [hi]])


def axis_from_edges(edges: np.ndarray, *, period: float | None = None) -> Axis:
    centres = 0.5 * (edges[:-1] + edges[1:])
    return Axis(jnp.asarray(centres), edges=jnp.asarray(edges), period=period)


def grid_from_edges(*edges: np.ndarray) -> RectilinearGrid:
    return RectilinearGrid(tuple(axis_from_edges(e) for e in edges))


def cell_areas(*edges: np.ndarray) -> np.ndarray:
    lengths = [np.abs(np.diff(e)) for e in edges]
    return np.prod(np.meshgrid(*lengths, indexing="ij"), axis=0).ravel()


class TestConservation:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_integrals_are_preserved_on_random_rectilinear_grids(
        self, seed: int
    ) -> None:
        # gate G3-conserve: ∑ᵢ |Cᵢ| ŷᵢ = ∑ⱼ |cⱼ| yⱼ when the target covers the source
        rng = np.random.default_rng(seed)
        source = (random_edges(rng, 13, 0.0, 5.0), random_edges(rng, 17, -2.0, 3.0))
        target = (random_edges(rng, 6, 0.0, 5.0), random_edges(rng, 9, -2.0, 3.0))
        plan = Conservative().plan(grid_from_edges(*source), grid_from_edges(*target))
        field = rng.normal(size=(4, 13 * 17))
        remapped = plan(jnp.asarray(field))
        source_integral = field @ cell_areas(*source)
        target_integral = np.asarray(remapped) @ cell_areas(*target)
        assert_matches_oracle(target_integral, source_integral, rtol=1e-12)

    def test_a_wider_target_conserves_under_extrapolate(self) -> None:
        rng = np.random.default_rng(3)
        source = (random_edges(rng, 9, 0.0, 4.0), random_edges(rng, 11, 0.0, 4.0))
        target = (random_edges(rng, 5, -1.0, 5.0), random_edges(rng, 4, -1.0, 5.0))
        plan = Conservative().plan(
            grid_from_edges(*source),
            grid_from_edges(*target),
            policies=Policies(outside="extrapolate", fill_value=0.0),
        )
        field = rng.normal(size=9 * 11)
        remapped = np.asarray(plan(jnp.asarray(field)))
        assert_matches_oracle(
            remapped @ cell_areas(*target), field @ cell_areas(*source), rtol=1e-12
        )

    @pytest.mark.parametrize("factor", [2, 4])
    def test_integer_factor_coarsening_is_a_block_mean_exactly(
        self, factor: int
    ) -> None:
        # dyadic edges and integer data: every product and sum is exact
        fine = RectilinearGrid(
            (Axis(jnp.arange(8.0) + 0.5), Axis(jnp.arange(16.0) + 0.5))
        )
        coarse = RectilinearGrid(
            (
                Axis(jnp.arange(8.0 / factor) * factor + factor / 2),
                Axis(jnp.arange(16.0 / factor) * factor + factor / 2),
            )
        )
        rng = np.random.default_rng(4)
        field = rng.integers(-50, 50, size=(8, 16)).astype(np.float64)
        plan = Conservative().plan(fine, coarse)
        remapped = plan(jnp.asarray(field.ravel()))
        block_mean = field.reshape(8 // factor, factor, 16 // factor, factor).mean(
            axis=(1, 3)
        )
        np.testing.assert_array_equal(np.asarray(remapped), block_mean.ravel())
        assert plan.read.n_taps == factor * factor

    def test_a_factor_of_three_is_a_block_mean_to_rounding(self) -> None:
        fine = Axis(jnp.arange(9.0) + 0.5)
        coarse = Axis(jnp.array([1.5, 4.5, 7.5]))
        field = np.random.default_rng(5).normal(size=9)
        remapped = Conservative().plan(fine, coarse)(jnp.asarray(field))
        # 1/3 is not exact, so the weights round once per tap
        assert_matches_oracle(remapped, field.reshape(3, 3).mean(axis=1), rtol=1e-15)

    def test_refining_then_coarsening_returns_the_field(self) -> None:
        rng = np.random.default_rng(6)
        coarse_edges = (random_edges(rng, 5, 0.0, 1.0), random_edges(rng, 4, 0.0, 1.0))
        # every coarse edge is a fine edge, so each fine cell sits in one coarse cell
        fine_edges = tuple(
            np.unique(np.concatenate([e, rng.uniform(0.0, 1.0, 9)]))
            for e in coarse_edges
        )
        coarse = grid_from_edges(*coarse_edges)
        fine = grid_from_edges(*fine_edges)
        field = jnp.asarray(rng.normal(size=5 * 4))
        refined = Conservative().plan(coarse, fine)(field)
        back = Conservative().plan(fine, coarse)(refined)
        assert_matches_oracle(back, field, rtol=1e-13)


class TestAdjoint:
    def test_dot_product(self) -> None:
        rng = np.random.default_rng(7)
        source = grid_from_edges(
            random_edges(rng, 7, 0.0, 3.0), random_edges(rng, 9, 0.0, 3.0)
        )
        target = grid_from_edges(
            random_edges(rng, 4, -0.5, 3.5), random_edges(rng, 5, 0.5, 2.5)
        )
        plan = Conservative().plan(source, target)
        assert_adjoint(
            plan, plan.adjoint, 7 * 9, 4 * 5, key=jax.random.key(0), rtol=1e-12
        )
        assert_transpose_matches(plan, plan.adjoint, jnp.zeros(7 * 9))


class TestGeometries:
    def test_a_periodic_source_wraps_across_the_seam(self) -> None:
        longitude = Axis(jnp.arange(5.0, 360.0, 10.0), period=360.0)
        shifted = Axis(jnp.array([0.0, 180.0]), edges=jnp.array([-10.0, 10.0, 350.0]))
        field = jnp.arange(36.0)
        remapped = Conservative().plan(longitude, shifted)(field)
        # [−10, 10] reads the cells at 355° and 5°, [10, 350] the 34 between
        assert_matches_oracle(remapped, [17.5, 17.5], rtol=1e-15)

    def test_the_plan_windows_a_grid_target(self) -> None:
        source = RectilinearGrid((Axis(jnp.arange(8.0)),) * 2)
        target = RectilinearGrid((Axis(jnp.arange(0.5, 7.0, 2.0)),) * 2)
        plan = Conservative().plan(source, target)
        field = jnp.arange(64.0)
        window = plan[1:3, 0:2](field)
        expected = einx.id("(y x) -> y x", plan(field), y=4)[1:3, 0:2]
        np.testing.assert_array_equal(np.asarray(window), np.asarray(expected).ravel())


def test_jit_and_vmap_agree_with_a_loop() -> None:
    rng = np.random.default_rng(10)
    source = grid_from_edges(
        random_edges(rng, 5, 0.0, 1.0), random_edges(rng, 6, 0.0, 1.0)
    )
    target = grid_from_edges(
        random_edges(rng, 3, 0.0, 1.0), random_edges(rng, 4, 0.0, 1.0)
    )
    plan = Conservative().plan(source, target, policies=Policies(missing="renormalise"))
    fields = rng.normal(size=(3, 30))
    fields[rng.uniform(size=fields.shape) < 0.2] = np.nan
    assert_traces(lambda field: plan(field), jnp.asarray(fields))


def test_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(11)
    source = grid_from_edges(
        random_edges(rng, 4, 0.0, 1.0), random_edges(rng, 5, 0.0, 1.0)
    )
    target = grid_from_edges(
        random_edges(rng, 3, 0.0, 1.0), random_edges(rng, 2, 0.0, 1.0)
    )
    plan = Conservative().plan(source, target)
    assert_grad_matches_fd(plan, jnp.asarray(rng.normal(size=20)))
