"""Tests for `Plan`, `Nearest`, `Linear` and `Tensor`. The fast tier:
behaviour, validation and small closed-form checks, each well under a second.
"""

from __future__ import annotations

import math

import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import (
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_transpose_matches,
)
from scipy.ndimage import map_coordinates

from functax import GeometryError, OutsideDomainError, PolicyError, ValidationError
from functax.geometry import AffineGrid, Axis, Points, RectilinearGrid
from functax.interp import Fit, Linear, Nearest, Plan, Policies, Tensor


def _rectilinear(
    shape: tuple[int, ...], seed: int
) -> tuple[RectilinearGrid, list[np.ndarray]]:
    """A grid with random, uneven spacing on every axis."""
    rng = np.random.default_rng(seed)
    nodes = [np.cumsum(rng.uniform(0.5, 1.5, size=n)) for n in shape]
    return RectilinearGrid(tuple(Axis(jnp.asarray(values)) for values in nodes)), nodes


def _interior(nodes: list[np.ndarray], count: int, seed: int) -> np.ndarray:
    """Random targets strictly inside every axis."""
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(n[0], n[-1], size=count) for n in nodes])


class TestOracle:
    @pytest.mark.parametrize(
        ("scheme", "order"), [(Linear(), 1), (Nearest(), 0)], ids=["linear", "nearest"]
    )
    @pytest.mark.parametrize(
        "shape", [(9,), (6, 7), (5, 4, 6), (4, 3, 5, 3)], ids=["1d", "2d", "3d", "4d"]
    )
    def test_equals_map_coordinates(
        self, scheme, order: int, shape: tuple[int, ...]
    ) -> None:  # type: ignore[no-untyped-def]
        # integer node coordinates, so grid coordinates are index positions
        grid = RectilinearGrid(tuple(Axis(jnp.arange(float(n))) for n in shape))
        field = np.random.default_rng(30 + len(shape)).normal(size=shape)
        rng = np.random.default_rng(40 + len(shape))
        targets = np.column_stack([rng.uniform(0.0, n - 1.0, size=200) for n in shape])
        plan = scheme.plan(grid, Points(jnp.asarray(targets)))
        expected = map_coordinates(field, targets.T, order=order)
        flat = einx.id("... -> (...)", jnp.asarray(field))
        assert_matches_oracle(plan(flat), expected, rtol=1e-12, atol=1e-14)


class TestAdjoint:
    def test_adjoint_equals_the_lineax_transpose(self) -> None:
        grid, nodes = _rectilinear((4, 5), seed=63)
        plan = Linear().plan(grid, Points(jnp.asarray(_interior(nodes, 12, seed=64))))
        operator = plan.operator()
        assert_transpose_matches(lambda y: operator.mv(y), plan.adjoint, jnp.zeros(20))

    def test_invalid_rows_ignore_their_residual(self) -> None:
        plan = Linear().plan(Axis(jnp.arange(4.0)), Points(jnp.array([[1.5], [9.0]])))
        assert plan.valid.tolist() == [True, False]
        # the NaN the forward pass wrote at the invalid row does not spread
        gradient = plan.adjoint(jnp.array([1.0, jnp.nan]))
        assert gradient.tolist() == [0.0, 0.5, 0.5, 0.0]


class TestTraces:
    def test_grad_in_the_target_positions(self) -> None:
        grid, nodes = _rectilinear((4, 5), seed=76)
        field = jax.random.normal(jax.random.key(77), (20,))
        xy = jnp.asarray(_interior(nodes, 6, seed=78))

        def read(positions: jax.Array) -> jax.Array:
            return Linear().plan(grid, Points(positions))(field)

        assert_grad_matches_fd(read, xy)


class TestSchemes:
    def test_tensor_rejects_bad_schemes(self) -> None:
        with pytest.raises(ValidationError, match="'schemes' must hold one scheme"):
            Tensor(())
        with pytest.raises(
            ValidationError, match=r"'schemes\[0\]' must be a one-axis local scheme"
        ):
            Tensor((Tensor((Linear(),)),))
        grid, _ = _rectilinear((3, 3), seed=94)
        with pytest.raises(
            GeometryError, match="Tensor has 1 schemes but the source has 2 axes"
        ):
            Tensor((Linear(),)).plan(grid, Points(jnp.zeros((1, 2))))

    def test_periodic_longitude_reads_across_the_dateline(self) -> None:
        longitude = Axis(jnp.arange(0.0, 360.0, 10.0), period=360.0)
        field = jnp.arange(36.0)
        plan = Linear().plan(longitude, Points(jnp.array([[355.0], [-5.0], [715.0]])))
        # halfway between node 35 (350°) and node 0 (360° ≡ 0°)
        assert plan(field).tolist() == [17.5, 17.5, 17.5]

    def test_targets_on_grids_and_axes(self) -> None:
        grid, nodes = _rectilinear((3, 4), seed=95)
        field = jax.random.normal(jax.random.key(96), (12,))
        # a grid read at its own nodes is the field itself
        assert_matches_oracle(
            Linear().plan(grid, grid)(field), field, rtol=1e-14, atol=1e-15
        )
        axis = Axis(jnp.asarray(nodes[0]))
        assert_matches_oracle(
            Linear().plan(axis, axis)(field[:3]), field[:3], rtol=1e-14, atol=1e-15
        )
        affine = AffineGrid([2.0, 0.0, 10.0, 0.0, -2.0, 30.0], (3, 4))
        assert_matches_oracle(
            Linear().plan(affine, affine)(field), field, rtol=1e-14, atol=1e-15
        )

    def test_rejects_mismatched_geometries(self) -> None:
        grid, _ = _rectilinear((3, 4), seed=97)
        with pytest.raises(GeometryError, match="'target' has 3 coordinates per point"):
            Linear().plan(grid, Points(jnp.zeros((2, 3))))
        with pytest.raises(GeometryError, match="'source' must be an Axis"):
            Linear().plan(Points(jnp.zeros((2, 2))), Points(jnp.zeros((2, 2))))
        with pytest.raises(GeometryError, match="'target' must be Points"):
            Linear().plan(grid, AffineGrid([1.0, 0.0, 0.0, 0.0, -1.0, 0.0], (2, 2)))


class TestOutside:
    axis = Axis(jnp.arange(4.0))
    field = jnp.array([1.0, 2.0, 4.0, 8.0])
    targets = Points(jnp.array([[-1.0], [1.5], [5.0]]))

    def test_nan_writes_the_fill_value(self) -> None:
        plan = Linear().plan(self.axis, self.targets)
        values = plan(self.field)
        assert bool(jnp.isnan(values[0])) and bool(jnp.isnan(values[2]))
        filled = Linear().plan(
            self.axis, self.targets, policies=Policies(fill_value=-9.0)
        )
        assert filled(self.field).tolist() == [-9.0, 3.0, -9.0]

    def test_clamp_reads_the_boundary(self) -> None:
        plan = Linear().plan(
            self.axis, self.targets, policies=Policies(outside="clamp")
        )
        assert plan(self.field).tolist() == [1.0, 3.0, 8.0]
        assert plan.valid.tolist() == [True, True, True]

    def test_extrapolate_continues_the_edge_cells(self) -> None:
        plan = Linear().plan(
            self.axis, self.targets, policies=Policies(outside="extrapolate")
        )
        # left cell slope 1, right cell slope 4
        assert plan(self.field).tolist() == [0.0, 3.0, 16.0]

    def test_raise(self) -> None:
        with pytest.raises(OutsideDomainError, match="2 of 3 targets"):
            Linear().plan(self.axis, self.targets, policies=Policies(outside="raise"))

    def test_non_finite_targets_are_invalid_under_every_policy(self) -> None:
        targets = Points(jnp.array([[jnp.nan], [1.0]]))
        for outside in ("nan", "clamp", "extrapolate"):
            plan = Linear().plan(self.axis, targets, policies=Policies(outside=outside))
            assert plan.valid.tolist() == [False, True]


class TestMissing:
    axis = Axis(jnp.arange(4.0))
    field = jnp.array([1.0, jnp.nan, 4.0, 8.0])
    targets = Points(jnp.array([[0.25], [2.5], [1.0]]))

    def test_propagate(self) -> None:
        values = Linear().plan(self.axis, self.targets)(self.field)
        assert (
            bool(jnp.isnan(values[0]))
            and values[1] == 6.0
            and bool(jnp.isnan(values[2]))
        )

    def test_fill(self) -> None:
        plan = Linear().plan(
            self.axis, self.targets, policies=Policies(missing="fill", fill_value=0.0)
        )
        assert plan(self.field).tolist() == [0.75, 6.0, 0.0]

    def test_renormalise(self) -> None:
        plan = Linear().plan(
            self.axis, self.targets, policies=Policies(missing="renormalise")
        )
        values = plan(self.field)
        # 0.25: the NaN tap is dropped and the row divided by the weight left
        assert values[:2].tolist() == [1.0, 6.0]
        # 1.0 sits on the missing node, so no weight is left: the fill value
        assert bool(jnp.isnan(values[2]))

    def test_renormalise_with_every_tap_missing_is_invalid(self) -> None:
        field = jnp.array([jnp.nan, jnp.nan, 4.0, 8.0])
        plan = Linear().plan(
            self.axis,
            Points(jnp.array([[0.5], [2.5]])),
            policies=Policies(missing="renormalise", fill_value=-1.0),
        )
        assert plan(field).tolist() == [-1.0, 6.0]

    def test_renormalise_over_a_batch(self) -> None:
        plan = Linear().plan(
            self.axis, self.targets, policies=Policies(missing="renormalise")
        )
        batch = jnp.stack([self.field, jnp.array([1.0, 2.0, 4.0, 8.0])])
        values = plan(batch)
        assert values.shape == (2, 3)
        assert values[1].tolist() == [1.25, 6.0, 2.0]

    def test_linearize_under_renormalise(self) -> None:
        plan = Linear().plan(
            self.axis, self.targets, policies=Policies(missing="renormalise")
        )
        with pytest.raises(PolicyError, match="needs the data"):
            plan.linearize()
        operator = plan.linearize(self.field)
        filled = jnp.nan_to_num(self.field)
        assert operator.mv(filled).tolist() == [1.0, 6.0, 0.0]

    def test_linearize_of_a_linear_plan_is_its_operator(self) -> None:
        plan = Linear().plan(self.axis, self.targets)
        assert plan.linearize().mv(jnp.arange(4.0)).tolist() == [0.25, 2.5, 1.0]

    def test_renormalise_needs_a_local_plan(self) -> None:
        plan = Linear().plan(self.axis, self.targets)

        class Doubled(Fit):
            def apply(self, data):  # type: ignore[no-untyped-def]
                return 2.0 * data

            def apply_adjoint(self, coefficients):  # type: ignore[no-untyped-def]
                return 2.0 * coefficients

        odd = Plan(Doubled(), plan.read, Policies(missing="renormalise"))
        with pytest.raises(PolicyError, match="needs a local plan"):
            odd(self.field)


def test_default_policies_are_nan_and_mirror() -> None:
    plan = Linear().plan(Axis(jnp.arange(3.0)), Points(jnp.zeros((1, 1))))
    assert plan.policies == Policies()
    assert math.isnan(plan.policies.fill_value)
