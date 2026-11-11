"""Tests for `Plan`, `Nearest`, `Linear` and `Tensor`. The fast tier:
behaviour, validation and small closed-form checks, each well under a second.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from functax import GeometryError, OutsideDomainError, PolicyError, ValidationError
from functax.geometry import AffineGrid, Axis, Points, RectilinearGrid
from functax.interp import Cubic, Fit, Linear, Nearest, Plan, Policies, Tensor


def _rectilinear(
    shape: tuple[int, ...], seed: int
) -> tuple[RectilinearGrid, list[np.ndarray]]:
    """A grid with random, uneven spacing on every axis."""
    rng = np.random.default_rng(seed)
    nodes = [np.cumsum(rng.uniform(0.5, 1.5, size=n)) for n in shape]
    return RectilinearGrid(tuple(Axis(jnp.asarray(values)) for values in nodes)), nodes


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


ZERO_TARGET_SOURCES = [
    (Axis(jnp.linspace(0.0, 1.0, 6)), 1),
    (
        RectilinearGrid(
            (Axis(jnp.linspace(0.0, 1.0, 6)), Axis(jnp.linspace(0.0, 1.0, 5)))
        ),
        2,
    ),
]


@pytest.mark.parametrize("scheme", [Nearest(), Linear(), Cubic()])
@pytest.mark.parametrize(("source", "n_dim"), ZERO_TARGET_SOURCES)
def test_a_plan_on_zero_targets_reads_and_sends_back_nothing(
    scheme, source, n_dim: int
) -> None:
    # einx rejects zero-length axes; an empty target set must still plan,
    # as gaussx's interpolation weights did (gaussx#703, gaussx#813)
    plan = scheme.plan(source, Points(jnp.zeros((0, n_dim))))
    n_source = 6 if n_dim == 1 else 30
    assert plan(jnp.ones((3, n_source))).shape == (3, 0)
    adjoint = plan.adjoint(jnp.ones((3, 0)))
    assert adjoint.shape == (3, n_source)
    assert not jnp.any(adjoint)
    indices, weights = plan.to_gaussx()
    taps = plan.read.n_taps
    assert indices.shape == weights.shape == (0, taps)
    assert plan.operator().mv(jnp.ones(n_source)).shape == (0,)


class TestOneNodeAxis:
    """An axis of one node x₀ reads its node with weight 1 (#118)."""

    @pytest.mark.parametrize("scheme", [Nearest(), Linear(), Cubic()])
    def test_reads_the_node_and_nan_elsewhere(self, scheme) -> None:  # type: ignore[no-untyped-def]
        plan = scheme.plan(Axis(jnp.array([2.0])), Points(jnp.array([[2.0], [2.5]])))
        values = plan(jnp.array([7.0]))
        assert values[0] == 7.0
        assert math.isnan(float(values[1]))
        assert plan.valid.tolist() == [True, False]

    @pytest.mark.parametrize("outside", ["clamp", "extrapolate"])
    def test_clamp_and_extrapolate_continue_the_node(self, outside: str) -> None:
        plan = Linear().plan(
            Axis(jnp.array([2.0])),
            Points(jnp.array([[-1.0], [2.0], [9.0]])),
            policies=Policies(outside=outside),
        )
        assert plan(jnp.array([7.0])).tolist() == [7.0, 7.0, 7.0]

    def test_cells_extent_reads_the_whole_cell(self) -> None:
        axis = Axis(jnp.array([1.0]), edges=jnp.array([0.0, 4.0]))
        plan = Nearest().plan(
            axis,
            Points(jnp.array([[0.0], [3.0], [4.0], [4.5]])),
            policies=Policies(extent="cells"),
        )
        assert plan.valid.tolist() == [True, True, True, False]
        assert plan(jnp.array([7.0]))[:3].tolist() == [7.0, 7.0, 7.0]

    def test_cells_extent_without_edges_raises(self) -> None:
        grid = RectilinearGrid((Axis(jnp.array([1.0])), Axis(jnp.arange(3.0))))
        targets = Points(jnp.array([[1.0, 0.5]]))
        cells = Policies(extent="cells")
        with pytest.raises(GeometryError, match="axis 0 has one node"):
            Linear().plan(grid, targets, policies=cells)
        field = Linear().fit(grid, jnp.zeros(3), policies=cells)
        with pytest.raises(GeometryError, match="axis 0 has one node"):
            field(targets)
