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
from functax.interp import (
    Conservative,
    Cubic,
    Fit,
    Linear,
    Nearest,
    Plan,
    Policies,
    Tensor,
)


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

    @pytest.mark.parametrize("outside", ["clamp", "extrapolate"])
    @pytest.mark.parametrize("edges", [None, [0.0, 4.0], [2.0, 4.0]])
    def test_non_finite_targets_stay_invalid(
        self, outside: str, edges: list[float] | None
    ) -> None:
        axis = Axis(jnp.array([2.0]), edges=None if edges is None else jnp.array(edges))
        targets = Points(jnp.array([[jnp.inf], [-jnp.inf], [jnp.nan], [2.0]]))
        plan = Linear().plan(axis, targets, policies=Policies(outside=outside))
        assert plan.valid.tolist() == [False, False, False, True]

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


class TestRepeatedNodes:
    """Nearest and Linear read an axis with repeats; the others refuse it (#117)."""

    def test_linear_equals_numpy_interp(self) -> None:
        nodes = np.array([0.0, 0.5, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0])
        values = np.random.default_rng(90).normal(size=nodes.size)
        targets = np.concatenate([np.linspace(0.0, 3.0, 25), nodes])
        axis = Axis(jnp.asarray(nodes), repeats="allow")
        plan = Linear().plan(axis, Points(jnp.asarray(targets)[:, None]))
        np.testing.assert_allclose(
            plan(jnp.asarray(values)),
            np.interp(targets, nodes, values),
            rtol=1e-14,
            atol=1e-14,
        )

    def test_linear_reads_the_last_node_of_a_repeat(self) -> None:
        axis = Axis(jnp.array([0.0, 0.5, 1.0, 1.0]), repeats="allow")
        plan = Linear().plan(axis, Points(jnp.array([[0.75], [1.0]])))
        assert plan(jnp.array([0.0, 1.0, 2.0, 5.0])).tolist() == [1.5, 5.0]

    def test_nearest_is_right_continuous(self) -> None:
        axis = Axis(jnp.array([0.0, 1.0, 1.0, 2.0]), repeats="allow")
        plan = Nearest().plan(axis, Points(jnp.array([[0.9], [1.0], [1.4]])))
        assert plan(jnp.array([0.0, 1.0, 3.0, 4.0])).tolist() == [1.0, 3.0, 3.0]

    def test_extrapolation_holds_a_repeated_end(self) -> None:
        axis = Axis(jnp.array([0.0, 0.5, 1.0, 1.0]), repeats="allow")
        plan = Linear().plan(
            axis,
            Points(jnp.array([[-0.5], [1.5]])),
            policies=Policies(outside="extrapolate"),
        )
        # below: the line through the first cell; above: the one-node piece
        assert plan(jnp.array([0.0, 1.0, 2.0, 5.0])).tolist() == [-1.0, 5.0]

    def test_a_held_end_has_zero_slope(self) -> None:
        axis = Axis(jnp.array([0.0, 1.0, 2.0, 3.0, 3.0]), repeats="allow")
        plan = Linear().plan(
            axis,
            Points(jnp.array([[2.5], [3.0], [3.5]])),
            policies=Policies(outside="extrapolate"),
        )
        slopes = plan.derivative(0)(jnp.array([1.0, 2.0, 4.0, 7.0, 20.0]))
        assert slopes.tolist() == [3.0, 0.0, 0.0]

    def test_other_schemes_name_the_repeat(self) -> None:
        axis = Axis(jnp.array([0.0, 1.0, 1.0, 2.0]), repeats="allow")
        targets = Points(jnp.array([[0.5]]))
        with pytest.raises(GeometryError, match=r"CubicBasis .* x\[1\] = x\[2\]"):
            Cubic().plan(axis, targets)
        with pytest.raises(GeometryError, match="Conservative cannot read repeated"):
            Conservative().plan(axis, Axis(jnp.array([0.5, 1.5])))
        # Tensor checks each axis with its own scheme
        grid = RectilinearGrid((axis, Axis(jnp.arange(4.0), repeats="allow")))
        Tensor((Linear(), Cubic())).plan(grid, Points(jnp.array([[0.5, 1.5]])))
        with pytest.raises(GeometryError, match="axis 0 repeats"):
            Tensor((Cubic(), Linear())).plan(grid, Points(jnp.array([[0.5, 1.5]])))


def test_nearest_ties_go_to_the_lower_coordinate() -> None:
    # #129: the same field on the same nodes, in either order
    targets = Points(jnp.array([[0.5], [1.5]]))
    up = Nearest().plan(Axis(jnp.array([0.0, 1.0, 2.0])), targets)
    down = Nearest().plan(Axis(jnp.array([2.0, 1.0, 0.0])), targets)
    assert up(jnp.array([10.0, 11.0, 12.0])).tolist() == [10.0, 11.0]
    assert down(jnp.array([12.0, 11.0, 10.0])).tolist() == [10.0, 11.0]
