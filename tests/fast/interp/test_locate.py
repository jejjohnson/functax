"""Tests for `locate`, `index_positions`, `remap_taps` and `Policies`. The fast tier:
behaviour, validation and small closed-form checks, each well under a second.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from functax import OutsideDomainError, PolicyError
from functax.geometry import Axis
from functax.interp import (
    Policies,
    check_outside,
    index_positions,
    locate,
    remap_taps,
)


class TestIndexPositions:
    def test_any_target_shape(self) -> None:
        located = index_positions(Axis(jnp.arange(5.0)), jnp.full((2, 3), 1.5))
        assert located.shape == (2, 3)


class TestLocate:
    def test_extent_nodes_and_cells(self) -> None:
        axis = Axis(jnp.arange(4.0))
        targets = jnp.array([-0.6, -0.4, 0.0, 3.0, 3.4, 3.6])
        _, _, nodes = locate(axis, targets)
        _, _, cells = locate(axis, targets, extent="cells")
        assert nodes.tolist() == [False, False, True, True, False, False]
        assert cells.tolist() == [False, True, True, True, True, False]

    def test_non_finite_targets_are_outside(self) -> None:
        _, _, inside = locate(Axis(jnp.arange(4.0)), jnp.array([jnp.nan, jnp.inf, 1.0]))
        assert inside.tolist() == [False, False, True]

    def test_rejects_an_unknown_extent(self) -> None:
        with pytest.raises(PolicyError, match="'extent' must be one of"):
            locate(Axis(jnp.arange(3.0)), jnp.zeros(1), extent="pixels")


class TestRemapTaps:
    @pytest.mark.parametrize(
        ("boundary", "mode"),
        [
            ("mirror", "mirror"),
            ("wrap", "grid-wrap"),
            ("edge", "nearest"),
            ("constant", "grid-constant"),
        ],
    )
    @pytest.mark.parametrize("n_nodes", [1, 2, 5])
    def test_matches_scipy_ndimage(
        self, boundary: str, mode: str, n_nodes: int
    ) -> None:
        signal = np.arange(1.0, n_nodes + 1.0)
        taps = np.arange(-3 * n_nodes - 2, 4 * n_nodes + 3)
        indices, kept = remap_taps(jnp.asarray(taps)[None], n_nodes, boundary)
        read = np.where(np.asarray(kept[0]), signal[np.asarray(indices[0])], 0.0)
        expected = map_coordinates(
            signal, taps[None].astype(float), order=0, mode=mode, cval=0.0
        )
        np.testing.assert_array_equal(read, expected)

    def test_a_periodic_axis_always_wraps(self) -> None:
        indices, kept = remap_taps(jnp.array([[-1, 4]]), 4, "constant", periodic=True)
        assert indices.tolist() == [[3, 0]]
        assert kept.tolist() == [[True, True]]

    def test_rejects_an_unknown_boundary(self) -> None:
        with pytest.raises(PolicyError, match="'boundary' must be one of"):
            remap_taps(jnp.zeros((1, 1), dtype=int), 3, "reflect")


class TestPolicies:
    def test_defaults(self) -> None:
        policies = Policies()
        assert (
            policies.outside,
            policies.missing,
            policies.boundary,
            policies.extent,
        ) == (
            "nan",
            "propagate",
            "mirror",
            "nodes",
        )
        assert math.isnan(policies.fill_value)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"outside": "wrap"}, "'outside' must be one of"),
            ({"missing": "drop"}, "'missing' must be one of"),
            ({"boundary": "reflect"}, "'boundary' must be one of"),
            ({"extent": "pixels"}, "'extent' must be one of"),
        ],
    )
    def test_rejects_unknown_options(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        with pytest.raises(PolicyError, match=match):
            Policies(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("fill_value", ["zero", True])
    def test_rejects_a_non_number_fill(self, fill_value: object) -> None:
        # the type hook in conftest.py rejects these before __check_init__, so
        # run the check itself on an instance built without __init__
        policies = object.__new__(Policies)
        for name, value in vars(Policies()).items():
            object.__setattr__(policies, name, value)
        object.__setattr__(policies, "fill_value", fill_value)
        with pytest.raises(PolicyError, match="'fill_value' must be a number"):
            policies.__check_init__()

    def test_a_numeric_fill_is_kept(self) -> None:
        assert Policies(fill_value=0.0).fill_value == 0.0
        assert Policies(fill_value=-9999).fill_value == -9999

    def test_nan_fills_give_one_tree_structure(self) -> None:
        # two NaN objects must not make equal policies look different to jit
        first = Policies(fill_value=float("nan"))
        second = Policies(fill_value=float("nan"))
        assert jax.tree.structure(first) == jax.tree.structure(second)
        assert jax.tree.structure(first) == jax.tree.structure(Policies())


class TestCheckOutside:
    def test_raise_reports_the_count_and_the_first(self) -> None:
        with pytest.raises(
            OutsideDomainError, match=r"2 of 4 targets .* the first is target 1"
        ):
            check_outside(
                jnp.array([True, False, True, False]), Policies(outside="raise")
            )

    def test_raise_passes_when_all_inside(self) -> None:
        check_outside(jnp.ones(3, dtype=bool), Policies(outside="raise"))

    def test_raise_refuses_traced_targets(self) -> None:
        def build(inside: jax.Array) -> jax.Array:
            check_outside(inside, Policies(outside="raise"))
            return inside

        with pytest.raises(PolicyError, match="needs concrete targets"):
            jax.jit(build)(jnp.ones(3, dtype=bool))

    @pytest.mark.parametrize("outside", ["nan", "clamp", "extrapolate"])
    def test_other_policies_do_not_check(self, outside: str) -> None:
        check_outside(jnp.zeros(2, dtype=bool), Policies(outside=outside))
