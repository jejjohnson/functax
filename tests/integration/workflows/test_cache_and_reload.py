"""Caching plans: a plan written to bytes and read back into a template applies bit-
identically, for every plan kind.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import (
    BSpline,
    Linear,
    Nearest,
    Pchip,
    Plan,
    Policies,
    Tensor,
)


GRID = RectilinearGrid((Axis(jnp.arange(3.0)), Axis(jnp.linspace(0.0, 1.0, 5))))


def _template(
    build: Callable[[], Plan],
) -> Plan:
    """The abstract plan ``build`` would return, traced but not run."""
    return eqx.filter_eval_shape(build)


SERIALIZE_CASES: dict[str, tuple[Any, Policies]] = {
    "nearest": (Nearest(), Policies()),
    "linear-clamp-wrap": (Linear(), Policies(outside="clamp", boundary="wrap")),
    "linear-renormalise": (Linear(), Policies(missing="renormalise", fill_value=-1.0)),
    "tensor-extrapolate-cells": (
        Tensor((Nearest(), Linear())),
        Policies(outside="extrapolate", extent="cells"),
    ),
}


class TestCacheAndReload:
    targets = Points(jnp.array([[0.5, 0.25], [1.5, 0.8], [2.4, -0.3], [9.0, 0.5]]))
    field = jnp.array([1.0, 2.0, jnp.nan, 4.0, 8.0] * 3)

    @pytest.mark.parametrize("case", SERIALIZE_CASES)
    def test_round_trip_applies_bit_identically(self, case: str) -> None:
        scheme, policies = SERIALIZE_CASES[case]

        def build() -> Plan:
            return scheme.plan(GRID, self.targets, policies=policies)

        plan = build()
        restored = Plan.deserialize(plan.serialize(), like=_template(build))
        for original, loaded in zip(
            jax.tree_util.tree_leaves(plan),
            jax.tree_util.tree_leaves(restored),
            strict=True,
        ):
            assert original.dtype == loaded.dtype
            assert np.asarray(original).tobytes() == np.asarray(loaded).tobytes()
        np.testing.assert_array_equal(restored(self.field), plan(self.field))
        residual = jnp.arange(4.0)
        np.testing.assert_array_equal(
            restored.adjoint(residual), plan.adjoint(residual)
        )
        assert restored.policies == policies

    @pytest.mark.parametrize(
        "scheme",
        [BSpline(3), Linear(prior_variance_rate=0.5), Pchip()],
        ids=["bspline", "gaussian", "pchip"],
    )
    def test_plans_with_later_fields_round_trip_through_bytes(self, scheme) -> None:  # type: ignore[no-untyped-def]
        # the geometry, posterior and slope fits added after #33 travel as leaves
        axis = Axis(jnp.linspace(0.0, 4.0, 9))
        targets = Points(jnp.array([[0.3], [1.7], [3.9]]))
        plan = scheme.plan(axis, targets)
        template = scheme.plan(axis, Points(jnp.zeros((3, 1))))
        restored = type(plan).deserialize(plan.serialize(), like=template)
        data = jnp.sin(axis.values)
        assert restored(data).tolist() == plan(data).tolist()
