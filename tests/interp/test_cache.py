"""Tests for caching plans: `plan_key`, `Plan.serialize` and `Plan.deserialize`."""

from __future__ import annotations

import inspect
import string
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from functax import ValidationError
from functax._src.interp import _cache
from functax.geometry import AffineGrid, Axis, CurvilinearGrid, Points, RectilinearGrid
from functax.interp import (
    BSpline,
    Linear,
    Nearest,
    Pchip,
    Plan,
    Policies,
    Tensor,
    plan_key,
)


def _up(array: jax.Array, index: int = 0) -> jax.Array:
    """``array`` with one entry moved up by one ULP."""
    flat = np.array(array).ravel()
    flat[index] = np.nextafter(flat[index], np.inf)
    return jnp.asarray(flat.reshape(np.shape(array)))


AXIS = Axis(jnp.linspace(0.0, 3.0, 4))
GRID = RectilinearGrid((Axis(jnp.arange(3.0)), Axis(jnp.linspace(0.0, 1.0, 5))))
AFFINE = AffineGrid([10.0, 0.0, 500.0, 0.0, -10.0, 900.0], (4, 5))
CURVILINEAR = CurvilinearGrid(
    x=jnp.array([[0.0, 1.0], [0.1, 1.1]]), y=jnp.array([[0.0, 0.0], [1.0, 1.0]])
)
POINTS = Points(
    jnp.array([[0.5, 0.25], [1.5, 0.75]]),
    z=jnp.array([1.0, 2.0]),
    valid=jnp.array([True, True]),
)
SCHEME = Linear()


# Each geometry with one of its values, dtype, shape or static fields changed.
CHANGED: dict[str, tuple[Any, Any]] = {
    "axis-ulp": (AXIS, Axis(_up(AXIS.values, 1))),
    "axis-dtype": (AXIS, Axis(AXIS.values.astype(jnp.float32))),
    "axis-length": (AXIS, Axis(jnp.linspace(0.0, 3.0, 5))),
    "axis-period": (AXIS, Axis(AXIS.values, period=10.0)),
    "axis-epoch": (
        Axis(AXIS.values, epoch="2026-01-01"),
        Axis(AXIS.values, epoch="2026-01-02"),
    ),
    "grid-ulp": (GRID, RectilinearGrid((GRID.axes[0], Axis(_up(GRID.axes[1].values))))),
    "grid-order": (GRID, RectilinearGrid(GRID.axes[::-1])),
    "affine-ulp": (AFFINE, AffineGrid(_up(AFFINE.transform, 2), AFFINE.shape)),
    "affine-shape": (AFFINE, AffineGrid(AFFINE.transform, (5, 4))),
    "curvilinear-x": (
        CURVILINEAR,
        CurvilinearGrid(x=_up(CURVILINEAR.x, 3), y=CURVILINEAR.y),
    ),
    "curvilinear-y": (
        CURVILINEAR,
        CurvilinearGrid(x=CURVILINEAR.x, y=_up(CURVILINEAR.y)),
    ),
    "points-xy": (POINTS, Points(_up(POINTS.xy, 3), z=POINTS.z, valid=POINTS.valid)),
    "points-z": (POINTS, Points(POINTS.xy, z=_up(POINTS.z), valid=POINTS.valid)),
    "points-no-z": (POINTS, Points(POINTS.xy, valid=POINTS.valid)),
    "points-valid": (
        POINTS,
        Points(POINTS.xy, z=POINTS.z, valid=jnp.array([True, False])),
    ),
    "points-metric": (
        POINTS,
        Points(POINTS.xy, z=POINTS.z, valid=POINTS.valid, metric="chord"),
    ),
    "points-signed-zero": (
        Points(jnp.array([[0.0]])),
        Points(jnp.array([[-0.0]])),
    ),
}


class TestPlanKey:
    def test_is_a_stable_sha256_hex(self) -> None:
        key = plan_key(SCHEME, GRID, POINTS)
        assert len(key) == 64
        assert set(key) <= set(string.hexdigits.lower())
        # equal inputs built afresh give the same key
        twin = RectilinearGrid((Axis(jnp.arange(3.0)), Axis(jnp.linspace(0.0, 1.0, 5))))
        assert plan_key(Linear(), twin, POINTS) == key

    @pytest.mark.parametrize("case", CHANGED)
    def test_changes_with_the_source(self, case: str) -> None:
        before, after = CHANGED[case]
        assert plan_key(SCHEME, before, POINTS) != plan_key(SCHEME, after, POINTS)

    @pytest.mark.parametrize("case", CHANGED)
    def test_changes_with_the_target(self, case: str) -> None:
        before, after = CHANGED[case]
        assert plan_key(SCHEME, GRID, before) != plan_key(SCHEME, GRID, after)

    def test_changes_when_source_and_target_swap(self) -> None:
        assert plan_key(SCHEME, AXIS, GRID) != plan_key(SCHEME, GRID, AXIS)

    def test_changes_with_the_scheme(self) -> None:
        keys = {
            plan_key(scheme, GRID, POINTS)
            for scheme in (
                Linear(),
                Nearest(),
                Tensor((Nearest(), Linear())),
                Tensor((Linear(), Nearest())),
            )
        }
        assert len(keys) == 4

    def test_changes_with_the_policies(self) -> None:
        variants = [
            Policies(),
            Policies(outside="clamp"),
            Policies(missing="fill"),
            Policies(boundary="edge"),
            Policies(extent="cells"),
            Policies(fill_value=0.0),
            Policies(fill_value=0),
        ]
        keys = {plan_key(SCHEME, GRID, POINTS, policies=p) for p in variants}
        assert len(keys) == len(variants)
        # None is the default policies, as in Scheme.plan
        assert plan_key(SCHEME, GRID, POINTS) == plan_key(
            SCHEME, GRID, POINTS, policies=Policies()
        )

    def test_changes_with_the_functax_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        key = plan_key(SCHEME, GRID, POINTS)
        monkeypatch.setattr(_cache, "__version__", "99.0.0")
        assert plan_key(SCHEME, GRID, POINTS) != key

    def test_refuses_traced_geometries(self) -> None:
        def key_of(xy: jax.Array) -> jax.Array:
            plan_key(SCHEME, GRID, Points(xy))
            return xy

        with pytest.raises(ValidationError, match=r"'target.xy' is traced"):
            jax.jit(key_of)(POINTS.xy)

    @pytest.mark.parametrize(
        ("arguments", "match"),
        [
            ((object(), GRID, POINTS), "'scheme' must be a Scheme"),
            ((SCHEME, np.zeros(3), POINTS), "'source' must be a geometry"),
            ((SCHEME, GRID, [[0.5, 0.5]]), "'target' must be a geometry"),
        ],
        ids=["scheme", "source", "target"],
    )
    def test_rejects_bad_arguments(
        self, arguments: tuple[Any, ...], match: str
    ) -> None:
        # the type hook would preempt the module's own checks, so skip it
        with pytest.raises(ValidationError, match=match):
            inspect.unwrap(plan_key)(*arguments)

    def test_rejects_bad_policies(self) -> None:
        with pytest.raises(ValidationError, match="'policies' must be Policies"):
            inspect.unwrap(plan_key)(SCHEME, GRID, POINTS, policies="nan")

    def test_rejects_a_field_without_a_json_form(self) -> None:
        class Odd(eqx.Module):
            when: object = eqx.field(static=True, default_factory=object)

        with pytest.raises(ValidationError, match=r"'target\.when' has no strict-JSON"):
            plan_key(SCHEME, GRID, Odd())

    @pytest.mark.slow
    def test_is_identical_across_processes(self) -> None:
        code = (
            "import jax\n"
            "jax.config.update('jax_enable_x64', True)\n"
            "import jax.numpy as jnp\n"
            "from functax.geometry import AffineGrid, Points\n"
            "from functax.interp import Linear, Policies, plan_key\n"
            "grid = AffineGrid([10.0, 0.0, 500.0, 0.0, -10.0, 900.0], (4, 5))\n"
            "points = Points(jnp.array([[515.0, 885.0], [540.0, 870.0]]))\n"
            "policies = Policies(extent='cells', boundary='edge')\n"
            "print(plan_key(Linear(), grid, points, policies=policies))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        points = Points(jnp.array([[515.0, 885.0], [540.0, 870.0]]))
        policies = Policies(extent="cells", boundary="edge")
        expected = plan_key(Linear(), AFFINE, points, policies=policies)
        assert result.stdout.strip() == expected


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


class TestSerialize:
    targets = Points(jnp.array([[0.5, 0.25], [1.5, 0.8], [2.4, -0.3], [9.0, 0.5]]))
    field = jnp.array([1.0, 2.0, jnp.nan, 4.0, 8.0] * 3)

    @pytest.mark.slow
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

    def test_a_concrete_plan_is_a_template_too(self) -> None:
        plan = Linear().plan(GRID, self.targets)
        restored = Plan.deserialize(plan.serialize(), like=plan)
        np.testing.assert_array_equal(restored(self.field), plan(self.field))

    @pytest.mark.slow
    def test_rejects_a_template_of_another_shape(self) -> None:
        data = Linear().plan(GRID, self.targets).serialize()
        other = Linear().plan(GRID, Points(jnp.zeros((2, 2))))
        with pytest.raises(ValidationError, match="'data' does not match 'like'"):
            Plan.deserialize(data, like=other)

    def test_rejects_truncated_and_padded_data(self) -> None:
        plan = Linear().plan(GRID, self.targets)
        data = plan.serialize()
        with pytest.raises(ValidationError, match="'data' does not match 'like'"):
            Plan.deserialize(data[: len(data) // 2], like=plan)
        with pytest.raises(ValidationError, match="holds more leaves than 'like'"):
            Plan.deserialize(data + data, like=plan)

    def test_refuses_traced_and_abstract_plans(self) -> None:
        def serialize_inside(xy: jax.Array) -> jax.Array:
            Linear().plan(GRID, Points(xy)).serialize()
            return xy

        with pytest.raises(ValidationError, match="must hold concrete arrays"):
            jax.jit(serialize_inside)(self.targets.xy)
        template = _template(lambda: Linear().plan(GRID, self.targets))
        with pytest.raises(ValidationError, match="got a ShapeDtypeStruct leaf"):
            template.serialize()

    def test_rejects_bad_arguments(self) -> None:
        plan = Linear().plan(GRID, self.targets)
        # the type hook would preempt the module's own checks, so skip it
        deserialize = inspect.unwrap(Plan.deserialize)
        with pytest.raises(ValidationError, match="'data' must be bytes"):
            deserialize(Plan, "not bytes", like=plan)
        with pytest.raises(ValidationError, match="'like' must be a Plan"):
            deserialize(Plan, plan.serialize(), like=plan.read)


@pytest.mark.parametrize(
    "scheme",
    [BSpline(3), Linear(prior_variance_rate=0.5), Pchip()],
    ids=["bspline", "gaussian", "pchip"],
)
def test_plans_with_later_fields_round_trip_through_bytes(scheme) -> None:  # type: ignore[no-untyped-def]
    # the geometry, posterior and slope fits added after #33 travel as leaves
    axis = Axis(jnp.linspace(0.0, 4.0, 9))
    targets = Points(jnp.array([[0.3], [1.7], [3.9]]))
    plan = scheme.plan(axis, targets)
    template = scheme.plan(axis, Points(jnp.zeros((3, 1))))
    restored = type(plan).deserialize(plan.serialize(), like=template)
    data = jnp.sin(axis.values)
    assert restored(data).tolist() == plan(data).tolist()
