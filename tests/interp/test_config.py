"""Tests for the strict-JSON configs of `Scheme` and `Policies`."""

from __future__ import annotations

import inspect
import itertools
import json
import math
from typing import Any

import jax.numpy as jnp
import pytest

from functax import PolicyError, ValidationError
from functax.interp import (
    Akima,
    Basis,
    BSpline,
    Cubic,
    CubicSpline,
    Linear as LinearScheme,
    LinearBasis,
    LocalScheme,
    Nearest,
    Pchip,
    Policies,
    Scheme,
    Steffen,
    Tensor,
)


class Scaled(LocalScheme):
    """A scheme defined outside functax, with a float field: no registration."""

    factor: float = 1.0

    def bases(self, n_axes: int) -> tuple[Basis, ...]:
        return (LinearBasis(),) * n_axes


class Holding(LocalScheme):
    """A scheme with a field strict JSON cannot hold."""

    weight: Any = None

    def bases(self, n_axes: int) -> tuple[Basis, ...]:
        return (LinearBasis(),) * n_axes


def _through_json(config: dict[str, Any]) -> dict[str, Any]:
    """``config`` written as strict JSON and read back."""
    return json.loads(json.dumps(config, allow_nan=False))


SCHEMES = [
    Nearest(),
    LinearScheme(),
    Tensor((Nearest(),)),
    Tensor((Nearest(), LinearScheme(), LinearScheme())),
    Scaled(2.5),
    Scaled(math.inf),
    Tensor((Scaled(-math.inf), Nearest())),
]

FILL_VALUES = [math.nan, math.inf, -math.inf, 0.0, -9, 2.5]


def _every_policy() -> list[Policies]:
    """Every outside × missing × boundary × extent × fill value combination."""
    combinations = itertools.product(
        ("nan", "clamp", "extrapolate", "raise"),
        ("propagate", "renormalise", "fill"),
        ("mirror", "wrap", "edge", "constant"),
        ("nodes", "cells"),
        FILL_VALUES,
    )
    return [
        Policies(outside=o, missing=m, boundary=b, extent=e, fill_value=f)
        for o, m, b, e, f in combinations
    ]


class TestSchemes:
    @pytest.mark.parametrize("scheme", SCHEMES, ids=repr)
    def test_round_trips_through_strict_json(self, scheme: Scheme) -> None:
        restored = Scheme.from_config(_through_json(scheme.get_config()))
        assert restored == scheme
        assert type(restored) is type(scheme)

    def test_the_config_names_the_class_and_every_field(self) -> None:
        assert LinearScheme().get_config() == {
            "class": "Linear",
            "prior_variance_rate": None,
        }
        assert Tensor((Nearest(), LinearScheme())).get_config() == {
            "class": "Tensor",
            "schemes": [
                {"class": "Nearest"},
                {"class": "Linear", "prior_variance_rate": None},
            ],
        }

    def test_a_non_finite_float_field_is_tagged(self) -> None:
        assert Scaled(math.inf).get_config() == {
            "class": "Scaled",
            "factor": {"float": "inf"},
        }

    def test_a_subclass_reads_its_own_config(self) -> None:
        assert LinearScheme.from_config({"class": "Linear"}) == LinearScheme()
        with pytest.raises(ValidationError, match="'Nearest', which is not a Linear"):
            LinearScheme.from_config({"class": "Nearest"})

    def test_an_omitted_default_takes_its_default(self) -> None:
        assert Scheme.from_config({"class": "Scaled"}) == Scaled()

    def test_a_field_without_a_json_form_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="'weight' has no strict-JSON form"):
            Holding(jnp.ones(2)).get_config()


class TestPolicies:
    def test_round_trips_through_strict_json_for_every_policy(self) -> None:
        for policies in _every_policy():
            config = _through_json(policies.get_config())
            assert Policies.from_config(config) == policies

    @pytest.mark.parametrize(
        ("fill_value", "encoded"),
        [
            (math.nan, {"float": "nan"}),
            (math.inf, {"float": "inf"}),
            (-math.inf, {"float": "-inf"}),
            (-9, -9),
            (2.5, 2.5),
        ],
        ids=["nan", "inf", "-inf", "int", "float"],
    )
    def test_the_fill_value_encoding(self, fill_value: float, encoded: Any) -> None:
        config = Policies(fill_value=fill_value).get_config()
        assert config["fill_value"] == encoded
        restored = Policies.from_config(_through_json(config)).fill_value
        assert type(restored) is type(fill_value)

    def test_a_decoded_nan_is_the_canonical_nan(self) -> None:
        restored = Policies.from_config(
            {"class": "Policies", "fill_value": {"float": "nan"}}
        )
        assert restored.fill_value is math.nan

    def test_a_bad_option_is_a_policy_error(self) -> None:
        with pytest.raises(PolicyError, match="'outside' must be one of"):
            Policies.from_config({"class": "Policies", "outside": "wrap"})


class TestRejects:
    @pytest.mark.parametrize(
        ("config", "match"),
        [
            ({"scheme": "Linear"}, "must be a dict with a 'class' name"),
            ({"class": 3}, "must be a dict with a 'class' name"),
            ({"class": "NoSuchScheme"}, "'NoSuchScheme', which is not a loaded Scheme"),
            ({"class": "LocalScheme"}, "'LocalScheme', which is not a loaded Scheme"),
            ({"class": "Policies"}, "'Policies', which is not a loaded Scheme"),
            ({"class": "Linear", "order": 1}, r"keys \['order'\] that Linear"),
            ({"class": "Tensor"}, r"missing \['schemes'\], which Tensor needs"),
            (
                {"class": "Tensor", "schemes": [{"name": "Linear"}]},
                r"'schemes\[0\]' must be a config with a 'class'",
            ),
            (
                {"class": "Scaled", "factor": {"float": "1.5"}},
                "'factor' must be a config with a 'class' or a non-finite float",
            ),
            ({"class": "Scaled", "factor": {1.0}}, "'factor' must be a JSON value"),
        ],
        ids=[
            "no-class",
            "class-not-str",
            "unknown",
            "abstract",
            "other-root",
            "unknown-key",
            "missing-key",
            "bad-nested",
            "bad-float-tag",
            "not-json",
        ],
    )
    def test_malformed_configs(self, config: dict[str, Any], match: str) -> None:
        with pytest.raises(ValidationError, match=match):
            Scheme.from_config(config)

    def test_a_config_that_is_not_a_dict(self) -> None:
        # the type hook would preempt the module's own check, so skip it
        from_config = inspect.unwrap(Scheme.from_config)
        with pytest.raises(ValidationError, match="'config' must be a dict"):
            from_config(Scheme, ["Linear"])

    def test_an_ambiguous_class_name(self) -> None:
        def define() -> type:
            class Ambiguous(LocalScheme):
                def bases(self, n_axes: int) -> tuple[Basis, ...]:
                    return (LinearBasis(),) * n_axes

            return Ambiguous

        # both classes stay alive (and loaded) while the config is read
        first, second = define(), define()
        assert first is not second
        with pytest.raises(ValidationError, match="'Ambiguous', which is ambiguous"):
            Scheme.from_config({"class": "Ambiguous"})


ALL_SCHEMES = [
    Nearest(),
    LinearScheme(),
    LinearScheme(prior_variance_rate=0.5),
    Cubic(a=-0.75),
    *(BSpline(order) for order in range(6)),
    Tensor((Nearest(), BSpline(3))),
    CubicSpline("periodic"),
    Pchip(),
    Akima(),
    Steffen(),
]


@pytest.mark.parametrize("scheme", ALL_SCHEMES, ids=lambda s: type(s).__name__)
def test_every_scheme_round_trips_through_strict_json(scheme) -> None:  # type: ignore[no-untyped-def]
    # every scheme in functax today, including the ones added after #33
    text = json.dumps(scheme.get_config(), allow_nan=False)
    restored = type(scheme).from_config(json.loads(text))
    assert restored == scheme
    assert Scheme.from_config(json.loads(text)) == scheme
