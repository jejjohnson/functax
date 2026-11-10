"""Tests for the strict-JSON configs of `Scheme` and `Policies`. The slow tier:
sweeps, oracle and dot-product checks, gradient and trace checks, anything over
0.3 s on CI (mostly einx and XLA compiling new shapes).
"""

from __future__ import annotations

import itertools
import json
import math
from typing import Any

from functax.interp import (
    Policies,
)


def _through_json(config: dict[str, Any]) -> dict[str, Any]:
    """``config`` written as strict JSON and read back."""
    return json.loads(json.dumps(config, allow_nan=False))


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


class TestPolicies:
    def test_round_trips_through_strict_json_for_every_policy(self) -> None:
        for policies in _every_policy():
            config = _through_json(policies.get_config())
            assert Policies.from_config(config) == policies
