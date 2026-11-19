"""Tests for `Variational`. The fast tier: validation, the float32 path and the
smoothing weight as a leaf, on small dense operators. The solves against dense
references, the adjoint and the gradients are in tests/slow.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

from functax import ValidationError
from functax.project import Variational


TIGHT = lx.CG(rtol=1e-13, atol=1e-13)


def _first_differences(n_cell: int) -> jax.Array:
    """D, (N − 1, N), with (Dx)ₖ = xₖ₊₁ − xₖ."""
    return jnp.asarray(np.diff(np.eye(n_cell), axis=0))


def _observation() -> lx.MatrixLinearOperator:
    """H, (7, 12): the checks read only its shape, so no plan is built."""
    return lx.MatrixLinearOperator(jnp.ones((7, 12)))


def _fit(weight: float = 0.7) -> Variational:
    roughness = lx.MatrixLinearOperator(_first_differences(12))
    return Variational(
        _observation(),
        roughness,
        observation_precision=lx.DiagonalLinearOperator(jnp.ones(7)),
        smoothing_weight=weight,
        solver=TIGHT,
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"roughness": lx.MatrixLinearOperator(jnp.ones((3, 5)))}, "roughness"),
        (
            {"observation_precision": lx.DiagonalLinearOperator(jnp.ones(3))},
            "observation_precision",
        ),
        ({"smoothing_weight": -1.0}, "non-negative"),
        ({"smoothing_weight": jnp.ones(2)}, "scalar"),
    ],
)
def test_variational_validates(kwargs, match) -> None:
    arguments = {
        "observation": _observation(),
        "roughness": lx.MatrixLinearOperator(_first_differences(12)),
    }
    with pytest.raises(ValidationError, match=match):
        Variational(**(arguments | kwargs))


def test_apply_checks_the_data_length() -> None:
    with pytest.raises(ValidationError, match="'data'"):
        _fit().apply(jnp.ones(5))
    with pytest.raises(ValidationError, match="'coefficients'"):
        _fit().apply_adjoint(jnp.ones(5))


def test_float32_is_kept() -> None:
    structure = jax.ShapeDtypeStruct((4,), jnp.float32)
    identity = lx.IdentityLinearOperator(structure)
    fit = Variational(identity, identity, smoothing_weight=0.5)
    assert fit(jnp.ones(4, dtype=jnp.float32)).dtype == jnp.float32


def test_the_smoothing_weight_is_a_leaf() -> None:
    assert isinstance(_fit().smoothing_weight, jax.Array)
