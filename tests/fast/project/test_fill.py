"""Tests for `fill_gaps`: the 1-D checks of #42, the mirror-edge and periodic
Laplacians, dtype and validation. Larger images, gradients and traces are in tests/slow.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax import ValidationError
from functax._src.project._fill import mirror_laplacian, periodic_laplacian
from functax.project import fill_gaps


def test_mirror_laplacian_is_minus_d_transpose_d() -> None:
    # Δ = −DᵀD with forward differences along each axis
    n_row, n_col = 4, 5
    difference_row = np.diff(np.eye(n_row), axis=0)
    difference_col = np.diff(np.eye(n_col), axis=0)
    expected = -np.kron(difference_row.T @ difference_row, np.eye(n_col)) - np.kron(
        np.eye(n_row), difference_col.T @ difference_col
    )
    columns = [
        np.asarray(mirror_laplacian(jnp.asarray(e.reshape(n_row, n_col)))).ravel()
        for e in np.eye(n_row * n_col)
    ]
    assert_matches_oracle(np.stack(columns, axis=1), expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    ("image", "mask", "method", "match"),
    [
        (jnp.ones(4), jnp.ones(4, dtype=bool), "laplace", "'image'"),
        (
            jnp.ones((2, 2), dtype=jnp.int32),
            jnp.ones((2, 2), dtype=bool),
            "laplace",
            "image",
        ),
        (jnp.ones((2, 2)), jnp.ones((2, 3), dtype=bool), "laplace", "'mask'"),
        (jnp.ones((2, 2)), jnp.ones((2, 2)), "laplace", "'mask'"),
        (jnp.ones((2, 2)), jnp.ones((2, 2), dtype=bool), "kriging", "'method'"),
        (jnp.ones((2, 2)), jnp.zeros((2, 2), dtype=bool), "laplace", "observed"),
    ],
)
def test_fill_gaps_validates(image, mask, method, match) -> None:
    with pytest.raises(ValidationError, match=match):
        inspect.unwrap(fill_gaps)(image, mask=mask, method=method)


def test_periodic_laplacian_is_symmetric_with_constant_null_space() -> None:
    n_row, n_col = 3, 5
    columns = [
        np.asarray(
            periodic_laplacian(
                jnp.asarray(e.reshape(n_row, n_col)), periodic=(False, True)
            )
        ).ravel()
        for e in np.eye(n_row * n_col)
    ]
    laplacian = np.stack(columns, axis=1)
    assert_matches_oracle(laplacian, laplacian.T, rtol=0, atol=0)
    null = laplacian @ np.ones(n_row * n_col)
    assert_matches_oracle(null, np.zeros(n_row * n_col), rtol=0, atol=1e-15)
    # without a periodic axis it is the mirror Laplacian
    image = jnp.asarray(np.random.default_rng(0).normal(size=(n_row, n_col)))
    assert_matches_oracle(
        periodic_laplacian(image, periodic=(False, False)),
        mirror_laplacian(image),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("periodic", [(True,), (True, False, True), ("yes", False)])
def test_fill_gaps_validates_periodic(periodic) -> None:  # type: ignore[no-untyped-def]
    image = jnp.array([[0.0, jnp.nan, 3.0]])
    with pytest.raises(ValidationError, match="'periodic' must be two booleans"):
        inspect.unwrap(fill_gaps)(image, mask=jnp.isfinite(image), periodic=periodic)
