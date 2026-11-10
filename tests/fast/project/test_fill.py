"""Tests for `fill_gaps`: the 1-D checks of #42, the mirror-edge Laplacian, dtype
and validation. Larger images, gradients and traces are in tests/slow.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax import ValidationError
from functax._src.project._fill import mirror_laplacian
from functax.project import fill_gaps


TIGHT = lx.CG(rtol=1e-14, atol=1e-14)
ROW = jnp.array([[0.0, 1.0, jnp.nan, jnp.nan, 0.0]])


@pytest.mark.parametrize(
    ("method", "expected"),
    [("laplace", [2.0 / 3.0, 1.0 / 3.0]), ("biharmonic", [1.0, 0.5])],
)
def test_the_one_dimensional_checks(method: str, expected: list[float]) -> None:
    # gate of #42: [0, 1, ?, ?, 0] with mirror edges, to 1e-12
    filled = fill_gaps(ROW, mask=jnp.isfinite(ROW), method=method, solver=TIGHT)
    assert_matches_oracle(filled[0, 2:4], np.array(expected), rtol=1e-12)
    # the same row as a column
    column = fill_gaps(ROW.T, mask=jnp.isfinite(ROW.T), method=method, solver=TIGHT)
    assert_matches_oracle(column[2:4, 0], np.array(expected), rtol=1e-12)


def test_a_fully_observed_image_is_returned_unchanged() -> None:
    image = jnp.arange(12.0).reshape(3, 4)
    filled = fill_gaps(image, mask=jnp.ones((3, 4), dtype=bool))
    assert bool(jnp.all(filled == image))


def test_float32_is_kept() -> None:
    image = ROW.astype(jnp.float32)
    filled = fill_gaps(image, mask=jnp.isfinite(image))
    assert filled.dtype == jnp.float32
    assert_matches_oracle(
        filled[0, 2:4], np.array([2.0 / 3.0, 1.0 / 3.0]), rtol=0, atol=1e-5
    )


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
