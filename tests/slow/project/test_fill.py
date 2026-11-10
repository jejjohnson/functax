"""Slow tests for `fill_gaps`: the observed pixels kept, NaN holes never read,
linear fields reproduced, the hole equations, gradients through the solve against
finite differences, and jit/vmap agreement.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces

from functax._src.project._fill import mirror_laplacian
from functax.project import fill_gaps


TIGHT = lx.CG(rtol=1e-14, atol=1e-14)


ROW = jnp.array([[0.0, 1.0, jnp.nan, jnp.nan, 0.0]])


def _image_and_mask(seed: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    image = jnp.asarray(rng.normal(size=(5, 6)))
    mask = jnp.asarray(rng.uniform(size=(5, 6)) > 0.4)
    return image, mask


@pytest.mark.parametrize("method", ["laplace", "biharmonic"])
def test_observed_pixels_are_kept_and_hole_values_never_read(method: str) -> None:
    rng = np.random.default_rng(0)
    image = jnp.asarray(rng.normal(size=(6, 7)))
    mask = jnp.asarray(rng.uniform(size=(6, 7)) > 0.3)
    with_nan = jnp.where(mask, image, jnp.nan)
    with_junk = jnp.where(mask, image, 1e6)
    filled = fill_gaps(with_nan, mask=mask, method=method, solver=TIGHT)
    assert bool(jnp.all(jnp.where(mask, filled == image, True)))
    assert bool(jnp.all(jnp.isfinite(filled)))
    other = fill_gaps(with_junk, mask=mask, method=method, solver=TIGHT)
    assert_matches_oracle(other, filled, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("method", ["laplace", "biharmonic"])
def test_a_linear_field_is_reproduced_in_an_interior_hole(method: str) -> None:
    rows, cols = jnp.meshgrid(jnp.arange(9.0), jnp.arange(11.0), indexing="ij")
    field = 1.0 + 0.5 * rows - 0.25 * cols
    mask = jnp.ones(field.shape, dtype=bool).at[3:6, 4:8].set(False)
    filled = fill_gaps(field, mask=mask, method=method, solver=TIGHT)
    assert_matches_oracle(filled, field, rtol=1e-12)


@pytest.mark.parametrize("method", ["laplace", "biharmonic"])
def test_the_holes_satisfy_their_equation(method: str) -> None:
    rng = np.random.default_rng(1)
    image = jnp.asarray(rng.normal(size=(8, 8)))
    mask = jnp.asarray(rng.uniform(size=(8, 8)) > 0.4)
    filled = fill_gaps(image, mask=mask, method=method, solver=TIGHT)
    residual = mirror_laplacian(filled)
    if method == "biharmonic":
        residual = mirror_laplacian(residual)
    assert float(jnp.max(jnp.where(mask, 0.0, jnp.abs(residual)))) < 1e-11


@pytest.mark.parametrize("method", ["laplace", "biharmonic"])
def test_gradient_in_the_image_matches_finite_differences(method: str) -> None:
    image, mask = _image_and_mask(0)
    assert_grad_matches_fd(
        lambda field: fill_gaps(field, mask=mask, method=method, solver=TIGHT),
        image,
        rtol=1e-6,
    )


@pytest.mark.parametrize("method", ["laplace", "biharmonic"])
def test_jit_and_vmap_agree_with_a_loop(method: str) -> None:
    pairs = [_image_and_mask(seed) for seed in range(3)]
    images = jnp.stack([image for image, _ in pairs])
    masks = jnp.stack([mask for _, mask in pairs])
    assert_traces(
        lambda field, observed: fill_gaps(
            field, mask=observed, method=method, solver=TIGHT
        ),
        images,
        masks,
        rtol=1e-10,
    )


def test_the_default_solver_works_under_jit() -> None:
    image, mask = _image_and_mask(2)
    eager = fill_gaps(image, mask=mask)
    jitted = jax.jit(lambda field, observed: fill_gaps(field, mask=observed))
    assert_matches_oracle(jitted(image, mask), eager, rtol=1e-12, atol=1e-14)


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
