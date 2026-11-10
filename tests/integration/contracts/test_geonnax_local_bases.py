"""functax's local bases (#19) against geonnax, pinned at geonnax be5537d as data.

`spherical_rbf_basis` measures geodesic distance with functax's atan2 form where
geonnax used a guarded arccos (#19 asks for the switch where parity holds); the
fixture includes a coincident and an antipodal centre, the two ends where arccos
is least accurate.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax.basis.local import (
    gabor_frame,
    gabor_frame_grid,
    rbf_basis,
    spherical_rbf_basis,
    wavelet_basis_1d,
    wavelet_basis_2d,
    wendland_c2,
    wendland_c4,
)


GEONNAX = np.load(Path(__file__).parent / "data" / "geonnax_local_bases.npz")


def _array(key: str) -> jnp.ndarray:
    return jnp.asarray(GEONNAX[key])


def test_wendland_kernels_equal_geonnax() -> None:
    assert_matches_oracle(
        wendland_c2(_array("r")), GEONNAX["wendland_c2"], rtol=1e-12, atol=1e-15
    )
    assert_matches_oracle(
        wendland_c4(_array("r")), GEONNAX["wendland_c4"], rtol=1e-12, atol=1e-15
    )


@pytest.mark.parametrize("kernel", ["gaussian", "wendland_c2", "wendland_c4"])
def test_rbf_bases_equal_geonnax(kernel: str) -> None:
    values = rbf_basis(_array("x"), _array("centers"), _array("widths"), kernel=kernel)
    assert_matches_oracle(values, GEONNAX[f"rbf_{kernel}"], rtol=1e-12, atol=1e-15)
    spherical = spherical_rbf_basis(
        _array("unit"), _array("unit_centers"), _array("sphere_widths"), kernel=kernel
    )
    assert_matches_oracle(
        spherical, GEONNAX[f"spherical_rbf_{kernel}"], rtol=1e-12, atol=1e-15
    )


def test_gabor_frames_equal_geonnax() -> None:
    args = (_array("x"), _array("centers"), _array("scales"), _array("wavenumbers"))
    assert_matches_oracle(
        gabor_frame(*args), GEONNAX["gabor_radial"], rtol=1e-12, atol=1e-15
    )
    directional = gabor_frame(*args, orientations=_array("orientations"))
    assert_matches_oracle(
        directional, GEONNAX["gabor_directional"], rtol=1e-12, atol=1e-15
    )
    grid = gabor_frame_grid(
        _array("x"),
        jnp.array([[-1.0, 1.0], [-1.0, 0.5]]),
        n_scales=2,
        base_scale=0.4,
        oversample=1.5,
    )
    for value, key in zip(
        grid, ("phi", "centers", "scales", "wavenumbers"), strict=True
    ):
        assert_matches_oracle(value, GEONNAX[f"grid_{key}"], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("wavelet", ["haar", "db2", "db4"])
def test_wavelet_bases_equal_geonnax(wavelet: str) -> None:
    assert_matches_oracle(
        wavelet_basis_1d(16, wavelet=wavelet),
        GEONNAX[f"wavelet_1d_{wavelet}"],
        rtol=1e-12,
        atol=1e-15,
    )


def test_partial_and_2d_wavelet_bases_equal_geonnax() -> None:
    partial = wavelet_basis_1d(16, wavelet="db2", levels=2)
    assert_matches_oracle(partial, GEONNAX["wavelet_1d_levels"], rtol=1e-12, atol=1e-15)
    assert_matches_oracle(
        wavelet_basis_2d(4, 8, wavelet="db2"),
        GEONNAX["wavelet_2d"],
        rtol=1e-12,
        atol=1e-15,
    )
