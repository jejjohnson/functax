"""functax against geonnax, pinned at geonnax be5537d as data: the lon/lat lift
against lonlat_to_cartesian3d, and the feature maps moved in by #21.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_matches_oracle

from functax.basis.features import (
    fourier_features,
    gaussian_window_features,
    rff_cosine_forward,
    rff_forward,
    seasonal_features,
)
from functax.geometry import (
    lonlat_to_unit,
)


GEONNAX = np.load(Path(__file__).parent / "data" / "geonnax_lonlat_to_cartesian3d.npz")


class TestLonlatToUnit:
    @pytest.mark.parametrize(
        ("inputs", "outputs", "unit"),
        [
            ("lonlat_degrees", "unit_from_degrees", "degrees"),
            ("lonlat_radians", "unit_from_radians", "radians"),
        ],
    )
    def test_equals_geonnax(self, inputs: str, outputs: str, unit: str) -> None:
        lifted = lonlat_to_unit(jnp.asarray(GEONNAX[inputs]), unit=unit)
        assert_matches_oracle(lifted, GEONNAX[outputs], rtol=0.0, atol=1e-15)


FEATURES = np.load(Path(__file__).parent / "data" / "geonnax_features.npz")


class TestFeatures:
    """The feature maps of #21 against geonnax ``randfeat`` and ``basis``."""

    def test_rff_forward_equals_geonnax(self) -> None:
        W = jnp.asarray(FEATURES["W"])
        lengthscale = float(FEATURES["lengthscale"])
        features = jnp.stack(
            [
                rff_forward(W, lengthscale, 5, point)
                for point in jnp.asarray(FEATURES["points"])
            ]
        )
        assert_matches_oracle(features, FEATURES["rff_forward"], rtol=1e-12)

    def test_rff_cosine_forward_equals_geonnax(self) -> None:
        W = jnp.asarray(FEATURES["W"])
        phases = jnp.asarray(FEATURES["b"])
        lengthscale = float(FEATURES["lengthscale"])
        features = jnp.stack(
            [
                rff_cosine_forward(W, phases, lengthscale, 5, point)
                for point in jnp.asarray(FEATURES["points"])
            ]
        )
        assert_matches_oracle(
            features, FEATURES["rff_cosine_forward"], rtol=1e-12, atol=1e-15
        )

    @pytest.mark.parametrize(
        ("rescale", "key"),
        [(False, "fourier_features"), (True, "fourier_features_rescaled")],
    )
    def test_fourier_features_equal_geonnax(self, rescale: bool, key: str) -> None:
        features = fourier_features(jnp.asarray(FEATURES["x"]), 4, rescale=rescale)
        assert_matches_oracle(features, FEATURES[key], rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize(
        ("rescale", "key"),
        [(False, "seasonal_features"), (True, "seasonal_features_rescaled")],
    )
    def test_seasonal_features_equal_geonnax(self, rescale: bool, key: str) -> None:
        features = seasonal_features(
            jnp.asarray(FEATURES["t"]), (7.0, 365.25), (2, 3), rescale=rescale
        )
        assert_matches_oracle(features, FEATURES[key], rtol=1e-12, atol=1e-15)

    def test_gaussian_window_features_equal_geonnax(self) -> None:
        windows = gaussian_window_features(
            jnp.asarray(FEATURES["t"]),
            jnp.asarray(FEATURES["centers"]),
            jnp.asarray(FEATURES["widths"]),
        )
        assert_matches_oracle(windows, FEATURES["gaussian_window_features"], rtol=1e-12)
