"""The spherical-harmonic transform: dense analysis and synthesis on a lat/lon grid.

A matrix transform built from `real_spherical_harmonics` and a Gauss–Legendre
latitude rule, differentiable and with no external SHT package. Analysis is
c_lm = ∑_q w_q y(x_q) Y_lm(x_q) and synthesis y(x) = ∑_{l ≤ l_max} ∑_m c_lm Y_lm(x);
on a grid with n_lat > l_max and n_lon > 2 l_max, analysis after synthesis is
exact for band-limited fields. Moved from geonnax ``layers/_spherical_transform``
at ``be5537d``, apart from the spectral layers, which stay in geonnax.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from functax._src.basis._legendre import _check_degree
from functax._src.basis._spherical_harmonics import (
    harmonic_degrees,
    real_spherical_harmonics,
)
from functax._src.quad._gauss_legendre import gauss_legendre
from functax.exceptions import ValidationError


__all__ = ["SphericalHarmonicTransform"]


def _sht_grid(n_lat: int, n_lon: int) -> tuple[np.ndarray, np.ndarray]:
    """The transform's grid: unit vectors and quadrature weights, latitude-major.

    The sines of latitude are the Gauss–Legendre nodes (ascending, south to
    north) and the longitudes are λ_j = 2πj/n_lon from 0, geonnax's layout.
    `gauss_legendre_grid` starts its longitudes at −π instead, which would roll
    every field by half a turn, so the transform keeps its own longitudes.
    """
    z, latitude_weights = gauss_legendre(n_lat)
    cos_latitude = np.sqrt(np.clip(1.0 - z**2, 0.0, None))
    longitude = 2.0 * np.pi * np.arange(n_lon) / n_lon
    # u = (cos φ cos λ, cos φ sin λ, sin φ) on the (lat, lon) grid
    x = np.outer(cos_latitude, np.cos(longitude))
    y = np.outer(cos_latitude, np.sin(longitude))
    sin_latitude = np.repeat(z, n_lon).reshape(n_lat, n_lon)
    xyz = np.stack([x, y, sin_latitude], axis=-1).reshape(-1, 3)
    weights = np.repeat(latitude_weights * (2.0 * np.pi / n_lon), n_lon)
    return xyz, weights


class SphericalHarmonicTransform(eqx.Module):
    """Dense, differentiable spherical-harmonic analysis and synthesis on a grid.

    Built for one Gauss–Legendre (n_lat, n_lon) grid and band limit l_max,
    with the coefficients in the layout of `real_spherical_harmonics`
    ((l_max + 1)² real coefficients, degree l outermost). The tables are
    construction-time constants and pass through ``lax.stop_gradient``, so
    training a model that holds a transform never updates them.

    spectraldiffx's complex transform û(l, m) (an FFT in longitude, then
    orthonormal P̃_l^|m| in latitude, rows north to south) holds the same
    information on the same grid: c_l0 = √(2π)/n_lon Re û(l, 0), and for m > 0,
    c_lm = 2√π/n_lon Re û(l, m) and c_l,−m = −2√π/n_lon Im û(l, m).

    Attributes:
        synth: The synthesis matrix, from coefficients to grid values.
        analy: The analysis matrix, from grid values to coefficients.
        degrees: The degree l of each coefficient.
        n_lat: The number of Gauss–Legendre latitudes.
        n_lon: The number of equispaced longitudes.
        l_max: The band limit.

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from functax.transform import SphericalHarmonicTransform
        >>> sht = SphericalHarmonicTransform.init(n_lat=12, n_lon=24, l_max=8)
        >>> sht.forward(jnp.ones((2, 12, 24))).shape
        (2, 81)
        >>> coefficients = jr.normal(jr.key(0), (2, 81))
        >>> bool(
        ...     jnp.allclose(
        ...         sht.forward(sht.inverse(coefficients)), coefficients, atol=1e-12
        ...     )
        ... )
        True
    """

    synth: Float[Array, "node feature"]
    analy: Float[Array, "feature node"]
    degrees: tuple[int, ...] = eqx.field(static=True)
    n_lat: int = eqx.field(static=True)
    n_lon: int = eqx.field(static=True)
    l_max: int = eqx.field(static=True)

    @classmethod
    def init(cls, n_lat: int, n_lon: int, l_max: int) -> SphericalHarmonicTransform:
        """The transform matrices for a Gauss–Legendre lat/lon grid.

        The grid must resolve the band limit, or analysis and synthesis are
        not inverse on l_max (high modes alias silently): Gauss–Legendre needs
        n_lat > l_max (it is exact to degree 2 n_lat − 1), and equispaced
        longitudes need n_lon > 2 l_max to hold every order m.

        Args:
            n_lat: The number of Gauss–Legendre latitudes, above l_max.
            n_lon: The number of equispaced longitudes, above 2 l_max.
            l_max: The band limit.

        Returns:
            The `SphericalHarmonicTransform`, its tables in the default dtype.

        Raises:
            ValidationError: If ``l_max`` is not an integer of at least 0, or
                the grid does not resolve it.

        Examples:
            >>> from functax.transform import SphericalHarmonicTransform
            >>> SphericalHarmonicTransform.init(5, 9, 4).synth.shape
            (45, 25)
        """
        degree = _check_degree(l_max, "l_max")
        if n_lat <= degree:
            msg = (
                "'n_lat' must exceed l_max for an exact Gauss–Legendre rule, got "
                f"n_lat={n_lat} and l_max={degree}"
            )
            raise ValidationError(msg)
        if n_lon <= 2 * degree:
            msg = (
                "'n_lon' must exceed 2 l_max to hold every order m, got "
                f"n_lon={n_lon} and l_max={degree}"
            )
            raise ValidationError(msg)
        xyz, weights = _sht_grid(n_lat, n_lon)
        # the tables are construction-time constants, so they are built in
        # float64 (finite and orthonormal far past float32's limits) and cast
        # to the default dtype once
        with jax.enable_x64(True):
            synth = np.asarray(
                real_spherical_harmonics(jnp.asarray(xyz, dtype=jnp.float64), degree)
            )
        analy = (synth * weights[:, None]).T
        return cls(
            synth=jnp.asarray(synth),
            analy=jnp.asarray(analy),
            degrees=harmonic_degrees(degree),
            n_lat=n_lat,
            n_lon=n_lon,
            l_max=degree,
        )

    def forward(
        self, field: Float[Array, "channel lat lon"]
    ) -> Float[Array, "channel feature"]:
        """Analysis, c_lm = ∑_q w_q y(x_q) Y_lm(x_q), for each channel.

        Args:
            field: The values on the (n_lat, n_lon) grid, one field per channel.

        Returns:
            The (l_max + 1)² coefficients of each channel.

        Examples:
            >>> import math
            >>> import jax.numpy as jnp
            >>> from functax.transform import SphericalHarmonicTransform
            >>> sht = SphericalHarmonicTransform.init(4, 8, 2)
            >>> # a constant 1 has c₀₀ = √(4π) and nothing else
            >>> coefficients = sht.forward(jnp.ones((1, 4, 8)))
            >>> bool(jnp.allclose(coefficients[0, 0], math.sqrt(4 * math.pi)))
            True
        """
        flat = einx.id("channel lat lon -> channel (lat lon)", field)
        analysis = jax.lax.stop_gradient(self.analy)
        return einx.dot("channel node, feature node -> channel feature", flat, analysis)

    def inverse(
        self, coeffs: Float[Array, "channel feature"]
    ) -> Float[Array, "channel lat lon"]:
        """Synthesis, y(x) = ∑_{l ≤ l_max} ∑_m c_lm Y_lm(x), for each channel.

        Args:
            coeffs: The (l_max + 1)² coefficients of each channel.

        Returns:
            The values on the (n_lat, n_lon) grid.

        Examples:
            >>> import jax.numpy as jnp
            >>> from functax.transform import SphericalHarmonicTransform
            >>> sht = SphericalHarmonicTransform.init(4, 8, 2)
            >>> sht.inverse(jnp.zeros((3, 9))).shape
            (3, 4, 8)
        """
        synthesis = jax.lax.stop_gradient(self.synth)
        flat = einx.dot(
            "channel feature, node feature -> channel node", coeffs, synthesis
        )
        return einx.id(
            "channel (lat lon) -> channel lat lon", flat, lat=self.n_lat, lon=self.n_lon
        )
