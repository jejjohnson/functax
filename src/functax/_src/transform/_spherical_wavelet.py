"""The spherical wavelet (scale-discretised, needlet) transform on a lat/lon grid.

A multiresolution analysis built on `SphericalHarmonicTransform`: each scale j
is a band-pass window hⱼ(l) in harmonic space, and the windowed coefficients are
synthesised back to a full-resolution map, one per scale (Marinucci et al. 2008;
Wiaux et al. 2008). The squared windows partition unity, ∑ⱼ hⱼ(l)² = 1, so
synthesis after analysis is exact up to the harmonic transform's own error.
Moved from geonnax ``layers/_spherical_wavelet`` at ``6e32651``, apart from
`SphericalWaveletConv`, which learns weights per scale and stays in geonnax.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from functax._src.basis._legendre import _check_degree
from functax._src.transform._sht import SphericalHarmonicTransform


__all__ = ["SphericalWaveletTransform"]


def _meyer_nu(t: np.ndarray) -> np.ndarray:
    """The Meyer polynomial ν: 0 for t ≤ 0, 1 for t ≥ 1, smooth in between.

    ν(t) = t⁴ (35 − 84t + 70t² − 20t³) on [0, 1], and ν(t) + ν(1 − t) = 1,
    which makes the sine and cosine tapers of `_partition_filters` a squared
    partition of unity.
    """
    t = np.clip(t, 0.0, 1.0)
    return t**4 * (35.0 - 84.0 * t + 70.0 * t**2 - 20.0 * t**3)


def _partition_filters(l_max: int, n_scales: int | None = None) -> np.ndarray:
    """Dyadic harmonic windows hⱼ(l) over the degrees l = 0, …, l_max.

    With x = log₂ max(l, 1) and u = x − j, scale j rises as
    hⱼ = sin(π/2 · ν(u + 1)) for −1 < u ≤ 0 and falls as
    hⱼ = cos(π/2 · ν(u)) for 0 < u < 1, and is 0 elsewhere; scale 0 is also 1
    for x ≤ 0, as it has no coarser neighbour. Two scales overlap on each
    octave, so ∑ⱼ hⱼ(l)² = 1 for every l when the scales cover every octave
    (the default, ⌈log₂ max(l_max, 2)⌉ + 1 of them). Fewer scales leave the
    highest degrees uncovered.

    Returns:
        The windows, (n_scales, l_max + 1), non-negative, in float64.
    """
    degrees = np.arange(l_max + 1)
    x = np.log2(np.maximum(degrees, 1))
    j_max = int(np.ceil(np.log2(max(l_max, 2))))
    n = j_max + 1 if n_scales is None else n_scales

    filters = np.zeros((n, l_max + 1))
    for k in range(n):
        u = x - k
        g = np.zeros_like(x)
        # the hand-off from the coarser scale
        rising = (u > -1.0) & (u <= 0.0)
        g[rising] = np.sin(0.5 * np.pi * _meyer_nu(u[rising] + 1.0))
        # the hand-off to the finer scale
        falling = (u > 0.0) & (u < 1.0)
        g[falling] = np.cos(0.5 * np.pi * _meyer_nu(u[falling]))
        if k == 0:
            g[x <= 0.0] = 1.0
        filters[k] = g
    return filters


class SphericalWaveletTransform(eqx.Module):
    """Scale-discretised (needlet) wavelet analysis and synthesis on the sphere.

    Analysis maps a field y to one band-pass map per scale,
    bⱼ(x) = ∑_{lm} hⱼ(l) c_lm Y_lm(x) with c = SHT(y), and synthesis sums the
    maps back through the windows again, y = ∑ⱼ SHT⁻¹(hⱼ · SHT(bⱼ)). The
    squared windows partition unity, so synthesis after analysis is the
    identity on fields band-limited to l_max, up to the SHT's own error. The
    windows pass through ``lax.stop_gradient``, as the SHT tables do.

    Attributes:
        sht: The spherical-harmonic transform it wraps.
        filters: The windows hⱼ(l), (n_scales, l_max + 1).
        degree_index: The degree l of each harmonic coefficient.
        n_scales: S, the number of scales.

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from functax.transform import (
        ...     SphericalHarmonicTransform,
        ...     SphericalWaveletTransform,
        ... )
        >>> sht = SphericalHarmonicTransform.init(n_lat=12, n_lon=24, l_max=8)
        >>> swt = SphericalWaveletTransform.init(sht)
        >>> field = sht.inverse(jr.normal(jr.key(0), (2, 81)))
        >>> bands = swt.forward(field)
        >>> bands.shape
        (4, 2, 12, 24)
        >>> bool(jnp.allclose(swt.inverse(bands), field, atol=1e-12))
        True

    References:
        Marinucci, D., Pietrobon, D., Balbi, A., Baldi, P., Cabella, P.,
        Kerkyacharian, G., Natoli, P., Picard, D., Vittorio, N. (2008).
        Spherical needlets for cosmic microwave background data analysis.
        Monthly Notices of the Royal Astronomical Society, 383(2), 539–545.

        Wiaux, Y., McEwen, J. D., Vandergheynst, P., Blanc, O. (2008).
        Exact reconstruction with directional wavelets on the sphere.
        Monthly Notices of the Royal Astronomical Society, 388(2), 770–788.
    """

    sht: SphericalHarmonicTransform
    filters: Float[Array, "scale degree"]
    degree_index: Int[Array, " feature"]
    n_scales: int = eqx.field(static=True)

    @classmethod
    def init(
        cls, sht: SphericalHarmonicTransform, *, n_scales: int | None = None
    ) -> SphericalWaveletTransform:
        """The wavelet windows for the band limit of ``sht``.

        Args:
            sht: The spherical-harmonic transform to wrap.
            n_scales: S, the number of dyadic scales. The default,
                ⌈log₂ max(l_max, 2)⌉ + 1, covers every octave up to l_max, so
                the round trip is exact; fewer scales leave the highest
                degrees out.

        Returns:
            The `SphericalWaveletTransform`, its windows in the default dtype.

        Raises:
            ValidationError: If ``n_scales`` is not an integer of at least 1.

        Examples:
            >>> from functax.transform import (
            ...     SphericalHarmonicTransform,
            ...     SphericalWaveletTransform,
            ... )
            >>> sht = SphericalHarmonicTransform.init(5, 9, 4)
            >>> SphericalWaveletTransform.init(sht).filters.shape
            (3, 5)
        """
        if n_scales is not None:
            n_scales = _check_degree(n_scales, "n_scales", minimum=1)
        filters = _partition_filters(sht.l_max, n_scales)
        degree_index = jnp.asarray(sht.degrees, dtype=jnp.int32)
        return cls(
            sht=sht,
            filters=jnp.asarray(filters),
            degree_index=degree_index,
            n_scales=filters.shape[0],
        )

    def _windows(self, dtype: jnp.dtype) -> Float[Array, "scale feature"]:
        """hⱼ(l) at each harmonic coefficient in ``dtype``, the gradient stopped."""
        filters = jax.lax.stop_gradient(self.filters).astype(dtype)
        return einx.get_at(
            "scale [degree], feature -> scale feature", filters, self.degree_index
        )

    def forward(
        self, field: Float[Array, "channel lat lon"]
    ) -> Float[Array, "scale channel lat lon"]:
        """Analysis, bⱼ = SHT⁻¹(hⱼ · SHT(y)), one band-pass map per scale.

        Args:
            field: The values on the (n_lat, n_lon) grid, one field per channel.

        Returns:
            The band-pass maps, (n_scales, channel, n_lat, n_lon).

        Examples:
            >>> import jax.numpy as jnp
            >>> from functax.transform import (
            ...     SphericalHarmonicTransform,
            ...     SphericalWaveletTransform,
            ... )
            >>> sht = SphericalHarmonicTransform.init(5, 9, 4)
            >>> swt = SphericalWaveletTransform.init(sht)
            >>> # a constant is degree 0, which only the coarsest scale holds
            >>> bands = swt.forward(jnp.ones((1, 5, 9)))
            >>> bool(jnp.allclose(bands[0], 1.0)), bool(jnp.allclose(bands[1:], 0.0))
            (True, True)
        """
        # c = SHT(y), (channel, feature)
        coeffs = self.sht.forward(field)
        # hⱼ(l) c_lm for each scale
        windowed = einx.multiply(
            "scale feature, channel feature -> scale channel feature",
            self._windows(coeffs.dtype),
            coeffs,
        )
        return jax.vmap(self.sht.inverse)(windowed)

    def inverse(
        self, scales: Float[Array, "scale channel lat lon"]
    ) -> Float[Array, "channel lat lon"]:
        """Synthesis, y = SHT⁻¹(∑ⱼ hⱼ · SHT(bⱼ)), from the band-pass maps.

        Args:
            scales: The band-pass maps, (n_scales, channel, n_lat, n_lon).

        Returns:
            The field on the (n_lat, n_lon) grid.

        Examples:
            >>> import jax.numpy as jnp
            >>> from functax.transform import (
            ...     SphericalHarmonicTransform,
            ...     SphericalWaveletTransform,
            ... )
            >>> sht = SphericalHarmonicTransform.init(5, 9, 4)
            >>> swt = SphericalWaveletTransform.init(sht)
            >>> swt.inverse(jnp.zeros((3, 2, 5, 9))).shape
            (2, 5, 9)
        """
        # SHT(bⱼ), (scale, channel, feature)
        coeffs = jax.vmap(self.sht.forward)(scales)
        # ∑ⱼ hⱼ(l) SHT(bⱼ)_lm
        combined = einx.dot(
            "scale feature, scale channel feature -> channel feature",
            self._windows(coeffs.dtype),
            coeffs,
        )
        return self.sht.inverse(combined)
