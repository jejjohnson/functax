"""Feature maps of coordinates: Fourier features at given frequencies, and windows.

Each map evaluates a fixed set of basis functions φ(x) at the inputs. The
frequencies, periods and window centres are arguments: drawing random
frequencies from a kernel's spectral density is sampling, which stays with the
library that owns the kernel (kernellib, geonnax). Moved from geonnax
``randfeat`` and ``basis`` at ``be5537d`` with their names and signatures.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

from collections.abc import Sequence

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float, Real

from functax._src._dtypes import promote_integers
from functax.exceptions import ValidationError


__all__ = [
    "fourier_features",
    "gaussian_window_features",
    "rff_cosine_forward",
    "rff_forward",
    "seasonal_features",
    "seasonal_frequencies",
]


def rff_forward(
    W: Float[Array, "dim feature"],
    lengthscale: float | Float[Array, ""],
    n_features: int,
    x: Float[Array, " dim"],
) -> Float[Array, " 2*feature"]:
    """Random Fourier features at given frequencies, cosines then sines.

    φ(x) = √(1/F) [cos(xᵀW/ℓ), sin(xᵀW/ℓ)], so φ(x)ᵀφ(x′) approximates the
    stationary kernel whose spectral density W was drawn from. Concatenating
    cosines and sines doubles the width to 2F.

    Args:
        W: The frequencies, one column per feature.
        lengthscale: ℓ, which divides every projection.
        n_features: F in the normalisation √(1/F), normally the number of
            columns of ``W``.
        x: One input point.

    Returns:
        φ(x), the cosines of the F projections followed by their sines.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.features import rff_forward
        >>> W = jnp.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.5]])
        >>> rff_forward(W, 1.0, 3, jnp.array([0.3, -0.2])).shape
        (6,)
        >>> features = rff_forward(W, 1.0, 3, jnp.zeros(2))
        >>> bool(jnp.allclose(features @ features, 1.0))
        True
    """
    # z = xᵀW / ℓ
    projection = einx.dot("dim, dim feature -> feature", x, W) / lengthscale
    scale = jnp.sqrt(1.0 / n_features)
    # φ = √(1/F) [cos z, sin z]
    features = einx.id(
        "feature, feature -> (feature + feature)",
        jnp.cos(projection),
        jnp.sin(projection),
    )
    return scale * features


def rff_cosine_forward(
    W: Float[Array, "dim feature"],
    b: Float[Array, " feature"],
    lengthscale: float | Float[Array, ""],
    n_features: int,
    x: Float[Array, " dim"],
) -> Float[Array, " feature"]:
    """Random Fourier features with random phases, one cosine per frequency.

    φ(x) = √(2/F) cos(xᵀW/ℓ + b): the single-cosine form, whose width is the
    number of frequencies F.

    Args:
        W: The frequencies, one column per feature.
        b: The phases, one per feature (uniform on [0, 2π) for the usual
            estimator).
        lengthscale: ℓ, which divides every projection.
        n_features: F in the normalisation √(2/F), normally the number of
            columns of ``W``.
        x: One input point.

    Returns:
        φ(x), one cosine per frequency.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.features import rff_cosine_forward
        >>> W = jnp.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.5]])
        >>> rff_cosine_forward(W, jnp.zeros(3), 1.0, 3, jnp.ones(2)).shape
        (3,)
        >>> features = rff_cosine_forward(W, jnp.zeros(3), 1.0, 3, jnp.zeros(2))
        >>> bool(jnp.allclose(features, jnp.sqrt(2.0 / 3.0)))
        True
    """
    # z = xᵀW / ℓ + b
    projection = einx.dot("dim, dim feature -> feature", x, W) / lengthscale + b
    # φ = √(2/F) cos z
    return jnp.sqrt(2.0 / n_features) * jnp.cos(projection)


def fourier_features(
    x: Real[Array, " sample"],
    max_degree: int,
    *,
    rescale: bool = False,
) -> Float[Array, "sample feature"]:
    """Cosines and sines of a coordinate at dyadic frequencies.

    φ_d,cos(x) = cos(2π 2ᵈ x) and φ_d,sin(x) = sin(2π 2ᵈ x) for
    d = 0, …, D − 1, laid out as [cos₀, …, cos_{D−1}, sin₀, …, sin_{D−1}]
    (the layout of Google's bayesnf).

    Args:
        x: The coordinate at each sample. Integers become float32.
        max_degree: D, the number of dyadic frequencies; the output has 2D
            columns.
        rescale: Divide each (cos_d, sin_d) pair by d + 1, which biases a
            prior on the coefficients toward the low frequencies.

    Returns:
        The features, 2D columns per sample, in the dtype of ``x``.

    Raises:
        ValidationError: If ``max_degree`` is negative.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.features import fourier_features
        >>> fourier_features(jnp.linspace(0.0, 1.0, 5), max_degree=3).shape
        (5, 6)
        >>> features = fourier_features(jnp.array([0.0]), max_degree=2)
        >>> features.tolist()
        [[1.0, 1.0, 0.0, 0.0]]
    """
    if max_degree < 0:
        msg = f"'max_degree' must be non-negative, got {max_degree}"
        raise ValidationError(msg)
    x = promote_integers(x)
    if max_degree == 0:
        # einx cannot solve for an axis of length zero
        return jnp.zeros((x.shape[0], 0), dtype=x.dtype)
    # the degrees in the dtype of x, so a float32 input stays float32 under x64
    degrees = jnp.arange(max_degree, dtype=x.dtype)
    # ω_d = 2π 2ᵈ
    frequencies = 2.0 * jnp.pi * 2.0**degrees
    # z = x ω_d
    phase = einx.multiply("sample, degree -> sample degree", x, frequencies)
    features = einx.id(
        "sample degree, sample degree -> sample (degree + degree)",
        jnp.cos(phase),
        jnp.sin(phase),
    )
    if rescale:
        weights = degrees + 1
        divisor = einx.id("degree, degree -> (degree + degree)", weights, weights)
        features = einx.divide("sample feature, feature", features, divisor)
    return features


def seasonal_frequencies(
    periods: Sequence[float],
    harmonics: Sequence[int],
) -> tuple[list[int], list[float]]:
    """The frequencies of the seasonal features, as Python lists.

    For each period τ_p with H_p harmonics, the frequencies are
    f_p,h = h / τ_p for h = 1, …, H_p, F = ∑_p H_p in all. The inputs and the
    result are Python values, so this runs while tracing and never needs a
    concrete array under ``jax.jit``.

    Args:
        periods: The periods τ_p.
        harmonics: H_p, the number of harmonics of each period.

    Returns:
        ``(period_index, frequency)``: the index p and the frequency of each of
        the F features.

    Raises:
        ValidationError: If ``periods`` and ``harmonics`` differ in length.

    Examples:
        >>> from functax.basis.features import seasonal_frequencies
        >>> seasonal_frequencies([7.0, 365.0], [1, 2])[0]
        [0, 1, 1]
        >>> seasonal_frequencies([4.0], [2])[1]
        [0.25, 0.5]
    """
    if len(periods) != len(harmonics):
        msg = (
            "'periods' and 'harmonics' must have the same length, got "
            f"{len(periods)} and {len(harmonics)}"
        )
        raise ValidationError(msg)
    period_index: list[int] = []
    frequencies: list[float] = []
    for index, (period, count) in enumerate(zip(periods, harmonics, strict=True)):
        for harmonic in range(1, int(count) + 1):
            period_index.append(index)
            frequencies.append(float(harmonic) / float(period))
    return period_index, frequencies


def seasonal_features(
    x: Real[Array, " sample"],
    periods: Sequence[float],
    harmonics: Sequence[int],
    *,
    rescale: bool = False,
) -> Float[Array, "sample feature"]:
    """Cosines and sines of a time at the harmonics of given periods.

    φ_p,h,cos(x) = cos(2π h x / τ_p) and φ_p,h,sin(x) = sin(2π h x / τ_p) for
    h = 1, …, H_p, laid out as the F cosines followed by the F sines, with
    F = ∑_p H_p. The periods and harmonic counts are Python values, so the map
    traces under ``jax.jit`` and ``lax.scan``.

    Args:
        x: The time at each sample. Integers become float32.
        periods: The periods τ_p, in the units of ``x``.
        harmonics: H_p, the number of harmonics of each period.
        rescale: Divide each (cos, sin) pair by its harmonic h within its
            period, which biases a prior toward the long wavelengths.

    Returns:
        The features, 2F columns per sample, in the dtype of ``x`` (zero
        columns when there are no harmonics).

    Raises:
        ValidationError: If ``periods`` and ``harmonics`` differ in length.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.features import seasonal_features
        >>> x = jnp.linspace(0.0, 10.0, 4)
        >>> seasonal_features(x, [7.0, 365.0], [1, 2]).shape
        (4, 6)
        >>> seasonal_features(x, [7.0], [0]).shape
        (4, 0)
    """
    x = promote_integers(x)
    _, frequency_list = seasonal_frequencies(periods, harmonics)
    if not frequency_list:
        return jnp.zeros((x.shape[0], 0), dtype=x.dtype)
    # the frequencies in the dtype of x: a float64 time (seconds since an
    # epoch) keeps a float64 phase rather than a float32-rounded 1/τ
    frequencies = jnp.asarray(frequency_list, dtype=x.dtype)
    # z = 2π f x
    phase = einx.multiply(
        "sample, frequency -> sample frequency", x, 2.0 * jnp.pi * frequencies
    )
    features = einx.id(
        "sample frequency, sample frequency -> sample (frequency + frequency)",
        jnp.cos(phase),
        jnp.sin(phase),
    )
    if rescale:
        within: list[float] = []
        for count in harmonics:
            within.extend(range(1, int(count) + 1))
        weights = jnp.asarray(within, dtype=x.dtype)
        divisor = einx.id(
            "frequency, frequency -> (frequency + frequency)", weights, weights
        )
        features = einx.divide("sample feature, feature", features, divisor)
    return features


def gaussian_window_features(
    t: Float[Array, " sample"],
    centers: Float[Array, " feature"],
    widths: Float[Array, " feature"],
) -> Float[Array, "sample feature"]:
    """Gaussian windows in time, the temporal analogue of an RBF column.

    χ_a(t) = exp(−(t − τ_a)² / (2 T_a²)): a soft gate centred at τ_a with width
    T_a, the localised counterpart of `seasonal_features`.

    Args:
        t: The time at each sample.
        centers: The window centres τ_a.
        widths: The window widths T_a.

    Returns:
        χ_a(t), one column per window.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.features import gaussian_window_features
        >>> t = jnp.linspace(0.0, 10.0, 5)
        >>> gaussian_window_features(
        ...     t, jnp.array([2.0, 8.0]), jnp.array([1.0, 1.0])
        ... ).shape
        (5, 2)
        >>> window = gaussian_window_features(
        ...     jnp.array([5.0]), jnp.array([5.0]), jnp.array([2.0])
        ... )
        >>> window.tolist()
        [[1.0]]
    """
    # z = (t − τ_a) / T_a
    offset = einx.subtract("sample, feature -> sample feature", t, centers)
    scaled = einx.divide("sample feature, feature", offset, widths)
    # χ = exp(−z²/2)
    return jnp.exp(-0.5 * scaled**2)
