"""Orthonormal discrete-wavelet-transform (DWT) basis matrices.

The columns are an orthonormal multiscale wavelet basis, Φ = Wᵀ for the
periodic DWT analysis operator W built by cascading convolve-and-downsample
levels, so Φᵀ Φ = I and a field is reconstructed exactly as Φ (Φᵀ f). This is
the critically sampled counterpart of the overcomplete Gabor frames: periodic
boundaries, power-of-two sizes, and the Haar, db2 and db4 filters, built once
on the host. Moved from geonnax ``_wavelet`` at ``be5537d``.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from functax.exceptions import ValidationError


__all__ = [
    "wavelet_basis_1d",
    "wavelet_basis_2d",
]


# the orthonormal scaling (low-pass) filters, ∑h = √2 and ∑h² = 1, in the
# PyWavelets ``dec_lo`` convention
_FILTERS: dict[str, tuple[float, ...]] = {
    "haar": (0.7071067811865476, 0.7071067811865476),
    "db2": (
        -0.12940952255092145,
        0.22414386804185735,
        0.836516303737469,
        0.48296291314469025,
    ),
    "db4": (
        -0.010597401784997278,
        0.032883011666982945,
        0.030841381835986965,
        -0.18703481171888114,
        -0.02798376941698385,
        0.6308807679295904,
        0.7148465705525415,
        0.23037781330885523,
    ),
}


def _qmf(h: np.ndarray) -> np.ndarray:
    """The high-pass quadrature-mirror filter g_k = (−1)^k h_{L−1−k}."""
    k = np.arange(h.shape[0])
    return ((-1.0) ** k) * h[::-1]


def _level_matrix(m: int, h: np.ndarray, g: np.ndarray) -> np.ndarray:
    """One periodic convolve-and-downsample level as an m × m matrix.

    The first m/2 rows are the low-pass (approximation) outputs and the rest
    the high-pass (detail) outputs; circular boundaries make it orthonormal for
    an orthonormal mirror pair.
    """
    half = m // 2
    length = h.shape[0]
    w = np.zeros((m, m))
    for i in range(half):
        for k in range(length):
            col = (2 * i + k) % m
            w[i, col] += h[k]
            w[half + i, col] += g[k]
    return w


def wavelet_basis_1d(
    n: int,
    *,
    wavelet: str = "haar",
    levels: int | None = None,
) -> Float[Array, "row col"]:
    """The orthonormal 1-D DWT synthesis basis on a periodic grid of length n.

    Args:
        n: The grid length, a power of two of at least 2.
        wavelet: ``"haar"``, ``"db2"`` or ``"db4"``.
        levels: The number of decomposition levels; the full cascade log₂ n by
            default, and capped there.

    Returns:
        Φ, n × n with the orthonormal basis functions as columns (Φᵀ Φ = I).

    Raises:
        ValidationError: If ``n`` is not a power of two of at least 2,
            ``wavelet`` is unknown, or ``levels`` is below 1.

    References:
        Daubechies, I. (1992). Ten Lectures on Wavelets. SIAM.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.local import wavelet_basis_1d
        >>> Phi = wavelet_basis_1d(8, wavelet="db2")
        >>> bool(jnp.allclose(Phi.T @ Phi, jnp.eye(8), atol=1e-12))
        True
        >>> wavelet_basis_1d(4, levels=1).shape
        (4, 4)
    """
    if n < 2 or (n & (n - 1)) != 0:
        msg = f"'n' must be a power of two of at least 2, got {n!r}"
        raise ValidationError(msg)
    if wavelet not in _FILTERS:
        msg = f"'wavelet' must be one of {list(_FILTERS)}, got {wavelet!r}"
        raise ValidationError(msg)
    # log₂ n exactly, since n is a power of two
    max_levels = n.bit_length() - 1
    levels = max_levels if levels is None else levels
    if levels < 1:
        msg = f"'levels' must be at least 1, got {levels!r}"
        raise ValidationError(msg)
    levels = min(levels, max_levels)

    h = np.asarray(_FILTERS[wavelet], dtype=np.float64)
    g = _qmf(h)
    w = np.eye(n)
    size = n
    for _ in range(levels):
        if size < 2:
            break
        level = np.eye(n)
        level[:size, :size] = _level_matrix(size, h, g)
        w = level @ w
        size //= 2
    # the transform is orthonormal, so the synthesis basis is the transpose
    return jnp.asarray(w.T)


def wavelet_basis_2d(
    ny: int,
    nx: int | None = None,
    *,
    wavelet: str = "haar",
    levels: int | None = None,
) -> Float[Array, "row col"]:
    """The separable orthonormal 2-D DWT basis, the Kronecker product Φ_y ⊗ Φ_x.

    Its rows index the row-major flattened (ny, nx) grid (row iy · nx + ix),
    and it stays orthonormal.

    Args:
        ny: The number of rows, a power of two of at least 2.
        nx: The number of columns, a power of two of at least 2; ``ny`` by
            default.
        wavelet: ``"haar"``, ``"db2"`` or ``"db4"``.
        levels: The decomposition levels per axis; the full cascade by
            default.

    Returns:
        Φ, of size ny · nx × ny · nx.

    Raises:
        ValidationError: If an axis or the options are invalid (see
            `wavelet_basis_1d`).

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.local import wavelet_basis_2d
        >>> wavelet_basis_2d(4, 4, wavelet="haar").shape
        (16, 16)
        >>> Phi = wavelet_basis_2d(2, 4)
        >>> bool(jnp.allclose(Phi.T @ Phi, jnp.eye(8), atol=1e-12))
        True
    """
    nx = ny if nx is None else nx
    phi_y = wavelet_basis_1d(ny, wavelet=wavelet, levels=levels)
    phi_x = wavelet_basis_1d(nx, wavelet=wavelet, levels=levels)
    return jnp.kron(phi_y, phi_x)
