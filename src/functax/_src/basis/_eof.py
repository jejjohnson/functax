"""Empirical orthogonal functions (EOF, PCA): the data-driven basis of a field.

EOFs are the dominant spatial patterns of a dataset, the right singular vectors
of a (sample, grid) matrix, so the eigenvectors of its spatial covariance.
Unlike the closed-form eigenbases, they adapt to the covariance in the data
(the DINEOF construction for ocean gap filling). With T samples, mode a carries
variance σ_a²/(T − 1); the singular values are returned and the variance
convention stays with the caller. Moved from geonnax ``_eof`` at ``be5537d``.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float

from functax.exceptions import ValidationError


__all__ = ["eof_basis"]


def eof_basis(
    data: Float[Array, "sample cell"],
    n_modes: int,
    *,
    center: bool = True,
) -> tuple[Float[Array, "cell feature"], Float[Array, " feature"]]:
    """The leading ``n_modes`` empirical orthogonal functions of a data matrix.

    The economy SVD X = U S Vᵀ of the data X ∈ ℝ^{T×N} (centred on the sample
    mean of each grid point by default) gives the spatial patterns as the
    columns of V ∈ ℝ^{N×min(T, N)}, orthonormal, with their singular values
    σ_a, a = 1, …, min(T, N); the prior variance per mode is σ_a²/(T − 1).

    Algorithm:
        1. Optionally centre each grid point on its sample mean, X − x̄ with
           x̄ ∈ ℝ^N.
        2. Take the economy SVD X = U S Vᵀ.
        3. Keep the first M rows of Vᵀ as the columns of Φ ∈ ℝ^{N×M}, with
           their singular values.

    Args:
        data: X, T samples (time snapshots, say) over N grid points.
        n_modes: M, the number of leading EOFs.
        center: Subtract the per-grid-point sample mean first, so the EOFs
            describe anomalies about the mean.

    Returns:
        ``(Phi, singular_values)``: the M orthonormal patterns over the grid,
        and their singular values, descending.

    Raises:
        ValidationError: If ``data`` is not 2-D or ``n_modes`` is outside
            [1, min(T, N)].

    References:
        Beckers, J.-M., Rixen, M. (2003). EOF calculations and data filling
        from incomplete oceanographic datasets. Journal of Atmospheric and
        Oceanic Technology 20(12), 1839–1856.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.eigen import eof_basis
        >>> data = jnp.arange(24.0).reshape(4, 6)
        >>> Phi, singular_values = eof_basis(data, n_modes=2)
        >>> Phi.shape, singular_values.shape
        ((6, 2), (2,))
        >>> bool(jnp.allclose(Phi.T @ Phi, jnp.eye(2), atol=1e-12))
        True

        Rank-one data has one non-zero singular value, when it is not centred:

        >>> rank_one = jnp.outer(jnp.arange(1.0, 5.0), jnp.ones(6))
        >>> _, values = eof_basis(rank_one, n_modes=2, center=False)
        >>> bool(jnp.allclose(values[1], 0.0, atol=1e-12))
        True

        Asking for more modes than min(T, N) is rejected:

        >>> eof_basis(data, n_modes=5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValidationError: 'n_modes' must lie in [1, 4], got 5
    """
    if data.ndim != 2:
        msg = f"'data' must be 2-D (T, N), got shape {data.shape}"
        raise ValidationError(msg)
    n_samples, n_grid = data.shape
    max_modes = min(n_samples, n_grid)
    if not 1 <= n_modes <= max_modes:
        msg = f"'n_modes' must lie in [1, {max_modes}], got {n_modes!r}"
        raise ValidationError(msg)
    if center:
        # x̄ = (1/T) ∑ₜ xₜ, over the rows xₜ ∈ ℝ^N of X
        sample_mean = einx.mean("[sample] cell", data)
        # X − x̄
        data = einx.subtract("sample cell, cell", data, sample_mean)
    # X = U S Vᵀ, the economy SVD: the rows of Vᵀ are the spatial EOFs
    _, singular_values, right_vectors = jnp.linalg.svd(data, full_matrices=False)
    leading = right_vectors[:n_modes]
    # Φ, the first M columns of V
    patterns = einx.id("feature cell -> cell feature", leading)
    return patterns, singular_values[:n_modes]
