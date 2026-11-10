"""Overcomplete multiscale Gabor frames: Gaussian envelopes times oscillations.

A Gabor atom multiplies a Gaussian envelope of width L_a by a cosine of
wavenumber k_a, radial or along an orientation. A dyadic bank of them over a
box is a redundant frame, the overcomplete counterpart of the orthonormal
wavelet basis. Only Φ and the per-atom geometry are returned: recovering
coefficients from a field is a regulariser-dependent frame-dual problem that
stays downstream. Moved from geonnax ``_gabor`` at ``be5537d``.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import einx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from functax._src.geometry._sphere import _safe_norm
from functax.exceptions import ValidationError


__all__ = [
    "gabor_frame",
    "gabor_frame_grid",
]


# the softening of the orientation normalisation, for a zero vector
_EPS = 1e-12


def gabor_frame(
    x: Float[Array, "obs dim"],
    centers: Float[Array, "feature dim"],
    scales: Float[Array, " feature"],
    wavenumbers: Float[Array, " feature"],
    *,
    orientations: Float[Array, "feature dim"] | None = None,
) -> Float[Array, "obs feature"]:
    """A bank of Gabor atoms, Gaussian envelopes times oscillations.

    φ_a(x) = exp(−‖x − c_a‖²/(2L_a²)) cos(k_a r_a), with r_a = ‖x − c_a‖ for
    radial atoms (``orientations`` is None) or r_a = (x − c_a) · θ_a for
    directional ones (the orientations θ_a are normalised to unit length).

    Args:
        x: The evaluation points.
        centers: c_a, the centre of each atom.
        scales: L_a, the envelope width of each atom.
        wavenumbers: k_a, the oscillation wavenumber of each atom.
        orientations: θ_a, the direction of each directional atom; None gives
            radial atoms.

    Returns:
        Φ, each atom at each point.

    References:
        Mallat, S. (2008). A Wavelet Tour of Signal Processing, 3rd edition.
        Academic Press.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.local import gabor_frame
        >>> gabor_frame(
        ...     jnp.zeros((4, 2)), jnp.ones((6, 2)), jnp.ones(6), jnp.ones(6)
        ... ).shape
        (4, 6)
        >>> # at its centre an atom is exp(0) cos(0) = 1
        >>> at_centre = gabor_frame(
        ...     jnp.zeros((1, 2)), jnp.zeros((1, 2)), jnp.ones(1), jnp.ones(1)
        ... )
        >>> at_centre.tolist()
        [[1.0]]
    """
    difference = einx.subtract("obs dim, feature dim -> obs feature dim", x, centers)
    # ‖x − c_a‖²
    squared = einx.sum("obs feature [dim]", difference**2)
    envelope = jnp.exp(-0.5 * einx.divide("obs feature, feature", squared, scales**2))
    if orientations is None:
        # the radial distance, with a zero gradient at the centre
        distance = _safe_norm(squared)
        phase = einx.multiply("obs feature, feature", distance, wavenumbers)
    else:
        norm = jnp.sqrt(einx.sum("feature [dim]", orientations**2))
        unit = einx.divide("feature dim, feature", orientations, norm + _EPS)
        # (x − c_a) · θ_a
        projection = einx.dot(
            "obs feature dim, feature dim -> obs feature", difference, unit
        )
        phase = einx.multiply("obs feature, feature", projection, wavenumbers)
    return envelope * jnp.cos(phase)


def gabor_frame_grid(
    x: Float[Array, "obs dim"],
    bounds: Float[Array, "dim 2"],
    *,
    n_scales: int,
    base_scale: float,
    oversample: float = 1.0,
) -> tuple[
    Float[Array, "obs feature"],
    Float[Array, "feature dim"],
    Float[Array, " feature"],
    Float[Array, " feature"],
]:
    """A dyadic radial Gabor frame over a box.

    The scales are L_s = base_scale · 2^s for s = 0, …, n_scales − 1; at each
    scale the centres lie on a regular grid of spacing L_s/oversample over
    ``bounds``, with wavenumber k_s = 2π/L_s. The centre layout is built on the
    host (few scales, data-independent), so only `gabor_frame` is on the hot
    path.

    Args:
        x: The evaluation points.
        bounds: The [lo, hi] of each axis, a concrete array (it lays out the
            grid on the host), not a traced one.
        n_scales: The number of dyadic scales.
        base_scale: L₀, the finest scale.
        oversample: The grid density per scale; above 1 packs the centres
            tighter than one envelope width.

    Returns:
        ``(Phi, centers, scales, wavenumbers)``: the atoms at each point and
        the geometry of each atom, so a consumer can both synthesise
        ``Phi @ w`` and weight the atoms by scale.

    Raises:
        ValidationError: If ``n_scales`` is below 1, ``base_scale`` or
            ``oversample`` is not positive, or ``bounds`` is not (d, 2) with
            lo < hi.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.local import gabor_frame_grid
        >>> bounds = jnp.array([[0.0, 1.0], [0.0, 1.0]])
        >>> Phi, centers, scales, wavenumbers = gabor_frame_grid(
        ...     jnp.zeros((10, 2)), bounds, n_scales=2, base_scale=0.25
        ... )
        >>> Phi.shape[0], centers.shape[1], scales.shape == wavenumbers.shape
        (10, 2, True)
    """
    if n_scales < 1:
        msg = f"'n_scales' must be at least 1, got {n_scales!r}"
        raise ValidationError(msg)
    if not base_scale > 0:
        msg = f"'base_scale' must be positive, got {base_scale!r}"
        raise ValidationError(msg)
    if not oversample > 0:
        msg = f"'oversample' must be positive, got {oversample!r}"
        raise ValidationError(msg)
    bounds_np = np.asarray(bounds, dtype=np.float64)
    if bounds_np.ndim != 2 or bounds_np.shape[1] != 2:
        msg = f"'bounds' must have shape (d, 2), got {bounds_np.shape}"
        raise ValidationError(msg)
    if np.any(bounds_np[:, 1] <= bounds_np[:, 0]):
        msg = "'bounds' must have lo < hi on every axis"
        raise ValidationError(msg)

    d = bounds_np.shape[0]
    centers_blocks, scale_blocks, wavenumber_blocks = [], [], []
    for scale_index in range(n_scales):
        length = base_scale * 2.0**scale_index
        spacing = length / oversample
        # floor the step count so every centre stays inside [lo, hi] when the
        # spacing does not divide the interval (the 1e-9 catches an endpoint
        # that rounding puts just under)
        axes = [
            lo + spacing * np.arange(int(np.floor((hi - lo) / spacing + 1e-9)) + 1)
            for lo, hi in bounds_np
        ]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, d)
        centers_blocks.append(grid)
        scale_blocks.append(np.full(grid.shape[0], length))
        wavenumber_blocks.append(np.full(grid.shape[0], 2.0 * np.pi / length))

    centers = jnp.asarray(np.concatenate(centers_blocks), dtype=x.dtype)
    scales = jnp.asarray(np.concatenate(scale_blocks), dtype=x.dtype)
    wavenumbers = jnp.asarray(np.concatenate(wavenumber_blocks), dtype=x.dtype)
    return gabor_frame(x, centers, scales, wavenumbers), centers, scales, wavenumbers
