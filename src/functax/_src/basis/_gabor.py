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
        >>> bool(jnp.allclose(at_centre, 1.0))
        True

        With a very wide envelope, a directional atom does not change
        perpendicular to θ_a:

        >>> along_y = jnp.array([[0.0, 0.0], [0.0, 3.0]])
        >>> atom = gabor_frame(
        ...     along_y,
        ...     jnp.zeros((1, 2)),
        ...     jnp.full(1, 1e6),
        ...     jnp.ones(1),
        ...     orientations=jnp.array([[1.0, 0.0]]),
        ... )
        >>> bool(jnp.allclose(atom[0], atom[1]))
        True
    """
    # x − c_a
    difference = einx.subtract("obs dim, feature dim -> obs feature dim", x, centers)
    # ‖x − c_a‖²
    squared = einx.sum("obs feature [dim]", difference**2)
    squared_scales = scales**2
    # ‖x − c_a‖²/L_a²
    scaled_sq = einx.divide("obs feature, feature", squared, squared_scales)
    # exp(−‖x − c_a‖²/(2L_a²))
    exponent = -0.5 * scaled_sq
    envelope = jnp.exp(exponent)
    if orientations is None:
        # r_a = ‖x − c_a‖, with a zero gradient at the centre
        distance = _safe_norm(squared)
        # k_a r_a
        phase = einx.multiply("obs feature, feature", distance, wavenumbers)
    else:
        # ‖θ_a‖²
        squared_norm = einx.sum("feature [dim]", orientations**2)
        norm = jnp.sqrt(squared_norm)
        softened = norm + _EPS
        # θ_a/‖θ_a‖
        unit = einx.divide("feature dim, feature", orientations, softened)
        # r_a = (x − c_a) · θ_a
        projection = einx.dot(
            "obs feature dim, feature dim -> obs feature", difference, unit
        )
        # k_a r_a
        phase = einx.multiply("obs feature, feature", projection, wavenumbers)
    oscillation = jnp.cos(phase)
    # φ_a = exp(−‖x − c_a‖²/(2L_a²)) cos(k_a r_a)
    return envelope * oscillation


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

    Algorithm:
        1. For each scale s, set L_s = base_scale · 2^s and the spacing
           L_s/oversample.
        2. Lay the centres on a regular grid of that spacing from each lower
           bound, with as many steps as stay inside the box.
        3. Join the scales and evaluate the atoms with `gabor_frame`.

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

        One scale of width ½ over the unit square has 3 × 3 centres:

        >>> Phi, centers, scales, wavenumbers = gabor_frame_grid(
        ...     jnp.zeros((10, 2)), bounds, n_scales=1, base_scale=0.5
        ... )
        >>> centers.shape, bool(jnp.allclose(wavenumbers, 4 * jnp.pi))
        ((9, 2), True)
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
    lower = bounds_np[:, 0]
    upper = bounds_np[:, 1]
    if np.any(upper <= lower):
        msg = "'bounds' must have lo < hi on every axis"
        raise ValidationError(msg)

    centers_blocks, scale_blocks, wavenumber_blocks = [], [], []
    for scale_index in range(n_scales):
        # L_s = L₀ 2^s
        length = base_scale * 2.0**scale_index
        spacing = length / oversample
        axes = []
        for lo, hi in bounds_np:
            # floor the step count so every centre stays inside [lo, hi] when
            # the spacing does not divide the interval (the 1e-9 catches an
            # endpoint that rounding puts just under)
            steps = np.floor((hi - lo) / spacing + 1e-9)
            offsets = spacing * np.arange(int(steps) + 1)
            axes.append(lo + offsets)
        grids = np.meshgrid(*axes, indexing="ij")
        stacked = np.stack(grids)
        grid = einx.id("dim ... -> (...) dim", stacked)
        centers_blocks.append(grid)
        scale_blocks.append(np.full(grid.shape[0], length))
        # k_s = 2π/L_s
        wavenumber = 2.0 * np.pi / length
        wavenumber_blocks.append(np.full(grid.shape[0], wavenumber))

    # the scales have different numbers of centres, so they are joined as lists
    all_centers = np.concatenate(centers_blocks)
    all_scales = np.concatenate(scale_blocks)
    all_wavenumbers = np.concatenate(wavenumber_blocks)
    centers = jnp.asarray(all_centers, dtype=x.dtype)
    scales = jnp.asarray(all_scales, dtype=x.dtype)
    wavenumbers = jnp.asarray(all_wavenumbers, dtype=x.dtype)
    atoms = gabor_frame(x, centers, scales, wavenumbers)
    return atoms, centers, scales, wavenumbers
