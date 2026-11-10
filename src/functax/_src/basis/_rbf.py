"""Placeable radial bases: Gaussian and compactly supported Wendland bumps.

A radial basis puts a kernel at a centre c_a with a width ℓ_a and evaluates
φ_a(x) = K(‖x − c_a‖/ℓ_a). Unlike the spectral bases, its columns are local
bumps placed where the caller wants them (a river mouth, a front) without
touching the rest of the domain. A Gaussian gives a smooth global bump; the
Wendland kernels are exactly zero past their width, so the basis and its Gram
matrix are sparse by construction. Moved from geonnax ``_rbf`` at ``be5537d``.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float

from functax._src.geometry._sphere import _safe_norm, geodesic_distance
from functax.exceptions import ValidationError


__all__ = [
    "rbf_basis",
    "spherical_rbf_basis",
    "wendland_c2",
    "wendland_c4",
]


_KERNELS: tuple[str, ...] = ("gaussian", "wendland_c2", "wendland_c4")


def _check_kernel(kernel: str) -> None:
    if kernel not in _KERNELS:
        msg = f"'kernel' must be one of {list(_KERNELS)}, got {kernel!r}"
        raise ValidationError(msg)


def wendland_c2(r: Float[Array, ...]) -> Float[Array, ...]:
    """The Wendland C² kernel K(r) = (1 − r)₊⁴ (4r + 1).

    Positive definite in dimensions d ≤ 3.

    Args:
        r: Scaled distances, any shape.

    Returns:
        K(r), with the shape of ``r``; zero for r ≥ 1.

    References:
        Wendland, H. (2004). Scattered Data Approximation. Cambridge
        University Press.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.local import wendland_c2
        >>> wendland_c2(jnp.array([0.0, 1.5])).tolist()
        [1.0, 0.0]
    """
    return jnp.where(r < 1.0, (1.0 - r) ** 4 * (4.0 * r + 1.0), 0.0)


def wendland_c4(r: Float[Array, ...]) -> Float[Array, ...]:
    """The Wendland C⁴ kernel K(r) = (1 − r)₊⁶ (35r² + 18r + 3)/3.

    Positive definite in dimensions d ≤ 3.

    Args:
        r: Scaled distances, any shape.

    Returns:
        K(r), with the shape of ``r``; zero for r ≥ 1.

    References:
        Wendland, H. (2004). Scattered Data Approximation. Cambridge
        University Press.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.local import wendland_c4
        >>> wendland_c4(jnp.array([0.0, 1.5])).tolist()
        [1.0, 0.0]
    """
    return jnp.where(
        r < 1.0, (1.0 - r) ** 6 * (35.0 * r**2 + 18.0 * r + 3.0) / 3.0, 0.0
    )


def rbf_basis(
    x: Float[Array, "obs dim"],
    centers: Float[Array, "feature dim"],
    widths: Float[Array, " feature"],
    *,
    kernel: str = "gaussian",
) -> Float[Array, "obs feature"]:
    """A placeable radial basis at given centres and widths.

    Column a is φ_a(x) = K(‖x − c_a‖/ℓ_a). Only Φ is returned: the prior
    variance per centre is the caller's choice, and the geometry (``centers``,
    ``widths``) is already theirs. Wendland columns are exactly zero past
    their width.

    Args:
        x: The evaluation points.
        centers: c_a, the centre of each atom.
        widths: ℓ_a, the Gaussian length scale or the Wendland support radius
            of each atom.
        kernel: ``"gaussian"`` (a smooth global bump) or ``"wendland_c2"`` /
            ``"wendland_c4"`` (compact support, positive definite for d ≤ 3).

    Returns:
        Φ, each atom at each point.

    Raises:
        ValidationError: If ``kernel`` is unknown.

    References:
        Buhmann, M. D. (2003). Radial Basis Functions. Cambridge University
        Press.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.local import rbf_basis
        >>> rbf_basis(
        ...     jnp.zeros((5, 2)),
        ...     jnp.ones((3, 2)),
        ...     jnp.ones(3),
        ...     kernel="wendland_c2",
        ... ).shape
        (5, 3)
        >>> # every atom peaks at 1 on its centre
        >>> centres = jnp.array([[0.0, 0.0], [2.0, 1.0]])
        >>> bool(
        ...     jnp.allclose(
        ...         jnp.diag(rbf_basis(centres, centres, jnp.ones(2))), 1.0
        ...     )
        ... )
        True
    """
    _check_kernel(kernel)
    difference = einx.subtract("obs dim, feature dim -> obs feature dim", x, centers)
    # ‖x − c_a‖²
    squared = einx.sum("obs feature [dim]", difference**2)
    if kernel == "gaussian":
        # exp(−‖x − c‖²/(2ℓ²)) from the squared distance, so the gradient is
        # clean at r = 0 with no square root
        scaled_sq = einx.divide("obs feature, feature", squared, widths**2)
        return jnp.exp(-0.5 * scaled_sq)
    # r = ‖x − c‖, with a zero gradient at r = 0 rather than NaN
    distance = _safe_norm(squared)
    scaled = einx.divide("obs feature, feature", distance, widths)
    return wendland_c2(scaled) if kernel == "wendland_c2" else wendland_c4(scaled)


def spherical_rbf_basis(
    unit_xyz: Float[Array, "obs 3"],
    centers: Float[Array, "feature 3"],
    widths: Float[Array, " feature"],
    *,
    kernel: str = "gaussian",
) -> Float[Array, "obs feature"]:
    """A placeable radial basis on the unit sphere, in geodesic distance.

    The spherical counterpart of `rbf_basis`: φ_a(x) = K(d(x, c_a)/ℓ_a) with d
    the great-circle distance, so an atom's support is a geodesic cap,
    rotation-invariant on the sphere. The distance is `geodesic_distance`'s
    atan2(‖x × c‖, x · c), accurate at every angle and with a zero gradient at a
    coincident centre (geonnax used a guarded arccos, which loses precision
    near 0 and π). The inputs are assumed to lie on the unit sphere.

    Args:
        unit_xyz: The evaluation directions.
        centers: c_a, the centre of each atom, on the unit sphere.
        widths: ℓ_a, the geodesic length scale or Wendland cap radius of each
            atom, in radians.
        kernel: ``"gaussian"`` (a smooth cap) or ``"wendland_c2"`` /
            ``"wendland_c4"`` (compact geodesic support).

    Returns:
        Φ, each atom at each point.

    Raises:
        ValidationError: If ``kernel`` is unknown.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.local import spherical_rbf_basis
        >>> spherical_rbf_basis(jnp.eye(3)[:2], jnp.eye(3), jnp.ones(3)).shape
        (2, 3)
        >>> # a quarter turn away is π/2 radians
        >>> value = spherical_rbf_basis(
        ...     jnp.eye(3)[:1], jnp.eye(3)[1:2], jnp.ones(1)
        ... )
        >>> bool(jnp.allclose(value, jnp.exp(-0.5 * (jnp.pi / 2) ** 2)))
        True
    """
    _check_kernel(kernel)
    distance = geodesic_distance(unit_xyz, centers)
    if kernel == "gaussian":
        scaled_sq = einx.divide("obs feature, feature", distance**2, widths**2)
        return jnp.exp(-0.5 * scaled_sq)
    scaled = einx.divide("obs feature, feature", distance, widths)
    return wendland_c2(scaled) if kernel == "wendland_c2" else wendland_c4(scaled)
