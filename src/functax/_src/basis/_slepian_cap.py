"""Spherical Slepian functions of an axisymmetric cap.

Slepian functions are band-limited combinations of the real spherical
harmonics that concentrate the most energy inside a region. For a cap the
concentration operator is block-diagonal in the azimuthal order m, so it is
built and diagonalised block by block, and the retained modes are evaluated
against `real_spherical_harmonics`. Moved from geonnax ``_slepian`` at
``be5537d``.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import math
from typing import Any

import einx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from functax._src.basis._legendre import (
    _check_degree,
    associated_legendre,
    associated_legendre_indices,
)
from functax._src.basis._spherical_harmonics import (
    _check_unit_points,
    real_spherical_harmonics,
)
from functax._src.geometry._frames import rotation_to_pole
from functax._src.quad._rules import gauss_legendre
from functax.exceptions import ValidationError


__all__ = [
    "SlepianCapBasis",
    "shannon_number",
    "slepian_cap_basis",
    "slepian_cap_eigh_per_m",
    "slepian_concentration_matrix",
]


def _check_cap_radius(cap_radius: Any) -> None:
    if isinstance(cap_radius, int | float):
        radius = float(cap_radius)
        if not 0.0 < radius <= math.pi:
            msg = f"'cap_radius' must lie in (0, π], got {radius!r}"
            raise ValidationError(msg)


def _sh_index(ell: int, m: int) -> int:
    """The column of Y_lm in the layout of `real_spherical_harmonics`."""
    return ell * ell + (m + ell)


def _cap_quadrature(
    cap_radius: float | Float[Array, ""], num_quadrature: int
) -> tuple[Float[Array, " node"], Float[Array, " node"]]:
    """Gauss–Legendre on z = cos θ over the cap, [cos Θ, 1].

    The nodes and weights on [−1, 1] are host constants for each
    ``num_quadrature``; the scaling by Θ stays in JAX, so Θ may be traced.
    """
    nodes, weights = gauss_legendre(num_quadrature)
    nodes = jnp.asarray(nodes)
    weights = jnp.asarray(weights)
    radius = jnp.asarray(cap_radius)
    # z_min = cos Θ
    z_min = jnp.cos(radius)
    # z = ½(1 − cos Θ) s + ½(1 + cos Θ), the nodes s mapped onto [cos Θ, 1]
    z = 0.5 * (1.0 - z_min) * nodes + 0.5 * (1.0 + z_min)
    # w = ½(1 − cos Θ) w_s
    w = 0.5 * (1.0 - z_min) * weights
    return z, w


def _slepian_block(
    l_max: int,
    cap_radius: float | Float[Array, ""],
    m: int,
    num_quadrature: int,
) -> Float[Array, "row col"]:
    """The block D^m_ll′ = ∫_cap Y_l^m Y_l′^m dΩ for l, l′ = |m|, …, l_max."""
    m_abs = abs(m)
    z, w = _cap_quadrature(cap_radius, num_quadrature)
    # the orthonormal P̄_l^m (unit L² norm on [−1, 1]) for l = |m|, …, l_max; the
    # real-harmonic normalisation is P̄/√(2π) for m = 0 and √2 P̄/√(2π) otherwise
    columns = [
        i
        for i, (_, order) in enumerate(associated_legendre_indices(l_max))
        if order == m_abs
    ]
    legendre = associated_legendre(z, l_max)
    p_bar = legendre[:, jnp.asarray(columns)]
    factor = (math.sqrt(2.0) if m_abs > 0 else 1.0) / math.sqrt(2.0 * math.pi)
    transposed = einx.id("node row -> row node", p_bar)
    # the latitude parts of Y_l^m
    rows = factor * transposed
    # w_q P_l(z_q)
    weighted = einx.multiply("row node, node", rows, w)
    # the azimuthal integral: 2π for m = 0, π otherwise
    phi_factor = 2.0 * jnp.pi if m_abs == 0 else jnp.pi
    # ∫ P_l P_l′ dz ≈ ∑_q w_q P_l(z_q) P_l′(z_q)
    latitude_integral = einx.dot("row node, col node -> row col", weighted, rows)
    # D^m_ll′ = (∫ dφ) ∫ P_l P_l′ dz
    return phi_factor * latitude_integral


def shannon_number(l_max: int, area: float | Float[Array, ""]) -> Float[Array, ""]:
    """The Shannon number N = (l_max + 1)² A / (4π), the count of concentrated modes.

    Args:
        l_max: The highest spherical-harmonic degree.
        area: A, the region's area on the unit sphere.

    Returns:
        N, the expected number of well-concentrated Slepian functions.

    Raises:
        ValidationError: If ``l_max`` is not an integer of at least 0.

    Examples:
        >>> import math
        >>> import jax.numpy as jnp
        >>> from functax.basis.slepian import shannon_number
        >>> # the whole sphere holds all (l_max + 1)² modes
        >>> bool(jnp.allclose(shannon_number(3, 4.0 * math.pi), 16.0))
        True
        >>> bool(jnp.allclose(shannon_number(3, math.pi), 4.0))
        True
        >>> # a traced area works too
        >>> bool(jnp.allclose(shannon_number(0, jnp.asarray(2.0 * math.pi)), 0.5))
        True
    """
    degree = _check_degree(l_max, "l_max")
    area_array = jnp.asarray(area)
    # N = (l_max + 1)² A / (4π)
    return ((degree + 1) ** 2) * area_array / (4.0 * jnp.pi)


def slepian_concentration_matrix(
    l_max: int,
    cap_radius: float | Float[Array, ""],
    *,
    num_quadrature: int | None = None,
) -> Float[Array, "row col"]:
    """The concentration matrix D_ij = ∫_cap Y_i Y_j dΩ of a north-pole cap.

    The cap is axisymmetric, so D is block-diagonal in the order m.

    Algorithm:
        1. Lay a Gauss–Legendre rule on z = cos θ over [cos Θ, 1].
        2. For each order m, integrate the products of the P̄_l^m by
           quadrature and scale by the azimuthal integral.
        3. Scatter each block into the rows and columns of its Y_lm.

    Args:
        l_max: The highest spherical-harmonic degree.
        cap_radius: Θ, the cap half-angle in radians, in (0, π].
        num_quadrature: The Gauss–Legendre order over cos θ; by default
            max(2 l_max + 3, 8), exact for the products up to degree 2 l_max.

    Returns:
        D, over the (l_max + 1)² harmonics in the order of
        `real_spherical_harmonics`.

    Raises:
        ValidationError: If ``l_max`` is not an integer of at least 0, a
            concrete Θ is outside (0, π], or ``num_quadrature`` is below 1.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.slepian import slepian_concentration_matrix
        >>> slepian_concentration_matrix(1, 0.5).shape
        (4, 4)
        >>> whole = slepian_concentration_matrix(2, jnp.pi)
        >>> bool(jnp.allclose(whole, jnp.eye(9), atol=1e-12))
        True
        >>> # its trace is the Shannon number of the cap, area 2π(1 − cos Θ)
        >>> from functax.basis.slepian import shannon_number
        >>> matrix = slepian_concentration_matrix(3, 0.5)
        >>> area = 2 * jnp.pi * (1 - jnp.cos(0.5))
        >>> bool(jnp.allclose(jnp.trace(matrix), shannon_number(3, area)))
        True
    """
    degree = _check_degree(l_max, "l_max")
    _check_cap_radius(cap_radius)
    n_quad = num_quadrature or max(2 * degree + 3, 8)
    if n_quad < 1:
        msg = f"'num_quadrature' must be at least 1, got {n_quad!r}"
        raise ValidationError(msg)
    n_harmonics = (degree + 1) ** 2
    matrix = jnp.zeros((n_harmonics, n_harmonics))
    for m in range(-degree, degree + 1):
        block = _slepian_block(degree, cap_radius, m, n_quad)
        indices = jnp.asarray([_sh_index(ell, m) for ell in range(abs(m), degree + 1)])
        rows_and_cols = jnp.ix_(indices, indices)
        matrix = matrix.at[rows_and_cols].set(block)
    return matrix


def slepian_cap_eigh_per_m(
    l_max: int,
    cap_radius: float | Float[Array, ""],
    *,
    num_quadrature: int | None = None,
) -> tuple[Float[Array, " feature"], Float[Array, "harmonic feature"]]:
    """The cap Slepian eigenproblem D g = λ g, solved one azimuthal block at a time.

    Algorithm:
        1. For each order m, build the block D^m and solve D^m g = λ g.
        2. Sort each block's pairs by decreasing λ, and scatter its
           eigenvectors into the rows of its Y_lm.
        3. Join the blocks, sort every pair by decreasing λ, and clip λ to
           [0, 1].

    Args:
        l_max: The highest spherical-harmonic degree.
        cap_radius: Θ, the cap half-angle in radians, in (0, π].
        num_quadrature: The Gauss–Legendre order over cos θ (see
            `slepian_concentration_matrix`).

    Returns:
        ``(eigenvalues, coeffs)``: the (l_max + 1)² concentrations λ in [0, 1],
        descending, and the harmonic coefficients of each Slepian function as
        the matching columns.

    Raises:
        ValidationError: If ``l_max`` is not an integer of at least 0, or a
            concrete Θ is outside (0, π].

    Examples:
        >>> from functax.basis.slepian import slepian_cap_eigh_per_m
        >>> eigenvalues, coeffs = slepian_cap_eigh_per_m(1, 0.5)
        >>> eigenvalues.shape, coeffs.shape
        ((4,), (4, 4))
        >>> bool((eigenvalues[:-1] >= eigenvalues[1:]).all())
        True
        >>> # the whole sphere concentrates every mode fully
        >>> import jax.numpy as jnp
        >>> whole, _ = slepian_cap_eigh_per_m(1, jnp.pi)
        >>> bool(jnp.allclose(whole, 1.0))
        True
    """
    degree = _check_degree(l_max, "l_max")
    _check_cap_radius(cap_radius)
    n_quad = num_quadrature or max(2 * degree + 3, 8)
    n_harmonics = (degree + 1) ** 2
    eigenvalue_blocks = []
    coeff_blocks = []
    # the eigen-solves and sorts run in this order on purpose: in float32 the
    # first concentrations round to exactly 1.0, so their ties keep the order
    # these sorts leave (see `slepian_cap_basis`)
    for m in range(-degree, degree + 1):
        block = _slepian_block(degree, cap_radius, m, n_quad)
        # D^m g = λ g
        values, vectors = jnp.linalg.eigh(block)
        # descending λ within the block
        ascending = jnp.argsort(values)
        order = ascending[::-1]
        values = values[order]
        vectors = vectors[:, order]
        indices = jnp.asarray([_sh_index(ell, m) for ell in range(abs(m), degree + 1)])
        coeff_block = jnp.zeros((n_harmonics, vectors.shape[1]))
        coeff_block = coeff_block.at[indices, :].set(vectors)
        eigenvalue_blocks.append(values)
        coeff_blocks.append(coeff_block)
    # the blocks have different sizes, so they are joined as lists
    values = jnp.hstack(eigenvalue_blocks)
    coeffs = jnp.hstack(coeff_blocks)
    # descending λ over every block
    ascending = jnp.argsort(values)
    order = ascending[::-1]
    sorted_values = values[order]
    # λ ∈ [0, 1], clipped against round-off
    concentrations = jnp.clip(sorted_values, 0.0, 1.0)
    return concentrations, coeffs[:, order]


def _centered_coordinates(
    unit_xyz: Float[Array, "obs 3"], lonlat_centre: Float[Array, " 2"]
) -> Float[Array, "obs 3"]:
    # rotate so the centre sits at the north pole (rows east, north, up)
    rotation = rotation_to_pole(lonlat_centre, unit="radians")
    return einx.dot("obs dim, axis dim -> obs axis", unit_xyz, rotation)


def _select_modes(
    values: Float[Array, " feature"],
    coeffs: Float[Array, "harmonic feature"],
    n_modes: int | None,
    eig_threshold: float | None,
) -> tuple[Float[Array, " kept"], Float[Array, "harmonic kept"]]:
    """The modes (by decreasing concentration) above the threshold, then the first n."""
    if eig_threshold is not None:
        above = values > eig_threshold
        # padded to a static size with −1, then trimmed on the host
        (keep,) = jnp.nonzero(above, size=values.shape[0], fill_value=-1)
        keep = keep[keep >= 0]
        if n_modes is not None and int(keep.shape[0]) < n_modes:
            msg = (
                f"'eig_threshold'={eig_threshold} keeps only {int(keep.shape[0])} "
                f"modes but 'n_modes'={n_modes} were asked for; lower the threshold "
                "or reduce n_modes"
            )
            raise ValidationError(msg)
        values = values[keep]
        coeffs = coeffs[:, keep]
    if n_modes is not None:
        values = values[:n_modes]
        coeffs = coeffs[:, :n_modes]
    return values, coeffs


class SlepianCapBasis(eqx.Module):
    """A Slepian cap eigendecomposition in the real spherical-harmonic basis.

    Each retained Slepian function is g_k = ∑_i C_ik Y_i over the
    (l_max + 1)² harmonics, evaluated in the frame where the cap centre is the
    north pole.

    Attributes:
        l_max: The highest spherical-harmonic degree.
        coeffs: C, one column per retained mode, sorted by decreasing
            concentration.
        eigenvalues: The concentration λ_k of each mode.
        cap_radius: Θ, the cap half-angle in radians.
        lonlat_centre: (λ, φ) of the cap centre, in radians.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.slepian import slepian_cap_basis
        >>> basis = slepian_cap_basis(3, 0.35, n_modes=2, eig_threshold=0.0)
        >>> basis.evaluate(jnp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])).shape
        (2, 2)
        >>> basis.coeffs.shape, basis.eigenvalues.shape
        ((16, 2), (2,))
    """

    l_max: int = eqx.field(static=True)
    coeffs: Float[Array, "harmonic feature"]
    eigenvalues: Float[Array, " feature"]
    cap_radius: Float[Array, ""]
    lonlat_centre: Float[Array, " 2"]

    @property
    def num_modes(self) -> int:
        """The number of retained Slepian functions.

        Examples:
            >>> from functax.basis.slepian import slepian_cap_basis
            >>> slepian_cap_basis(3, 0.35, n_modes=2, eig_threshold=0.0).num_modes
            2
        """
        return int(self.eigenvalues.shape[0])

    def centred_coordinates(
        self, unit_xyz: Float[Array, "obs 3"]
    ) -> Float[Array, "obs 3"]:
        """The points rotated into the frame where the cap centre is the north pole.

        A consumer that evaluates the same harmonics in the cap frame (pyrox's
        Slepian inducing features) feeds these to `real_spherical_harmonics`
        to stay consistent with `evaluate`.

        Raises:
            ValidationError: If ``unit_xyz`` is not (N, 3).

        Examples:
            >>> import jax.numpy as jnp
            >>> from functax.basis.slepian import slepian_cap_basis
            >>> basis = slepian_cap_basis(3, 0.35, n_modes=2, eig_threshold=0.0)
            >>> xyz = jnp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
            >>> basis.centred_coordinates(xyz).shape
            (2, 3)
        """
        unit_xyz = jnp.asarray(unit_xyz)
        _check_unit_points(unit_xyz, "unit_xyz")
        return _centered_coordinates(unit_xyz, self.lonlat_centre)

    def evaluate(self, unit_xyz: Float[Array, "obs 3"]) -> Float[Array, "obs feature"]:
        """The retained Slepian functions g_k at points on the unit sphere.

        Raises:
            ValidationError: If ``unit_xyz`` is not (N, 3).

        Examples:
            >>> import jax.numpy as jnp
            >>> from functax.basis.slepian import slepian_cap_basis
            >>> basis = slepian_cap_basis(3, 0.35, n_modes=2, eig_threshold=0.0)
            >>> xyz = jnp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
            >>> basis.evaluate(xyz).shape
            (2, 2)

            Moving the cap moves the functions with it:

            >>> moved = basis.rotate_to(jnp.array([0.0, 0.0]))
            >>> at_pole = basis.evaluate(jnp.array([[0.0, 0.0, 1.0]]))
            >>> at_new_centre = moved.evaluate(jnp.array([[1.0, 0.0, 0.0]]))
            >>> bool(jnp.allclose(at_pole, at_new_centre, atol=1e-6))
            True
        """
        centered = self.centred_coordinates(unit_xyz)
        harmonics = real_spherical_harmonics(centered, self.l_max)
        # g_k = ∑_i C_ik Y_i
        return einx.dot(
            "obs harmonic, harmonic feature -> obs feature", harmonics, self.coeffs
        )

    def rotate_to(self, lonlat_centre: Float[Array, " 2"]) -> SlepianCapBasis:
        """The same cap basis centred at ``lonlat_centre`` ((λ, φ) in radians).

        Raises:
            ValidationError: If ``lonlat_centre`` is not (2,).

        Examples:
            >>> import jax.numpy as jnp
            >>> from functax.basis.slepian import slepian_cap_basis
            >>> basis = slepian_cap_basis(3, 0.35, n_modes=2, eig_threshold=0.0)
            >>> basis.rotate_to(jnp.array([0.1, 0.2])).num_modes
            2
        """
        centre = jnp.asarray(lonlat_centre)
        if centre.shape != (2,):
            msg = f"'lonlat_centre' must have shape (2,), got {centre.shape}"
            raise ValidationError(msg)
        return eqx.tree_at(lambda basis: basis.lonlat_centre, self, centre)


def slepian_cap_basis(
    l_max: int,
    cap_radius: float | Float[Array, ""],
    *,
    n_modes: int | None = None,
    eig_threshold: float | None = None,
    lonlat_centre: Float[Array, " 2"] | None = None,
    num_quadrature: int | None = None,
) -> SlepianCapBasis:
    """A Slepian basis for an axisymmetric spherical cap.

    Algorithm:
        1. Solve the cap eigenproblem block by block with
           `slepian_cap_eigh_per_m`.
        2. Keep the modes above ``eig_threshold``, then the first
           ``n_modes``.
        3. Store the cap centre, the north pole by default.

    Args:
        l_max: The highest spherical-harmonic degree.
        cap_radius: Θ, the cap half-angle in radians, in (0, π].
        n_modes: At most this many modes. With ``eig_threshold`` too, the
            basis is first trimmed by threshold and then cut to ``n_modes``;
            a threshold that leaves fewer than ``n_modes`` raises, since a
            silently smaller basis would break static shapes downstream.
        eig_threshold: Keep the modes with concentration above this value.
        lonlat_centre: (λ, φ) of the cap centre in radians; the north pole by
            default.
        num_quadrature: The Gauss–Legendre order over cos θ.

    Returns:
        A `SlepianCapBasis`, its columns sorted by decreasing concentration.
        Modes whose concentrations are equal in the working precision keep
        whatever order the sort leaves them in, which can differ from the
        float64 order: for a 40° cap at ``l_max=20`` the first five
        concentrations are all 1.0 in float32, and mode 1 is not the
        axisymmetric one. Enable float64 when that order matters.

    Raises:
        ValidationError: If ``n_modes`` is below 1, the threshold keeps fewer
            than ``n_modes``, ``lonlat_centre`` is not (2,), or an argument of
            `slepian_cap_eigh_per_m` is invalid.

    Examples:
        >>> from functax.basis.slepian import slepian_cap_basis
        >>> slepian_cap_basis(3, 0.35, n_modes=2, eig_threshold=0.0).num_modes
        2
        >>> # by default every one of the (l_max + 1)² modes is kept
        >>> slepian_cap_basis(2, 0.5).num_modes
        9
        >>> # a threshold that keeps too few modes is rejected
        >>> slepian_cap_basis(
        ...     2, 0.5, n_modes=9, eig_threshold=0.5
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValidationError: 'eig_threshold'=0.5 keeps only 2 modes
    """
    if n_modes is not None and n_modes < 1:
        msg = f"'n_modes' must be at least 1, got {n_modes!r}"
        raise ValidationError(msg)
    values, coeffs = slepian_cap_eigh_per_m(
        l_max, cap_radius, num_quadrature=num_quadrature
    )
    values, coeffs = _select_modes(values, coeffs, n_modes, eig_threshold)
    centre = (
        jnp.asarray([0.0, 0.5 * jnp.pi])
        if lonlat_centre is None
        else jnp.asarray(lonlat_centre)
    )
    if centre.shape != (2,):
        msg = f"'lonlat_centre' must have shape (2,), got {centre.shape}"
        raise ValidationError(msg)
    return SlepianCapBasis(
        l_max=l_max,
        coeffs=coeffs,
        eigenvalues=values,
        cap_radius=jnp.asarray(cap_radius),
        lonlat_centre=centre,
    )
