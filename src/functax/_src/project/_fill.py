"""Laplace and biharmonic gap fill of an image, with a five-point Laplacian.

Each axis has mirror edges, or is periodic (a longitude that wraps).

The two fills are the σ → 0 limits of the variational fit with P selecting
the observed pixels: the roughness D (first differences, DᵀD = −Δ) gives
Δx = 0 in the holes, and L = Δ gives Δ²x = 0. The solve runs over the hole
pixels only, with the observed pixels as Dirichlet values.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, ArrayLike, Bool, Float

from functax._src._checks import check_values, require_floating
from functax._src.project._variational import default_solver
from functax.exceptions import ValidationError


__all__ = [
    "FILL_METHODS",
    "fill_gaps",
    "mirror_laplacian",
    "periodic_laplacian",
]


FILL_METHODS: tuple[str, ...] = ("laplace", "biharmonic")
"""The gap fills: Δx = 0 or Δ²x = 0 in the holes."""


def mirror_laplacian(image: Float[Array, "row col"]) -> Float[Array, "row col"]:
    """Δx, the five-point Laplacian with mirror edges (the ghost pixel is the edge).

    (Δx)ᵢⱼ = xᵢ₋₁ⱼ + xᵢ₊₁ⱼ + xᵢⱼ₋₁ + xᵢⱼ₊₁ − 4xᵢⱼ, with x₋₁ⱼ = x₀ⱼ and its
    three companions at the other edges, so Δ = −DᵀD for the forward
    differences D, symmetric negative semi-definite with the constants as its
    null space. It is functax's only difference operator, there for
    `fill_gaps`; richer ones arrive as arguments (finitevolX, spectraldiffx).
    """
    padded = jnp.pad(image, 1, mode="edge")
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    return up + down + left + right - 4 * image


def periodic_laplacian(
    image: Float[Array, "row col"], *, periodic: tuple[bool, bool]
) -> Float[Array, "row col"]:
    """Δx, the five-point Laplacian with a periodic edge on each axis marked True.

    On a periodic axis of length n the neighbours of index j are j ± 1 mod n,

        (Δx)ᵢⱼ = xᵢ₋₁ⱼ + xᵢ₊₁ⱼ + xᵢ,₍ⱼ₋₁₎ mod n + xᵢ,₍ⱼ₊₁₎ mod n − 4xᵢⱼ

    for periodic columns, and an axis that is not periodic has mirror edges,
    as in `mirror_laplacian`. Δ stays symmetric negative semi-definite with
    the constants as its null space.

    Args:
        image: x, the image.
        periodic: Whether the rows and the columns wrap, ``(row, col)``.

    Returns:
        Δx, the shape of ``image``.
    """
    padded = image
    for axis, wraps in enumerate(periodic):
        widths = [(0, 0), (0, 0)]
        widths[axis] = (1, 1)
        padded = jnp.pad(padded, widths, mode="wrap" if wraps else "edge")
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    return up + down + left + right - 4 * image


def fill_gaps(
    image: Float[ArrayLike, "row col"],
    *,
    mask: Bool[ArrayLike, "row col"],
    method: str = "laplace",
    periodic: tuple[bool, bool] = (False, False),
    solver: lx.AbstractLinearSolver | None = None,
) -> Float[Array, "row col"]:
    """The image with its holes filled: Δx = 0 (Laplace) or Δ²x = 0 (biharmonic).

    With P the observed pixels and Q = I − P the holes, the fill keeps x = y on
    P and minimises the roughness over the holes: ‖Dx‖² = −xᵀΔx for
    ``"laplace"`` or ‖Δx‖² for ``"biharmonic"``, both with mirror edges
    (`mirror_laplacian`), or with an axis that wraps (`periodic_laplacian`),
    so a hole can reach observed pixels across the seam of a global
    longitude grid. Writing x = Py + z with z zero on P, z solves

        (QKQ + P) z = −QKPy,   K = −Δ (laplace) or Δ² (biharmonic),

    symmetric positive definite when every connected hole touches an observed
    pixel, so CG solves it matrix-free. The holes then satisfy Δx = 0 or
    Δ²x = 0, the two named cases of `Variational` as the observation error
    goes to zero. Hole values in ``image`` (NaN or not) are never read.

    Algorithm:
        1. y₀ = Py, the observed pixels with the holes zeroed.
        2. b = −Q K y₀.
        3. z = (QKQ + P)⁻¹ b by ``solver``.
        4. x = y₀ + z.

    Args:
        image: y, the image; its hole pixels are ignored.
        mask: True at the observed pixels P.
        method: ``"laplace"`` or ``"biharmonic"``.
        periodic: Whether the rows and the columns wrap, ``(row, col)``;
            ``(False, True)`` joins the east and west edges of a global
            longitude grid. Mirror edges on both axes by default.
        solver: The lineax solver; CG at √ε of the image dtype when ``None``.

    Returns:
        x, the observed pixels unchanged and the holes filled.

    Raises:
        ValidationError: If ``image`` is not a floating 2-D array, ``mask`` is
            not boolean of its shape, ``method`` is unknown, ``periodic`` is
            not two booleans, or no pixel is observed (checked when run under
            ``jit``).

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.project import fill_gaps
        >>> # one row [0, 1, ?, ?, 0]: linear between the samples, or a cubic
        >>> image = jnp.array([[0.0, 1.0, jnp.nan, jnp.nan, 0.0]])
        >>> observed = jnp.isfinite(image)
        >>> filled = fill_gaps(image, mask=observed)
        >>> bool(jnp.allclose(filled, jnp.array([[0.0, 1.0, 2 / 3, 1 / 3, 0.0]])))
        True
        >>> smooth = fill_gaps(image, mask=observed, method="biharmonic")
        >>> bool(jnp.allclose(smooth, jnp.array([[0.0, 1.0, 1.0, 0.5, 0.0]])))
        True
        >>> # a ring of four [0, ?, 3, ?]: each hole sits between 0 and 3
        >>> ring = jnp.array([[0.0, jnp.nan, 3.0, jnp.nan]])
        >>> wrapped = fill_gaps(
        ...     ring, mask=jnp.isfinite(ring), periodic=(False, True)
        ... )
        >>> bool(jnp.allclose(wrapped, jnp.array([[0.0, 1.5, 3.0, 1.5]])))
        True
    """
    image = jnp.asarray(image)
    mask = jnp.asarray(mask)
    if image.ndim != 2:
        msg = f"'image' must be 2-D (row, col), got shape {image.shape}"
        raise ValidationError(msg)
    require_floating(image, "image")
    if mask.shape != image.shape or mask.dtype != jnp.bool_:
        msg = (
            f"'mask' must be boolean with the shape of 'image' {image.shape}, got "
            f"{mask.dtype} {mask.shape}"
        )
        raise ValidationError(msg)
    if method not in FILL_METHODS:
        msg = f"'method' must be one of {FILL_METHODS}, got {method!r}"
        raise ValidationError(msg)
    periodic = tuple(periodic)
    if len(periodic) != 2 or not all(isinstance(wraps, bool) for wraps in periodic):
        msg = f"'periodic' must be two booleans (row, col), got {periodic!r}"
        raise ValidationError(msg)
    mask = check_values(
        mask, ~jnp.any(mask), "'mask' must mark at least one observed pixel"
    )
    zero = jnp.zeros((), image.dtype)

    def five_point(field: jax.Array) -> jax.Array:
        """Δx with the edges ``periodic`` asks for."""
        if any(periodic):
            return periodic_laplacian(field, periodic=periodic)
        return mirror_laplacian(field)

    def roughness(field: jax.Array) -> jax.Array:
        """K x: −Δx for the Laplace fill, Δ²x for the biharmonic one."""
        laplacian = five_point(field)
        if method == "laplace":
            return -laplacian
        return five_point(laplacian)

    def holes_only(field: jax.Array) -> jax.Array:
        """Q x, the field zeroed at the observed pixels."""
        return jnp.where(mask, zero, field)

    def system(hole_values: jax.Array) -> jax.Array:
        # (QKQ + P) z
        hole_rows = holes_only(roughness(holes_only(hole_values)))
        observed_rows = jnp.where(mask, hole_values, zero)
        return hole_rows + observed_rows

    # y₀ = Py, with the holes' values (NaN or not) never read
    observed = jnp.where(mask, image, zero)
    # b = −QKPy
    rhs = -holes_only(roughness(observed))
    structure = jax.ShapeDtypeStruct(image.shape, image.dtype)
    operator = lx.FunctionLinearOperator(
        system, structure, (lx.symmetric_tag, lx.positive_semidefinite_tag)
    )
    solver = default_solver(image.dtype) if solver is None else solver
    hole_values = lx.linear_solve(operator, rhs, solver).value
    return observed + hole_values
