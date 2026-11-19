"""Laplace and biharmonic gap fill of an image, with a five-point Laplacian.

Each axis has mirror edges, or is periodic (a longitude that wraps). On this
page x is the image, ∇ the forward differences on its pixel grid and
∇² = −∇ᵀ∇ the five-point Laplacian.

The two fills are the limits of the variational fit (`Variational`) as the
observation-error covariance Σ_ε goes to zero, with P selecting the observed
pixels: the roughness L = ∇ (LᵀL = −∇²) gives ∇²x = 0 in the holes, and
L = ∇² gives ∇⁴x = 0. The solve runs over the hole pixels only, with the
observed pixels as Dirichlet values.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import einx
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
"""The gap fills: ∇²x = 0 or ∇⁴x = 0 in the holes."""


def mirror_laplacian(image: Float[Array, "row col"]) -> Float[Array, "row col"]:
    """∇²x, the five-point Laplacian with mirror edges (the ghost pixel is the edge).

    (∇²x)ᵢⱼ = xᵢ₋₁ⱼ + xᵢ₊₁ⱼ + xᵢⱼ₋₁ + xᵢⱼ₊₁ − 4xᵢⱼ at row i and column j, with
    x₋₁ⱼ = x₀ⱼ and its three companions at the other edges, so ∇² = −∇ᵀ∇ for
    the forward differences ∇, symmetric negative semi-definite with the
    constants as its null space. It is functax's only difference operator,
    there for `fill_gaps`; richer ones arrive as arguments (finitevolX,
    spectraldiffx).

    Args:
        image: x, the image.

    Returns:
        ∇²x, the shape of ``image``.

    Examples:
        An impulse: −4 at its pixel, 1 at each neighbour, and a zero sum, since
        the constants span the null space of the symmetric ∇²:

        >>> import einx
        >>> import jax.numpy as jnp
        >>> from functax._src.project._fill import mirror_laplacian
        >>> impulse = jnp.zeros((3, 3)).at[1, 1].set(1.0)
        >>> laplacian = mirror_laplacian(impulse)
        >>> total = einx.sum("[row col]", laplacian)
        >>> bool(laplacian[1, 1] == -4.0), bool(jnp.allclose(total, 0.0))
        (True, True)

        A constant image, even a single pixel, has ∇²x = 0:

        >>> pixel = jnp.full((1, 1), 2.0)
        >>> bool(jnp.allclose(mirror_laplacian(pixel), 0.0))
        True
    """
    # x with a ghost ring that repeats the edge, x₋₁ⱼ = x₀ⱼ
    padded = jnp.pad(image, 1, mode="edge")
    # xᵢ₋₁ⱼ, xᵢ₊₁ⱼ, xᵢⱼ₋₁ and xᵢⱼ₊₁
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    # (∇²x)ᵢⱼ = xᵢ₋₁ⱼ + xᵢ₊₁ⱼ + xᵢⱼ₋₁ + xᵢⱼ₊₁ − 4xᵢⱼ
    laplacian = up + down + left + right - 4 * image
    return laplacian


def periodic_laplacian(
    image: Float[Array, "row col"], *, periodic: tuple[bool, bool]
) -> Float[Array, "row col"]:
    """∇²x, the five-point Laplacian with a periodic edge on each axis marked True.

    On a periodic axis of length N the neighbours of index j are j ± 1 mod N,

        (∇²x)ᵢⱼ = xᵢ₋₁ⱼ + xᵢ₊₁ⱼ + xᵢ,₍ⱼ₋₁₎ mod N + xᵢ,₍ⱼ₊₁₎ mod N − 4xᵢⱼ

    for periodic columns (row i, column j), and an axis that is not periodic
    has mirror edges, as in `mirror_laplacian`. ∇² stays symmetric negative
    semi-definite with the constants as its null space.

    Args:
        image: x, the image.
        periodic: Whether the rows and the columns wrap, ``(row, col)``.

    Returns:
        ∇²x, the shape of ``image``.

    Examples:
        One row [1, 0, 0, 0] with periodic columns: the impulse reaches the
        last column across the seam:

        >>> import jax.numpy as jnp
        >>> from functax._src.project._fill import (
        ...     mirror_laplacian,
        ...     periodic_laplacian,
        ... )
        >>> row = jnp.array([[1.0, 0.0, 0.0, 0.0]])
        >>> laplacian = periodic_laplacian(row, periodic=(False, True))
        >>> bool(jnp.allclose(laplacian, jnp.array([[-2.0, 1.0, 0.0, 1.0]])))
        True

        With no axis periodic it is `mirror_laplacian`:

        >>> image = jnp.array([[0.0, 1.0, 4.0], [2.0, -1.0, 3.0]])
        >>> mirrored = periodic_laplacian(image, periodic=(False, False))
        >>> bool(jnp.allclose(mirrored, mirror_laplacian(image)))
        True
    """
    # x with a ghost ring: x₋₁ = x_{N−1} on a periodic axis, x₋₁ = x₀ otherwise
    padded = image
    for axis_index, wraps in enumerate(periodic):
        widths = [(0, 0), (0, 0)]
        widths[axis_index] = (1, 1)
        mode = "wrap" if wraps else "edge"
        padded = jnp.pad(padded, widths, mode=mode)
    # xᵢ₋₁ⱼ, xᵢ₊₁ⱼ, xᵢⱼ₋₁ and xᵢⱼ₊₁, each index mod N on a periodic axis
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    # (∇²x)ᵢⱼ = xᵢ₋₁ⱼ + xᵢ₊₁ⱼ + xᵢⱼ₋₁ + xᵢⱼ₊₁ − 4xᵢⱼ
    laplacian = up + down + left + right - 4 * image
    return laplacian


def fill_gaps(
    image: Float[ArrayLike, "row col"],
    *,
    mask: Bool[ArrayLike, "row col"],
    method: str = "laplace",
    periodic: tuple[bool, bool] = (False, False),
    solver: lx.AbstractLinearSolver | None = None,
) -> Float[Array, "row col"]:
    """The image with its holes filled: ∇²x = 0 (Laplace) or ∇⁴x = 0 (biharmonic).

    With P = diag(mask) the projector onto the observed pixels and Q = I − P
    onto the holes, the fill keeps x = y on P and minimises the roughness
    ‖Lx‖₂² over the holes: ‖∇x‖₂² = −xᵀ∇²x for ``"laplace"`` or ‖∇²x‖₂² for
    ``"biharmonic"``, both with mirror edges
    (`mirror_laplacian`), or with an axis that wraps (`periodic_laplacian`),
    so a hole can reach observed pixels across the seam of a global
    longitude grid. Writing x = Py + z with z zero on P, z solves

        (QLᵀLQ + P) z = −QLᵀLPy,   LᵀL = −∇² (laplace) or ∇⁴ (biharmonic),

    symmetric positive definite when every connected hole touches an observed
    pixel, so CG solves it matrix-free. The holes then satisfy ∇²x = 0 or
    ∇⁴x = 0, the two named cases of `Variational` as the observation error
    goes to zero. Hole values in ``image`` (NaN or not) are never read.

    Algorithm:
        1. y₀ = Py, the observed pixels with the holes zeroed.
        2. b = −QLᵀL y₀.
        3. z = (QLᵀLQ + P)⁻¹ b by ``solver``.
        4. x = y₀ + z.

    Args:
        image: y, the image; its hole pixels are ignored.
        mask: True at the observed pixels P.
        method: ``"laplace"`` or ``"biharmonic"``.
        periodic: Whether the rows and the columns wrap, ``(row, col)``;
            ``(False, True)`` joins the east and west edges of a global
            longitude grid. Mirror edges on both axes by default.
        solver: The lineax solver; CG at √eps of the image dtype when ``None``.

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
    # P = 0: no observed pixel to fill from
    any_observed = einx.any("[row col]", mask)
    none_observed = ~any_observed
    mask = check_values(
        mask, none_observed, "'mask' must mark at least one observed pixel"
    )
    zero = jnp.zeros((), image.dtype)

    def five_point(field: Float[Array, "row col"]) -> Float[Array, "row col"]:
        """∇²x with the edges ``periodic`` asks for."""
        if any(periodic):
            return periodic_laplacian(field, periodic=periodic)
        return mirror_laplacian(field)

    def roughness(field: Float[Array, "row col"]) -> Float[Array, "row col"]:
        """LᵀLx: −∇²x for the Laplace fill, ∇⁴x for the biharmonic one."""
        # ∇²x
        laplacian = five_point(field)
        if method == "laplace":
            # LᵀLx = −∇²x
            return -laplacian
        # LᵀLx = ∇⁴x
        return five_point(laplacian)

    def holes_only(field: Float[Array, "row col"]) -> Float[Array, "row col"]:
        """Q x, the field zeroed at the observed pixels."""
        return jnp.where(mask, zero, field)

    def system(hole_values: Float[Array, "row col"]) -> Float[Array, "row col"]:
        # Qz
        hole_part = holes_only(hole_values)
        # LᵀLQz
        hole_roughness = roughness(hole_part)
        # QLᵀLQz
        hole_rows = holes_only(hole_roughness)
        # Pz
        observed_rows = jnp.where(mask, hole_values, zero)
        # (QLᵀLQ + P) z
        return hole_rows + observed_rows

    # --- right-hand side ---
    # y₀ = Py, with the holes' values (NaN or not) never read
    observed = jnp.where(mask, image, zero)
    # LᵀL y₀
    observed_roughness = roughness(observed)
    # QLᵀL y₀
    hole_roughness = holes_only(observed_roughness)
    # b = −QLᵀLPy
    rhs = -hole_roughness

    # --- solve over the holes ---
    structure = jax.ShapeDtypeStruct(image.shape, image.dtype)
    operator = lx.FunctionLinearOperator(
        system, structure, (lx.symmetric_tag, lx.positive_semidefinite_tag)
    )
    solver = default_solver(image.dtype) if solver is None else solver
    # z = (QLᵀLQ + P)⁻¹ b
    solution = lx.linear_solve(operator, rhs, solver)
    hole_values = solution.value
    # x = y₀ + z
    filled = observed + hole_values
    return filled
