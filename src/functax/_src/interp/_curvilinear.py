"""Locate targets on a curvilinear grid: a nearest-cell search, then Newton steps.

Cell (r, c) of a `CurvilinearGrid` maps the unit square onto its four corner
nodes by the bilinear map

    x(s, t) = (1 − s)(1 − t) x_{r,c} + s(1 − t) x_{r+1,c}
              + (1 − s) t x_{r,c+1} + s t x_{r+1,c+1},

so a target x* sits at the fractional index position u = (r + s, c + t)
once x(s, t) = x* is solved, and (s, t) then play the role of the offset on a
regular axis. Past the edge of the grid the edge cell's map continues, as
`index_positions` continues the end cells of an axis.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import math

import einx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from functax._src.geometry._metric import MAX_PAIRS
from functax._src.geometry._types import CurvilinearGrid
from functax._src.interp._locate import domain_bounds
from functax.exceptions import ValidationError


__all__ = [
    "N_NEWTON_STEPS",
    "N_WALKS",
    "curvilinear_positions",
    "locate_curvilinear",
]


N_NEWTON_STEPS: int = 4
"""Newton steps per cell; two to four reach float64 precision on smooth grids."""

N_WALKS: int = 3
"""Moves to a neighbouring cell when (s, t) leaves the unit square."""


def _cell_centres(nodes: Float[Array, "row col"]) -> Float[Array, " cell"]:
    """x̄_{r,c} = ¼ (x_{r,c} + x_{r+1,c} + x_{r,c+1} + x_{r+1,c+1}), flat row-major."""
    total = nodes[:-1, :-1] + nodes[1:, :-1] + nodes[:-1, 1:] + nodes[1:, 1:]
    return einx.id("row col -> (row col)", 0.25 * total)


def _nearest_cell(
    grid: CurvilinearGrid, coordinates: Float[Array, "target 2"]
) -> tuple[Int[Array, " target"], Int[Array, " target"]]:
    """The (row, col) of the cell whose centre is closest to each target.

    A brute-force search over every cell, in chunks of targets with
    ``lax.map`` so the (chunk, cell) distances stay near ``MAX_PAIRS``
    entries; it traces, so a plan can be built under ``jit``.
    """
    n_cols = grid.shape[1]
    centres = einx.id(
        "cell, cell -> cell (1 + 1)", _cell_centres(grid.x), _cell_centres(grid.y)
    )
    n_cells = centres.shape[0]

    def nearest(target: Float[Array, " 2"]) -> Int[Array, ""]:
        # ‖x̄ − x*‖² for every cell
        difference = einx.subtract("cell c, c -> cell c", centres, target)
        squared = einx.sum("cell [c]", difference**2)
        index = einx.argmin("[cell]", squared)
        return einx.id("1 ->", index)

    batch_size = max(1, MAX_PAIRS // n_cells)
    flat = jax.lax.map(nearest, coordinates, batch_size=batch_size)
    # flat cell k = r (N_c − 1) + c
    return flat // (n_cols - 1), flat % (n_cols - 1)


def _bilinear_terms(
    nodes: Float[Array, "row col"],
    row: Int[Array, " target"],
    col: Int[Array, " target"],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """(a, b, d, e) of one coordinate, so that x(s, t) = a + b s + d t + e s t."""
    corner_00 = nodes[row, col]
    corner_10 = nodes[row + 1, col]
    corner_01 = nodes[row, col + 1]
    corner_11 = nodes[row + 1, col + 1]
    along_row = corner_10 - corner_00
    along_col = corner_01 - corner_00
    # e = x_{r+1,c+1} − x_{r+1,c} − x_{r,c+1} + x_{r,c}, zero on a parallelogram
    twist = corner_11 - corner_10 - corner_01 + corner_00
    return corner_00, along_row, along_col, twist


def _misfit(
    terms: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    row_offset: jax.Array,
    col_offset: jax.Array,
    target: jax.Array,
) -> jax.Array:
    """F = x(s, t) − x* for one coordinate."""
    origin, along_row, along_col, twist = terms
    # x(s, t) = a + b s + d t + e s t
    mapped = (
        origin + along_row * row_offset + along_col * col_offset
    ) + twist * row_offset * col_offset
    return mapped - target


def _jacobian(
    terms_x: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    terms_y: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    row_offset: jax.Array,
    col_offset: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """The entries of J = [∂x/∂s ∂x/∂t] at (s, t), and det J."""
    _, row_edge_x, col_edge_x, twist_x = terms_x
    _, row_edge_y, col_edge_y, twist_y = terms_y
    # J = [b + e t, d + e s]
    ds_x = row_edge_x + twist_x * col_offset
    ds_y = row_edge_y + twist_y * col_offset
    dt_x = col_edge_x + twist_x * row_offset
    dt_y = col_edge_y + twist_y * row_offset
    determinant = ds_x * dt_y - dt_x * ds_y
    return ds_x, ds_y, dt_x, dt_y, determinant


def _newton(
    grid: CurvilinearGrid,
    row: Int[Array, " target"],
    col: Int[Array, " target"],
    coordinates: Float[Array, "target 2"],
    n_steps: int,
) -> tuple[jax.Array, jax.Array, Bool[Array, " target"]]:
    """(s, t) solving x(s, t) = x* in cell (row, col), and whether it converged."""
    target_x = coordinates[:, 0]
    target_y = coordinates[:, 1]
    terms_x = _bilinear_terms(grid.x, row, col)
    terms_y = _bilinear_terms(grid.y, row, col)
    dtype = jnp.result_type(terms_x[0], target_x)
    # Newton starts at the cell centre (½, ½)
    row_offset = jnp.full(target_x.shape, 0.5, dtype=dtype)
    col_offset = jnp.full(target_x.shape, 0.5, dtype=dtype)
    one = jnp.ones((), dtype)
    for _ in range(n_steps):
        misfit_x = _misfit(terms_x, row_offset, col_offset, target_x)
        misfit_y = _misfit(terms_y, row_offset, col_offset, target_y)
        ds_x, ds_y, dt_x, dt_y, determinant = _jacobian(
            terms_x, terms_y, row_offset, col_offset
        )
        # det J set to 1 where J is singular; such a target fails below
        safe_determinant = jnp.where(determinant != 0, determinant, one)
        # (Δs, Δt) = J⁻¹ F, the closed-form 2 × 2 solve (Cramer's rule)
        row_step = (misfit_x * dt_y - misfit_y * dt_x) / safe_determinant
        col_step = (ds_x * misfit_y - ds_y * misfit_x) / safe_determinant
        # (s, t) ← (s, t) − J⁻¹ F
        row_offset = row_offset - row_step
        col_offset = col_offset - col_step
    misfit_x = _misfit(terms_x, row_offset, col_offset, target_x)
    misfit_y = _misfit(terms_y, row_offset, col_offset, target_y)
    *_, determinant = _jacobian(terms_x, terms_y, row_offset, col_offset)
    # converged when |F| reaches √ε of the cell size, above the rounding
    # floor 8 ε |x| of the coordinates themselves, with J invertible there
    eps = jnp.finfo(dtype).eps
    _, row_edge_x, col_edge_x, _ = terms_x
    _, row_edge_y, col_edge_y, _ = terms_y
    cell_size = jnp.maximum(
        jnp.maximum(jnp.abs(row_edge_x), jnp.abs(col_edge_x)),
        jnp.maximum(jnp.abs(row_edge_y), jnp.abs(col_edge_y)),
    )
    magnitude = jnp.maximum(
        jnp.maximum(jnp.abs(target_x), jnp.abs(terms_x[0])),
        jnp.maximum(jnp.abs(target_y), jnp.abs(terms_y[0])),
    )
    tolerance = jnp.sqrt(eps) * cell_size + 8 * eps * magnitude
    misfit = jnp.maximum(jnp.abs(misfit_x), jnp.abs(misfit_y))
    converged = (misfit <= tolerance) & (determinant != 0)
    return row_offset, col_offset, converged


def _walk(
    cell: Int[Array, " target"],
    offset: Float[Array, " target"],
    n_cells: int,
    slack: float,
) -> Int[Array, " target"]:
    """Move ⌊s⌋ cells along one index where s leaves [0, 1], kept on the grid."""
    leaves = (offset < -slack) | (offset > 1 + slack)
    # clipped before the cast, so a diverged offset cannot overflow int32
    finite_offset = jnp.where(jnp.isfinite(offset), offset, 0.0)
    jump = jnp.floor(jnp.clip(finite_offset, -n_cells, n_cells))
    jump = jnp.where(leaves, jump, 0.0)
    return jnp.clip(cell + jump.astype(jnp.int32), 0, n_cells - 1)


def _settled(
    cell: Int[Array, " target"],
    offset: Float[Array, " target"],
    n_cells: int,
    slack: float,
) -> Bool[Array, " target"]:
    """Whether s lies in [0, 1], or past the edge of the grid on the edge cell."""
    above_low = (offset >= -slack) | (cell == 0)
    below_high = (offset <= 1 + slack) | (cell == n_cells - 1)
    return above_low & below_high


def curvilinear_positions(
    grid: CurvilinearGrid,
    coordinates: Float[Array, "target 2"],
    *,
    n_steps: int = N_NEWTON_STEPS,
    n_walks: int = N_WALKS,
) -> Float[Array, "target 2"]:
    """The (M, 2) index positions u = (r + s, c + t) of targets (x, y) on a grid.

    NaN where the locate failed: Newton did not converge, or the walk ran out
    with (s, t) still outside the unit square of an inner cell. The policies
    treat such rows as non-finite targets, which are invalid under every
    ``outside`` rule.
    """
    n_rows, n_cols = grid.shape
    row, col = _nearest_cell(grid, coordinates)
    dtype = jnp.result_type(grid.x, coordinates)
    # an offset this close past 0 or 1 sits on the shared edge, not beyond it
    slack = math.sqrt(float(np.finfo(dtype).eps))
    for _ in range(n_walks):
        row_offset, col_offset, _ = _newton(grid, row, col, coordinates, n_steps)
        row = _walk(row, row_offset, n_rows - 1, slack)
        col = _walk(col, col_offset, n_cols - 1, slack)
    row_offset, col_offset, converged = _newton(grid, row, col, coordinates, n_steps)
    settled = _settled(row, row_offset, n_rows - 1, slack) & _settled(
        col, col_offset, n_cols - 1, slack
    )
    found = converged & settled
    # u = (r + s, c + t)
    positions = einx.id(
        "target, target -> target (1 + 1)", row + row_offset, col + col_offset
    )
    nan = jnp.asarray(jnp.nan, dtype=positions.dtype)
    return einx.where("target, target c, -> target c", found, positions, nan)


def locate_curvilinear(
    grid: CurvilinearGrid,
    positions: Float[Array, "target 2"],
    *,
    extent: str = "nodes",
    n_steps: int = N_NEWTON_STEPS,
    n_walks: int = N_WALKS,
) -> tuple[Int[Array, "target 2"], Float[Array, "target 2"], Bool[Array, " target"]]:
    """The cell (r, c), the offset (s, t) and the inside flag of targets on a swath.

    Each cell maps the unit square onto its corners by the bilinear map
    x(s, t) = (1 − s)(1 − t) x₀₀ + s(1 − t) x₁₀ + (1 − s) t x₀₁ + s t x₁₁, and
    the target x* sits at index position u = (r + s, c + t) where
    x(s, t) = x*. As in `locate`, the cell is ⌊u⌋ and the offset u − ⌊u⌋, and
    a target is inside when u is finite and lies in the domain of ``extent``
    on both indices.

    Algorithm:
        1. Start each target in the cell whose centre is nearest, by a
           brute-force search over the cells (chunked, and traceable).
        2. Run ``n_steps`` Newton steps from (½, ½),
           (s, t) ← (s, t) − J⁻¹ (x(s, t) − x*), with J = ∂x/∂(s, t).
        3. Where s or t leaves [0, 1], move ⌊s⌋ (or ⌊t⌋) cells, kept on the
           grid, and repeat, ``n_walks`` times; past the edge of the grid the
           edge cell's map is continued.
        4. Mark the target not found (u = NaN, so not inside) when Newton did
           not converge or the walk ended outside an inner cell.

    Args:
        grid: The `CurvilinearGrid` source, nodes x_{r,c} = (x, y)[r, c].
        positions: x*, the M target coordinates (x, y).
        extent: ``"nodes"`` or ``"cells"`` (see `Policies`).
        n_steps: The Newton steps per cell, at least 1.
        n_walks: The moves to a neighbouring cell, at least 0.

    Returns:
        ``(cell, offset, inside)``: cell (M, 2) and offset (M, 2) along (row,
        col), and inside (M,).

    Raises:
        ValidationError: If ``positions`` is not (M, 2), or ``n_steps`` or
            ``n_walks`` is out of range.
        PolicyError: If ``extent`` is unknown.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import CurvilinearGrid
        >>> from functax.interp import locate_curvilinear
        >>> rows, cols = jnp.meshgrid(
        ...     jnp.arange(3.0), jnp.arange(4.0), indexing="ij"
        ... )
        >>> sheared = CurvilinearGrid(x=cols + 0.5 * rows, y=rows)
        >>> cell, offset, inside = locate_curvilinear(
        ...     sheared, jnp.array([[1.5, 0.5], [9.0, 0.0]])
        ... )
        >>> cell.tolist(), offset.tolist(), inside.tolist()
        ([[0, 1], [0, 9]], [[0.5, 0.25], [0.0, 0.0]], [True, False])
        >>> warped = CurvilinearGrid(x=cols + 0.1 * rows**2, y=rows + 0.1 * cols)
        >>> _, offset, inside = locate_curvilinear(warped, jnp.array([[1.2, 1.3]]))
        >>> offset.shape, inside.tolist()
        ((1, 2), [True])
    """
    positions = jnp.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 2:
        msg = f"'positions' must be (M, 2) coordinates (x, y), got {positions.shape}"
        raise ValidationError(msg)
    if isinstance(n_steps, bool) or n_steps < 1:
        msg = f"'n_steps' must be a positive integer, got {n_steps!r}"
        raise ValidationError(msg)
    if isinstance(n_walks, bool) or n_walks < 0:
        msg = f"'n_walks' must be a non-negative integer, got {n_walks!r}"
        raise ValidationError(msg)
    n_rows, n_cols = grid.shape
    row_lower, row_upper = domain_bounds(n_rows, extent)
    col_lower, col_upper = domain_bounds(n_cols, extent)
    index = curvilinear_positions(grid, positions, n_steps=n_steps, n_walks=n_walks)
    cell_float = jnp.floor(index)
    offset = index - cell_float
    cell = cell_float.astype(jnp.int32)
    row_index = index[:, 0]
    col_index = index[:, 1]
    # NaN compares False, so a target not found is never inside
    inside_rows = (row_index >= row_lower) & (row_index <= row_upper)
    inside_cols = (col_index >= col_lower) & (col_index <= col_upper)
    return cell, offset, inside_rows & inside_cols
