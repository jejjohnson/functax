"""Tensor-product stencils: one local basis per axis, H = H₁ ⊗ … ⊗ H_D.

A target with index position u_d on axis d reads S_d taps there; across D
axes it reads S = ∏_d S_d taps at flat, row-major indices
∑_d j_d ∏_{d′ > d} n_{d′} (the order of ``einx.id("... y x -> ... (y x)")``),
with weights ∏_d w_d. The policies decide what happens off the domain and
past the ends.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import math
from typing import Any

import einx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from functax._src.geometry._affine import pixel_centres, to_pixel
from functax._src.geometry._types import AffineGrid, Axis, Points, RectilinearGrid
from functax._src.interp._bases import Basis
from functax._src.interp._locate import (
    check_outside,
    domain_bounds,
    index_positions,
    remap_taps,
)
from functax._src.interp._policies import Policies
from functax._src.interp._stencil import Stencil
from functax.exceptions import GeometryError


__all__ = [
    "source_layout",
    "stencil_from_positions",
    "target_positions",
    "tensor_stencil",
]


def source_layout(source: Any) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """The node counts (n₁, …, n_D) of a source, and which of its axes are periodic.

    Raises:
        GeometryError: If ``source`` is not an `Axis`, `RectilinearGrid` or
            `AffineGrid`.
    """
    if isinstance(source, Axis):
        return (source.values.shape[0],), (source.period is not None,)
    if isinstance(source, RectilinearGrid):
        periodic = tuple(axis.period is not None for axis in source.axes)
        return source.shape, periodic
    if isinstance(source, AffineGrid):
        return source.shape, (False, False)
    msg = (
        "'source' must be an Axis, a RectilinearGrid or an AffineGrid, got "
        f"{type(source).__name__}"
    )
    raise GeometryError(msg)


def _target_coordinates(source: Any, target: Any) -> jax.Array:
    """The (M, D) target coordinates in the source's frame, before locating."""
    if isinstance(target, Points):
        return target.xy
    if isinstance(target, Axis):
        return einx.id("target -> target 1", target.values)
    if isinstance(target, RectilinearGrid):
        nodes = jnp.meshgrid(*(axis.values for axis in target.axes), indexing="ij")
        flat = [einx.id("... -> (...)", node) for node in nodes]
        return jnp.column_stack(flat)
    if isinstance(target, AffineGrid) and isinstance(source, AffineGrid):
        x, y = pixel_centres(target)
        return einx.id("row col, row col -> (row col) (1 + 1)", x, y)
    msg = (
        f"'target' must be Points, an Axis or a RectilinearGrid (or an AffineGrid "
        f"on an AffineGrid source), got {type(target).__name__}"
    )
    raise GeometryError(msg)


def target_positions(source: Any, target: Any) -> Float[Array, "target dim"]:
    """The (M, D) index positions u of the targets on each source axis.

    `index_positions` per axis for an `Axis` or a `RectilinearGrid` source,
    and `to_pixel` (row, col) for an `AffineGrid`, whose targets are map
    coordinates (x, y).

    Raises:
        GeometryError: If the source or target kind is unsupported, or the
            target has the wrong number of coordinates.
    """
    shape, _ = source_layout(source)
    coordinates = _target_coordinates(source, target)
    if coordinates.shape[1] != len(shape):
        msg = (
            f"'target' has {coordinates.shape[1]} coordinates per point, but the "
            f"source has {len(shape)} axes"
        )
        raise GeometryError(msg)
    if isinstance(source, AffineGrid):
        return to_pixel(source.transform, coordinates)
    axes = (source,) if isinstance(source, Axis) else source.axes
    columns = [index_positions(axis, coordinates[:, d]) for d, axis in enumerate(axes)]
    return jnp.column_stack(columns)


def _axis_stencil(
    position: Float[Array, " target"],
    n_nodes: int,
    periodic: bool,
    basis: Basis,
    policies: Policies,
) -> tuple[jax.Array, jax.Array, tuple[jax.Array, jax.Array]]:
    """Taps, weights, and (row kept, target inside) on one axis, after the policies."""
    lower, upper = domain_bounds(n_nodes, policies.extent)
    finite = jnp.isfinite(position)
    inside = finite if periodic else finite & (position >= lower) & (position <= upper)
    # non-finite positions read node 0 and are masked; nothing else can read
    position = jnp.where(finite, position, 0.0)
    if policies.outside == "clamp" and not periodic:
        position = jnp.clip(position, lower, upper)
    cell = jnp.floor(position)
    if policies.outside == "extrapolate" and not periodic:
        # keep every tap on the axis, so the edge cell's weights continue
        low_tap, high_tap = basis.support
        first = -low_tap
        last = max(n_nodes - 1 - high_tap, first)
        cell = jnp.clip(cell, first, last)
    offset = position - cell
    relative, weights = basis.stencil(offset)
    taps = einx.add(
        "target, target tap -> target tap", cell.astype(jnp.int32), relative
    )
    indices, kept = remap_taps(taps, n_nodes, policies.boundary, periodic=periodic)
    weights = jnp.where(kept, weights, jnp.zeros((), weights.dtype))
    # clamp and extrapolate keep every finite row; nan keeps the inside ones
    row_ok = finite if policies.outside in ("clamp", "extrapolate") else inside
    return indices, weights, (row_ok, inside)


def stencil_from_positions(
    positions: Float[Array, "target dim"],
    shape: tuple[int, ...],
    periodic: tuple[bool, ...],
    bases: tuple[Basis, ...],
    *,
    policies: Policies,
) -> Stencil:
    """The tensor stencil of index positions u on a grid of ``shape``.

    Rows outside the domain under ``outside="nan"`` and rows at non-finite
    positions under any policy are invalid, with zero weights.

    Raises:
        OutsideDomainError: Under ``outside="raise"`` when a target is outside.
        PolicyError: Under ``outside="raise"`` on traced positions.
    """
    per_axis = [
        _axis_stencil(positions[:, d], n_nodes, is_periodic, basis, policies)
        for d, (n_nodes, is_periodic, basis) in enumerate(
            zip(shape, periodic, bases, strict=True)
        )
    ]
    indices, weights, (row_ok, inside_all) = per_axis[0]
    for n_nodes, (axis_indices, axis_weights, (axis_ok, axis_inside)) in zip(
        shape[1:], per_axis[1:], strict=True
    ):
        # flat index ((j₁ n₂ + j₂) n₃ + j₃) …, weight w₁ w₂ w₃ …
        scaled = indices * n_nodes
        indices = einx.add("target a, target b -> target (a b)", scaled, axis_indices)
        weights = einx.multiply(
            "target a, target b -> target (a b)", weights, axis_weights
        )
        row_ok = row_ok & axis_ok
        inside_all = inside_all & axis_inside
    check_outside(inside_all, policies)
    zero = jnp.zeros((), weights.dtype)
    weights = einx.where("target, target tap, -> target tap", row_ok, weights, zero)
    return Stencil(indices, weights, n_source=math.prod(shape), valid=row_ok)


def tensor_stencil(
    source: Any,
    target: Any,
    bases: tuple[Basis, ...],
    *,
    policies: Policies,
) -> Stencil:
    """H = H₁ ⊗ … ⊗ H_D: one local basis per source axis, read at the targets.

    Args:
        source: An `Axis`, a `RectilinearGrid` or an `AffineGrid`; its values
            are flattened row-major, last axis fastest.
        target: `Points` (one coordinate per source axis, or (x, y) map
            coordinates on an `AffineGrid`), or a grid read at its nodes.
        bases: One `Basis` per source axis.
        policies: The outside and boundary rules.

    Returns:
        The `Stencil`, with S = ∏_d S_d taps per target.

    Raises:
        GeometryError: If the geometries are unsupported or mismatched, or the
            number of bases differs from the number of axes.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points, RectilinearGrid
        >>> from functax.interp import LinearBasis, Policies, tensor_stencil
        >>> grid = RectilinearGrid((Axis(jnp.arange(3.0)), Axis(jnp.arange(4.0))))
        >>> H = tensor_stencil(
        ...     grid,
        ...     Points(jnp.array([[0.5, 1.5]])),
        ...     (LinearBasis(),) * 2,
        ...     policies=Policies(),
        ... )
        >>> H.indices.tolist(), H.weights.tolist()
        ([[1, 2, 5, 6]], [[0.25, 0.25, 0.25, 0.25]])
    """
    shape, periodic = source_layout(source)
    if len(bases) != len(shape):
        msg = (
            f"'bases' must have one basis per source axis ({len(shape)}), "
            f"got {len(bases)}"
        )
        raise GeometryError(msg)
    positions = target_positions(source, target)
    return stencil_from_positions(positions, shape, periodic, bases, policies=policies)
