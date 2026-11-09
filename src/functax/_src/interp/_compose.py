"""The product of two stencils, H = H_outer H_inner, as one stencil.

Row k of H_outer reads rows iᵒₖₛ of H_inner with weights wᵒₖₛ, so row k of
the product reads every tap of those rows,

    Hₖⱼ = ∑ₛ wᵒₖₛ (H_inner)_{iᵒₖₛ, j},

at S_outer · S_inner taps: indices iⁱ_{iᵒₖₛ, r} and weights wᵒₖₛ wⁱ_{iᵒₖₛ, r}.
Repeated indices add, as in any stencil, and can be merged on the host.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import einx
import jax.numpy as jnp
import numpy as np

from functax._src._backend import is_tracer
from functax._src.interp._stencil import Stencil
from functax.exceptions import ValidationError


__all__ = [
    "compose",
]


def compose(outer: Stencil, inner: Stencil, *, merge: bool = False) -> Stencil:
    """H = H_outer H_inner: apply ``inner`` first, then ``outer``, as one stencil.

    Hₖⱼ = ∑ₛ wᵒₖₛ (H_inner)_{iᵒₖₛ, j}, with S_outer · S_inner taps per row
    before merging. A row of ``outer`` that reads an invalid row of ``inner``
    is invalid, with zero weights, at any weight (0 · NaN is NaN, so applying
    the two stencils in turn under a NaN fill gives NaN there too). The same
    product is ``outer @ inner``.

    Algorithm:
        1. Gather the rows of ``inner`` that each tap of ``outer`` reads.
        2. Multiply each gathered weight by the weight of the tap that reads it.
        3. Flatten the two tap axes into one; mark rows that read an invalid
           row invalid and zero their weights.
        4. If ``merge``, sum the weights of repeated indices row by row on the
           host and pad to the largest row.

    Args:
        outer: H_outer, an (M, P) stencil, applied second.
        inner: H_inner, a (P, N) stencil, applied first.
        merge: Whether to merge repeated indices in each row (host only); the
            product then has as many taps as its fullest row.

    Returns:
        The (M, N) `Stencil`; its weights take the promoted dtype of both.

    Raises:
        ValidationError: If ``outer`` does not read ``inner.n_target`` sources,
            or ``merge`` is set on traced stencils.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.interp import Stencil, compose
        >>> # midpoints of 4 samples, then midpoints of those
        >>> inner = Stencil(
        ...     jnp.array([[0, 1], [1, 2], [2, 3]]),
        ...     jnp.full((3, 2), 0.5),
        ...     n_source=4,
        ... )
        >>> outer = Stencil(
        ...     jnp.array([[0, 1], [1, 2]]), jnp.full((2, 2), 0.5), n_source=3
        ... )
        >>> product = compose(outer, inner)
        >>> product.n_taps, product.to_dense().tolist()
        (4, [[0.25, 0.5, 0.25, 0.0], [0.0, 0.25, 0.5, 0.25]])
        >>> merged = compose(outer, inner, merge=True)
        >>> merged.indices.tolist(), merged.weights.tolist()
        ([[0, 1, 2], [1, 2, 3]], [[0.25, 0.5, 0.25], [0.25, 0.5, 0.25]])
        >>> bool(jnp.allclose((outer @ inner).to_dense(), product.to_dense()))
        True
    """
    if outer.n_source != inner.n_target:
        msg = (
            f"'outer' reads {outer.n_source} sources but 'inner' has "
            f"{inner.n_target} rows"
        )
        raise ValidationError(msg)
    # iⁱ_{iᵒₖₛ, r}: the indices of the inner rows each outer tap reads
    gathered_indices = inner.indices[outer.indices]
    # wⁱ_{iᵒₖₛ, r}
    gathered_weights = inner.weights[outer.indices]
    indices = einx.id("target tap inner -> target (tap inner)", gathered_indices)
    # wᵒₖₛ wⁱ_{iᵒₖₛ, r}
    weights = einx.multiply(
        "target tap, target tap inner -> target (tap inner)",
        outer.weights,
        gathered_weights,
    )
    # a row is valid when it is and every inner row it reads is
    reads_valid = einx.all("target [tap]", inner.valid[outer.indices])
    valid = outer.valid & reads_valid
    zero = jnp.zeros((), weights.dtype)
    weights = einx.where("target, target tap, -> target tap", valid, weights, zero)
    product = Stencil(indices, weights, n_source=inner.n_source, valid=valid)
    if merge:
        return _merge_taps(product)
    return product


def _merge_taps(stencil: Stencil) -> Stencil:
    """The stencil with each row's repeated indices summed into one tap, on the host."""
    if is_tracer(stencil.indices) or is_tracer(stencil.weights):
        msg = "'merge' needs concrete stencils: compose outside jit, or merge=False"
        raise ValidationError(msg)
    indices = np.asarray(stencil.indices)
    weights = np.asarray(stencil.weights)
    merged_rows = []
    for row_indices, row_weights in zip(indices, weights, strict=True):
        columns, tap_to_column = np.unique(row_indices, return_inverse=True)
        sums = np.zeros(columns.size, dtype=weights.dtype)
        # wₖⱼ = ∑ₛ [iₖₛ = j] wₖₛ
        np.add.at(sums, tap_to_column, row_weights)
        merged_rows.append((columns, sums))
    n_taps = max((columns.size for columns, _ in merged_rows), default=0)
    merged_indices = np.zeros((len(merged_rows), n_taps), dtype=indices.dtype)
    merged_weights = np.zeros((len(merged_rows), n_taps), dtype=weights.dtype)
    for row, (columns, sums) in enumerate(merged_rows):
        merged_indices[row, : columns.size] = columns
        # padding repeats a column the row already reads, with weight 0
        merged_indices[row, columns.size :] = columns[0] if columns.size else 0
        merged_weights[row, : columns.size] = sums
    return Stencil(
        jnp.asarray(merged_indices),
        jnp.asarray(merged_weights),
        n_source=stencil.n_source,
        valid=stencil.valid,
    )
