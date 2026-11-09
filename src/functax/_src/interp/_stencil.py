"""The sparse read-out H: S taps per target, applied by gather and by scatter-add.

Row k of H has weights wₖₛ at source indices iₖₛ for s = 1, …, S, so

    f̂ₖ = ∑ₛ wₖₛ c[iₖₛ]  and  (Hᵀr)ⱼ = ∑ₖ ∑ₛ [iₖₛ = j] wₖₛ rₖ,

with [·] equal to 1 when its condition holds. Every local scheme (nearest,
linear, cubic, B-spline read-out, inverse distance, Delaunay, conservative)
produces this layout, and the hand-written adjoint is checked against
``jax.linear_transpose`` by the dot-product test.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import abc
from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from functax._src._checks import check_values, require_floating
from functax.exceptions import ValidationError


__all__ = [
    "Readout",
    "Stencil",
]


class Readout(eqx.Module):
    """c → f̂(X*): the read-out of a plan, from coefficients to target values.

    A `Stencil` is the sparse read-out of every local scheme; dense kernel
    read-outs K(X*, X) arrive with the global schemes of Wave 3.
    """

    valid: eqx.AbstractVar[Bool[Array, " target"]]

    @abc.abstractmethod
    def apply(
        self, coefficients: Float[Array, "*batch source"]
    ) -> Float[Array, "*batch target"]:
        """f̂ = H c, over any leading batch dims."""

    @abc.abstractmethod
    def apply_adjoint(
        self, residual: Float[Array, "*batch target"]
    ) -> Float[Array, "*batch source"]:
        """g = Hᵀ r, over any leading batch dims."""

    @abc.abstractmethod
    def operator(self) -> lx.AbstractLinearOperator:
        """H as an (M, N) lineax operator."""


class Stencil(Readout):
    """A sparse read-out H: row k reads S source values at flat indices.

    Repeated indices in one row are allowed and add. Padding taps carry weight
    0, so shapes stay static; they repeat an index the row already reads, so a
    NaN reaches a row only through a source it reads anyway (0 · NaN is NaN,
    and `apply` stays linear, so no value-dependent mask is possible). Rows
    with ``valid=False`` (targets outside the domain) carry zero weights, and
    the plan writes its fill value there. Weights keep their dtype; `apply`
    never casts.

    Args:
        indices: iₖₛ, the flat source index of each tap, in [0, ``n_source``).
        weights: wₖₛ, the weight of each tap, the same shape as ``indices``.
        n_source: N, the length of the flat source.
        valid: Whether each row is a real read-out; all true when ``None``.

    Raises:
        ValidationError: If the shapes disagree, ``indices`` is not integer or
            ``weights`` not floating, ``n_source`` is not positive, or an index
            is out of range (checked when run under ``jit``).

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.interp import Stencil
        >>> # the midpoints of a 4-sample signal: rows (0, 1), (1, 2), (2, 3)
        >>> H = Stencil(
        ...     jnp.array([[0, 1], [1, 2], [2, 3]]),
        ...     jnp.full((3, 2), 0.5),
        ...     n_source=4,
        ... )
        >>> H.apply(jnp.array([0.0, 2.0, 4.0, 6.0])).tolist()
        [1.0, 3.0, 5.0]
        >>> H.apply_adjoint(jnp.ones(3)).tolist()
        [0.5, 1.0, 1.0, 0.5]
    """

    indices: Int[Array, "target tap"]
    weights: Float[Array, "target tap"]
    valid: Bool[Array, " target"]
    n_source: int = eqx.field(static=True)

    def __init__(
        self,
        indices: ArrayLike,
        weights: ArrayLike,
        *,
        n_source: int,
        valid: ArrayLike | None = None,
    ) -> None:
        indices = jnp.asarray(indices)
        weights = jnp.asarray(weights)
        if indices.ndim != 2:
            msg = f"'indices' must be 2-D (target, tap), got shape {indices.shape}"
            raise ValidationError(msg)
        if weights.shape != indices.shape:
            msg = (
                f"'weights' must have the shape of 'indices' {indices.shape}, "
                f"got {weights.shape}"
            )
            raise ValidationError(msg)
        if not jnp.issubdtype(indices.dtype, jnp.integer):
            msg = f"'indices' must have an integer dtype, got {indices.dtype}"
            raise ValidationError(msg)
        require_floating(weights, "weights")
        if isinstance(n_source, bool) or not isinstance(n_source, int | np.integer):
            msg = f"'n_source' must be an integer, got {n_source!r}"
            raise ValidationError(msg)
        if n_source < 1:
            msg = f"'n_source' must be positive, got {n_source}"
            raise ValidationError(msg)
        n_target = indices.shape[0]
        if valid is None:
            valid = jnp.ones(n_target, dtype=bool)
        valid = jnp.asarray(valid)
        if valid.shape != (n_target,) or valid.dtype != jnp.bool_:
            msg = (
                f"'valid' must be boolean with shape ({n_target},), got "
                f"{valid.dtype} {valid.shape}"
            )
            raise ValidationError(msg)
        out_of_range = jnp.any((indices < 0) | (indices >= n_source))
        indices = check_values(
            indices, out_of_range, f"'indices' must lie in [0, {int(n_source)})"
        )
        self.indices = indices
        self.weights = weights
        self.valid = valid
        self.n_source = int(n_source)

    @property
    def n_target(self) -> int:
        """M, the number of rows."""
        return self.indices.shape[0]

    @property
    def n_taps(self) -> int:
        """S, the number of taps per row."""
        return self.indices.shape[1]

    def apply(
        self, coefficients: Float[Array, "*batch source"]
    ) -> Float[Array, "*batch target"]:
        """f̂ₖ = ∑ₛ wₖₛ c[iₖₛ], over any leading batch dims.

        Args:
            coefficients: c, with the flat source last.

        Returns:
            f̂, with the targets last.

        Raises:
            ValidationError: If the last axis of ``coefficients`` is not N.
        """
        self._check_last_axis(coefficients, self.n_source, "coefficients")
        # c[iₖₛ]
        gathered = coefficients[..., self.indices]
        # f̂ₖ = ∑ₛ wₖₛ c[iₖₛ]
        return einx.dot(
            "... target tap, target tap -> ... target", gathered, self.weights
        )

    def apply_adjoint(
        self, residual: Float[Array, "*batch target"]
    ) -> Float[Array, "*batch source"]:
        """(Hᵀr)ⱼ = ∑ₖ ∑ₛ [iₖₛ = j] wₖₛ rₖ, over any leading batch dims.

        Args:
            residual: r, with the targets last.

        Returns:
            Hᵀ r, with the flat source last.

        Raises:
            ValidationError: If the last axis of ``residual`` is not M.
        """
        self._check_last_axis(residual, self.n_target, "residual")
        # wₖₛ rₖ, one contribution per tap
        contributions = einx.multiply(
            "... target, target tap -> ... target tap", residual, self.weights
        )
        batch_shape = residual.shape[:-1]
        zeros = jnp.zeros((*batch_shape, self.n_source), contributions.dtype)
        # (Hᵀr)ⱼ = ∑ₖ ∑ₛ [iₖₛ = j] wₖₛ rₖ: the scatter adds repeated indices
        return zeros.at[..., self.indices].add(contributions)

    def operator(self) -> lx.FunctionLinearOperator:
        """H as an (M, N) lineax operator.

        Its transpose goes through ``jax.linear_transpose``, which equals
        `apply_adjoint` (the dot-product test checks it).

        Returns:
            A ``lineax.FunctionLinearOperator`` from ℝᴺ to ℝᴹ.
        """
        structure = jax.ShapeDtypeStruct((self.n_source,), self.weights.dtype)
        return lx.FunctionLinearOperator(self.apply, structure)

    def __matmul__(self, inner: Stencil) -> Stencil:
        """H_self H_inner, the stencil of ``inner`` then ``self``: see `compose`.

        Examples:
            >>> import jax.numpy as jnp
            >>> from functax.interp import Stencil
            >>> average = Stencil(
            ...     jnp.array([[0, 1]]), jnp.full((1, 2), 0.5), n_source=2
            ... )
            >>> swap = Stencil(jnp.array([[1], [0]]), jnp.ones((2, 1)), n_source=2)
            >>> (average @ swap).indices.tolist()
            [[1, 0]]
        """
        # _compose imports this module, so the import waits for the first call
        from functax._src.interp._compose import compose

        return compose(self, inner)

    def to_dense(self) -> Float[Array, "target source"]:
        """H as a dense (M, N) matrix, repeated indices summed.

        Examples:
            >>> import jax.numpy as jnp
            >>> from functax.interp import Stencil
            >>> Stencil(
            ...     jnp.array([[1, 1]]), jnp.array([[0.25, 0.5]]), n_source=3
            ... ).to_dense().tolist()
            [[0.0, 0.75, 0.0]]
        """
        rows = einx.id(
            "target -> target tap", jnp.arange(self.n_target), tap=self.n_taps
        )
        dense = jnp.zeros((self.n_target, self.n_source), self.weights.dtype)
        return dense.at[rows, self.indices].add(self.weights)

    @classmethod
    def from_dense(cls, matrix: Any, *, max_taps: int) -> Stencil:
        """A stencil holding each row's non-zero entries, padded to ``max_taps``.

        Built on the host, so ``matrix`` must be concrete.

        Args:
            matrix: A dense (M, N) matrix, NumPy or JAX.
            max_taps: S, the number of taps per row.

        Returns:
            The `Stencil`, in the dtype of ``matrix``; padding taps repeat the
            row's first index with weight 0 (index 0 for an empty row).

        Raises:
            ValidationError: If ``matrix`` is not 2-D or floating, or a row has
                more than ``max_taps`` non-zeros.

        Examples:
            >>> import numpy as np
            >>> from functax.interp import Stencil
            >>> H = Stencil.from_dense(
            ...     np.array([[0.0, 2.0, 0.0], [1.0, 0.0, 3.0]]), max_taps=2
            ... )
            >>> H.indices.tolist(), H.weights.tolist()
            ([[1, 1], [0, 2]], [[2.0, 0.0], [1.0, 3.0]])
        """
        dense = np.asarray(matrix)
        if dense.ndim != 2:
            msg = f"'matrix' must be 2-D (target, source), got shape {dense.shape}"
            raise ValidationError(msg)
        if not np.issubdtype(dense.dtype, np.floating):
            msg = f"'matrix' must have a floating dtype, got {dense.dtype}"
            raise ValidationError(msg)
        n_target, n_source = dense.shape
        indices = np.zeros((n_target, max_taps), dtype=np.int32)
        weights = np.zeros((n_target, max_taps), dtype=dense.dtype)
        for row in range(n_target):
            (columns,) = np.nonzero(dense[row])
            if columns.size > max_taps:
                msg = (
                    f"'matrix' row {row} has {columns.size} non-zeros, more than "
                    f"max_taps={max_taps}"
                )
                raise ValidationError(msg)
            indices[row, : columns.size] = columns
            # padding repeats a column the row already reads
            indices[row, columns.size :] = columns[0] if columns.size else 0
            weights[row, : columns.size] = dense[row, columns]
        return cls(jnp.asarray(indices), jnp.asarray(weights), n_source=n_source)

    @staticmethod
    def _check_last_axis(array: jax.Array, size: int, name: str) -> None:
        if array.ndim < 1 or array.shape[-1] != size:
            msg = (
                f"'{name}' must have a last axis of length {size}, "
                f"got shape {array.shape}"
            )
            raise ValidationError(msg)
