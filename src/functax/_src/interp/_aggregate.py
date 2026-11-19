"""Aggregation: binning and cloud-in-cell gridding as adjoints of a sampling stencil.

Sampling reads M grid cells at N sample positions through a stencil
H ∈ ℝ^{N×M} (one tap per row for nearest and 2ᴰ on D grid axes for linear).
Aggregation runs the other way: for the samples y ∈ ℝ^N the sums in each
cell are Hᵀ y, the counts are (N₁, …, N_M) = Hᵀ 1, and the means are
ȳⱼ = (Hᵀ y)ⱼ / Nⱼ in cell j. The counts depend only on the positions, so for
fixed positions the sum and the mean are linear in y and form an ordinary
plan with an adjoint.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

from typing import Any, ClassVar

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float

from functax._src.geometry._types import (
    AffineGrid,
    Axis,
    CurvilinearGrid,
    Points,
    RectilinearGrid,
)
from functax._src.interp._bases import LinearBasis, NearestBasis
from functax._src.interp._fit import Identity
from functax._src.interp._locate import check_outside
from functax._src.interp._neighbourhood import check_scattered_policies
from functax._src.interp._plan import Plan
from functax._src.interp._policies import Policies
from functax._src.interp._schemes import Scheme
from functax._src.interp._stencil import Readout, Stencil
from functax._src.interp._tensor import tensor_stencil
from functax.exceptions import (
    GeometryError,
    NonlinearSchemeError,
    PolicyError,
    ValidationError,
)


__all__ = [
    "Binning",
    "BinningPlan",
    "TransposedStencil",
]


class TransposedStencil(Readout):
    """f̂ = diag(w) Hᵀ y: a sampling stencil H used backwards, scaled per cell.

    H ∈ ℝ^{N×M} reads M cells at N sample positions (``stencil``, N rows);
    this read-out maps the N samples y to the M cells by its transpose, a
    scatter-add, and scales cell j by wⱼ (1 for sums, 1/Nⱼ for means, with
    Nⱼ the samples in cell j). Its adjoint is H diag(w), the gather of the
    stencil.

    Args:
        stencil: H, with one row per sample and ``n_source`` = M cells.
        scale: w ∈ ℝ^M, one factor per cell.
        valid: Whether each cell is a real read-out (false for an empty cell
            under the mean).

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.interp import Stencil, TransposedStencil
        >>> # three samples, in cells 0, 1 and 1
        >>> H = Stencil(jnp.array([[0], [1], [1]]), jnp.ones((3, 1)), n_source=2)
        >>> sums = TransposedStencil(H, jnp.ones(2), jnp.ones(2, dtype=bool))
        >>> sums.apply(jnp.array([1.0, 2.0, 4.0])).tolist()
        [1.0, 6.0]
        >>> sums.apply_adjoint(jnp.array([1.0, 10.0])).tolist()
        [1.0, 10.0, 10.0]
    """

    stencil: Stencil
    scale: Float[Array, " cell"]
    valid: Bool[Array, " cell"]

    # an explicit __init__, since the inherited AbstractVar would put valid first
    def __init__(
        self,
        stencil: Stencil,
        scale: Float[Array, " cell"],
        valid: Bool[Array, " cell"],
    ) -> None:
        self.stencil = stencil
        self.scale = scale
        self.valid = valid

    def apply(
        self, coefficients: Float[Array, "*batch obs"]
    ) -> Float[Array, "*batch cell"]:
        """f̂ⱼ = wⱼ (Hᵀ y)ⱼ, over any leading batch dims."""
        # Hᵀ y
        gathered = self.stencil.apply_adjoint(coefficients)
        # f̂ⱼ = wⱼ (Hᵀ y)ⱼ
        return einx.multiply("... cell, cell -> ... cell", gathered, self.scale)

    def apply_adjoint(
        self, residual: Float[Array, "*batch cell"]
    ) -> Float[Array, "*batch obs"]:
        """H diag(w) r for a residual r ∈ ℝ^M, over any leading batch dims."""
        # diag(w) r
        scaled = einx.multiply("... cell, cell -> ... cell", residual, self.scale)
        # H diag(w) r
        return self.stencil.apply(scaled)

    def operator(self) -> lx.FunctionLinearOperator:
        """diag(w) Hᵀ as an (M, N) lineax operator."""
        structure = jax.ShapeDtypeStruct(
            (self.stencil.n_target,), self.stencil.weights.dtype
        )
        return lx.FunctionLinearOperator(self.apply, structure)


class BinningPlan(Plan):
    """A plan from N samples to M cells: the sums, means or counts per cell.

    For ``"sum"`` and ``"mean"`` it is an ordinary linear plan: ``read`` is
    diag(w) Hᵀ, its `adjoint` is H diag(w), and an empty cell under the mean is
    an invalid row that reads ``fill_value`` (NaN by default), with a zero row
    in `operator`, as a target outside the domain is for sampling. A count
    does not depend on the data, so under ``"count"`` the plan returns Hᵀ 1
    for every field and has no linear map: `operator`, `adjoint` and
    `linearize` raise `NonlinearSchemeError`.

    Args:
        counts: (N₁, …, N_M) = Hᵀ 1, the (weighted) number of samples in
            each cell.
        statistic: ``"mean"``, ``"sum"`` or ``"count"``.

    Examples:
        Three samples binned into three cells, the middle one empty:

        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points
        >>> from functax.interp import Binning
        >>> cells = Axis(jnp.arange(3.0))
        >>> samples = Points(jnp.array([[0.1], [0.4], [2.2]]))
        >>> means = Binning("mean").plan(samples, cells)
        >>> means.counts.tolist(), means.valid.tolist()
        ([2.0, 0.0, 1.0], [True, False, True])

        A batch of fields, and the adjoint of the sums, which gathers:

        >>> means(jnp.ones((2, 3))).shape
        (2, 3)
        >>> sums = Binning("sum").plan(samples, cells)
        >>> sums.adjoint(jnp.array([1.0, 10.0, 100.0])).tolist()
        [1.0, 1.0, 100.0]

        A count has no linear map:

        >>> Binning("count").plan(
        ...     samples, cells
        ... ).operator()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        NonlinearSchemeError: a count does not depend on the data
    """

    counts: Float[Array, " cell"] = eqx.field(kw_only=True)
    statistic: str = eqx.field(static=True, kw_only=True)

    def __call__(self, data: Float[Array, "*batch obs"]) -> Float[Array, "*batch cell"]:
        """The statistic of the samples in each cell, over any leading batch dims.

        Args:
            data: y, one value per sample, with the samples last; a count reads
                only its shape.

        Returns:
            The statistic per cell, with the cells last.
        """
        if self.statistic == "count":
            # Hᵀ 1, whatever the data
            samples = jnp.asarray(data)
            data = jnp.ones_like(samples)
        return super().__call__(data)

    def _require_linear(self, what: str) -> None:
        if self.statistic == "count":
            msg = (
                f"a count does not depend on the data, so it has no {what}; use "
                "statistic='sum' for Hᵀ"
            )
            raise NonlinearSchemeError(msg)

    def adjoint(
        self, residual: Float[Array, "*batch cell"]
    ) -> Float[Array, "*batch obs"]:
        """H diag(w) r; invalid (empty) cells contribute nothing.

        Raises:
            NonlinearSchemeError: Under ``"count"``.
        """
        self._require_linear("adjoint")
        return super().adjoint(residual)

    def operator(self) -> lx.FunctionLinearOperator:
        """diag(w) Hᵀ as an (M, N) lineax operator, with zero rows at empty means.

        Raises:
            NonlinearSchemeError: Under ``"count"``.
        """
        self._require_linear("operator")
        return super().operator()

    def linearize(
        self, data: Float[Array, " obs"] | None = None
    ) -> lx.FunctionLinearOperator:
        """`operator`, since the sum and the mean are linear in y.

        Raises:
            NonlinearSchemeError: Under ``"count"``.
        """
        self._require_linear("linear map")
        return super().linearize(data)


class Binning(Scheme):
    """Aggregate scattered samples into grid cells: Hᵀ y, normalised for the mean.

    With H ∈ ℝ^{N×M} the sampling stencil from the M cells to the N sample
    positions and the samples y ∈ ℝ^N, cell j = 1, …, M holds

        sums  (Hᵀ y)ⱼ,   counts  Nⱼ = (Hᵀ 1)ⱼ,   means  ȳⱼ = (Hᵀ y)ⱼ / Nⱼ,

    with NaN (``fill_value``) where Nⱼ = 0. With ``weights="nearest"`` H has
    one tap per sample, so each sample falls in the cell of its nearest node
    (the cell's boundaries sit half-way between nodes, and half a cell past the
    end nodes) and this is a histogram, as ``scipy.stats.binned_statistic_dd``.
    With ``weights="linear"`` H is the 2ᴰ-tap linear stencil on D grid axes,
    so a sample spreads over the nodes around it: cloud-in-cell gridding,
    which conserves the total and is differentiable in the positions.

    Algorithm:
        1. Build H by `tensor_stencil` at the sample positions, with
           ``outside="nan"``: a sample off the grid gets zero weights and is
           dropped from every cell, never clamped onto the edge.
        2. Nⱼ = (Hᵀ 1)ⱼ by the stencil's scatter-add (`Stencil.apply_adjoint`).
        3. Read Hᵀ y through a `TransposedStencil`, scaled by 1/Nⱼ for the
           mean (by 1 where Nⱼ = 0, then marked invalid).

    The domain is the union of the cells for nearest weights, and the hull of
    the nodes for linear ones (a sample past the end nodes is dropped, so the
    deposited total equals the total of the samples inside). The plan traces
    in the positions, so ``jax.grad`` of a cloud-in-cell loss reaches them.

    Args:
        statistic: ``"mean"``, ``"sum"`` or ``"count"``.
        weights: ``"nearest"`` (binning) or ``"linear"`` (cloud-in-cell).

    Raises:
        ValidationError: If ``statistic`` or ``weights`` is unknown.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points
        >>> from functax.interp import Binning
        >>> cells = Axis(jnp.arange(4.0))
        >>> samples = Points(jnp.array([[0.1], [0.4], [2.2], [9.0]]))
        >>> values = jnp.array([1.0, 3.0, 5.0, 7.0])
        >>> # the last sample is off the grid; cells 1 and 3 are empty
        >>> Binning("mean").plan(samples, cells)(values).tolist()
        [2.0, nan, 5.0, nan]
        >>> Binning("count").plan(samples, cells)(values).tolist()
        [2.0, 0.0, 1.0, 0.0]
        >>> # cloud-in-cell: each sample splits between its two nodes
        >>> cic = Binning("sum", weights="linear").plan(samples, cells)
        >>> [round(v, 12) for v in cic(jnp.ones(4)).tolist()]
        [1.5, 0.5, 0.8, 0.2]
    """

    STATISTICS: ClassVar[tuple[str, ...]] = ("mean", "sum", "count")
    WEIGHTS: ClassVar[tuple[str, ...]] = ("nearest", "linear")

    statistic: str = eqx.field(static=True, default="mean")
    weights: str = eqx.field(static=True, default="nearest")

    def __check_init__(self) -> None:
        if self.statistic not in self.STATISTICS:
            msg = (
                f"'statistic' must be one of {list(self.STATISTICS)}, got "
                f"{self.statistic!r}"
            )
            raise ValidationError(msg)
        if self.weights not in self.WEIGHTS:
            msg = f"'weights' must be one of {list(self.WEIGHTS)}, got {self.weights!r}"
            raise ValidationError(msg)

    def plan(
        self, source: Any, target: Any, *, policies: Policies | None = None
    ) -> BinningPlan:
        """The plan from samples at ``source`` positions to the cells of ``target``.

        Args:
            source: `Points` with the ``"euclidean"`` metric, one coordinate per
                grid axis (map coordinates in (``x``, ``y``) order on an
                `AffineGrid` or a `CurvilinearGrid`); padding points
                (``valid`` False) count in no cell.
            target: The grid: an `Axis`, a `RectilinearGrid`, an `AffineGrid`
                or a `CurvilinearGrid`, its cells flattened row-major.
            policies: ``outside`` (``"nan"`` or ``"raise"``) acts on samples
                off the grid, ``fill_value`` is written at empty cells under
                the mean, and ``missing`` (``"propagate"`` or ``"fill"``) at
                non-finite samples; ``extent`` and ``boundary`` are not used,
                since ``weights`` fixes the domain.

        Returns:
            The `BinningPlan`; a grid keeps its shape as ``target_shape``.

        Raises:
            GeometryError: If the source is not euclidean `Points` or the
                target is not a grid, or their dimensions differ.
            PolicyError: Under ``outside="clamp"`` or ``"extrapolate"``, or
                ``missing="renormalise"``.
            OutsideDomainError: Under ``outside="raise"`` when a sample is off
                the grid.
        """
        policies = Policies() if policies is None else policies
        check_scattered_policies(policies, "Binning")
        if policies.missing == "renormalise":
            msg = "Binning supports missing='propagate' or 'fill', got 'renormalise'"
            raise PolicyError(msg)
        if not isinstance(source, Points) or source.metric != "euclidean":
            msg = (
                "'source' must be Points with the 'euclidean' metric, got "
                f"{type(source).__name__}"
            )
            raise GeometryError(msg)
        if not isinstance(
            target, Axis | RectilinearGrid | AffineGrid | CurvilinearGrid
        ):
            msg = (
                "'target' must be a grid (an Axis, a RectilinearGrid, an AffineGrid "
                f"or a CurvilinearGrid), got {type(target).__name__}"
            )
            raise GeometryError(msg)

        # --- H, from the cells to the samples; samples off the grid drop out ---
        nearest = self.weights == "nearest"
        basis = NearestBasis() if nearest else LinearBasis()
        sampling = Policies(
            outside="nan",
            extent="cells" if nearest else "nodes",
            boundary="constant",
        )
        n_axes = 1 if isinstance(target, Axis) else len(target.shape)
        stencil = tensor_stencil(target, source, (basis,) * n_axes, policies=sampling)
        inside = stencil.valid
        if source.valid is not None:
            # padding counts in no cell, and is never outside
            zero = jnp.zeros((), stencil.weights.dtype)
            weights = einx.where(
                "obs, obs tap, -> obs tap", source.valid, stencil.weights, zero
            )
            real_valid = stencil.valid & source.valid
            stencil = Stencil(
                stencil.indices,
                weights,
                n_source=stencil.n_source,
                valid=real_valid,
                check=False,
            )
            padding = ~source.valid
            inside = inside | padding
        check_outside(inside, policies)

        # --- Nⱼ = (Hᵀ 1)ⱼ, and the scale of each cell ---
        ones = jnp.ones(stencil.n_target, stencil.weights.dtype)
        # Nⱼ = (Hᵀ 1)ⱼ
        counts = stencil.apply_adjoint(ones)
        if self.statistic == "mean":
            occupied = counts > 0
            one = jnp.ones((), counts.dtype)
            safe_counts = jnp.where(occupied, counts, one)
            zero = jnp.zeros((), counts.dtype)
            # 1/Nⱼ
            reciprocal = 1.0 / safe_counts
            # wⱼ = 1/Nⱼ, and 0 at an empty cell, whose row the plan marks invalid
            scale = jnp.where(occupied, reciprocal, zero)
        else:
            occupied = jnp.ones(counts.shape, dtype=bool)
            scale = jnp.ones(counts.shape, counts.dtype)
        read = TransposedStencil(stencil, scale, occupied)
        target_shape = None if isinstance(target, Axis) else target.shape
        return BinningPlan(
            Identity(),
            read,
            policies,
            None,
            target_shape,
            counts=counts,
            statistic=self.statistic,
        )
