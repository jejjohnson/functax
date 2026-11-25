"""Posterior covariances Σ_{*∣y} of a field at the targets of a Gaussian plan.

A scheme with a probabilistic reading has a posterior mean (its plan) and a
posterior covariance Σ_{*∣y} ∈ ℝ^{M×M} of f_* = f(X_*) at the M targets X_*,
given the samples y. Linear interpolation on one axis is the
posterior mean under a Brownian-motion prior, and its covariance is the
Brownian bridge of each cell. Kriging's covariance, `KernelCovariance`, is
dense: the prior kernel less what the samples explain, through the same
saddle-point solve as the mean.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import abc

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Int, PRNGKeyArray

from functax._src._checks import check_values
from functax._src.geometry._types import Axis
from functax._src.interp._fit import KernelSolve
from functax._src.interp._kernels import Kernel, kernel_matrix
from functax._typing import Shape
from functax.exceptions import ValidationError


__all__ = [
    "NEGATIVE_VARIANCE_RTOL",
    "BrownianBridge",
    "KernelCovariance",
    "PosteriorCovariance",
]


NEGATIVE_VARIANCE_RTOL = 1e-10
"""How far below 0 rounding may push a posterior variance, relative to k(x_*, x_*)."""


class PosteriorCovariance(eqx.Module):
    """Σ_{*∣y}, the (M, M) posterior covariance of the field at M targets.

    Each entry is Cov[f(x_*), f(x_*′) ∣ y] for a pair of targets x_*, x_*′
    and the samples y; a subclass gives M, the diagonal σ_*², the dense matrix
    and draws f̃ ~ 𝒩(0, Σ_{*∣y}).

    Examples:
        Linear interpolation's Brownian bridge is one:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis
        >>> from functax.interp import BrownianBridge, PosteriorCovariance
        >>> bridge = BrownianBridge.from_positions(
        ...     Axis(jnp.array([0.0, 1.0])), jnp.array([0.5]), rate=1.0
        ... )
        >>> isinstance(bridge, PosteriorCovariance), bridge.n_target
        (True, 1)

        A subclass for independent targets, with a diagonal Σ_{*∣y}:

        >>> class Independent(PosteriorCovariance):
        ...     variances: jax.Array
        ...
        ...     @property
        ...     def n_target(self):
        ...         return self.variances.shape[0]
        ...
        ...     def variance(self):
        ...         return self.variances
        ...
        ...     def covariance(self):
        ...         return jnp.diag(self.variances)
        ...
        ...     def sample(self, batch_shape, *, key):
        ...         shape = (*batch_shape, self.n_target)
        ...         noise = jax.random.normal(key, shape)
        ...         return noise * jnp.sqrt(self.variances)
        >>> independent = Independent(jnp.array([1.0, 4.0]))
        >>> independent.covariance().tolist()
        [[1.0, 0.0], [0.0, 4.0]]
        >>> independent.sample((3,), key=jax.random.key(0)).shape
        (3, 2)
    """

    @property
    @abc.abstractmethod
    def n_target(self) -> int:
        """M, the number of targets."""

    @abc.abstractmethod
    def variance(self) -> Float[Array, " target"]:
        """σ_*² = diag(Σ_{*∣y}), the posterior variance at each target."""

    @abc.abstractmethod
    def covariance(self) -> Float[Array, "target target"]:
        """Σ_{*∣y} as a dense (M, M) matrix."""

    @abc.abstractmethod
    def sample(
        self, batch_shape: Shape, *, key: PRNGKeyArray
    ) -> Float[Array, "*batch target"]:
        """f̃ ~ 𝒩(0, Σ_{*∣y}), one independent draw per entry of ``batch_shape``."""


def _safe_sqrt(value: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    """√v for v ≥ 0, with a zero (not infinite) gradient at v = 0."""
    positive = value > 0
    one = jnp.ones((), value.dtype)
    zero = jnp.zeros((), value.dtype)
    # the untaken branch reads 1, so √ never sees 0 and its gradient stays finite
    safe_value = jnp.where(positive, value, one)
    # √v
    root = jnp.sqrt(safe_value)
    return jnp.where(positive, root, zero)


class BrownianBridge(PosteriorCovariance):
    """The Brownian bridge between the knots of one axis, cell by cell.

    Let f be Brownian motion with variance rate σ², observed exactly as y at
    the N knots x₀, …, x_{N−1}. Conditioned on them, f on cell i (width
    Δᵢ = |xᵢ₊₁ − xᵢ|) is a Brownian bridge: its mean is linear interpolation,
    and for two targets x_* and x_*′ in the same cell, at offsets
    t ≤ t′ ∈ [0, 1],

        Cov[f(x_*), f(x_*′) ∣ y] = σ² Δᵢ t (1 − t′),

    so σ_*² = σ² Δᵢ t (1 − t), zero at the knots and σ² Δᵢ / 4 at mid-cell.
    Targets in different cells are independent, because Brownian motion is
    Markov: the knots between two cells screen them off. On a closed axis
    of period T_period the wrap cell between the last node and the first one
    has width T_period − |x_{N−1} − x₀|.

    Algorithm:
        `sample` draws every cell's bridge at once, in O(M log M):

        1. Sort the targets by cell, then by offset, so that m indexes a
           cell's targets in order of their offsets tₘ.
        2. Draw a standard Brownian motion b(t) on each cell at the sorted
           offsets, b(tₘ) = b(tₘ₋₁) + √(tₘ − tₘ₋₁) zₘ with zₘ ~ 𝒩(0, 1) (from
           b(0) = 0 at the first target of a cell), as one cumulative sum
           minus its value before the cell starts.
        3. Continue each cell to b(1) = b(t_last) + √(1 − t_last) ζ, with
           ζ ~ 𝒩(0, 1) and t_last the cell's last offset.
        4. Pin the far knot: f̃ = √(σ² Δᵢ) (b(t) − t b(1)), which has the
           bridge covariance above, and undo the sort.

    Args:
        cell: i, the cell of each target, in [0, N − 1) (or [0, N) on a
            closed periodic axis).
        offset: t ∈ [0, 1], the offset of each target in its cell.
        scale: σ² Δᵢ, the variance rate times the width of each target's cell.

    Raises:
        ValidationError: If the three are not 1-D of one length, ``cell`` is not
            integer or ``offset`` and ``scale`` not floating.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis
        >>> from functax.interp import BrownianBridge
        >>> # two cells of widths 1 and 2, variance rate σ² = 4
        >>> bridge = BrownianBridge.from_positions(
        ...     Axis(jnp.array([0.0, 1.0, 3.0])),
        ...     jnp.array([0.0, 0.5, 1.25, 1.5]),
        ...     rate=4.0,
        ... )
        >>> bridge.variance().tolist()
        [0.0, 1.0, 1.5, 2.0]
        >>> bridge.covariance()[2].tolist()
        [0.0, 0.0, 1.5, 1.0]
        >>> bridge.sample((5,), key=jax.random.key(0)).shape
        (5, 4)

    References:
        Kimeldorf, G. S. and Wahba, G. (1970). A correspondence between
        Bayesian estimation on stochastic processes and smoothing by splines.
        The Annals of Mathematical Statistics, 41(2), 495-502.
    """

    cell: Int[Array, " target"]
    offset: Float[Array, " target"]
    scale: Float[Array, " target"]

    def __init__(self, cell: ArrayLike, offset: ArrayLike, scale: ArrayLike) -> None:
        cell = jnp.asarray(cell)
        offset = jnp.asarray(offset)
        scale = jnp.asarray(scale)
        if offset.ndim != 1:
            msg = f"'offset' must be 1-D (target,), got shape {offset.shape}"
            raise ValidationError(msg)
        for name, array in (("cell", cell), ("scale", scale)):
            if array.shape != offset.shape:
                msg = (
                    f"'{name}' must have the shape of 'offset' {offset.shape}, "
                    f"got {array.shape}"
                )
                raise ValidationError(msg)
        if not jnp.issubdtype(cell.dtype, jnp.integer):
            msg = f"'cell' must have an integer dtype, got {cell.dtype}"
            raise ValidationError(msg)
        for name, array in (("offset", offset), ("scale", scale)):
            if not jnp.issubdtype(array.dtype, jnp.floating):
                msg = f"'{name}' must have a floating dtype, got {array.dtype}"
                raise ValidationError(msg)
        self.cell = cell
        self.offset = offset
        self.scale = scale

    @classmethod
    def from_positions(
        cls, axis: Axis, positions: Float[Array, " target"], *, rate: float | int
    ) -> BrownianBridge:
        """The bridge of each target from its index position u on ``axis``.

        The cell is i = ⌊u⌋ and the offset t = u − i, with u clipped to the
        axis ([0, N − 1]) so that a target off it reads a knot; a plan marks
        those rows invalid. On a closed axis i = ⌊u⌋ mod N. Non-finite
        positions read knot 0.

        Args:
            axis: The source axis, increasing or decreasing, maybe periodic.
            positions: u, the index positions of the M targets
                (`index_positions`).
            rate: σ², the variance rate of the Brownian-motion prior, in squared
                field units per coordinate unit.

        Returns:
            The `BrownianBridge`.
        """
        positions = jnp.asarray(positions)
        values = axis.values
        n_nodes = values.shape[0]
        # xᵢ₊₁ − xᵢ
        spacing = jnp.diff(values)
        # Δᵢ = |xᵢ₊₁ − xᵢ|
        widths = jnp.abs(spacing)
        finite = jnp.isfinite(positions)
        zero = jnp.zeros((), positions.dtype)
        positions = jnp.where(finite, positions, zero)
        # ⌊u⌋
        lower = jnp.floor(positions)
        if axis.period is not None and axis.closed:
            # x_{N−1} − x₀
            span = values[-1] - values[0]
            # the wrap cell x_{N−1} → x₀ + T_period has width
            # T_period − |x_{N−1} − x₀|
            wrap = axis.period - jnp.abs(span)
            widths = jnp.append(widths, wrap)
            # i = ⌊u⌋ mod N
            wrapped = jnp.mod(lower, n_nodes)
            cell = wrapped.astype(jnp.int32)
            # t = u − ⌊u⌋
            offset = positions - lower
        else:
            # i ∈ [0, N − 2], so the last knot is t = 1 of the last cell
            clipped = jnp.clip(lower, 0, n_nodes - 2)
            cell = clipped.astype(jnp.int32)
            # t = u − i, clipped to the cell
            unclipped_offset = positions - cell
            offset = jnp.clip(unclipped_offset, 0.0, 1.0)
        # σ² Δᵢ
        scale = rate * widths[cell]
        return cls(cell, offset, scale)

    @property
    def n_target(self) -> int:
        """M, the number of targets."""
        return self.offset.shape[0]

    def variance(self) -> Float[Array, " target"]:
        """σ_*² = σ² Δᵢ t (1 − t), the variance of f(x_*) ∣ y.

        Returns:
            The variance at each target.
        """
        # σ² Δᵢ t (1 − t)
        return self.scale * self.offset * (1.0 - self.offset)

    def covariance(self) -> Float[Array, "target target"]:
        """Σ_{*∣y}, with σ² Δᵢ min(t, t′) (1 − max(t, t′)) within a cell, else 0.

        The entry of two targets at offsets t and t′ in the same cell i is
        σ² Δᵢ min(t, t′) (1 − max(t, t′)), so Σ_{*∣y} is block diagonal by cell
        (once the targets are sorted by cell), symmetric and positive
        semi-definite.

        Returns:
            Σ_{*∣y}, dense (M, M).
        """
        same_cell = einx.equal("row, col -> row col", self.cell, self.cell)
        earlier = einx.minimum("row, col -> row col", self.offset, self.offset)
        later = einx.maximum("row, col -> row col", self.offset, self.offset)
        # t (1 − t′) for t ≤ t′
        shape = earlier * (1.0 - later)
        # σ² Δᵢ t (1 − t′)
        bridge = einx.multiply("row, row col -> row col", self.scale, shape)
        zero = jnp.zeros((), bridge.dtype)
        return jnp.where(same_cell, bridge, zero)

    def sample(
        self, batch_shape: Shape, *, key: PRNGKeyArray
    ) -> Float[Array, "*batch target"]:
        """f̃ ~ 𝒩(0, Σ_{*∣y}), drawn cell by cell as b(t) − t b(1).

        Args:
            batch_shape: The leading shape of independent draws, () for one.
            key: The PRNG key.

        Returns:
            f̃, of shape (*batch_shape, M), in the dtype of ``scale``.
        """
        key_walk, key_end = jax.random.split(key)
        shape = (*batch_shape, self.n_target)
        dtype = self.scale.dtype
        walk_noise = jax.random.normal(key_walk, shape, dtype)
        end_noise = jax.random.normal(key_end, shape, dtype)

        # --- sort by cell, then offset, and find each cell's first and last ---
        order = jnp.lexsort((self.offset, self.cell))
        cells = self.cell[order]
        offsets = self.offset[order]
        scales = self.scale[order]
        index = jnp.arange(self.n_target)
        changes = cells[1:] != cells[:-1]
        true = jnp.ones(1, dtype=bool)
        first = einx.id("1, target -> (1 + target)", true, changes)
        last = einx.id("target, 1 -> (target + 1)", changes, true)
        zero = jnp.zeros(1, offsets.dtype)
        # tₘ₋₁, from 0 at the first target of a cell
        shifted = einx.id("1, target -> (1 + target)", zero, offsets[:-1])
        previous = jnp.where(first, 0.0, shifted)
        # the index of the first and of the last target of each target's cell
        first_index = jnp.where(first, index, 0)
        start = jax.lax.cummax(first_index)
        last_index = jnp.where(last, index, self.n_target - 1)
        stop = jax.lax.cummin(last_index, reverse=True)
        # tₘ − tₘ₋₁
        increments = offsets - previous
        # √(tₘ − tₘ₋₁)
        step_scale = _safe_sqrt(increments)
        # 1 − t_last
        remainder = 1.0 - offsets[stop]
        # √(1 − t_last)
        end_scale = _safe_sqrt(remainder)
        # √(σ² Δᵢ)
        bridge_scale = _safe_sqrt(scales)

        def draw(
            walk: Float[Array, " target"], end: Float[Array, " target"]
        ) -> Float[Array, " target"]:
            # b(tₘ) − b(tₘ₋₁) = √(tₘ − tₘ₋₁) zₘ
            steps = step_scale * walk
            total = jnp.cumsum(steps)
            # b(tₘ): the sum since the first target of the cell
            before = total - steps
            brownian = total - before[start]
            # b(1) = b(t_last) + √(1 − t_last) ζ
            brownian_end = brownian[stop] + end_scale * end[stop]
            # b(t) − t b(1)
            pinned = brownian - offsets * brownian_end
            # f̃ = √(σ² Δᵢ) (b(t) − t b(1)), back in the targets' order
            bridge = bridge_scale * pinned
            unsorted = jnp.zeros_like(bridge)
            return unsorted.at[order].set(bridge)

        draw_batched = jnp.vectorize(draw, signature="(target),(target)->(target)")
        return draw_batched(walk_noise, end_noise)


class KernelCovariance(PosteriorCovariance):
    """Σ_{*∣y} = K_** − H A⁻¹ Hᵀ, the posterior covariance of a kernel fit.

    With the N sources X, the samples' covariance Σ_y = K_XX + σ_ε² I (the
    nugget σ_ε² on the diagonal) and the drift matrix P ∈ ℝ^{N×N_tail}, A is
    the saddle-point matrix [[Σ_y, P], [Pᵀ, 0]] of the fit. The read-out
    H = [K_*X P_*] ∈ ℝ^{M×(N+N_tail)} has at a target x_* the row
    h_*ᵀ = [k_*ᵀ p_*ᵀ], the cross-kernel k_* = k(X, x_*) and the drift p_*.
    Written out, with the drift's correction p̃_* = p_* − Pᵀ Σ_y⁻¹ k_*, the
    diagonal is the kriging variance

        σ_*² = k(x_*, x_*) − k_*ᵀ Σ_y⁻¹ k_* + p̃_*ᵀ (Pᵀ Σ_y⁻¹ P)⁻¹ p̃_*,

    which drops the last term without a drift (simple kriging, a Gaussian
    process posterior). It is the covariance of the latent field: the nugget
    sits on the sources' diagonal only. Rounding can push σ_*² slightly below
    0 where the samples pin the field (at the sources, with no nugget); a
    negative part above ``NEGATIVE_VARIANCE_RTOL`` · k(x_*, x_*) raises, and
    the rest is clipped to 0.

    Algorithm:
        1. W = A⁻¹ Hᵀ, one `KernelSolve.solve` per target row of H.
        2. Σ_{*∣y} = K_** − H W, symmetrised; σ_*² = k(x_*, x_*) − h_*ᵀ A⁻¹ h_*,
           row by row.
        3. `sample` draws f̃ = V diag(√γ₊) z, z ~ 𝒩(0, I), from the
           eigendecomposition Σ_{*∣y} = V diag(γ) Vᵀ with eigenvalues
           γ ∈ ℝ^M, whose negative entries (rounding) are set to 0 in γ₊,
           since Σ_{*∣y} is singular wherever a target sits on a noiseless
           source.

    Args:
        fit: The `KernelSolve` of the plan, A factorised.
        read: H, the (M, N + N_tail) read-out [K_*X P_*].
        targets: X_*, the M target coordinates.
        kernel: k, the prior covariance.

    Raises:
        ValidationError: If ``read`` and ``targets`` do not have one row per
            target, or ``read`` does not have N + N_tail columns.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Points
        >>> from functax.interp import Kriging, Wendland
        >>> sources = Points(jnp.array([[0.0], [1.0], [2.0]]))
        >>> targets = Points(jnp.array([[0.0], [0.5]]))
        >>> plan = Kriging(Wendland(2, support=3.0), drift=None).plan(
        ...     sources, targets
        ... )
        >>> variance = plan.posterior.variance()
        >>> # zero at a noiseless source, positive between two
        >>> bool(jnp.allclose(variance[0], 0.0)), bool(variance[1] > 0.0)
        (True, True)
        >>> plan.posterior.covariance().shape
        (2, 2)
    """

    fit: KernelSolve
    read: Float[Array, "target coefficient"]
    targets: Float[Array, "target dim"]
    # a pytree field, not static: a kernel holds arrays (kernellib's
    # hyperparameters), which must trace and differentiate with the plan
    kernel: Kernel

    def __check_init__(self) -> None:
        n_target = self.read.shape[0]
        if self.targets.ndim != 2 or self.targets.shape[0] != n_target:
            msg = (
                f"'targets' must have one row per row of 'read' ({n_target}), "
                f"got shape {self.targets.shape}"
            )
            raise ValidationError(msg)
        n_coefficients = self.fit.n_source + self.fit.n_terms
        if self.read.shape[1] != n_coefficients:
            msg = (
                f"'read' must have the fit's {n_coefficients} coefficients as "
                f"columns, got shape {self.read.shape}"
            )
            raise ValidationError(msg)

    @property
    def n_target(self) -> int:
        """M, the number of targets."""
        return self.read.shape[0]

    def _weights(self) -> Float[Array, "target coefficient"]:
        """W = A⁻¹ Hᵀ, stored with the column A⁻¹ h_* of each target as a row."""
        return self.fit.solve(self.read)

    def _prior_variance(self) -> Float[Array, " target"]:
        """k(x_*, x_*) at each target x_*."""

        def at(target: Float[Array, " dim"]) -> Float[Array, ""]:
            point = einx.id("dim -> 1 dim", target)
            # k(x_*, x_*)
            gram = kernel_matrix(self.kernel, point, point)
            return gram[0, 0]

        at_every_target = jax.vmap(at)
        return at_every_target(self.targets)

    def variance(self) -> Float[Array, " target"]:
        """σ_*² = k(x_*, x_*) − h_*ᵀ A⁻¹ h_*, clipped at 0 after the rounding check.

        Returns:
            The posterior variance at each target.

        Raises:
            ValidationError: If σ_*² < −``NEGATIVE_VARIANCE_RTOL`` · k(x_*, x_*)
                at a target (checked when run under ``jit``): the kernel is
                not positive definite on the sources, or needs a nugget.
        """
        # k(x_*, x_*)
        prior = self._prior_variance()
        # W = A⁻¹ Hᵀ
        weights = self._weights()
        # h_*ᵀ A⁻¹ h_*, row by row
        explained = einx.dot(
            "target coefficient, target coefficient -> target", self.read, weights
        )
        # σ_*² = k(x_*, x_*) − h_*ᵀ A⁻¹ h_*
        variance = prior - explained
        # tol · |k(x_*, x_*)|
        tolerance = NEGATIVE_VARIANCE_RTOL * jnp.abs(prior)
        below = variance < -tolerance
        too_negative = jnp.any(below)
        variance = check_values(
            variance,
            too_negative,
            "the kernel posterior variance must be non-negative up to "
            f"{NEGATIVE_VARIANCE_RTOL:g} · k(x*, x*); is the kernel positive "
            "definite on the sources, or does it need a nugget",
        )
        zero = jnp.zeros((), variance.dtype)
        # max(σ_*², 0)
        return jnp.maximum(variance, zero)

    def covariance(self) -> Float[Array, "target target"]:
        """Σ_{*∣y} = K_** − H A⁻¹ Hᵀ, dense (M, M) and symmetric.

        Returns:
            Σ_{*∣y}.
        """
        # K_**
        prior = kernel_matrix(self.kernel, self.targets, self.targets)
        # W = A⁻¹ Hᵀ
        weights = self._weights()
        # H A⁻¹ Hᵀ
        explained = einx.dot(
            "row coefficient, col coefficient -> row col", self.read, weights
        )
        # Σ_{*∣y} = K_** − H A⁻¹ Hᵀ
        covariance = prior - explained
        transposed = einx.id("row col -> col row", covariance)
        # ½ (Σ_{*∣y} + Σ_{*∣y}ᵀ)
        return (covariance + transposed) / 2

    def sample(
        self, batch_shape: Shape, *, key: PRNGKeyArray
    ) -> Float[Array, "*batch target"]:
        """f̃ ~ 𝒩(0, Σ_{*∣y}), from the eigendecomposition of Σ_{*∣y}.

        Args:
            batch_shape: The leading shape of independent draws, () for one.
            key: The PRNG key.

        Returns:
            f̃, of shape (*batch_shape, M), in the dtype of ``read``.
        """
        covariance = self.covariance()
        # Σ_{*∣y} = V diag(γ) Vᵀ
        eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
        zero = jnp.zeros((), eigenvalues.dtype)
        # γ₊ = max(γ, 0): the negative eigenvalues are rounding on a singular Σ_{*∣y}
        clipped = jnp.maximum(eigenvalues, zero)
        # √γ₊
        root = _safe_sqrt(clipped)
        shape = (*batch_shape, self.n_target)
        # z ~ 𝒩(0, I)
        noise = jax.random.normal(key, shape, covariance.dtype)
        # diag(√γ₊) z
        scaled = einx.multiply("... mode, mode -> ... mode", noise, root)
        # f̃ = V diag(√γ₊) z
        return einx.dot("... mode, target mode -> ... target", scaled, eigenvectors)
