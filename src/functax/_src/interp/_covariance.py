"""Posterior covariances Σ* of a field at the targets of a Gaussian plan.

A scheme with a probabilistic reading has a posterior mean (its plan) and a
posterior covariance at the targets. Linear interpolation on one axis is the
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
"""How far below 0 rounding may push a kernel posterior variance, relative to k**."""


class PosteriorCovariance(eqx.Module):
    """Σ*, the (M, M) posterior covariance of the field at M targets."""

    @property
    @abc.abstractmethod
    def n_target(self) -> int:
        """M, the number of targets."""

    @abc.abstractmethod
    def variance(self) -> Float[Array, " target"]:
        """diag Σ*, the posterior variance at each target."""

    @abc.abstractmethod
    def covariance(self) -> Float[Array, "target target"]:
        """Σ* as a dense (M, M) matrix."""

    @abc.abstractmethod
    def sample(
        self, batch_shape: Shape, *, key: PRNGKeyArray
    ) -> Float[Array, "*batch target"]:
        """ε ~ 𝒩(0, Σ*), one independent draw per entry of ``batch_shape``."""


def _safe_sqrt(value: jax.Array) -> jax.Array:
    """√v for v ≥ 0, with a zero (not infinite) gradient at v = 0."""
    positive = value > 0
    # the untaken branch reads 1, so √ never sees 0 and its gradient stays finite
    root = jnp.sqrt(jnp.where(positive, value, jnp.ones((), value.dtype)))
    return jnp.where(positive, root, jnp.zeros((), value.dtype))


class BrownianBridge(PosteriorCovariance):
    """The Brownian bridge between the knots of one axis, cell by cell.

    Let f be Brownian motion with variance rate σ², observed exactly at the
    knots x₀, …, x_{N−1}. Conditioned on them, f on cell i (width
    Δᵢ = |xᵢ₊₁ − xᵢ|) is a Brownian bridge: its mean is linear interpolation,
    and for targets at offsets t₁ ≤ t₂ ∈ [0, 1] in the same cell

        Cov[f(x₁), f(x₂) ∣ knots] = σ² Δᵢ t₁ (1 − t₂),

    so Var[f(x*) ∣ knots] = σ² Δᵢ t (1 − t), zero at the knots and σ² Δᵢ / 4 at
    mid-cell. Targets in different cells are independent, because Brownian
    motion is Markov: the knots between two cells screen them off. On a
    periodic axis the wrap cell between the last node and the first one has
    width P − |x_{N−1} − x₀|.

    Algorithm:
        `sample` draws every cell's bridge at once, in O(M log M):

        1. Sort the targets by cell, then by offset.
        2. Draw a standard Brownian motion W on each cell at the sorted
           offsets, W(tₖ) = W(tₖ₋₁) + √(tₖ − tₖ₋₁) zₖ (from W(0) = 0 at the
           first target of a cell), as one cumulative sum minus its value
           before the cell starts.
        3. Continue each cell to W(1) = W(t_last) + √(1 − t_last) ζ.
        4. Pin the far knot: ε = √(σ² Δᵢ) (W(t) − t W(1)), which has the bridge
           covariance above, and undo the sort.

    Args:
        cell: i, the cell of each target, in [0, N − 1) (or [0, N) on a
            periodic axis).
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
        those rows invalid. On a periodic axis i = ⌊u⌋ mod N. Non-finite
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
        # Δᵢ = |xᵢ₊₁ − xᵢ|
        widths = jnp.abs(jnp.diff(values))
        finite = jnp.isfinite(positions)
        positions = jnp.where(finite, positions, jnp.zeros((), positions.dtype))
        lower = jnp.floor(positions)
        if axis.period is not None:
            # the wrap cell x_{N−1} → x₀ + P has width P − |x_{N−1} − x₀|
            wrap = axis.period - jnp.abs(values[-1] - values[0])
            widths = jnp.append(widths, wrap)
            cell = jnp.mod(lower, n_nodes).astype(jnp.int32)
            offset = positions - lower
        else:
            # i ∈ [0, N − 2], so the last knot is t = 1 of the last cell
            cell = jnp.clip(lower, 0, n_nodes - 2).astype(jnp.int32)
            offset = jnp.clip(positions - cell, 0.0, 1.0)
        # σ² Δᵢ
        scale = rate * widths[cell]
        return cls(cell, offset, scale)

    @property
    def n_target(self) -> int:
        """M, the number of targets."""
        return self.offset.shape[0]

    def variance(self) -> Float[Array, " target"]:
        """Var[f(x*) ∣ knots] = σ² Δᵢ t (1 − t).

        Returns:
            The variance at each target.
        """
        # σ² Δᵢ t (1 − t)
        return self.scale * self.offset * (1.0 - self.offset)

    def covariance(self) -> Float[Array, "target target"]:
        """Σ*ₖₗ = σ² Δᵢ min(tₖ, tₗ) (1 − max(tₖ, tₗ)) in one cell, else 0.

        Block diagonal by cell (once the targets are sorted by cell),
        symmetric and positive semi-definite.

        Returns:
            Σ*, dense (M, M).
        """
        same_cell = einx.equal("row, col -> row col", self.cell, self.cell)
        earlier = einx.minimum("row, col -> row col", self.offset, self.offset)
        later = einx.maximum("row, col -> row col", self.offset, self.offset)
        # σ² Δᵢ t₁ (1 − t₂) for t₁ ≤ t₂
        bridge = einx.multiply(
            "row, row col -> row col", self.scale, earlier * (1.0 - later)
        )
        zero = jnp.zeros((), bridge.dtype)
        return jnp.where(same_cell, bridge, zero)

    def sample(
        self, batch_shape: Shape, *, key: PRNGKeyArray
    ) -> Float[Array, "*batch target"]:
        """ε ~ 𝒩(0, Σ*), drawn cell by cell as W(t) − t W(1).

        Args:
            batch_shape: The leading shape of independent draws, () for one.
            key: The PRNG key.

        Returns:
            ε, of shape (*batch_shape, M), in the dtype of ``scale``.
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
        first = jnp.concatenate([jnp.ones(1, dtype=bool), changes])
        last = jnp.concatenate([changes, jnp.ones(1, dtype=bool)])
        zero = jnp.zeros(1, offsets.dtype)
        previous = jnp.where(first, 0.0, jnp.concatenate([zero, offsets[:-1]]))
        # the index of the first and of the last target of each target's cell
        start = jax.lax.cummax(jnp.where(first, index, 0))
        stop = jax.lax.cummin(jnp.where(last, index, self.n_target - 1), reverse=True)
        # √(tₖ − tₖ₋₁) and √(1 − t_last)
        step_scale = _safe_sqrt(offsets - previous)
        end_scale = _safe_sqrt(1.0 - offsets[stop])
        # √(σ² Δᵢ)
        bridge_scale = _safe_sqrt(scales)

        def draw(walk: jax.Array, end: jax.Array) -> jax.Array:
            # W(tₖ) − W(tₖ₋₁) = √(tₖ − tₖ₋₁) zₖ
            steps = step_scale * walk
            total = jnp.cumsum(steps)
            # W(tₖ): the sum since the first target of the cell
            before = total - steps
            brownian = total - before[start]
            # W(1) = W(t_last) + √(1 − t_last) ζ
            brownian_end = brownian[stop] + end_scale * end[stop]
            # ε = √(σ² Δᵢ) (W(t) − t W(1)), back in the targets' order
            bridge = bridge_scale * (brownian - offsets * brownian_end)
            return jnp.zeros_like(bridge).at[order].set(bridge)

        draw_batched = jnp.vectorize(draw, signature="(target),(target)->(target)")
        return draw_batched(walk_noise, end_noise)


class KernelCovariance(PosteriorCovariance):
    """Σ*(x, x′) = k(x, x′) − h(x)ᵀ A⁻¹ h(x′), the posterior covariance of a kernel fit.

    A is the saddle-point matrix [[K + τ²I, F], [Fᵀ, 0]] of the fit and
    h(x) = [k(X, x); f(x)] the cross-kernel and the drift at a target. Written
    out, with the drift's correction u = f* − Fᵀ K⁻¹ k*, this is the kriging
    variance

        σ²(x*) = k** − k*ᵀ K⁻¹ k* + uᵀ (Fᵀ K⁻¹ F)⁻¹ u,

    which drops the last term without a drift (simple kriging, a Gaussian
    process posterior). It is the covariance of the latent field: the nugget
    sits on the sources' diagonal only. Rounding can push σ² slightly below
    0 where the samples pin the field (at the sources, with no nugget); a
    negative part above ``NEGATIVE_VARIANCE_RTOL`` · k** raises, and the
    rest is clipped to 0.

    Algorithm:
        1. W = A⁻¹ Hᵀ, one `KernelSolve.solve` per target row of H.
        2. Σ* = K** − H W, symmetrised; σ² = k** − ∑_c H ⊙ Wᵀ.
        3. `sample` draws ε = V √Λ₊ z from the eigendecomposition
           Σ* = V Λ Vᵀ, with negative eigenvalues (rounding) set to 0, since
           Σ* is singular wherever a target sits on a noiseless source.

    Args:
        fit: The `KernelSolve` of the plan, A factorised.
        read: H, the (M, N + P) read-out [K* F*].
        targets: x*, the M target coordinates.
        kernel: k, the prior covariance.

    Raises:
        ValidationError: If ``read`` and ``targets`` do not have one row per
            target, or ``read`` does not have N + P columns.

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
    kernel: Kernel = eqx.field(static=True)

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
        """W, with row k equal to A⁻¹ hₖ."""
        return self.fit.solve(self.read)

    def _prior_variance(self) -> Float[Array, " target"]:
        """k(x*ₖ, x*ₖ) at each target."""

        def at(target: jax.Array) -> jax.Array:
            point = einx.id("dim -> 1 dim", target)
            return kernel_matrix(self.kernel, point, point)[0, 0]

        return jax.vmap(at)(self.targets)

    def variance(self) -> Float[Array, " target"]:
        """σ²(x*) = k** − hᵀ A⁻¹ h, clipped at 0 after the rounding check.

        Returns:
            The posterior variance at each target.

        Raises:
            ValidationError: If σ² < −``NEGATIVE_VARIANCE_RTOL`` · k** at a
                target (checked when run under ``jit``): the kernel is not
                positive definite on the sources, or needs a nugget.
        """
        prior = self._prior_variance()
        # hᵀ A⁻¹ h, row by row
        explained = einx.dot(
            "target coefficient, target coefficient -> target",
            self.read,
            self._weights(),
        )
        variance = prior - explained
        tolerance = NEGATIVE_VARIANCE_RTOL * jnp.abs(prior)
        too_negative = jnp.any(variance < -tolerance)
        variance = check_values(
            variance,
            too_negative,
            "the kernel posterior variance must be non-negative up to "
            f"{NEGATIVE_VARIANCE_RTOL:g} · k(x*, x*); is the kernel positive "
            "definite on the sources, or does it need a nugget",
        )
        return jnp.maximum(variance, jnp.zeros((), variance.dtype))

    def covariance(self) -> Float[Array, "target target"]:
        """Σ* = K** − H A⁻¹ Hᵀ, dense (M, M) and symmetric.

        Returns:
            Σ*.
        """
        prior = kernel_matrix(self.kernel, self.targets, self.targets)
        explained = einx.dot(
            "row coefficient, col coefficient -> row col",
            self.read,
            self._weights(),
        )
        covariance = prior - explained
        transposed = einx.id("row col -> col row", covariance)
        return (covariance + transposed) / 2

    def sample(
        self, batch_shape: Shape, *, key: PRNGKeyArray
    ) -> Float[Array, "*batch target"]:
        """ε ~ 𝒩(0, Σ*), from the eigendecomposition of Σ*.

        Args:
            batch_shape: The leading shape of independent draws, () for one.
            key: The PRNG key.

        Returns:
            ε, of shape (*batch_shape, M), in the dtype of ``read``.
        """
        covariance = self.covariance()
        eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
        # √Λ₊: the negative eigenvalues are rounding on a singular Σ*
        root = _safe_sqrt(jnp.maximum(eigenvalues, jnp.zeros((), eigenvalues.dtype)))
        shape = (*batch_shape, self.n_target)
        noise = jax.random.normal(key, shape, covariance.dtype)
        scaled = einx.multiply("... mode, mode -> ... mode", noise, root)
        # ε = V √Λ₊ z
        return einx.dot("... mode, target mode -> ... target", scaled, eigenvectors)
