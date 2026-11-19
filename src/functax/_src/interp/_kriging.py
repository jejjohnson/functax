"""Kriging: simple, ordinary and universal, as a `GaussianPlan`.

Kriging is the kernel pattern of `RBF` read as a Gaussian process: the kernel
is a covariance, the tail a drift, the smoothing a nugget, and the same
saddle-point solve gives the mean and, through `KernelCovariance`, the
variance. The covariance's hyperparameters come from outside (a variogram fit
in kernellib, say), so the scheme takes the kernel as it is.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from functax._src.interp._covariance import KernelCovariance
from functax._src.interp._gaussian import GaussianPlan
from functax._src.interp._kernels import Kernel
from functax._src.interp._policies import Policies
from functax._src.interp._rbf import (
    PolynomialTail,
    Tail,
    _check_policies,
    _check_smoothing,
    kernel_system,
)
from functax._src.interp._schemes import Scheme
from functax.exceptions import ValidationError


__all__ = [
    "Kriging",
]


type Drift = str | int | Callable[[Float[Array, "obs dim"]], Float[Array, "obs term"]]
"""``"constant"``, a polynomial degree, or a callable P(X) giving the drift matrix."""


def _checked_drift(
    drift: Callable[[Float[Array, "obs dim"]], Float[Array, "obs term"]],
) -> Tail:
    """P(X), raising unless it returns one row of drift terms per point of X."""

    def terms(points: Float[Array, "obs dim"]) -> Float[Array, "obs term"]:
        # P(X)
        drift_terms = drift(points)
        values = jnp.asarray(drift_terms)
        if values.ndim != 2 or values.shape[0] != points.shape[0]:
            msg = (
                f"'drift' must return ({points.shape[0]}, term) drift terms, "
                f"got shape {values.shape}"
            )
            raise ValidationError(msg)
        return values

    return terms


class Kriging(Scheme):
    """Kriging: the best linear unbiased predictor under a covariance k and a drift.

    With the N samples y ∈ ℝ^N at the sources X, their covariance
    Σ_y = K_XX + σ_ε² I ∈ ℝ^{N×N} (the nugget σ_ε² on the diagonal, the K̃ of
    `KernelSolve`) and the drift matrix P = P(X) ∈ ℝ^{N×N_tail}, the mean
    solves the saddle-point system of `RBF`,

        [[Σ_y, P], [Pᵀ, 0]] [c; b] = [y; 0],  f̂(x_*) = k_*ᵀ c + p_*ᵀ b,

    with c ∈ ℝ^N, b ∈ ℝ^{N_tail}, and at a target x_* the cross-covariance
    k_* = k(X, x_*) ∈ ℝ^N and the drift p_* = P(x_*)ᵀ ∈ ℝ^{N_tail}. With the
    drift's correction p̃_* = p_* − Pᵀ Σ_y⁻¹ k_*, the posterior variance is

        σ_*² = k(x_*, x_*) − k_*ᵀ Σ_y⁻¹ k_* + p̃_*ᵀ (Pᵀ Σ_y⁻¹ P)⁻¹ p̃_*.

    Simple kriging (no drift, a known zero mean) is the Gaussian-process
    posterior of the latent field; ordinary kriging (P a column of ones) has
    weights that sum to 1; universal kriging takes a polynomial or any drift
    P(X). The plan
    is a `GaussianPlan` with a `KernelCovariance`, so ``plan(y)`` is the mean
    and ``plan.variance()``, ``covariance()`` and ``sample(y, key=…)`` add the
    uncertainty. The solve is Cholesky through lineax, in the dtype of the
    coordinates; the nugget is the only jitter, never added silently.

    Args:
        kernel: k, a positive definite covariance: any `Kernel`.
        drift: ``None`` (simple), ``"constant"`` (ordinary, the default), a
            polynomial degree ≥ 0 (universal, monomials up to that degree in
            coordinates scaled to the sources' bounding box), or a callable
            P(X) returning the drift matrix, one row per point of X
            (universal).
        nugget: σ_ε² ≥ 0, the noise variance of the samples.

    Raises:
        ValidationError: If ``kernel`` is not callable, ``drift`` is none of
            the above, or ``nugget`` is not a non-negative finite number.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Points
        >>> from functax.interp import Kriging, Wendland
        >>> sources = Points(
        ...     jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        ... )
        >>> targets = Points(jnp.array([[0.0, 0.0], [0.5, 0.5]]))
        >>> plan = Kriging(Wendland(2, support=2.0)).plan(sources, targets)
        >>> mean = plan(jnp.array([1.0, 2.0, 3.0, 4.0]))
        >>> # it passes through a noiseless sample, with zero variance there
        >>> bool(jnp.allclose(mean[0], 1.0))
        True
        >>> bool(jnp.allclose(plan.variance()[0], 0.0))
        True
        >>> # ordinary kriging weights sum to 1
        >>> bool(jnp.allclose(plan(jnp.ones(4)), 1.0))
        True
        >>> plan.sample(jnp.zeros((3, 4)), key=jax.random.key(0)).shape
        (3, 2)
    """

    kernel: Kernel
    # Drift | None rather than a narrower type, so a bad value reaches the check
    drift: Any = eqx.field(static=True, default="constant")
    nugget: float | int = eqx.field(static=True, default=0.0)

    def __check_init__(self) -> None:
        if not callable(self.kernel):
            msg = f"'kernel' must be callable k(targets, sources), got {self.kernel!r}"
            raise ValidationError(msg)
        drift = self.drift
        is_degree = isinstance(drift, int) and not isinstance(drift, bool)
        valid = (
            drift is None
            or drift == "constant"
            or (is_degree and drift >= 0)
            or (callable(drift) and not isinstance(drift, str))
        )
        if not valid:
            msg = (
                "'drift' must be None, 'constant', a degree ≥ 0 or a callable "
                f"F(x), got {drift!r}"
            )
            raise ValidationError(msg)
        _check_smoothing(self.nugget, "nugget")

    def _tail_for(self, sources: Float[Array, "source dim"]) -> Tail | None:
        """The drift P(X) of the points X, or ``None`` for simple kriging."""
        drift = self.drift
        if drift is None:
            return None
        if drift == "constant":
            return PolynomialTail.from_sources(sources, 0)
        if isinstance(drift, int):
            return PolynomialTail.from_sources(sources, drift)
        return _checked_drift(drift)

    def plan(
        self, source: Any, target: Any, *, policies: Policies | None = None
    ) -> GaussianPlan:
        """The plan of the kriging mean and covariance of ``source`` data at ``target``.

        Args:
            source: `Points` (no ``valid`` mask), the N sources.
            target: `Points` (padding rows are invalid, with NaN variance), an
                `Axis` or a `RectilinearGrid` read at its nodes.
            policies: The missing-sample and fill rules.

        Returns:
            The `GaussianPlan`: a `KernelSolve` fit, a `KernelReadout` and a
            `KernelCovariance`.

        Raises:
            GeometryError: If the geometries are unsupported.
            PolicyError: Under ``missing="renormalise"``.
            ValidationError: If the drift has as many terms as there are
                sources, or the kernel or drift returns the wrong shape.
        """
        policies = Policies() if policies is None else policies
        _check_policies(policies)
        system = kernel_system(
            self.kernel,
            source,
            target,
            tail_for=self._tail_for,
            diagonal=self.nugget,
            solver=None,
        )
        posterior = KernelCovariance(
            system.fit, system.read.matrix, system.targets, self.kernel
        )
        return GaussianPlan(
            system.fit,
            system.read,
            policies,
            None,
            system.target_shape,
            posterior=posterior,
        )
