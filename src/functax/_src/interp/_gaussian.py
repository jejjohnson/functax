"""Gaussian plans: a plan with a posterior mean and a posterior covariance.

A scheme with a probabilistic reading builds a `GaussianPlan`, so the type
says the variance exists; every other plan has no ``variance`` method, rather
than one that raises.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import einx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from functax._src.interp._covariance import PosteriorCovariance
from functax._src.interp._plan import Plan
from functax.exceptions import ValidationError


__all__ = [
    "GaussianPlan",
]


class GaussianPlan(Plan):
    """A plan read as a Gaussian posterior: f(X*) ∣ y ~ 𝒩(H 𝒮(y), Σ*).

    The mean is the plan itself, so `__call__`, `adjoint` and `operator` are
    those of `Plan`; ``posterior`` adds Σ*. Invalid rows (targets outside the
    domain under ``outside="nan"``) have NaN variance and covariance, and
    `sample` writes the plan's ``fill_value`` there, as the mean does.

    Args:
        fit: 𝒮, from samples to coefficients.
        read: H, from coefficients to target values.
        policies: The plan's outside, missing and boundary rules.
        posterior: Σ*, the posterior covariance at the targets (keyword only).

    Raises:
        ValidationError: If ``posterior`` has a different number of targets
            from ``read``.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points
        >>> from functax.interp import Linear
        >>> plan = Linear(prior_variance_rate=4.0).plan(
        ...     Axis(jnp.arange(3.0)), Points(jnp.array([[0.5], [1.0], [9.0]]))
        ... )
        >>> plan.mean(jnp.array([0.0, 2.0, 6.0])).tolist()
        [1.0, 2.0, nan]
        >>> plan.variance().tolist()
        [1.0, 0.0, nan]
        >>> plan.sample(jnp.zeros((8, 3)), key=jax.random.key(0)).shape
        (8, 3)
    """

    posterior: PosteriorCovariance = eqx.field(kw_only=True)

    def __check_init__(self) -> None:
        n_target = self.valid.shape[0]
        if self.posterior.n_target != n_target:
            msg = (
                f"'posterior' must have one target per row of 'read' ({n_target}), "
                f"got {self.posterior.n_target}"
            )
            raise ValidationError(msg)

    def mean(
        self, data: Float[Array, "*batch source"]
    ) -> Float[Array, "*batch target"]:
        """𝔼[f(X*) ∣ y] = H 𝒮(y), the same as calling the plan.

        Args:
            data: y, the samples, with the flat source last.

        Returns:
            The posterior mean, with the targets last.
        """
        return self(data)

    def variance(self) -> Float[Array, " target"]:
        """Var[f(x*) ∣ y] = diag Σ*, NaN at invalid rows.

        Returns:
            The posterior variance at each target.
        """
        variance = self.posterior.variance()
        nan = jnp.full((), jnp.nan, variance.dtype)
        return jnp.where(self.valid, variance, nan)

    def covariance(self) -> Float[Array, "target target"]:
        """Σ*, dense (M, M), with NaN rows and columns at invalid rows.

        Returns:
            The posterior covariance of the targets.
        """
        covariance = self.posterior.covariance()
        both_valid = einx.logical_and("row, col -> row col", self.valid, self.valid)
        nan = jnp.full((), jnp.nan, covariance.dtype)
        return jnp.where(both_valid, covariance, nan)

    def sample(
        self, data: Float[Array, "*batch source"], *, key: PRNGKeyArray
    ) -> Float[Array, "*batch target"]:
        """f(X*) ~ 𝒩(H 𝒮(y), Σ*), one independent draw per field in ``data``.

        Args:
            data: y, the samples, with the flat source last.
            key: The PRNG key.

        Returns:
            One draw per leading batch entry, with the targets last;
            ``fill_value`` at invalid rows.
        """
        mean = self(data)
        noise = self.posterior.sample(mean.shape[:-1], key=key)
        # f = 𝔼[f ∣ y] + ε, ε ~ 𝒩(0, Σ*)
        draw = mean + noise
        return einx.where(
            "target, ... target, ... target -> ... target", self.valid, draw, mean
        )
