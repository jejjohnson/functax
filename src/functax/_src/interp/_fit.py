"""Fits 𝒮: samples y to coefficients c, and the adjoint 𝒮ᵀ back.

Local schemes use the identity fit, c = y. The B-spline prefilter of #27 and
the global fits of Wave 3 are other fits with the same two methods.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import abc

import equinox as eqx
from jaxtyping import Array, Float


__all__ = [
    "Fit",
    "Identity",
]


class Fit(eqx.Module):
    """c = 𝒮(y), applied once per field before the read-out."""

    @abc.abstractmethod
    def apply(
        self, data: Float[Array, "*batch source"]
    ) -> Float[Array, "*batch source"]:
        """c = 𝒮(y), over any leading batch dims."""

    @abc.abstractmethod
    def apply_adjoint(
        self, coefficients: Float[Array, "*batch source"]
    ) -> Float[Array, "*batch source"]:
        """𝒮ᵀ g, over any leading batch dims."""

    def data_size(self, n_coefficients: int) -> int:
        """N, the length of the data that ``n_coefficients`` coefficients come from.

        The same for most fits; a Hermite fit returns values and slopes, 2N.
        """
        return n_coefficients


class Identity(Fit):
    """The fit of every local scheme: c = y.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.interp import Identity
        >>> Identity().apply(jnp.array([1.0, 2.0])).tolist()
        [1.0, 2.0]
    """

    def apply(
        self, data: Float[Array, "*batch source"]
    ) -> Float[Array, "*batch source"]:
        return data

    def apply_adjoint(
        self, coefficients: Float[Array, "*batch source"]
    ) -> Float[Array, "*batch source"]:
        return coefficients
