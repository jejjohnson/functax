"""Kernels k(x, x′) for the global schemes: a protocol and two radial families.

Any callable that takes the targets and the sources and returns their
(M, N) cross-kernel, as an array or as a lineax operator, is a `Kernel`, so a
kernel library hands its kernels to functax without functax importing it.
functax ships only the kernels that are about interpolation rather than about
Gaussian processes: the conditionally positive definite polyharmonic splines,
which need a polynomial tail, and the compactly supported Wendland functions.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import abc
import math
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from functax._src.basis._rbf import wendland_c2, wendland_c4
from functax._src.geometry._metric import _euclidean_distance
from functax._src.interp._config import build_from_config, config_of
from functax.exceptions import ValidationError


__all__ = [
    "Kernel",
    "Polyharmonic",
    "RadialKernel",
    "Wendland",
    "kernel_matrix",
]


@runtime_checkable
class Kernel(Protocol):
    """k(x, x′): the (M, N) cross-kernel of M targets and N sources.

    A plain function, an `eqx.Module` or another library's kernel object
    satisfies it, and it may return a dense array or a lineax operator (a
    ``lineax.MatrixLinearOperator`` Gram, say), which the global schemes
    materialise with ``as_matrix()``. Distances, metrics and length scales are
    the kernel's own business; the schemes only pass coordinates.
    """

    def __call__(
        self,
        targets: Float[Array, "target dim"],
        sources: Float[Array, "source dim"],
    ) -> Float[Array, "target source"] | lx.AbstractLinearOperator:
        """K with Kₖⱼ = k(x*ₖ, xⱼ)."""
        ...


def kernel_matrix(
    kernel: Kernel,
    targets: Float[Array, "target dim"],
    sources: Float[Array, "source dim"],
) -> Float[Array, "target source"]:
    """The dense (M, N) matrix of ``kernel`` at the targets and sources.

    Args:
        kernel: k, any `Kernel`.
        targets: x*, the M target coordinates.
        sources: x, the N source coordinates.

    Returns:
        K with Kₖⱼ = k(x*ₖ, xⱼ); a lineax operator is materialised.

    Raises:
        ValidationError: If the kernel returns anything but an (M, N) array or
            operator.
    """
    value = kernel(targets, sources)
    if isinstance(value, lx.AbstractLinearOperator):
        value = value.as_matrix()
    matrix = jnp.asarray(value)
    expected = (targets.shape[0], sources.shape[0])
    if matrix.shape != expected:
        msg = (
            f"'kernel' must return the {expected} cross-kernel of the targets "
            f"and sources, got shape {matrix.shape}"
        )
        raise ValidationError(msg)
    return matrix


class RadialKernel(eqx.Module):
    """k(x, x′) = φ(‖x − x′‖), Euclidean in the coordinates as given.

    A subclass gives the profile φ and, when it is only conditionally
    positive definite, the lowest polynomial degree its tail needs. Radial
    kernels are configuration, so they round-trip through strict JSON with
    `get_config` and `from_config` inside a scheme's config.
    """

    @abc.abstractmethod
    def profile(self, distance: Float[Array, "*batch"]) -> Float[Array, "*batch"]:
        """φ(r) at the distances r."""

    @property
    def min_degree(self) -> int:
        """The lowest degree of the polynomial tail φ needs, −1 for none."""
        return -1

    def __call__(
        self,
        targets: Float[Array, "target dim"],
        sources: Float[Array, "source dim"],
    ) -> Float[Array, "target source"]:
        """Kₖⱼ = φ(‖x*ₖ − xⱼ‖), with a zero gradient of r at coincident points."""
        # r = ‖x* − x‖, guarded at r = 0
        distance = _euclidean_distance(targets, sources)
        return self.profile(distance)

    def get_config(self) -> dict[str, Any]:
        """The kernel as strict JSON: ``{"class": name, field: value, …}``.

        Examples:
            >>> from functax.interp import Polyharmonic
            >>> Polyharmonic(3).get_config()
            {'class': 'Polyharmonic', 'order': 3}
        """
        return config_of(self)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RadialKernel:
        """The kernel a `get_config` config describes, found by its class name.

        Examples:
            >>> from functax.interp import RadialKernel, Wendland
            >>> kernel = Wendland(4, support=2.0)
            >>> RadialKernel.from_config(kernel.get_config()) == kernel
            True
        """
        return build_from_config(config, root=RadialKernel, expected=cls)


def _conditional_order(order: int) -> int:
    """m, the order of conditional positive definiteness of r^k (k odd) or r^k log r."""
    if order % 2 == 1:
        return (order + 1) // 2
    return order // 2 + 1


class Polyharmonic(RadialKernel):
    """The polyharmonic spline φ(r) = ±rᵏ for odd k, ±rᵏ log r for even k.

    It is conditionally positive definite of order m = ⌈k/2⌉ (odd k) or
    k/2 + 1 (even k), so the interpolant needs a polynomial tail of degree at
    least m − 1, and the sign (−1)ᵐ makes the kernel positive on the
    polynomials' null space. These are SciPy's ``RBFInterpolator`` kernels:
    k = 1 is ``"linear"`` (−r), 2 ``"thin_plate_spline"`` (r² log r), 3
    ``"cubic"`` (r³) and 5 ``"quintic"`` (−r⁵). The thin-plate kernel is
    guarded at r = 0 by the double ``where``, so its value there is 0 and its
    gradient finite.

    Args:
        order: k ≥ 1; 2 is the thin-plate spline, which minimises the bending
            energy in 2-D.

    Raises:
        ValidationError: If ``order`` is not a positive integer.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.interp import Polyharmonic
        >>> Polyharmonic(3).profile(jnp.array([0.0, 2.0])).tolist()
        [0.0, 8.0]
        >>> # r² log r is 0 at r = 0 and at r = 1
        >>> Polyharmonic(2).profile(jnp.array([0.0, 1.0])).tolist()
        [0.0, 0.0]
        >>> Polyharmonic(2).min_degree, Polyharmonic(5).min_degree
        (1, 2)
    """

    order: int = eqx.field(static=True, default=2)

    def __check_init__(self) -> None:
        order = self.order
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            msg = f"'order' must be a positive integer, got {order!r}"
            raise ValidationError(msg)

    @property
    def min_degree(self) -> int:
        """m − 1, the lowest degree of the polynomial tail."""
        return _conditional_order(self.order) - 1

    def profile(self, distance: Float[Array, "*batch"]) -> Float[Array, "*batch"]:
        """φ(r) = (−1)ᵐ rᵏ, or (−1)ᵐ rᵏ log r for even k."""
        sign = (-1.0) ** _conditional_order(self.order)
        power = distance**self.order
        if self.order % 2 == 1:
            return sign * power
        # the double where: log never sees r = 0, so the gradient stays finite
        positive = distance > 0
        safe_distance = jnp.where(positive, distance, jnp.ones((), distance.dtype))
        logarithm = jnp.log(safe_distance)
        # φ(r) = (−1)ᵐ rᵏ log r, and φ(0) = 0
        value = sign * safe_distance**self.order * logarithm
        return jnp.where(positive, value, jnp.zeros((), distance.dtype))


class Wendland(RadialKernel):
    """The compact Wendland kernel φ(r) = ψ(r/ρ), zero for r ≥ ρ.

    ψ is `wendland_c2`, (1 − s)₊⁴ (4s + 1), for ``smoothness=2``, and
    `wendland_c4`, (1 − s)₊⁶ (35s² + 18s + 3)/3, for ``smoothness=4``. Both are
    positive definite in up to three dimensions, so the interpolant needs no
    polynomial tail, and K is sparse when ρ is small (the plan still stores
    it dense).

    Args:
        smoothness: 2 (C²) or 4 (C⁴).
        support: ρ > 0, the support radius, in coordinate units.

    Raises:
        ValidationError: If ``smoothness`` is not 2 or 4, or a concrete
            ``support`` is not positive and finite.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.interp import Wendland
        >>> kernel = Wendland(2, support=2.0)
        >>> kernel.profile(jnp.array([0.0, 1.0, 3.0])).tolist()
        [1.0, 0.1875, 0.0]
        >>> kernel(jnp.zeros((1, 2)), jnp.array([[0.0, 0.0], [0.0, 2.0]])).tolist()
        [[1.0, 0.0]]
    """

    smoothness: int = eqx.field(static=True, default=2)
    support: float = 1.0

    def __check_init__(self) -> None:
        if self.smoothness not in (2, 4) or isinstance(self.smoothness, bool):
            msg = f"'smoothness' must be 2 or 4, got {self.smoothness!r}"
            raise ValidationError(msg)
        support = self.support
        # a traced or array support is checked by the caller
        concrete = isinstance(support, int | float) and not isinstance(support, bool)
        if concrete and not 0 < support < math.inf:
            msg = f"'support' must be positive and finite, got {support!r}"
            raise ValidationError(msg)

    def profile(self, distance: Float[Array, "*batch"]) -> Float[Array, "*batch"]:
        """φ(r) = ψ(r/ρ)."""
        # s = r/ρ
        scaled = distance / self.support
        if self.smoothness == 2:
            return wendland_c2(scaled)
        return wendland_c4(scaled)
