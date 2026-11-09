"""Schemes: configuration only, frozen into a `Plan` for one source and one target.

A local scheme reads each axis with a one-dimensional basis, and the tensor
product of those bases is its stencil, so mixed schemes come free:
``Tensor((Nearest(), Linear(), Linear()))`` is nearest in time and bilinear
in space, one stencil.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import abc
from typing import Any

import equinox as eqx

from functax._src.interp._bases import Basis, LinearBasis, NearestBasis
from functax._src.interp._fit import Identity
from functax._src.interp._plan import Plan
from functax._src.interp._policies import Policies
from functax._src.interp._tensor import source_layout, tensor_stencil
from functax.exceptions import GeometryError, ValidationError


__all__ = [
    "Linear",
    "LocalScheme",
    "Nearest",
    "Scheme",
    "Tensor",
]


class Scheme(eqx.Module):
    """An interpolation scheme: configuration, frozen into a `Plan` by `plan`."""

    @abc.abstractmethod
    def plan(
        self, source: Any, target: Any, *, policies: Policies | None = None
    ) -> Plan:
        """The plan that reads ``source`` data at ``target``."""


class LocalScheme(Scheme):
    """A scheme with one local basis per axis and the identity fit."""

    @abc.abstractmethod
    def bases(self, n_axes: int) -> tuple[Basis, ...]:
        """One `Basis` per source axis."""

    def plan(
        self, source: Any, target: Any, *, policies: Policies | None = None
    ) -> Plan:
        """H for ``source`` read at ``target``, with ``Policies()`` by default.

        Args:
            source: An `Axis`, a `RectilinearGrid` or an `AffineGrid`; data on
                it is flattened row-major (``einx.id("... y x -> ... (y x)")``).
            target: `Points`, or a grid read at its nodes (see
                `tensor_stencil`).
            policies: The outside, missing and boundary rules.

        Returns:
            The `Plan`.

        Raises:
            GeometryError: If the geometries are unsupported or mismatched.
            OutsideDomainError: Under ``outside="raise"`` when a target is
                outside.
        """
        policies = Policies() if policies is None else policies
        shape, _ = source_layout(source)
        stencil = tensor_stencil(
            source, target, self.bases(len(shape)), policies=policies
        )
        return Plan(Identity(), stencil, policies)


class Nearest(LocalScheme):
    """Nearest neighbour on every axis: one tap per target, weight 1.

    A tie goes to the lower node, as in SciPy's ``RegularGridInterpolator``.
    The gradient with respect to the target positions is 0.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points
        >>> from functax.interp import Nearest
        >>> plan = Nearest().plan(
        ...     Axis(jnp.arange(3.0)), Points(jnp.array([[0.4], [0.5], [1.6]]))
        ... )
        >>> plan(jnp.array([10.0, 20.0, 30.0])).tolist()
        [10.0, 10.0, 30.0]
    """

    def bases(self, n_axes: int) -> tuple[Basis, ...]:
        return (NearestBasis(),) * n_axes


class Linear(LocalScheme):
    """Linear on every axis: 2ᴰ taps per target, weights ∏_d (1 − t_d or t_d).

    Bilinear on a 2-D grid, trilinear on 3-D. It reproduces any affine
    function inside the domain, and its gradient in the target positions is
    piecewise constant.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import AffineGrid, Points
        >>> from functax.interp import Linear
        >>> grid = AffineGrid([1.0, 0.0, 0.0, 0.0, -1.0, 4.0], (4, 4))
        >>> plan = Linear().plan(grid, Points(jnp.array([[0.5, 3.5], [1.0, 3.0]])))
        >>> plan(jnp.arange(16.0)).tolist()
        [0.0, 2.5]
    """

    def bases(self, n_axes: int) -> tuple[Basis, ...]:
        return (LinearBasis(),) * n_axes


class Tensor(LocalScheme):
    """One local scheme per source axis, as one tensor stencil.

    Args:
        schemes: A `Nearest` or `Linear` (or any one-axis `LocalScheme`) per
            source axis, in the source's axis order.

    Raises:
        ValidationError: If ``schemes`` is empty or holds a non-local scheme.
        GeometryError: When planned, if the number of schemes differs from the
            number of source axes.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points, RectilinearGrid
        >>> from functax.interp import Linear, Nearest, Tensor
        >>> grid = RectilinearGrid((Axis(jnp.arange(2.0)), Axis(jnp.arange(3.0))))
        >>> plan = Tensor((Nearest(), Linear())).plan(
        ...     grid, Points(jnp.array([[0.8, 0.5]]))
        ... )
        >>> plan(jnp.arange(6.0)).tolist()
        [3.5]
    """

    schemes: tuple[LocalScheme, ...] = eqx.field(converter=tuple)

    def __check_init__(self) -> None:
        if not self.schemes:
            msg = "'schemes' must hold one scheme per axis, got ()"
            raise ValidationError(msg)
        for index, scheme in enumerate(self.schemes):
            if not isinstance(scheme, LocalScheme) or isinstance(scheme, Tensor):
                msg = (
                    f"'schemes[{index}]' must be a one-axis local scheme such as "
                    f"Nearest or Linear, got {type(scheme).__name__}"
                )
                raise ValidationError(msg)

    def bases(self, n_axes: int) -> tuple[Basis, ...]:
        if n_axes != len(self.schemes):
            msg = (
                f"Tensor has {len(self.schemes)} schemes but the source has "
                f"{n_axes} axes"
            )
            raise GeometryError(msg)
        return tuple(scheme.bases(1)[0] for scheme in self.schemes)
