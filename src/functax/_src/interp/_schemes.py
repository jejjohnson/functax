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

from functax._src.geometry._types import AffineGrid, RectilinearGrid
from functax._src.interp._bases import (
    Basis,
    BSplineBasis,
    CubicBasis,
    LinearBasis,
    NearestBasis,
)
from functax._src.interp._derivative import LocalGeometry
from functax._src.interp._fit import Identity
from functax._src.interp._plan import Plan
from functax._src.interp._policies import Policies
from functax._src.interp._prefilter import prefilter_for
from functax._src.interp._tensor import (
    coordinates_to_positions,
    source_layout,
    stencil_from_positions,
    target_coordinates,
)
from functax.exceptions import GeometryError, ValidationError


__all__ = [
    "BSpline",
    "Cubic",
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
            The `Plan`; a grid target keeps its shape as ``target_shape``, so
            the plan cuts into windows of it.

        Raises:
            GeometryError: If the geometries are unsupported or mismatched.
            OutsideDomainError: Under ``outside="raise"`` when a target is
                outside.
        """
        policies = Policies() if policies is None else policies
        shape, periodic = source_layout(source)
        bases = self.bases(len(shape))
        coordinates = target_coordinates(source, target)
        positions = coordinates_to_positions(source, coordinates)
        stencil = stencil_from_positions(
            positions, shape, periodic, bases, policies=policies
        )
        orders = tuple(basis.prefilter_order for basis in bases)
        prefilter = prefilter_for(orders, shape, periodic, policies.boundary)
        fit = Identity() if prefilter is None else prefilter
        local = LocalGeometry(source, coordinates, bases)
        # a grid target's rows are its nodes, row-major, so windows can cut it
        grid_target = isinstance(target, RectilinearGrid | AffineGrid)
        target_shape = target.shape if grid_target else None
        return Plan(fit, stencil, policies, local, target_shape)


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


class Cubic(LocalScheme):
    """Keys cubic convolution on every axis: 4ᴰ taps per target, no prefilter.

    Args:
        a: The kernel parameter; −½ by default (it then reproduces quadratics).

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points
        >>> from functax.interp import Cubic
        >>> plan = Cubic().plan(Axis(jnp.arange(6.0)), Points(jnp.array([[2.5]])))
        >>> plan(jnp.arange(6.0) ** 2).tolist()
        [6.25]
    """

    a: float = eqx.field(static=True, default=-0.5)

    def bases(self, n_axes: int) -> tuple[Basis, ...]:
        return (CubicBasis(self.a),) * n_axes


class BSpline(LocalScheme):
    """A B-spline of order k on every axis: (k + 1)ᴰ taps, after the prefilter.

    From order 2 the plan's fit is the `BandedPrefilter`, c = B⁻¹ y per axis,
    and its adjoint applies 𝒮ᵀ after Hᵀ. The prefilter uses the mirror
    boundary, or the wrap on a periodic axis or under ``boundary="wrap"``; it
    equals ``scipy.ndimage.spline_filter`` there. It reproduces polynomials of
    degree k only away from the ends: the mirror boundary's influence decays by
    |z| per node (|z| ≈ 0.27 for the cubic).

    Args:
        order: k, from 0 to 5 (3 is the cubic B-spline).

    Raises:
        ValidationError: If ``order`` is not an integer in [0, 5].

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points
        >>> from functax.interp import BSpline
        >>> # an interpolating spline: it passes through the samples at the knots
        >>> plan = BSpline(3).plan(
        ...     Axis(jnp.arange(6.0)), Points(jnp.array([[2.0], [3.0]]))
        ... )
        >>> [round(v, 12) for v in plan(jnp.arange(6.0) ** 3).tolist()]
        [8.0, 27.0]
    """

    order: int = eqx.field(static=True, default=3)

    def __check_init__(self) -> None:
        BSplineBasis(self.order)

    def bases(self, n_axes: int) -> tuple[Basis, ...]:
        return (BSplineBasis(self.order),) * n_axes


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
