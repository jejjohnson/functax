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
import math
from typing import Any

import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float

from functax._src.geometry._types import AffineGrid, Axis, RectilinearGrid
from functax._src.interp._bases import (
    Basis,
    BSplineBasis,
    CubicBasis,
    LinearBasis,
    NearestBasis,
)
from functax._src.interp._config import build_from_config, config_of
from functax._src.interp._covariance import BrownianBridge
from functax._src.interp._derivative import LocalGeometry
from functax._src.interp._fit import Identity
from functax._src.interp._gaussian import GaussianPlan
from functax._src.interp._interpolant import Interpolant, fit_interpolant
from functax._src.interp._kernels import RadialKernel
from functax._src.interp._locate import require_no_repeats
from functax._src.interp._plan import Plan
from functax._src.interp._policies import Policies
from functax._src.interp._prefilter import prefilter_for
from functax._src.interp._tensor import (
    check_local_source,
    coordinates_to_positions,
    descending_axes,
    source_layout,
    stencil_from_positions,
    target_coordinates,
    target_positions,
)
from functax.exceptions import GeometryError, PolicyError, ValidationError


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
    """An interpolation scheme: configuration, frozen into a `Plan` by `plan`.

    A scheme holds static fields only, so it round-trips through strict JSON
    with `get_config` and `from_config`. A new scheme needs no registration:
    `from_config` finds it by class name among the loaded subclasses. A
    nested `RadialKernel` or lineax solver (an `RBF`'s kernel and solver) is
    found the same way among its own kind.
    """

    @abc.abstractmethod
    def plan(
        self, source: Any, target: Any, *, policies: Policies | None = None
    ) -> Plan:
        """The plan that reads ``source`` data at ``target``."""

    def get_config(self) -> dict[str, Any]:
        """The scheme as strict JSON: ``{"class": name, field: value, …}``.

        Nested schemes become their own configs, tuples become lists, and a
        non-finite float becomes ``{"float": "nan"}`` (or ``"inf"``,
        ``"-inf"``), so ``json.dumps(config, allow_nan=False)`` succeeds.

        Returns:
            A dict of str, int, float, bool, None, lists and dicts.

        Raises:
            ValidationError: If a field has no strict-JSON form.

        Examples:
            >>> from functax.interp import Linear, Nearest, Tensor
            >>> Linear().get_config()
            {'class': 'Linear', 'prior_variance_rate': None}
            >>> Tensor((Nearest(), Linear())).get_config()["schemes"]
            [{'class': 'Nearest'}, {'class': 'Linear', 'prior_variance_rate': None}]
        """
        return config_of(self)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Scheme:
        """The scheme a `get_config` config describes, found by its class name.

        Args:
            config: A config, typically ``json.loads`` of a stored one.

        Returns:
            The scheme, equal to the one the config came from.

        Raises:
            ValidationError: If the class is not a loaded scheme (or not a
                ``cls``), its name is ambiguous, or a key is unknown or
                missing.

        Examples:
            >>> import json
            >>> from functax.interp import Linear, Nearest, Scheme, Tensor
            >>> scheme = Tensor((Nearest(), Linear(), Linear()))
            >>> text = json.dumps(scheme.get_config(), allow_nan=False)
            >>> Scheme.from_config(json.loads(text)) == scheme
            True
        """
        return build_from_config(
            config,
            root=Scheme,
            expected=cls,
            nested=(RadialKernel, lx.AbstractLinearSolver),
        )


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
            source: An `Axis`, a `RectilinearGrid`, an `AffineGrid` or a
                `CurvilinearGrid` (located by `locate_curvilinear`); data on
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
        check_local_source(source, bases, policies)
        coordinates = target_coordinates(source, target)
        extrapolate = policies.outside == "extrapolate"
        positions = coordinates_to_positions(
            source, coordinates, extrapolate=extrapolate
        )
        stencil = stencil_from_positions(
            positions,
            shape,
            periodic,
            bases,
            policies=policies,
            descending=descending_axes(source),
        )
        orders = tuple(basis.prefilter_order for basis in bases)
        prefilter = prefilter_for(orders, shape, periodic, policies.boundary)
        fit = Identity() if prefilter is None else prefilter
        local = LocalGeometry(source, coordinates, bases)
        # a grid target's rows are its nodes, row-major, so windows can cut it
        grid_target = isinstance(target, RectilinearGrid | AffineGrid)
        target_shape = target.shape if grid_target else None
        return Plan(fit, stencil, policies, local, target_shape)

    def fit(
        self,
        source: Any,
        data: Float[Array, "*batch source"],
        *,
        policies: Policies | None = None,
    ) -> Interpolant:
        """c = 𝒮(y) on ``source``, as an `Interpolant` read at targets chosen later.

        The interpolant equals ``self.plan(source, targets)(data)`` at any
        targets, and is read inside ``jit``, ``vmap`` and ``lax.scan`` without
        a rebuild.

        Args:
            source: An `Axis`, a `RectilinearGrid` or an `AffineGrid`.
            data: y, with the flat source last and any leading batch dims.
            policies: The outside, missing and boundary rules.

        Returns:
            The `Interpolant`.

        Raises:
            ValidationError: If the last axis of ``data`` is not the source size.
            PolicyError: Under ``missing="renormalise"`` with a prefilter.
        """
        policies = Policies() if policies is None else policies
        return fit_interpolant(self, source, data, policies)


class Nearest(LocalScheme):
    """Nearest neighbour on every axis: one tap per target, weight 1.

    A tie goes to the lower node, as in SciPy's ``RegularGridInterpolator``.
    The gradient with respect to the target positions is 0. On a
    `CurvilinearGrid` it rounds the index position (r + s, c + t), the
    nearest corner in the cell's own (s, t), which on a warped cell need not
    be the nearest node in distance.

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

    On one axis, linear interpolation is the posterior mean of Brownian motion
    with variance rate σ² observed exactly at the knots. With
    ``prior_variance_rate=σ²`` the plan is a `GaussianPlan` whose covariance
    is the Brownian bridge of each cell (`BrownianBridge`):

        Var[f(x*) ∣ y] = σ² Δᵢ t (1 − t),

    for a target at offset t in cell i of width Δᵢ.

    Args:
        prior_variance_rate: σ², the variance rate of the Brownian-motion
            prior (squared field units per coordinate unit), or ``None`` (the
            default) for a plain `Plan`. A float needs a 1-D `Axis` source and
            ``outside`` ``"nan"`` or ``"raise"``, ``extent="nodes"`` and no
            ``missing="renormalise"``, where the plan's mean is the bridge's.

    Raises:
        ValidationError: If ``prior_variance_rate`` is not ``None`` or a
            positive, finite number.
        GeometryError: When planned with a variance rate, if the source is not
            an `Axis`.
        PolicyError: When planned with a variance rate, under a policy whose
            mean is not the Brownian-bridge mean.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import AffineGrid, Axis, Points
        >>> from functax.interp import Linear
        >>> grid = AffineGrid([1.0, 0.0, 0.0, 0.0, -1.0, 4.0], (4, 4))
        >>> plan = Linear().plan(grid, Points(jnp.array([[0.5, 3.5], [1.0, 3.0]])))
        >>> plan(jnp.arange(16.0)).tolist()
        [0.0, 2.5]
        >>> # σ² Δ t (1 − t) on cells of width 2
        >>> gaussian = Linear(prior_variance_rate=1.0).plan(
        ...     Axis(jnp.array([0.0, 2.0, 4.0])), Points(jnp.array([[1.0], [3.5]]))
        ... )
        >>> gaussian.variance().tolist()
        [0.5, 0.375]
    """

    # float | int rather than float, so an int reaches the check below
    prior_variance_rate: float | int | None = eqx.field(static=True, default=None)

    def __check_init__(self) -> None:
        rate = self.prior_variance_rate
        if rate is None:
            return
        is_number = isinstance(rate, int | float) and not isinstance(rate, bool)
        if not is_number or not math.isfinite(rate) or rate <= 0:
            msg = (
                "'prior_variance_rate' must be None or a positive, finite number, "
                f"got {rate!r}"
            )
            raise ValidationError(msg)

    def bases(self, n_axes: int) -> tuple[Basis, ...]:
        return (LinearBasis(),) * n_axes

    def plan(
        self, source: Any, target: Any, *, policies: Policies | None = None
    ) -> Plan:
        """H for ``source`` read at ``target``; a `GaussianPlan` with a variance rate.

        Args:
            source: An `Axis`, a `RectilinearGrid` or an `AffineGrid` (only an
                `Axis` with ``prior_variance_rate``).
            target: `Points`, or a grid read at its nodes.
            policies: The outside, missing and boundary rules.

        Returns:
            The `Plan`, or a `GaussianPlan` when ``prior_variance_rate`` is set.

        Raises:
            GeometryError: If the geometries are unsupported or mismatched, or
                the source is not an `Axis` under a variance rate.
            PolicyError: Under a variance rate, if the policies change the mean
                away from the Brownian-bridge mean.
            OutsideDomainError: Under ``outside="raise"`` when a target is
                outside.
        """
        if self.prior_variance_rate is None:
            return super().plan(source, target, policies=policies)
        policies = Policies() if policies is None else policies
        _check_bridge(source, policies)
        require_no_repeats(source, "Linear(prior_variance_rate=...)")
        # the ordinary plan first, so the Gaussian one keeps its geometry
        # (derivative) and target shape (windows)
        plan = super().plan(source, target, policies=policies)
        positions = target_positions(source, target)
        posterior = BrownianBridge.from_positions(
            source, positions[:, 0], rate=self.prior_variance_rate
        )
        return GaussianPlan(
            plan.fit,
            plan.read,
            plan.policies,
            plan.local,
            plan.target_shape,
            posterior=posterior,
        )


def _check_bridge(source: Any, policies: Policies) -> None:
    """Raise unless the plan's mean is the Brownian-bridge mean of one axis."""
    if not isinstance(source, Axis):
        msg = (
            "'prior_variance_rate' needs a 1-D Axis source (a D-dimensional "
            f"prior is not available), got {type(source).__name__}"
        )
        raise GeometryError(msg)
    # clamp, extrapolate and the half-cell border read means the bridge does not
    if policies.outside not in ("nan", "raise"):
        msg = (
            "'prior_variance_rate' needs outside='nan' or 'raise', got "
            f"outside={policies.outside!r}"
        )
        raise PolicyError(msg)
    if policies.extent != "nodes":
        msg = (
            "'prior_variance_rate' needs extent='nodes', got "
            f"extent={policies.extent!r}"
        )
        raise PolicyError(msg)
    # renormalising drops a knot, so the bridge would span two cells
    if policies.missing == "renormalise":
        msg = "'prior_variance_rate' does not support missing='renormalise'"
        raise PolicyError(msg)


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
            # Tensor reads only the bases, so a variance rate would be dropped
            if isinstance(scheme, Linear) and scheme.prior_variance_rate is not None:
                msg = (
                    f"'schemes[{index}]' must not set prior_variance_rate; "
                    "Tensor builds a plain Plan"
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
