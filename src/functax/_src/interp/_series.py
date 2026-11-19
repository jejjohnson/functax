"""The 1-D series schemes: cubic splines, PCHIP, Akima and Steffen on one axis.

All four read with the cubic Hermite stencil of `hermite_stencil` and differ
in their slopes. The cubic spline's slopes are linear in y, so it builds an
ordinary `Plan` with an adjoint. The shape-preserving schemes' are not, so
they build a `NonlinearPlan`, whose ``linearize(y)`` gives the tangent-linear
stencil at y.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import abc
import dataclasses
from typing import Any, ClassVar

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from functax._src.geometry._types import AffineGrid, Axis, Points, RectilinearGrid
from functax._src.interp._hermite import (
    HermiteFit,
    HermiteGeometry,
    Slopes,
    ascending_axis,
    hermite_stencil,
)
from functax._src.interp._locate import require_no_repeats
from functax._src.interp._monotone import akima_slopes, pchip_slopes, steffen_slopes
from functax._src.interp._pieces import (
    PIECEWISE_SPLINE_SLOPES,
    piecewise_akima_slopes,
    piecewise_pchip_slopes,
    piecewise_steffen_slopes,
)
from functax._src.interp._plan import DerivativePlan, Plan
from functax._src.interp._policies import Policies
from functax._src.interp._schemes import Scheme
from functax._src.interp._spline1d import (
    MINIMUM_KNOTS,
    SPLINE_CONDITIONS,
    SPLINE_SLOPES,
)
from functax._src.interp._stencil import Stencil
from functax._src.interp._tensor import target_coordinates
from functax.exceptions import GeometryError, NonlinearSchemeError, ValidationError


__all__ = [
    "Akima",
    "CubicSpline",
    "NonlinearDerivativePlan",
    "NonlinearPlan",
    "Pchip",
    "SeriesInterpolant",
    "SeriesScheme",
    "Steffen",
]


class NonlinearPlan(Plan):
    """A plan whose fit depends on y: no fixed operator or adjoint, but `linearize`.

    Calling it reads f̂(X_*) = H 𝒮(y) as any plan does; `operator` and
    `adjoint` raise `NonlinearSchemeError`.

    Args:
        window: (l, r), the knots i − l … i + r that a target in cell i
            depends on.

    Raises:
        NonlinearSchemeError: From `operator` and `adjoint`.

    Examples:
        PCHIP builds one; it reads like any plan:

        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points
        >>> from functax.interp import NonlinearPlan, Pchip
        >>> plan = Pchip().plan(Axis(jnp.arange(5.0)), Points(jnp.array([[1.5]])))
        >>> isinstance(plan, NonlinearPlan)
        True
        >>> data = jnp.array([0.0, 1.0, 3.0, 4.0, 4.5])
        >>> plan(data).shape
        (1,)

        Its tangent-linear stencil at y has the taps i − 1 … i + 2; PCHIP is
        homogeneous of degree 1 in y, so J_f̂ y = f̂(y) with J_f̂ = ∂f̂/∂y (Euler):

        >>> tangent = plan.linearize(data)
        >>> tangent.weights.shape
        (1, 4)
        >>> bool(jnp.allclose(tangent.apply(data), plan(data)))
        True

        There is no fixed operator:

        >>> plan.operator()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        functax.exceptions.NonlinearSchemeError: a nonlinear plan has no ...
    """

    window: tuple[int, int] = eqx.field(static=True, kw_only=True)

    def derivative(self, axis: int, order: int = 1) -> DerivativePlan:
        """∂ᵐf̂/∂xᵐ at the targets, as a `NonlinearDerivativePlan`.

        It reads the closed-form derivative of the Hermite read-out (see
        `Plan.derivative`), and refuses a fixed operator, as this plan does.
        """
        derivative = super().derivative(axis, order)
        return NonlinearDerivativePlan(
            derivative.fit,
            derivative.read,
            derivative.policies,
            None,
            derivative.target_shape,
            axis=axis,
            order=order,
        )

    def operator(self) -> lx.FunctionLinearOperator:
        """Raises `NonlinearSchemeError`: the slopes depend on y."""
        msg = "a nonlinear plan has no fixed operator; use plan.linearize(y)"
        raise NonlinearSchemeError(msg)

    def adjoint(
        self, residual: Float[Array, "*batch target"]
    ) -> Float[Array, "*batch source"]:
        """Raises `NonlinearSchemeError`; ``linearize(y).apply_adjoint`` is it at y."""
        del residual
        msg = "a nonlinear plan has no fixed adjoint; use plan.linearize(y)"
        raise NonlinearSchemeError(msg)

    # a Stencil rather than Plan.linearize's lineax operator, as #34 specifies;
    # ``.operator()`` of the result is the same map as a lineax operator
    def linearize(  # ty: ignore[invalid-method-override]
        self, data: Float[Array, " source"] | None = None
    ) -> Stencil:
        """∂f̂(X_*)/∂y at y, as a stencil with the taps each target depends on.

        Target k in cell i depends only on y at knots i − l … i + r (l = 1,
        r = 2 for PCHIP and Steffen; l = 2, r = 3 for Akima), so row k of the
        Jacobian J_f̂ has those entries and the stencil holds them.

        Args:
            data: y, one vector of N samples.

        Returns:
            The tangent-linear `Stencil` (M, N).

        Raises:
            ValidationError: If ``data`` is missing or not one vector of N.
        """
        if data is None:
            msg = "linearize() of a nonlinear plan needs the data y"
            raise ValidationError(msg)
        data = jnp.asarray(data)
        fit = self.fit
        read = self.read
        assert isinstance(fit, HermiteFit)
        assert isinstance(read, Stencil)
        n_nodes = fit.nodes.shape[0]
        if data.shape != (n_nodes,):
            msg = f"'data' must be one vector of shape ({n_nodes},), got {data.shape}"
            raise ValidationError(msg)
        # J_f̂ = ∂f̂/∂y; its non-zeros sit in each target's window
        jacobian = jax.jacfwd(self.__call__)(data)
        left, right = self.window
        # i, the cell of each target (its first value tap)
        cell = read.indices[:, 0]
        offsets = jnp.arange(-left, right + 1, dtype=cell.dtype)
        # the knots i − l … i + r
        taps = einx.add("target, tap -> target tap", cell, offsets)
        in_range = (taps >= 0) & (taps <= n_nodes - 1)
        # back to the source's own order when its axis decreases
        source_taps = (n_nodes - 1) - taps if fit.reversed else taps
        indices = jnp.clip(source_taps, 0, n_nodes - 1)
        # wₖₛ = J_f̂[k, iₖₛ]
        weights = einx.get_at(
            "target [source], target tap -> target tap", jacobian, indices
        )
        zero = jnp.zeros((), weights.dtype)
        weights = jnp.where(in_range, weights, zero)
        return Stencil(indices, weights, n_source=n_nodes, valid=read.valid)


class NonlinearDerivativePlan(DerivativePlan):
    """∂ᵐf̂/∂xᵐ of a nonlinear series plan: it reads, but has no fixed operator.

    Made by ``NonlinearPlan.derivative``; the slopes still depend on y, so
    `operator` and `adjoint` raise `NonlinearSchemeError`, and ``jax.vjp`` of
    the call is its adjoint at y.

    Raises:
        NonlinearSchemeError: From `operator` and `adjoint`.

    Examples:
        The slope of a PCHIP field, read at the targets:

        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points
        >>> from functax.interp import NonlinearDerivativePlan, Pchip
        >>> plan = Pchip().plan(Axis(jnp.arange(4.0)), Points(jnp.array([[1.5]])))
        >>> derivative = plan.derivative(0)
        >>> isinstance(derivative, NonlinearDerivativePlan)
        True
        >>> bool(jnp.allclose(derivative(2.0 * jnp.arange(4.0)), 2.0))
        True

        Its adjoint at y comes from ``jax.vjp``, not from the plan:

        >>> derivative.adjoint(jnp.ones(1))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        functax.exceptions.NonlinearSchemeError: a nonlinear plan has no ...
    """

    def operator(self) -> lx.FunctionLinearOperator:
        """Raises `NonlinearSchemeError`: the slopes depend on y."""
        msg = "a nonlinear plan has no fixed operator; differentiate its call in y"
        raise NonlinearSchemeError(msg)

    def adjoint(
        self, residual: Float[Array, "*batch target"]
    ) -> Float[Array, "*batch source"]:
        """Raises `NonlinearSchemeError`; ``jax.vjp`` of the call is it at y."""
        del residual
        msg = "a nonlinear plan has no fixed adjoint; use jax.vjp of its call at y"
        raise NonlinearSchemeError(msg)


class SeriesInterpolant(eqx.Module):
    """A 1-D series field f̂ read at targets chosen inside the trace.

    Made by ``scheme.fit(axis, y)``; the fit runs in JAX, so gradients reach
    both y and the knot positions. It equals ``scheme.plan(axis, targets)(y)``.
    `derivative` gives the field ∂ᵐf̂/∂xᵐ, the derivative of the cubic Hermite
    read-out in closed form.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis
        >>> from functax.interp import Pchip
        >>> field = Pchip().fit(
        ...     Axis(jnp.array([0.0, 1.0, 2.0, 4.0])),
        ...     jnp.array([0.0, 1.0, 1.0, 3.0]),
        ... )
        >>> values = field(jnp.array([[1.5], [3.0]]))
        >>> bool(jnp.allclose(values, jnp.array([1.0, 1.5833333333333333])))
        True
        >>> # f̂′ at a knot is the knot's slope; PCHIP's is 0 at a flat step
        >>> slopes = field.derivative()(jnp.array([[1.0], [2.0]]))
        >>> bool(jnp.allclose(slopes, 0.0))
        True
    """

    data: Float[Array, "*batch source"]
    source: Axis
    scheme: SeriesScheme
    policies: Policies
    derivative_order: int = eqx.field(static=True, default=0)

    def __call__(
        self, targets: Float[Array, "target 1"] | Points
    ) -> Float[Array, "*batch target"]:
        """f̂ at the targets, ``fill_value`` where they are invalid."""
        if isinstance(targets, Points):
            points = targets
        else:
            points = Points(jnp.asarray(targets))
        return self.plan(points)(self.data)

    def plan(self, target: Any) -> Plan:
        """The plan that ``scheme.plan(axis, target)`` builds, or its derivative."""
        plan = self.scheme.plan(self.source, target, policies=self.policies)
        if self.derivative_order == 0:
            return plan
        return plan.derivative(0, self.derivative_order)

    def derivative(self, axis: int = 0, order: int = 1) -> SeriesInterpolant:
        """∂ᵐf̂/∂xᵐ as a field, with m = ``order``, read like f̂.

        Args:
            axis: The target coordinate, 0 (or −1), since the axis is 1-D.
            order: m ≥ 1; order 3 is piecewise constant and 4 and above zero.

        Raises:
            ValidationError: If ``axis`` is not 0 or −1, ``order`` is not a
                positive integer, or the field is already a derivative.
        """
        if self.derivative_order:
            msg = "derivative() of a derivative field: raise 'order' instead"
            raise ValidationError(msg)
        if isinstance(axis, bool) or axis not in (0, -1):
            msg = f"'axis' must be 0 (or -1) on a 1-D axis, got {axis!r}"
            raise ValidationError(msg)
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            msg = f"'order' must be a positive integer, got {order!r}"
            raise ValidationError(msg)
        return dataclasses.replace(self, derivative_order=order)


class SeriesScheme(Scheme):
    """A scheme on one axis with a cubic Hermite read-out and its own slopes.

    f̂ = φ₀₀(t) yᵢ + φ₁₀(t) hᵢ dᵢ + φ₀₁(t) yᵢ₊₁ + φ₁₁(t) hᵢ dᵢ₊₁ on cell i of
    width hᵢ, at offset t, with the cubic Hermite basis φ₀₀, φ₁₀, φ₀₁, φ₁₁
    (see `hermite_stencil`) and the slopes d = 𝒮_d(y) that each subclass
    supplies: `_slopes` for increasing knots and `_piecewise_slopes` for knots
    with repeats.

    Examples:
        A new scheme supplies only its slopes; here the centred secants:

        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points
        >>> from functax.interp import SeriesScheme
        >>> def centred_slopes(nodes, values):
        ...     secants = jnp.diff(values) / jnp.diff(nodes)
        ...     inner = 0.5 * (secants[1:] + secants[:-1])
        ...     return jnp.concatenate([secants[:1], inner, secants[-1:]])
        >>> class CentredSecants(SeriesScheme):
        ...     def _slopes(self):
        ...         return centred_slopes
        ...
        ...     def _piecewise_slopes(self):
        ...         return centred_slopes
        >>> axis = Axis(jnp.arange(4.0))
        >>> plan = CentredSecants().plan(axis, Points(jnp.array([[1.5]])))
        >>> bool(jnp.allclose(plan(2.0 * jnp.arange(4.0)), 3.0))
        True
    """

    window: ClassVar[tuple[int, int]] = (1, 2)
    linear: ClassVar[bool] = False

    @abc.abstractmethod
    def _slopes(self) -> Slopes:
        """d = slopes(x, y) for increasing knots."""

    @abc.abstractmethod
    def _piecewise_slopes(self) -> Slopes:
        """d = slopes(x, y) for non-decreasing knots, piece by piece between repeats."""

    def _minimum_knots(self) -> int:
        return 2

    def _periodic(self) -> bool:
        return False

    def plan(
        self, source: Any, target: Any, *, policies: Policies | None = None
    ) -> Plan:
        """The plan that reads data on the ``source`` axis at ``target``.

        On an axis with repeats (``Axis(..., repeats="allow")``), each piece
        between repeats gets the slopes the scheme gives it as an axis of its
        own, with the scheme's end rules at both of its ends, and a read-out
        at a repeat is right-continuous.

        Args:
            source: An `Axis`, increasing or decreasing, not periodic, maybe
                with repeats.
            target: `Points` with one coordinate, or an `Axis`.
            policies: The outside rules; ``boundary`` does not apply.

        Returns:
            A `Plan` for the cubic spline, a `NonlinearPlan` otherwise.

        Raises:
            GeometryError: If ``source`` is not an `Axis`, or a periodic
                cubic spline meets a repeat.
            ValidationError: If the axis has too few knots for the scheme.
        """
        policies = Policies() if policies is None else policies
        if not isinstance(source, Axis):
            msg = f"the 1-D series schemes read an Axis, got {type(source).__name__}"
            raise GeometryError(msg)
        ascending, reversed_axis = ascending_axis(source)
        n_nodes = ascending.values.shape[0]
        minimum = self._minimum_knots()
        if n_nodes < minimum:
            msg = f"{type(self).__name__} needs at least {minimum} knots, got {n_nodes}"
            raise ValidationError(msg)
        target_points = target_coordinates(source, target)
        local = HermiteGeometry(
            axis=ascending, coordinates=target_points, periodic=self._periodic()
        )
        coordinates = target_points[:, 0]
        if self._periodic():
            first = ascending.values[0]
            # P = x_{N−1} − x₀
            period = ascending.values[-1] - first
            # x ↦ x₀ + ((x − x₀) mod P), as SciPy does
            wrapped = jnp.mod(coordinates - first, period)
            coordinates = first + wrapped
        stencil = hermite_stencil(ascending, coordinates, policies)
        if source.repeats == "allow":
            if self._periodic():
                require_no_repeats(source, "CubicSpline(bc='periodic')")
            slopes = self._slopes() if self._periodic() else self._piecewise_slopes()
        else:
            slopes = self._slopes()
        fit = HermiteFit(ascending.values, slopes, reversed_axis, self.linear)
        is_grid = isinstance(target, RectilinearGrid | AffineGrid)
        target_shape = target.shape if is_grid else None
        if self.linear:
            return Plan(fit, stencil, policies, local, target_shape)
        return NonlinearPlan(
            fit, stencil, policies, local, target_shape, window=self.window
        )

    def fit(
        self,
        source: Axis,
        data: Float[Array, "*batch source"],
        *,
        policies: Policies | None = None,
    ) -> SeriesInterpolant:
        """The field on ``source`` from y, read later at targets chosen in the trace.

        Args:
            source: The `Axis` the data lie on.
            data: y, the samples last, with any leading batch dims.
            policies: The outside rules.

        Returns:
            The `SeriesInterpolant`.
        """
        policies = Policies() if policies is None else policies
        return SeriesInterpolant(jnp.asarray(data), source, self, policies)


class CubicSpline(SeriesScheme):
    """The interpolating C² cubic spline, as ``scipy.interpolate.CubicSpline``.

    Args:
        bc: The end condition: ``"not-a-knot"`` (the default), ``"natural"``,
            ``"clamped"`` (zero end slopes) or ``"periodic"`` (y₀ must equal
            y_{N−1}; targets wrap into one period x_{N−1} − x₀, as in SciPy).

    Raises:
        ValidationError: If ``bc`` is unknown.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points
        >>> from functax.interp import CubicSpline
        >>> knots = Axis(jnp.array([0.0, 1.0, 2.5, 3.0, 4.0]))
        >>> plan = CubicSpline().plan(knots, Points(jnp.array([[2.0]])))
        >>> # not-a-knot reproduces cubics: 2³ = 8
        >>> bool(jnp.allclose(plan(knots.values**3), 8.0))
        True

        The spline is linear in y, so its plan has an adjoint:

        >>> plan.adjoint(jnp.ones(1)).shape
        (5,)

        An unknown end condition is rejected:

        >>> CubicSpline("parabolic")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        functax.exceptions.ValidationError: 'bc' must be one of ...
    """

    bc: str = eqx.field(static=True, default="not-a-knot")
    linear: ClassVar[bool] = True

    def __check_init__(self) -> None:
        if self.bc not in SPLINE_CONDITIONS:
            msg = f"'bc' must be one of {list(SPLINE_CONDITIONS)}, got {self.bc!r}"
            raise ValidationError(msg)

    def _slopes(self) -> Slopes:
        return SPLINE_SLOPES[self.bc]

    def _piecewise_slopes(self) -> Slopes:
        return PIECEWISE_SPLINE_SLOPES[self.bc]

    def _minimum_knots(self) -> int:
        return MINIMUM_KNOTS[self.bc]

    def _periodic(self) -> bool:
        return self.bc == "periodic"


class Pchip(SeriesScheme):
    """Monotone piecewise cubic Hermite (Fritsch-Carlson), as SciPy's PCHIP.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points
        >>> from functax.interp import Pchip
        >>> plan = Pchip().plan(Axis(jnp.arange(4.0)), Points(jnp.array([[1.5]])))
        >>> bool(jnp.allclose(plan(jnp.array([0.0, 0.0, 1.0, 1.0])), 0.5))
        True

        On a decreasing axis it reads the same field:

        >>> flipped = Pchip().plan(
        ...     Axis(jnp.arange(4.0)[::-1]), Points(jnp.array([[1.5]]))
        ... )
        >>> bool(jnp.allclose(flipped(jnp.array([1.0, 1.0, 0.0, 0.0])), 0.5))
        True
    """

    def _slopes(self) -> Slopes:
        return pchip_slopes

    def _piecewise_slopes(self) -> Slopes:
        return piecewise_pchip_slopes


class Akima(SeriesScheme):
    """Akima's local cubic, as SciPy's ``Akima1DInterpolator(method="akima")``.

    A target reads y at the knots i − 2 … i + 3, so its tangent-linear stencil
    has 6 taps.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points
        >>> from functax.interp import Akima
        >>> plan = Akima().plan(Axis(jnp.arange(5.0)), Points(jnp.array([[2.5]])))
        >>> bool(jnp.allclose(plan(jnp.arange(5.0) * 2.0), 5.0))
        True

        Its tangent-linear stencil has the six taps i − 2 … i + 3:

        >>> plan.linearize(jnp.arange(5.0) ** 2).weights.shape
        (1, 6)
    """

    window: ClassVar[tuple[int, int]] = (2, 3)

    def _slopes(self) -> Slopes:
        return akima_slopes

    def _piecewise_slopes(self) -> Slopes:
        return piecewise_akima_slopes


class Steffen(SeriesScheme):
    """Steffen's monotone cubic: no overshoot between knots, by construction.

    SciPy has no Steffen interpolator; the tests check it against a direct
    port of Steffen's formulas and for monotonicity.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points
        >>> from functax.interp import Steffen
        >>> plan = Steffen().plan(
        ...     Axis(jnp.arange(4.0)), Points(jnp.array([[0.5], [2.5]]))
        ... )
        >>> values = plan(jnp.array([0.0, 1.0, 1.0, 5.0]))
        >>> bool(jnp.allclose(values, jnp.array([0.6875, 2.25])))
        True

        A line is reproduced, since every slope is then its secant:

        >>> line = plan(jnp.array([0.0, 1.0, 2.0, 3.0]))
        >>> bool(jnp.allclose(line, jnp.array([0.5, 2.5])))
        True
    """

    def _slopes(self) -> Slopes:
        return steffen_slopes

    def _piecewise_slopes(self) -> Slopes:
        return piecewise_steffen_slopes
