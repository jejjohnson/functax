"""Plans: a scheme frozen for one source and one target, f̂(X*) = H 𝒮(y).

A plan holds a fit 𝒮, a read-out H and the policies. It is built once (the
locate and the weights are paid then) and applied to every field, and its
adjoint 𝒮ᵀ Hᵀ sends residuals back. Building may use the host; applying
traces.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float

from functax._src.interp._derivative import LocalGeometry, derivative_stencil
from functax._src.interp._fit import Fit, Identity
from functax._src.interp._policies import Policies
from functax._src.interp._stencil import Readout, Stencil
from functax.exceptions import PolicyError, ValidationError


__all__ = [
    "DerivativePlan",
    "Plan",
]


class Plan(eqx.Module):
    """f̂(X*) = H 𝒮(y), fitted once, applied to many fields.

    Invalid rows (targets outside the domain under ``outside="nan"``, or with
    no weight left under ``missing="renormalise"``) read ``fill_value``.

    Args:
        fit: 𝒮, from samples to coefficients (`Identity` for local schemes).
        read: H, from coefficients to target values (a `Stencil` here).
        policies: The outside, missing and boundary rules the plan was built
            with; ``missing`` and ``fill_value`` act when it is applied.
        local: The source, target coordinates and bases of a local scheme's
            plan, which `derivative` needs; ``None`` for a plan built by hand.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points
        >>> from functax.interp import Linear
        >>> plan = Linear().plan(
        ...     Axis(jnp.arange(4.0)), Points(jnp.array([[0.5], [2.25], [9.0]]))
        ... )
        >>> plan(jnp.array([0.0, 10.0, 20.0, 30.0])).tolist()
        [5.0, 22.5, nan]
        >>> plan.adjoint(jnp.ones(3)).tolist()
        [0.5, 0.5, 0.75, 0.25]
    """

    fit: Fit
    read: Readout
    policies: Policies = eqx.field(default_factory=Policies)
    local: LocalGeometry | None = None

    @property
    def valid(self) -> Bool[Array, " target"]:
        """Whether each target row is a real read-out (before missing samples)."""
        return self.read.valid

    def _fill(self, data: jax.Array) -> jax.Array:
        """Replace non-finite samples by ``fill_value`` under ``missing="fill"``."""
        fill = jnp.asarray(self.policies.fill_value, dtype=data.dtype)
        return jnp.where(jnp.isfinite(data), data, fill)

    def __call__(
        self, data: Float[Array, "*batch source"]
    ) -> Float[Array, "*batch target"]:
        """f̂(X*) = H 𝒮(y) over any leading batch dims, ``fill_value`` at invalid rows.

        Args:
            data: y, the samples, with the flat source last.

        Returns:
            f̂ at the targets, with the targets last.
        """
        data = jnp.asarray(data)
        if self.policies.missing == "fill":
            data = self._fill(data)
        if self.policies.missing == "renormalise":
            values, supported = self._renormalised(data)
        else:
            coefficients = self.fit.apply(data)
            values = self.read.apply(coefficients)
            supported = jnp.ones(values.shape, dtype=bool)
        valid = einx.logical_and(
            "... target, target -> ... target", supported, self.valid
        )
        fill = jnp.asarray(self.policies.fill_value, dtype=values.dtype)
        return jnp.where(valid, values, fill)

    def _renormalised(self, data: jax.Array) -> tuple[jax.Array, jax.Array]:
        """∑ₛ wₖₛ yᵢ / ∑ₛ wₖₛ over the finite samples only, and where any are left."""
        stencil = self._local_stencil("missing='renormalise'")
        gathered = data[..., stencil.indices]
        finite = jnp.isfinite(gathered)
        zero = jnp.zeros((), data.dtype)
        # drop the taps at missing samples, then divide by the weight left
        weights = einx.where(
            "... target tap, target tap, -> ... target tap",
            finite,
            stencil.weights,
            zero,
        )
        safe = jnp.where(finite, gathered, zero)
        total = einx.sum("... target [tap]", weights)
        weighted = einx.dot(
            "... target tap, ... target tap -> ... target", safe, weights
        )
        supported = total != 0
        safe_total = jnp.where(supported, total, jnp.ones((), total.dtype))
        return weighted / safe_total, supported

    def _local_stencil(self, what: str) -> Stencil:
        """The plan's stencil, for policies defined only on local read-outs."""
        if not isinstance(self.fit, Identity) or not isinstance(self.read, Stencil):
            msg = f"{what} needs a local plan (identity fit, stencil read-out)"
            raise PolicyError(msg)
        return self.read

    def adjoint(
        self, residual: Float[Array, "*batch target"]
    ) -> Float[Array, "*batch source"]:
        """g = 𝒮ᵀ Hᵀ r over any leading batch dims; invalid rows contribute nothing.

        Residuals at invalid rows are zeroed first, so a NaN written there by
        the forward pass cannot spread through the scatter.

        Args:
            residual: r at the targets, with the targets last.

        Returns:
            g on the flat source.
        """
        residual = jnp.asarray(residual)
        zero = jnp.zeros((), residual.dtype)
        residual = einx.where(
            "target, ... target, -> ... target", self.valid, residual, zero
        )
        return self.fit.apply_adjoint(self.read.apply_adjoint(residual))

    def operator(self) -> lx.FunctionLinearOperator:
        """H 𝒮 as one (M, N) lineax operator, with zero rows where invalid.

        The fill value is not part of the linear map, so ``operator().mv(y)``
        is 0 where `__call__` writes ``fill_value``; its transpose is `adjoint`.

        Returns:
            A ``lineax.FunctionLinearOperator`` from ℝᴺ to ℝᴹ.
        """
        structure = self.read.operator().in_structure()

        def apply(data: jax.Array) -> jax.Array:
            return self.read.apply(self.fit.apply(data))

        return lx.FunctionLinearOperator(apply, structure)

    def linearize(
        self, data: Float[Array, " source"] | None = None
    ) -> lx.FunctionLinearOperator:
        """The linear map of the plan at ``data``: `operator` for a linear plan.

        Under ``missing="renormalise"`` the weights depend on which samples are
        finite, so ``data`` is needed and the result is the stencil of that
        mask, renormalised row by row.

        Raises:
            PolicyError: Under ``missing="renormalise"`` without ``data``.
        """
        if self.policies.missing != "renormalise":
            return self.operator()
        if data is None:
            msg = "linearize() under missing='renormalise' needs the data"
            raise PolicyError(msg)
        stencil = self._local_stencil("missing='renormalise'")
        finite = jnp.isfinite(jnp.asarray(data))[stencil.indices]
        zero = jnp.zeros((), stencil.weights.dtype)
        weights = jnp.where(finite, stencil.weights, zero)
        total = einx.sum("target [tap]", weights)
        supported = total != 0
        safe_total = jnp.where(supported, total, jnp.ones((), total.dtype))
        normalised = einx.divide(
            "target tap, target -> target tap", weights, safe_total
        )
        valid = stencil.valid & supported
        rows = Stencil(
            stencil.indices, normalised, n_source=stencil.n_source, valid=valid
        )
        return rows.operator()

    def derivative(self, axis: int, order: int = 1) -> DerivativePlan:
        """The plan of ∂ᵐf̂/∂x_cᵐ: H′ with H′ₖⱼ = ∂ᵐφⱼ/∂x_cᵐ at target k.

        c is a target coordinate (a column of ``Points.xy``: per source axis
        on an `Axis` or a `RectilinearGrid`, (x, y) on an `AffineGrid`). The
        result has the plan's fit, taps and validity, and its own adjoint.

        Args:
            axis: c; negative counts from the end.
            order: m ≥ 1.

        Returns:
            The `DerivativePlan`.

        Raises:
            ValidationError: If the plan has no local geometry (built by hand),
                ``axis`` is out of range, or ``order`` is not positive.
            PolicyError: Under ``missing="renormalise"``, whose division by the
                row's weight has no derivative counterpart.

        Examples:
            >>> import jax.numpy as jnp
            >>> from functax.geometry import Axis, Points
            >>> from functax.interp import Cubic
            >>> axis = Axis(jnp.linspace(0.0, 2.5, 6))
            >>> plan = Cubic().plan(axis, Points(jnp.array([[1.25]])))
            >>> # d/dx of x² is 2x
            >>> [round(v, 12) for v in plan.derivative(axis=0)(axis.values**2).tolist()]
            [2.5]
        """
        if self.local is None or not isinstance(self.read, Stencil):
            msg = "derivative() needs a plan built by a local scheme"
            raise ValidationError(msg)
        if self.policies.missing == "renormalise":
            msg = "derivative() is not defined under missing='renormalise'"
            raise PolicyError(msg)
        read = derivative_stencil(
            self.local, self.read, self.policies, axis=axis, order=order
        )
        return DerivativePlan(
            self.fit, read, self.policies, None, axis=axis, order=order
        )

    def __add__(self, other: Plan) -> Plan:
        """H₁ + H₂, as one stencil holding both plans' taps.

        Both plans must share the fit and the policies (as two derivatives of
        one plan do); a row is valid when it is valid in both.

        Raises:
            ValidationError: If either read-out is not a stencil, or the fits,
                policies or source sizes differ.
        """
        if not isinstance(other, Plan):
            return NotImplemented
        if not isinstance(self.read, Stencil) or not isinstance(other.read, Stencil):
            msg = "plans add when both read-outs are stencils"
            raise ValidationError(msg)
        if self.read.n_source != other.read.n_source:
            msg = (
                f"plans add on one source, got {self.read.n_source} and "
                f"{other.read.n_source} source values"
            )
            raise ValidationError(msg)
        same_structure = jax.tree.structure(self.fit) == jax.tree.structure(other.fit)
        if not same_structure or self.policies != other.policies:
            msg = "plans add when they share the fit and the policies"
            raise ValidationError(msg)
        indices = einx.id(
            "target a, target b -> target (a + b)",
            self.read.indices,
            other.read.indices,
        )
        weights = einx.id(
            "target a, target b -> target (a + b)",
            self.read.weights,
            other.read.weights,
        )
        valid = self.read.valid & other.read.valid
        zero = jnp.zeros((), weights.dtype)
        weights = einx.where("target, target tap, -> target tap", valid, weights, zero)
        read = Stencil(indices, weights, n_source=self.read.n_source, valid=valid)
        return Plan(self.fit, read, self.policies)


class DerivativePlan(Plan):
    """A plan whose read-out holds basis derivatives: ∂ᵐf̂/∂x_cᵐ at the targets.

    It is linear in the data, with its own `adjoint` and `operator`, and
    passes the dot-product test like any plan. Made by `Plan.derivative`.
    """

    axis: int = eqx.field(static=True, kw_only=True)
    order: int = eqx.field(static=True, kw_only=True)
