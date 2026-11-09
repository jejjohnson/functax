"""Plans: a scheme frozen for one source and one target, f̂(X*) = H 𝒮(y).

A plan holds a fit 𝒮, a read-out H and the policies. It is built once (the
locate and the weights are paid then) and applied to every field, and its
adjoint 𝒮ᵀ Hᵀ sends residuals back. Building may use the host; applying
traces. Two plans compose into one (``outer @ inner``, a `ComposedPlan`), a
plan on a grid target cuts to a window of it (``plan[window]``, a
`WindowedPlan`), and a local plan exports its stencil for gaussx
(`Plan.to_gaussx`).
"""

from __future__ import annotations

from functax._src._backend import is_tracer, require_jax


require_jax(__name__)

import dataclasses
import io
import math
from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from functax._src.interp._compose import compose
from functax._src.interp._derivative import LocalGeometry, derivative_stencil
from functax._src.interp._fit import Fit, Identity
from functax._src.interp._policies import Policies
from functax._src.interp._stencil import Readout, Stencil
from functax._typing import Shape
from functax.exceptions import PolicyError, ValidationError


__all__ = [
    "ComposedPlan",
    "DerivativePlan",
    "Plan",
    "WindowedPlan",
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
        target_shape: The shape of a grid target, whose nodes are the rows in
            row-major order (``einx.id("... y x -> ... (y x)")``); ``None``
            for a flat list of M targets. `__getitem__` cuts windows of it.

    Raises:
        ValidationError: If ``target_shape`` does not hold the plan's M rows.

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
    target_shape: Shape | None = eqx.field(static=True, default=None)

    def __check_init__(self) -> None:
        n_target = self.read.valid.shape[0]
        if self.target_shape is not None and math.prod(self.target_shape) != n_target:
            msg = (
                f"'target_shape' {self.target_shape} must hold the plan's "
                f"{n_target} target rows"
            )
            raise ValidationError(msg)

    @property
    def valid(self) -> Bool[Array, " target"]:
        """Whether each target row is a real read-out (before missing samples)."""
        return self.read.valid

    @property
    def grid_shape(self) -> Shape:
        """The shape of the targets: ``target_shape``, or (M,) for a flat list."""
        if self.target_shape is None:
            return (self.read.valid.shape[0],)
        return self.target_shape

    def __matmul__(self, inner: Plan) -> ComposedPlan:
        """``inner`` then this plan, as one `ComposedPlan`: H_self H_inner 𝒮_inner."""
        return ComposedPlan(self, inner)

    def __getitem__(self, window: slice | tuple[slice, ...]) -> WindowedPlan:
        """The rows of the targets inside ``window``, as a `WindowedPlan`."""
        return WindowedPlan(self, window)

    def to_gaussx(self) -> tuple[Int[Array, "target tap"], Float[Array, "target tap"]]:
        """The stencil arrays (iₖₛ, wₖₛ) in gaussx's row layout, invalid rows zeroed.

        Row k of W has weights wₖₛ at source indices iₖₛ, repeated indices
        adding: the ``(interp_indices, interp_values)`` of gaussx's
        ``InterpolatedOperator`` (W K Wᵀ), and the rows of a ``SparseOperator``.
        Invalid rows get index 0 and weight 0, so W is the linear map of
        `operator`. Nothing from gaussx is imported.

        Returns:
            The (M, S) indices and the (M, S) weights.

        Raises:
            PolicyError: If the plan is not local (identity fit, stencil
                read-out), or its weights depend on the data
                (``missing="renormalise"``).

        Examples:
            >>> import jax.numpy as jnp
            >>> from functax.geometry import Axis, Points
            >>> from functax.interp import Linear
            >>> plan = Linear().plan(
            ...     Axis(jnp.arange(4.0)), Points(jnp.array([[0.5], [9.0]]))
            ... )
            >>> indices, weights = plan.to_gaussx()
            >>> indices.tolist(), weights.tolist()
            ([[0, 1], [0, 0]], [[0.5, 0.5], [0.0, 0.0]])
        """
        stencil = self._local_stencil("to_gaussx()")
        if self.policies.missing == "renormalise":
            msg = (
                "to_gaussx() needs fixed weights, but under missing='renormalise' "
                "they depend on the data"
            )
            raise PolicyError(msg)
        valid = self.valid
        zero_index = jnp.zeros((), stencil.indices.dtype)
        zero_weight = jnp.zeros((), stencil.weights.dtype)
        indices = einx.where(
            "target, target tap, -> target tap", valid, stencil.indices, zero_index
        )
        weights = einx.where(
            "target, target tap, -> target tap", valid, stencil.weights, zero_weight
        )
        return indices, weights

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
        coefficients = self.read.operator().in_structure()
        n_data = self.fit.data_size(coefficients.shape[0])
        structure = jax.ShapeDtypeStruct((n_data,), coefficients.dtype)

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

    def serialize(self) -> bytes:
        """The plan's array leaves as bytes, by ``eqx.tree_serialise_leaves``.

        Only the arrays (the stencil's indices, weights and valid rows) are
        written; the static fields (N, the policies) come back from the
        template that `deserialize` is given. Runs on the host.

        Returns:
            The bytes.

        Raises:
            ValidationError: If a leaf is traced or abstract, as inside ``jit``
                or for a template from ``eqx.filter_eval_shape``.

        Examples:
            >>> import jax.numpy as jnp
            >>> from functax.geometry import Axis, Points
            >>> from functax.interp import Linear
            >>> plan = Linear().plan(Axis(jnp.arange(4.0)), Points(jnp.array([[0.5]])))
            >>> isinstance(plan.serialize(), bytes)
            True
        """
        for leaf in jax.tree_util.tree_leaves(self):
            if is_tracer(leaf) or not isinstance(leaf, jax.Array):
                msg = (
                    "'self' must hold concrete arrays to serialize, got a "
                    f"{type(leaf).__name__} leaf; call serialize outside jit"
                )
                raise ValidationError(msg)
        buffer = io.BytesIO()
        eqx.tree_serialise_leaves(buffer, self)
        return buffer.getvalue()

    @classmethod
    def deserialize(cls, data: bytes, *, like: Plan) -> Plan:
        """The plan whose array leaves `serialize` wrote, in the structure of ``like``.

        ``like`` fixes everything the bytes do not hold: the scheme's static
        fields, N and the policies, and the shape and dtype of every leaf. It
        need not be computed: ``eqx.filter_eval_shape(lambda:
        scheme.plan(source, target))`` traces the build without running it
        and gives a template with abstract leaves. Key the bytes by
        `plan_key` so they are only read into the template of the same plan.

        Args:
            data: Bytes from `serialize`.
            like: A plan, or an abstract template, of the same structure.

        Returns:
            The plan, applying bit-identically to the one serialized.

        Raises:
            ValidationError: If ``data`` is not bytes, ``like`` is not a
                `Plan`, or ``data`` does not match ``like`` (another shape or
                dtype, too few or too many leaves).

        Examples:
            >>> import equinox as eqx
            >>> import jax.numpy as jnp
            >>> from functax.geometry import Axis, Points
            >>> from functax.interp import Linear, Plan
            >>> axis, points = Axis(jnp.arange(4.0)), Points(jnp.array([[0.5]]))
            >>> data = Linear().plan(axis, points).serialize()
            >>> like = eqx.filter_eval_shape(lambda: Linear().plan(axis, points))
            >>> Plan.deserialize(data, like=like)(jnp.arange(4.0)).tolist()
            [0.5]
        """
        if not isinstance(data, bytes):
            msg = f"'data' must be bytes from Plan.serialize, got {type(data).__name__}"
            raise ValidationError(msg)
        if not isinstance(like, Plan):
            msg = f"'like' must be a Plan, got {type(like).__name__}"
            raise ValidationError(msg)
        buffer = io.BytesIO(data)
        try:
            plan: Any = eqx.tree_deserialise_leaves(buffer, like)
        except (RuntimeError, ValueError, EOFError) as err:
            msg = f"'data' does not match 'like': {err}"
            raise ValidationError(msg) from err
        if buffer.read(1):
            msg = "'data' holds more leaves than 'like'"
            raise ValidationError(msg)
        return plan

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
            self.fit,
            read,
            self.policies,
            None,
            self.target_shape,
            axis=axis,
            order=order,
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
        same_shape = self.target_shape == other.target_shape
        target_shape = self.target_shape if same_shape else None
        return Plan(self.fit, read, self.policies, None, target_shape)


class DerivativePlan(Plan):
    """A plan whose read-out holds basis derivatives: ∂ᵐf̂/∂x_cᵐ at the targets.

    It is linear in the data, with its own `adjoint` and `operator`, and
    passes the dot-product test like any plan. Made by `Plan.derivative`.
    """

    axis: int = eqx.field(static=True, kw_only=True)
    order: int = eqx.field(static=True, kw_only=True)


class ComposedPlan(Plan):
    """``outer @ inner``: H_outer H_inner 𝒮_inner, one plan with the product stencil.

    The product stencil is computed when the plan is built (`compose`), so
    applying it costs one gather of S_outer · S_inner taps per row. A row of
    ``outer`` that reads an invalid row of ``inner`` is invalid; on every row
    it marks valid, the composed plan equals ``outer(inner(y))``. It applies
    ``inner``'s missing policy to the samples and writes ``outer``'s fill value
    at invalid rows, and its adjoint is 𝒮ᵀ_inner Hᵀ_inner Hᵀ_outer.

    Args:
        outer: The plan applied second: a local plan (identity fit, stencil
            read-out) with ``missing="propagate"``.
        inner: The plan applied first: a stencil read-out after any fit, with
            ``missing="propagate"``, or ``"fill"`` with ``outer``'s fill value.
        merge: Whether to merge repeated indices of the product on the host
            (see `compose`).

    Raises:
        PolicyError: If the two plans cannot fold into one stencil: ``outer``
            is not local or has another missing policy, or ``inner`` has no
            stencil read-out, uses ``missing="renormalise"``, or fills missing
            samples with a value ``outer`` does not share. Their linear
            maps still compose in lineax, as
            ``outer.operator() @ inner.operator()``.
        ValidationError: If ``outer`` does not read ``inner``'s targets.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points
        >>> from functax.interp import Linear
        >>> # 5 fine samples read at 3 points, then those 3 read at 2 points
        >>> spectral = Linear().plan(
        ...     Axis(jnp.arange(5.0)), Points(jnp.array([[0.5], [2.0], [3.5]]))
        ... )
        >>> srf = Linear().plan(
        ...     Axis(jnp.arange(3.0)), Points(jnp.array([[0.5], [1.5]]))
        ... )
        >>> instrument = srf @ spectral
        >>> data = jnp.array([0.0, 2.0, 4.0, 6.0, 8.0])
        >>> instrument(data).tolist(), srf(spectral(data)).tolist()
        ([2.5, 5.5], [2.5, 5.5])
        >>> instrument.read.n_taps
        4
    """

    def __init__(self, outer: Plan, inner: Plan, *, merge: bool = False) -> None:
        outer_stencil = outer._local_stencil("the outer plan of a composition")
        if not isinstance(inner.read, Stencil):
            msg = (
                "the inner plan of a composition needs a stencil read-out; compose "
                "the operators in lineax instead (outer.operator() @ inner.operator())"
            )
            raise PolicyError(msg)
        if outer.policies.missing != "propagate":
            msg = (
                "the outer plan of a composition needs missing='propagate', got "
                f"{outer.policies.missing!r}, which would act on intermediate values"
            )
            raise PolicyError(msg)
        if inner.policies.missing == "renormalise":
            msg = (
                "the inner plan of a composition cannot use missing='renormalise', "
                "whose weights depend on the data"
            )
            raise PolicyError(msg)
        inner_fill = inner.policies.fill_value
        outer_fill = outer.policies.fill_value
        same_fill = inner_fill == outer_fill or (
            math.isnan(inner_fill) and math.isnan(outer_fill)
        )
        if inner.policies.missing == "fill" and not same_fill:
            # one fill value serves both the missing samples and invalid rows
            msg = (
                "under missing='fill' the inner plan of a composition must share "
                f"the outer plan's fill_value, got {inner_fill!r} and {outer_fill!r}"
            )
            raise PolicyError(msg)
        self.fit = inner.fit
        self.read = compose(outer_stencil, inner.read, merge=merge)
        self.policies = dataclasses.replace(
            inner.policies, fill_value=outer.policies.fill_value
        )
        self.local = None
        self.target_shape = outer.target_shape


class WindowedPlan(Plan):
    """``plan[window]``: the rows of a grid target inside a window, sources unchanged.

    The window holds one slice per target axis (a bare slice for a flat list
    of targets) and keeps the rows of the nodes inside it, in row-major
    order, with ``valid`` sliced alongside. Each row is the parent's row, so
    ``plan[window](y)`` equals the matching slice of ``plan(y)`` bit for bit:
    one plan on the destination grid, cut per chip, gives chips that agree
    with the full read-out.

    Args:
        plan: The plan to cut; its read-out must be a `Stencil`.
        window: One slice per axis of ``plan.grid_shape``.

    Raises:
        ValidationError: If ``window`` does not hold one slice per target axis,
            or selects no targets.
        PolicyError: If ``plan`` has no stencil read-out.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, RectilinearGrid
        >>> from functax.interp import Linear
        >>> source = RectilinearGrid((Axis(jnp.arange(3.0)), Axis(jnp.arange(4.0))))
        >>> target = RectilinearGrid(
        ...     (Axis(jnp.array([0.5, 1.5])), Axis(jnp.array([0.5, 1.5, 2.5])))
        ... )
        >>> plan = Linear().plan(source, target)
        >>> plan(jnp.arange(12.0)).tolist()
        [2.5, 3.5, 4.5, 6.5, 7.5, 8.5]
        >>> chip = plan[0:2, 1:3]
        >>> chip.grid_shape, chip(jnp.arange(12.0)).tolist()
        ((2, 2), [3.5, 4.5, 7.5, 8.5])
    """

    window: tuple[slice, ...] = eqx.field(static=True, default=())

    def __init__(self, plan: Plan, window: slice | tuple[slice, ...]) -> None:
        window = (window,) if isinstance(window, slice) else tuple(window)
        shape = plan.grid_shape
        all_slices = all(isinstance(axis_window, slice) for axis_window in window)
        if len(window) != len(shape) or not all_slices:
            msg = (
                f"'window' must hold one slice per target axis ({len(shape)}), "
                f"got {window!r}"
            )
            raise ValidationError(msg)
        if not isinstance(plan.read, Stencil):
            msg = "a window needs a plan with a stencil read-out"
            raise PolicyError(msg)
        # the nodes kept on each axis, then their flat row-major row numbers
        kept = [
            np.arange(n_nodes)[axis_window]
            for n_nodes, axis_window in zip(shape, window, strict=True)
        ]
        window_shape = tuple(axis_nodes.size for axis_nodes in kept)
        if math.prod(window_shape) == 0:
            msg = f"'window' {window!r} selects no targets of a grid of {shape}"
            raise ValidationError(msg)
        mesh = np.meshgrid(*kept, indexing="ij")
        rows = einx.id("... -> (...)", np.ravel_multi_index(mesh, shape))
        stencil = plan.read
        self.fit = plan.fit
        self.read = Stencil(
            stencil.indices[rows],
            stencil.weights[rows],
            n_source=stencil.n_source,
            valid=stencil.valid[rows],
        )
        self.policies = plan.policies
        self.local = None
        self.target_shape = window_shape
        self.window = window
