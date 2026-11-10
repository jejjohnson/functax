"""The variational fit: x = (HᵀR⁻¹H + λLᵀL)⁻¹ HᵀR⁻¹y, matrix-free through lineax.

H, R⁻¹ and L arrive as lineax operators, so a plan's ``operator()``, a
diagonal precision and a difference operator from finitevolX or spectraldiffx
plug in without functax importing them.
"""

from __future__ import annotations

from functax._src._backend import is_tracer, require_jax


require_jax(__name__)

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, ArrayLike, Float

from functax._src.interp._fit import Fit
from functax.exceptions import ValidationError


__all__ = [
    "Variational",
    "default_solver",
]


def default_solver(dtype: jnp.dtype) -> lx.CG:
    """CG with rtol = atol = √ε of ``dtype``: 1.5e-8 in float64, 3.5e-4 in float32."""
    # a host constant, so the default also works under jit
    tolerance = math.sqrt(float(jnp.finfo(dtype).eps))
    return lx.CG(rtol=tolerance, atol=tolerance)


def _vector_size(structure: object, name: str) -> int:
    """N of an operator side that must be one vector, ℝᴺ."""
    if not isinstance(structure, jax.ShapeDtypeStruct) or len(structure.shape) != 1:
        msg = f"'{name}' must act on vectors (one 1-D array), got {structure!r}"
        raise ValidationError(msg)
    return structure.shape[0]


class Variational(Fit):
    """x = argmin (Hx − y)ᵀR⁻¹(Hx − y) + λ‖Lx‖², the variational smoothing fit.

    The minimiser solves the normal equations

        (HᵀR⁻¹H + λLᵀL) x = HᵀR⁻¹y,

    whose operator is applied matrix-free and tagged symmetric positive
    semi-definite, so lineax's CG solves it. lineax differentiates the solve
    implicitly, so a gradient in y, λ or the operators' arrays costs one more
    solve. The fit is linear in y, with adjoint 𝒮ᵀg = R⁻¹H A⁻¹g for
    A = HᵀR⁻¹H + λLᵀL, so it sits in a `Plan` like any other fit.

    Algorithm:
        1. b = HᵀR⁻¹y.
        2. x = A⁻¹b by ``solver`` with A applied as x ↦ HᵀR⁻¹Hx + λLᵀLx.

    Args:
        observation: H, (M, N), from the grid to the observations (a plan's
            ``operator()``).
        roughness: L, (G, N), the roughness operator (a difference operator).
        observation_precision: R⁻¹, (M, M), symmetric positive definite; the
            identity when ``None``.
        smoothing_weight: λ ≥ 0, the weight of the roughness penalty.
        solver: The lineax solver for A; CG at √ε of the data dtype when
            ``None``. A must be positive definite for CG (H and L share no
            null vector), or the solve fails.

    Raises:
        ValidationError: If an operator does not act on vectors, the sizes
            disagree, or λ is negative (checked when concrete).

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from functax.project import Variational
        >>> grid = jnp.zeros(4)
        >>> identity = lx.IdentityLinearOperator(grid)
        >>> # H = I and λ = 0 return the data
        >>> fit = Variational(identity, identity, smoothing_weight=0.0)
        >>> y = jnp.array([1.0, -2.0, 0.5, 3.0])
        >>> bool(jnp.allclose(fit(y), y))
        True
        >>> # with L = I, x = y / (1 + λ)
        >>> ridge = Variational(identity, identity, smoothing_weight=1.0)
        >>> bool(jnp.allclose(ridge(y), y / 2.0))
        True
        >>> ridge.apply(jnp.ones((3, 4))).shape
        (3, 4)
    """

    observation: lx.AbstractLinearOperator
    roughness: lx.AbstractLinearOperator
    observation_precision: lx.AbstractLinearOperator | None = None
    smoothing_weight: Float[ArrayLike, ""] = eqx.field(
        default=1.0, converter=jnp.asarray
    )
    solver: lx.AbstractLinearSolver | None = eqx.field(default=None, static=True)

    def __check_init__(self) -> None:
        n_cell = _vector_size(self.observation.in_structure(), "observation")
        n_obs = _vector_size(self.observation.out_structure(), "observation")
        n_rough = _vector_size(self.roughness.in_structure(), "roughness")
        _vector_size(self.roughness.out_structure(), "roughness")
        if n_rough != n_cell:
            msg = (
                f"'roughness' must act on the {n_cell} grid values of "
                f"'observation', got {n_rough}"
            )
            raise ValidationError(msg)
        if self.observation_precision is not None:
            precision = self.observation_precision
            n_in = _vector_size(precision.in_structure(), "observation_precision")
            n_out = _vector_size(precision.out_structure(), "observation_precision")
            if (n_in, n_out) != (n_obs, n_obs):
                msg = (
                    f"'observation_precision' must be ({n_obs}, {n_obs}), got "
                    f"({n_out}, {n_in})"
                )
                raise ValidationError(msg)
        weight = jnp.asarray(self.smoothing_weight)
        if weight.ndim != 0:
            msg = f"'smoothing_weight' must be a scalar, got shape {weight.shape}"
            raise ValidationError(msg)
        if not is_tracer(weight) and bool(weight < 0):
            msg = f"'smoothing_weight' must be non-negative, got {float(weight)}"
            raise ValidationError(msg)

    def data_size(self, n_coefficients: int) -> int:
        """M, the number of observations the N grid values are fitted from."""
        del n_coefficients
        return self.observation.out_size()

    def _precision(self, residual: jax.Array) -> jax.Array:
        """R⁻¹ r, or r when R⁻¹ is the identity."""
        if self.observation_precision is None:
            return residual
        return self.observation_precision.mv(residual)

    def normal_operator(self) -> lx.FunctionLinearOperator:
        """A = HᵀR⁻¹H + λLᵀL, (N, N), matrix-free, tagged symmetric PSD."""
        observation = self.observation
        observation_t = observation.transpose()
        roughness = self.roughness
        roughness_t = roughness.transpose()

        def normal(grid: jax.Array) -> jax.Array:
            # HᵀR⁻¹H x
            weighted = self._precision(observation.mv(grid))
            data_term = observation_t.mv(weighted)
            # λ LᵀL x
            penalty = roughness_t.mv(roughness.mv(grid))
            return data_term + self.smoothing_weight * penalty

        return lx.FunctionLinearOperator(
            normal,
            observation.in_structure(),
            (lx.symmetric_tag, lx.positive_semidefinite_tag),
        )

    def _solve(self, rhs: jax.Array) -> jax.Array:
        """A⁻¹ b for one vector b."""
        solver = default_solver(rhs.dtype) if self.solver is None else self.solver
        return lx.linear_solve(self.normal_operator(), rhs, solver).value

    def apply(self, data: Float[Array, "*batch obs"]) -> Float[Array, "*batch cell"]:
        """x = A⁻¹HᵀR⁻¹y over any leading batch dims.

        Args:
            data: y, the M observations, last.

        Returns:
            x, the N grid values, last.

        Raises:
            ValidationError: If the last axis of ``data`` is not M.
        """
        data = jnp.asarray(data)
        n_obs = self.observation.out_size()
        if data.ndim < 1 or data.shape[-1] != n_obs:
            msg = (
                f"'data' must have a last axis of length {n_obs}, got shape "
                f"{data.shape}"
            )
            raise ValidationError(msg)
        observation_t = self.observation.transpose()

        def fit_one(observations: jax.Array) -> jax.Array:
            # b = HᵀR⁻¹y
            rhs = observation_t.mv(self._precision(observations))
            return self._solve(rhs)

        return jnp.vectorize(fit_one, signature="(m)->(n)")(data)

    def apply_adjoint(
        self, coefficients: Float[Array, "*batch cell"]
    ) -> Float[Array, "*batch obs"]:
        """𝒮ᵀg = R⁻¹H A⁻¹g over any leading batch dims (A and R⁻¹ are symmetric).

        Args:
            coefficients: g, the N grid values, last.

        Returns:
            𝒮ᵀg, the M observations, last.

        Raises:
            ValidationError: If the last axis of ``coefficients`` is not N.
        """
        coefficients = jnp.asarray(coefficients)
        n_cell = self.observation.in_size()
        if coefficients.ndim < 1 or coefficients.shape[-1] != n_cell:
            msg = (
                f"'coefficients' must have a last axis of length {n_cell}, got "
                f"shape {coefficients.shape}"
            )
            raise ValidationError(msg)

        def adjoint_one(grid: jax.Array) -> jax.Array:
            solved = self._solve(grid)
            return self._precision(self.observation.mv(solved))

        return jnp.vectorize(adjoint_one, signature="(n)->(m)")(coefficients)

    def __call__(self, data: Float[Array, "*batch obs"]) -> Float[Array, "*batch cell"]:
        """x = 𝒮(y), the same as `apply`."""
        return self.apply(data)
