"""The variational fit: c = (HᵀΛ_εH + βLᵀL)⁻¹ HᵀΛ_εy, matrix-free through lineax.

H, Λ_ε and L arrive as lineax operators, so a plan's ``operator()``, a
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
    """CG with rtol = atol = √eps of ``dtype``: 1.5e-8 in float64, 3.5e-4 in float32.

    eps is the machine epsilon, ``jnp.finfo(dtype).eps``.
    """
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
    """c = argmin_c (Hc − y)ᵀΛ_ε(Hc − y) + β‖Lc‖₂², the variational smoothing fit.

    Here c ∈ ℝᴺ are the grid values, y ∈ ℝᴹ the observations, H ∈ ℝ^{M×N}
    the observation operator, Λ_ε = Σ_ε⁻¹ ∈ ℝ^{M×M} the observation-error
    precision, L ∈ ℝ^{N_L×N} the roughness operator and β ≥ 0 the smoothing
    weight. In 3D-Var and DIVA letters, Λ_ε is R⁻¹ and βLᵀL plays the role of
    the prior precision B⁻¹. The minimiser solves the normal equations

        (HᵀΛ_εH + βLᵀL) c = HᵀΛ_εy,

    whose operator is applied matrix-free and tagged symmetric positive
    semi-definite, so lineax's CG solves it. lineax differentiates the solve
    implicitly, so a gradient in y, β or the operators' arrays costs one more
    solve. The fit is linear in y, with adjoint 𝒮ᵀg = Λ_εH A⁻¹g for
    A = HᵀΛ_εH + βLᵀL ∈ ℝ^{N×N} and a cotangent g ∈ ℝᴺ, so it sits in a
    `Plan` like any other fit.

    Algorithm:
        1. b = HᵀΛ_εy.
        2. c = A⁻¹b by ``solver`` with A applied as c ↦ HᵀΛ_εHc + βLᵀLc.

    Args:
        observation: H, (M, N), from the grid to the observations (a plan's
            ``operator()``).
        roughness: L, (N_L, N), the roughness operator (a difference operator).
        observation_precision: Λ_ε, (M, M), symmetric positive definite; the
            identity when ``None``.
        smoothing_weight: β ≥ 0, the weight of the roughness penalty.
        solver: The lineax solver for A; CG at √eps of the data dtype when
            ``None``. A must be positive definite for CG (H and L share no
            null vector), or the solve fails.

    Raises:
        ValidationError: If an operator does not act on vectors, the sizes
            disagree, or β is negative (checked when concrete).

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from functax.project import Variational
        >>> grid = jnp.zeros(4)
        >>> identity = lx.IdentityLinearOperator(grid)
        >>> # H = I and β = 0 return the data
        >>> fit = Variational(identity, identity, smoothing_weight=0.0)
        >>> y = jnp.array([1.0, -2.0, 0.5, 3.0])
        >>> bool(jnp.allclose(fit(y), y))
        True
        >>> # with L = I, c = y / (1 + β)
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

    def _precision(self, residual: Float[Array, " obs"]) -> Float[Array, " obs"]:
        """Λ_ε r, or r when Λ_ε is the identity."""
        if self.observation_precision is None:
            return residual
        return self.observation_precision.mv(residual)

    def normal_operator(self) -> lx.FunctionLinearOperator:
        """A = HᵀΛ_εH + βLᵀL, (N, N), matrix-free, tagged symmetric PSD."""
        observation = self.observation
        observation_t = observation.transpose()
        roughness = self.roughness
        roughness_t = roughness.transpose()

        def normal(grid: Float[Array, " cell"]) -> Float[Array, " cell"]:
            # Hc
            predicted = observation.mv(grid)
            # Λ_εHc
            weighted = self._precision(predicted)
            # HᵀΛ_εHc
            data_term = observation_t.mv(weighted)
            # Lc
            rough = roughness.mv(grid)
            # LᵀLc
            penalty = roughness_t.mv(rough)
            # Ac = HᵀΛ_εHc + βLᵀLc
            return data_term + self.smoothing_weight * penalty

        return lx.FunctionLinearOperator(
            normal,
            observation.in_structure(),
            (lx.symmetric_tag, lx.positive_semidefinite_tag),
        )

    def _solve(self, rhs: Float[Array, " cell"]) -> Float[Array, " cell"]:
        """A⁻¹ b for one vector b."""
        solver = default_solver(rhs.dtype) if self.solver is None else self.solver
        normal = self.normal_operator()
        # c = A⁻¹ b
        solution = lx.linear_solve(normal, rhs, solver)
        return solution.value

    def apply(self, data: Float[Array, "*batch obs"]) -> Float[Array, "*batch cell"]:
        """c = A⁻¹HᵀΛ_εy over any leading batch dims.

        Args:
            data: y, the M observations, last.

        Returns:
            c, the N grid values, last.

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

        def fit_one(observations: Float[Array, " obs"]) -> Float[Array, " cell"]:
            # Λ_εy
            weighted = self._precision(observations)
            # b = HᵀΛ_εy
            rhs = observation_t.mv(weighted)
            # c = A⁻¹ b
            return self._solve(rhs)

        fit_batch = jnp.vectorize(fit_one, signature="(m)->(n)")
        return fit_batch(data)

    def apply_adjoint(
        self, coefficients: Float[Array, "*batch cell"]
    ) -> Float[Array, "*batch obs"]:
        """𝒮ᵀg = Λ_εH A⁻¹g over any leading batch dims (A and Λ_ε are symmetric).

        Args:
            coefficients: g, the N grid values, last.

        Returns:
            𝒮ᵀg, the M observations, last.

        Raises:
            ValidationError: If the last axis of ``coefficients`` is not N.

        Examples:
            The dot-product test 𝒮(y)ᵀg = yᵀ𝒮ᵀg for a ridge fit:

            >>> import einx
            >>> import jax.numpy as jnp
            >>> import lineax as lx
            >>> from functax.project import Variational
            >>> identity = lx.IdentityLinearOperator(jnp.zeros(3))
            >>> fit = Variational(identity, identity, smoothing_weight=0.5)
            >>> y = jnp.array([1.0, -2.0, 0.5])
            >>> g = jnp.array([0.3, 0.1, -1.0])
            >>> forward = einx.dot("cell, cell ->", fit.apply(y), g)
            >>> adjoint = einx.dot("obs, obs ->", y, fit.apply_adjoint(g))
            >>> bool(jnp.allclose(forward, adjoint))
            True
        """
        coefficients = jnp.asarray(coefficients)
        n_cell = self.observation.in_size()
        if coefficients.ndim < 1 or coefficients.shape[-1] != n_cell:
            msg = (
                f"'coefficients' must have a last axis of length {n_cell}, got "
                f"shape {coefficients.shape}"
            )
            raise ValidationError(msg)

        def adjoint_one(grid: Float[Array, " cell"]) -> Float[Array, " obs"]:
            # A⁻¹g
            solved = self._solve(grid)
            # H A⁻¹g
            predicted = self.observation.mv(solved)
            # 𝒮ᵀg = Λ_εH A⁻¹g
            return self._precision(predicted)

        adjoint_batch = jnp.vectorize(adjoint_one, signature="(n)->(m)")
        return adjoint_batch(coefficients)

    def __call__(self, data: Float[Array, "*batch obs"]) -> Float[Array, "*batch cell"]:
        """c = 𝒮(y), the same as `apply`."""
        return self.apply(data)
