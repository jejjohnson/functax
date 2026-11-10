"""Fits 𝒮: samples y to coefficients c, and the adjoint 𝒮ᵀ back.

Local schemes use the identity fit, c = y. The B-spline prefilter of #27 and
the global fits of Wave 3 are other fits with the same two methods; the
kernel schemes (RBF, kriging) share `KernelSolve`, the factorised
saddle-point system of a kernel and a polynomial tail.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import abc
from typing import Any

import einx
import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from functax.exceptions import ValidationError


__all__ = [
    "Fit",
    "Identity",
    "KernelSolve",
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


class KernelSolve(Fit):
    """[c; b] = A⁻¹ [y; 0], the saddle-point system of a kernel and a tail.

    With the (N, N) kernel matrix K̃ = K + λI on the sources (smoothing or a
    nugget on the diagonal) and the (N, P) tail matrix P (monomials, a
    kriging drift),

        A = [[K̃, P], [Pᵀ, 0]],  K̃ c + P b = y  and  Pᵀ c = 0,

    so the N + P coefficients are the kernel weights c and the tail weights
    b. A is symmetric and indefinite; it is solved by the null-space method,
    which needs K̃ positive definite only on the null space of Pᵀ, as a
    conditionally positive definite kernel with a tail of high enough degree
    is.

    Algorithm:
        Built once:

        1. Factor P = Q R with the complete QR, Q = [Q₁ Q₂], so the columns of
           Q₂ (N × (N − P)) span the null space of Pᵀ.
        2. Form M = Q₂ᵀ K̃ Q₂ ((N − P) × (N − P), positive definite) and
           factor it with ``solver`` (lineax; Cholesky by default), once.

        Each solve of A [c; b] = [g; h]:

        1. The particular solution c₀ = Q₁ R⁻ᵀ h, with Pᵀ c₀ = h.
        2. α = M⁻¹ Q₂ᵀ (g − K̃ c₀) and c = c₀ + Q₂ α.
        3. b = R⁻¹ Q₁ᵀ (g − K̃ c).

        With no tail (P = 0), M = K̃ and c = K̃⁻¹ g.

    The fit is 𝒮 = A⁻¹ E with E = [I; 0], and A is symmetric, so its adjoint
    𝒮ᵀ g = Eᵀ A⁻¹ g is the same solve, restricted to the data block.

    Args:
        gram: K̃, the (N, N) kernel matrix with any smoothing on its diagonal.
        terms: P, the (N, P) tail matrix, or ``None`` for no tail.
        solver: The lineax solver for M; by default ``lineax.Cholesky()``
            when ``positive_definite``, else ``lineax.LU()``.
        positive_definite: Whether M is positive definite (the kernel is
            positive definite, or conditionally so of an order the tail
            covers); M is tagged positive semi-definite for lineax then, and
            only symmetric otherwise.

    Raises:
        ValidationError: If ``gram`` is not square and floating, or ``terms``
            does not have N rows and fewer than N columns.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.interp import KernelSolve
        >>> gram = jnp.array([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
        >>> fit = KernelSolve(gram, jnp.ones((3, 1)))
        >>> coefficients = fit.apply(jnp.array([1.0, 2.0, 3.0]))
        >>> # K̃ c + P b = y and Pᵀ c = 0
        >>> c, b = coefficients[:3], coefficients[3:]
        >>> bool(jnp.allclose(gram @ c + b, jnp.array([1.0, 2.0, 3.0])))
        True
        >>> bool(jnp.allclose(c.sum(), 0.0))
        True
        >>> fit.data_size(4)
        3
    """

    gram: Float[Array, "source source"] | None
    basis: Float[Array, "source source"] | None
    triangular: Float[Array, "term term"] | None
    inner: lx.MatrixLinearOperator
    state: Any
    solver: lx.AbstractLinearSolver
    n_source: int = eqx.field(static=True)
    n_terms: int = eqx.field(static=True)

    def __init__(
        self,
        gram: Float[Array, "source source"],
        terms: Float[Array, "source term"] | None = None,
        *,
        solver: lx.AbstractLinearSolver | None = None,
        positive_definite: bool = True,
    ) -> None:
        gram = jnp.asarray(gram)
        if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
            msg = f"'gram' must be square (source, source), got shape {gram.shape}"
            raise ValidationError(msg)
        if not jnp.issubdtype(gram.dtype, jnp.floating):
            msg = f"'gram' must have a floating dtype, got {gram.dtype}"
            raise ValidationError(msg)
        n_source = gram.shape[0]
        if solver is None:
            solver = lx.Cholesky() if positive_definite else lx.LU()
        if terms is not None:
            terms = jnp.asarray(terms)
            if terms.ndim != 2 or terms.shape[0] != n_source:
                msg = (
                    f"'terms' must be ({n_source}, term), one row per source, "
                    f"got shape {terms.shape}"
                )
                raise ValidationError(msg)
            if terms.shape[1] >= n_source:
                msg = (
                    f"'terms' must have fewer columns than the {n_source} sources, "
                    f"got {terms.shape[1]}"
                )
                raise ValidationError(msg)
        if terms is None or terms.shape[1] == 0:
            n_terms = 0
            inner_matrix = gram
            self.gram = None
            self.basis = None
            self.triangular = None
        else:
            n_terms = terms.shape[1]
            # P = Q R, complete: Q₂ = Q[:, P:] spans the null space of Pᵀ
            basis, triangular = jnp.linalg.qr(terms, mode="complete")
            null_basis = basis[:, n_terms:]
            # M = Q₂ᵀ K̃ Q₂
            projected = einx.dot("row free, row col -> free col", null_basis, gram)
            product = einx.dot(
                "free col, col other -> free other", projected, null_basis
            )
            # symmetric to the last bit, so 𝒮ᵀ is the same solve as 𝒮
            transposed = einx.id("free other -> other free", product)
            inner_matrix = (product + transposed) / 2
            self.gram = gram
            self.basis = basis
            self.triangular = triangular[:n_terms]
        tags = (
            (lx.symmetric_tag, lx.positive_semidefinite_tag)
            if positive_definite
            else lx.symmetric_tag
        )
        self.inner = lx.MatrixLinearOperator(inner_matrix, tags)
        self.state = solver.init(self.inner, {})
        self.solver = solver
        self.n_source = int(n_source)
        self.n_terms = int(n_terms)

    def _inner_solve(self, vector: Float[Array, " free"]) -> Float[Array, " free"]:
        """M⁻¹ v, with the factorisation computed at build time."""
        solution = lx.linear_solve(self.inner, vector, self.solver, state=self.state)
        return solution.value

    @staticmethod
    def _triangular_solve(
        triangular: Float[Array, "term term"],
        vector: Float[Array, " term"],
        *,
        transpose: bool,
    ) -> Float[Array, " term"]:
        """R⁻¹ v, or R⁻ᵀ v with ``transpose``."""
        operator = lx.MatrixLinearOperator(triangular, lx.upper_triangular_tag)
        if transpose:
            operator = operator.transpose()
        return lx.linear_solve(operator, vector, lx.Triangular()).value

    def _solve_one(
        self, rhs: Float[Array, " coefficient"]
    ) -> Float[Array, " coefficient"]:
        """[c; b] = A⁻¹ [g; h] for one right-hand side."""
        gram, basis, triangular = self.gram, self.basis, self.triangular
        if gram is None or basis is None or triangular is None:
            return self._inner_solve(rhs)
        n_terms = self.n_terms
        values = rhs[:-n_terms]
        constraints = rhs[-n_terms:]
        range_basis = basis[:, :n_terms]
        null_basis = basis[:, n_terms:]
        # c₀ = Q₁ R⁻ᵀ h
        lifted = self._triangular_solve(triangular, constraints, transpose=True)
        particular = einx.dot("source term, term -> source", range_basis, lifted)
        # α = M⁻¹ Q₂ᵀ (g − K̃ c₀)
        fitted = einx.dot("row col, col -> row", gram, particular)
        projected = einx.dot("source free, source -> free", null_basis, values - fitted)
        alpha = self._inner_solve(projected)
        # c = c₀ + Q₂ α
        weights = particular + einx.dot(
            "source free, free -> source", null_basis, alpha
        )
        # b = R⁻¹ Q₁ᵀ (g − K̃ c)
        residual = values - einx.dot("row col, col -> row", gram, weights)
        projected_residual = einx.dot(
            "source term, source -> term", range_basis, residual
        )
        tail = self._triangular_solve(triangular, projected_residual, transpose=False)
        return einx.id("a, b -> (a + b)", weights, tail)

    def solve(
        self, rhs: Float[Array, "*batch coefficient"]
    ) -> Float[Array, "*batch coefficient"]:
        """A⁻¹ [g; h] for any right-hand side, over any leading batch dims.

        Args:
            rhs: [g; h], N + P values with the coefficients last.

        Returns:
            [c; b], with the coefficients last.

        Raises:
            ValidationError: If the last axis of ``rhs`` is not N + P.
        """
        rhs = jnp.asarray(rhs)
        n_coefficients = self.n_source + self.n_terms
        if rhs.ndim < 1 or rhs.shape[-1] != n_coefficients:
            msg = (
                f"'rhs' must have a last axis of length {n_coefficients}, "
                f"got shape {rhs.shape}"
            )
            raise ValidationError(msg)
        solve = jnp.vectorize(self._solve_one, signature="(n)->(n)")
        return solve(rhs)

    def apply(
        self, data: Float[Array, "*batch source"]
    ) -> Float[Array, "*batch coefficient"]:
        """[c; b] = A⁻¹ [y; 0], over any leading batch dims."""
        data = jnp.asarray(data)
        if self.n_terms == 0:
            return self.solve(data)
        zeros = jnp.zeros((*data.shape[:-1], self.n_terms), data.dtype)
        rhs = einx.id("... a, ... b -> ... (a + b)", data, zeros)
        return self.solve(rhs)

    def apply_adjoint(
        self, coefficients: Float[Array, "*batch coefficient"]
    ) -> Float[Array, "*batch source"]:
        """𝒮ᵀ g = Eᵀ A⁻¹ g, the data block of the same symmetric solve."""
        solution = self.solve(coefficients)
        if self.n_terms == 0:
            return solution
        return solution[..., : -self.n_terms]

    def data_size(self, n_coefficients: int) -> int:
        """N = (N + P) − P."""
        return n_coefficients - self.n_terms
