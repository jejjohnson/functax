"""Laplacian eigenfunctions of the box [−L, L]^D under three boundary conditions.

In one dimension, the eigenpairs of −d²/dx² on [−L, L], each set orthonormal
on the interval and listed with non-decreasing eigenvalues:

- Dirichlet: φ_j = sin(jπ(x + L)/(2L))/√L, λ_j = (jπ/(2L))², j = 1, 2, …
- Neumann: φ₀ = 1/√(2L), φ_j = cos(jπ(x + L)/(2L))/√L, λ_j = (jπ/(2L))²,
  j = 0, 1, …
- Periodic: 1/√(2L), then cos(kπx/L)/√L and sin(kπx/L)/√L for k = 1, 2, …,
  λ = (kπ/L)², ordered constant, cos 1, sin 1, cos 2, sin 2, …

On the D-dimensional box the basis is their tensor product, flattened
row-major over the multi-index. These are the eigenfunctions behind the
variational Fourier features and the Hilbert-space GP approximation. Moved from
geonnax ``_fourier`` at ``be5537d``.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

from typing import Any

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float

from functax.exceptions import ValidationError


__all__ = [
    "fourier_basis",
    "fourier_basis_1d",
    "fourier_eigenvalues",
    "fourier_eigenvalues_1d",
]


_BOUNDARIES: tuple[str, ...] = ("dirichlet", "neumann", "periodic")


def _check_boundary(boundary: str) -> None:
    if boundary not in _BOUNDARIES:
        msg = f"'boundary' must be one of {list(_BOUNDARIES)}, got {boundary!r}"
        raise ValidationError(msg)


def _check_box(num_basis: int, L: float) -> None:
    if num_basis < 1:
        msg = f"'num_basis' must be at least 1, got {num_basis!r}"
        raise ValidationError(msg)
    if not L > 0:
        msg = f"'L' must be positive, got {L!r}"
        raise ValidationError(msg)


def _frequencies_1d(
    num_basis: int, L: float, boundary: str, dtype: Any
) -> Float[Array, " feature"]:
    """The angular frequency of each 1-D column (its eigenvalue is the square)."""
    if boundary == "dirichlet":
        return jnp.arange(1, num_basis + 1, dtype=dtype) * (jnp.pi / (2.0 * L))
    if boundary == "neumann":
        return jnp.arange(num_basis, dtype=dtype) * (jnp.pi / (2.0 * L))
    # k = 0, 1, 1, 2, 2, …
    k = (jnp.arange(num_basis) + 1) // 2
    return k.astype(dtype) * (jnp.pi / L)


def _basis_value_and_grad_1d(
    x: Float[Array, " obs"], num_basis: int, L: float, boundary: str
) -> tuple[Float[Array, "obs feature"], Float[Array, "obs feature"]]:
    """The 1-D eigenfunctions and their x-derivatives, for any boundary condition."""
    _check_box(num_basis, L)
    _check_boundary(boundary)
    frequency = _frequencies_1d(num_basis, L, boundary, x.dtype)
    inv_sqrt_l = 1.0 / jnp.sqrt(L)
    if boundary == "periodic":
        arg = einx.dot("obs, feature -> obs feature", x, frequency)
        is_cos = (jnp.arange(num_basis) % 2) == 1
        value = jnp.where(is_cos, jnp.cos(arg), jnp.sin(arg)) * inv_sqrt_l
        slope = jnp.where(is_cos, -jnp.sin(arg), jnp.cos(arg))
        grad = einx.multiply("obs feature, feature", slope, frequency) * inv_sqrt_l
        constant = jnp.arange(num_basis) == 0
    else:
        arg = einx.dot("obs, feature -> obs feature", x + L, frequency)
        if boundary == "dirichlet":
            value = jnp.sin(arg) * inv_sqrt_l
            slope = jnp.cos(arg)
        else:
            value = jnp.cos(arg) * inv_sqrt_l
            slope = -jnp.sin(arg)
        grad = einx.multiply("obs feature, feature", slope, frequency) * inv_sqrt_l
        constant = jnp.arange(num_basis) == (0 if boundary == "neumann" else -1)
    # the constant column (Neumann j = 0, periodic k = 0) is 1/√(2L)
    value = jnp.where(constant, 1.0 / jnp.sqrt(2.0 * L), value)
    grad = jnp.where(constant, 0.0, grad)
    return value, grad


def fourier_basis_1d(
    x: Float[Array, " obs"],
    num_basis: int,
    L: float,
    *,
    boundary: str = "dirichlet",
) -> Float[Array, "obs feature"]:
    """The first ``num_basis`` Laplacian eigenfunctions on [−L, L].

    Dirichlet by default, φ_j(x) = sin(jπ(x + L)/(2L))/√L; ``"neumann"`` and
    ``"periodic"`` are listed in the module docstring.

    Args:
        x: Points in [−L, L]; points outside are extrapolated silently, so
            callers check their own domain.
        num_basis: M, the number of eigenfunctions.
        L: The half-width of the interval, positive.
        boundary: ``"dirichlet"``, ``"neumann"`` or ``"periodic"``.

    Returns:
        Column m is the m-th eigenfunction (φ_{m+1} for Dirichlet) at each
        point.

    Raises:
        ValidationError: If ``num_basis`` is below 1, L is not positive, or
            ``boundary`` is unknown.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.fourier import fourier_basis_1d
        >>> fourier_basis_1d(jnp.linspace(-1.0, 1.0, 5), num_basis=3, L=1.0).shape
        (5, 3)
        >>> # the Neumann constant is 1/√(2L)
        >>> values = fourier_basis_1d(jnp.zeros(1), 2, 2.0, boundary="neumann")
        >>> bool(jnp.allclose(values[0, 0], 0.5))
        True
    """
    _check_box(num_basis, L)
    _check_boundary(boundary)
    if boundary != "dirichlet":
        return _basis_value_and_grad_1d(x, num_basis, L, boundary)[0]
    j = jnp.arange(1, num_basis + 1, dtype=x.dtype)
    # jπ(x + L)/(2L)
    arg = einx.dot("obs, feature -> obs feature", x + L, j) * (jnp.pi / (2.0 * L))
    return jnp.sin(arg) / jnp.sqrt(L)


def fourier_eigenvalues_1d(
    num_basis: int,
    L: float,
    *,
    dtype: Any = jnp.float32,
    boundary: str = "dirichlet",
) -> Float[Array, " feature"]:
    """The eigenvalues λ_j = (jπ/(2L))² of `fourier_basis_1d`, j = 1, …, num_basis.

    With ``boundary="neumann"`` or ``"periodic"``, the eigenvalues of those
    bases (see the module docstring).

    Args:
        num_basis: M, the number of eigenvalues.
        L: The half-width of the interval, positive.
        dtype: The dtype of the result.
        boundary: ``"dirichlet"``, ``"neumann"`` or ``"periodic"``.

    Returns:
        λ for each column of `fourier_basis_1d`, non-decreasing.

    Raises:
        ValidationError: If ``num_basis`` is below 1, L is not positive, or
            ``boundary`` is unknown.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.fourier import fourier_eigenvalues_1d
        >>> fourier_eigenvalues_1d(num_basis=4, L=1.0).shape
        (4,)
        >>> values = fourier_eigenvalues_1d(3, jnp.pi, boundary="periodic")
        >>> bool(jnp.allclose(values, jnp.array([0.0, 1.0, 1.0])))
        True
    """
    _check_box(num_basis, L)
    _check_boundary(boundary)
    if boundary != "dirichlet":
        return _frequencies_1d(num_basis, L, boundary, dtype) ** 2
    j = jnp.arange(1, num_basis + 1, dtype=dtype)
    return (j * jnp.pi / (2.0 * L)) ** 2


def _to_tuple(value: Any, D: int, name: str) -> tuple:
    """A scalar broadcast to a length-D tuple, or a sequence checked for length D."""
    if isinstance(value, tuple | list):
        out = tuple(value)
        if len(out) != D:
            msg = f"'{name}' must have length {D} (one per input dim), got {len(out)}"
            raise ValidationError(msg)
        return out
    return (value,) * D


def _tensor_product_sum(
    eigenvalues: list[Float[Array, " _"]], *, dtype: Any
) -> Float[Array, " feature"]:
    """λ_(j₁, …, j_D) = ∑_d λ_{j_d}, flattened row-major (the last index fastest)."""
    if not eigenvalues:
        return jnp.zeros(0, dtype=dtype)
    total = jnp.asarray(eigenvalues[0], dtype=dtype)
    for axis_values in eigenvalues[1:]:
        total = einx.add("left, right -> (left right)", total, axis_values)
    return total


def fourier_basis(
    x: Float[Array, "obs dim"],
    num_basis_per_dim: int | tuple[int, ...],
    L: float | tuple[float, ...],
    *,
    boundary: str | tuple[str, ...] = "dirichlet",
) -> tuple[Float[Array, "obs feature"], Float[Array, " feature"]]:
    """The tensor-product Laplacian eigenpairs on the box [−L, L]^D.

    Φ_(j₁, …, j_D)(x) = ∏_d φ_{j_d}(x_d) and λ_(j₁, …, j_D) = ∑_d λ_{j_d}, flattened
    row-major over the multi-index (the last dimension fastest), so there are
    M = ∏_d M_d columns.

    Args:
        x: Points in [−L, L]^D.
        num_basis_per_dim: M_d, the number of 1-D eigenfunctions per axis; an
            int applies to every axis.
        L: The half-width per axis; a number applies to every axis.
        boundary: ``"dirichlet"``, ``"neumann"`` or ``"periodic"``, one value
            for every axis or one per axis (periodic in longitude and
            Dirichlet in latitude, say).

    Returns:
        ``(Phi, lam)``: the M basis functions at each point and their
        eigenvalues.

    Raises:
        ValidationError: If ``x`` is not 2-D, a per-axis argument has the
            wrong length, or an axis's 1-D arguments are invalid.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.fourier import fourier_basis
        >>> Phi, lam = fourier_basis(jnp.zeros((4, 2)), num_basis_per_dim=3, L=1.0)
        >>> Phi.shape, lam.shape
        ((4, 9), (9,))
        >>> Phi, lam = fourier_basis(
        ...     jnp.zeros((4, 2)),
        ...     (2, 3),
        ...     (1.0, 2.0),
        ...     boundary=("periodic", "dirichlet"),
        ... )
        >>> Phi.shape
        (4, 6)
    """
    if x.ndim != 2:
        msg = f"'x' must be 2-D (N, D), got shape {x.shape}"
        raise ValidationError(msg)
    D = x.shape[-1]
    counts = _to_tuple(num_basis_per_dim, D, "num_basis_per_dim")
    widths = _to_tuple(L, D, "L")
    boundaries = _to_tuple(boundary, D, "boundary")
    bases = [
        fourier_basis_1d(x[:, d], counts[d], float(widths[d]), boundary=boundaries[d])
        for d in range(D)
    ]
    eigenvalues = [
        fourier_eigenvalues_1d(
            counts[d], float(widths[d]), dtype=x.dtype, boundary=boundaries[d]
        )
        for d in range(D)
    ]
    # Φ = ∏_d φ_{j_d}(x_d), flattened row-major
    product = bases[0]
    for axis_basis in bases[1:]:
        product = einx.multiply(
            "obs left, obs right -> obs (left right)", product, axis_basis
        )
    return product, _tensor_product_sum(eigenvalues, dtype=x.dtype)


def fourier_eigenvalues(
    num_basis_per_dim: int | tuple[int, ...],
    L: float | tuple[float, ...],
    D: int,
    *,
    dtype: Any = jnp.float32,
    boundary: str | tuple[str, ...] = "dirichlet",
) -> Float[Array, " feature"]:
    """The eigenvalues of `fourier_basis` alone, λ_(j₁, …, j_D) = ∑_d λ_{j_d}.

    For when only the spectrum is needed, such as building K_uu for inducing
    features without evaluating k_ux.

    Args:
        num_basis_per_dim: M_d per axis; an int applies to every axis.
        L: The half-width per axis; a number applies to every axis.
        D: The number of dimensions.
        dtype: The dtype of the result.
        boundary: One boundary condition, or one per axis.

    Returns:
        λ, flattened row-major as the columns of `fourier_basis`.

    Raises:
        ValidationError: If a per-axis argument has the wrong length, or an
            axis's 1-D arguments are invalid.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.fourier import fourier_eigenvalues
        >>> fourier_eigenvalues(num_basis_per_dim=3, L=1.0, D=2).shape
        (9,)
        >>> # with L = π/2 the Neumann eigenvalues are j², summed over the axes
        >>> values = fourier_eigenvalues(2, jnp.pi / 2, 2, boundary="neumann")
        >>> bool(jnp.allclose(values, jnp.array([0.0, 1.0, 1.0, 2.0])))
        True
    """
    counts = _to_tuple(num_basis_per_dim, D, "num_basis_per_dim")
    widths = _to_tuple(L, D, "L")
    boundaries = _to_tuple(boundary, D, "boundary")
    eigenvalues = [
        fourier_eigenvalues_1d(
            counts[d], float(widths[d]), dtype=dtype, boundary=boundaries[d]
        )
        for d in range(D)
    ]
    return _tensor_product_sum(eigenvalues, dtype=dtype)
