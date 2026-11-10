"""The low-frequency eigenpairs of a graph Laplacian, for graph and manifold features.

The eigendecomposition is a one-shot host computation in NumPy that returns JAX
arrays. Moved from geonnax ``_laplacian`` at ``be5537d``.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import operator

import einx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from functax.exceptions import ValidationError


__all__ = ["graph_laplacian_eigpairs"]


def graph_laplacian_eigpairs(
    adjacency: Float[ArrayLike, "row col"],
    num_basis: int,
    *,
    normalized: bool = True,
) -> tuple[Float[Array, " feature"], Float[Array, "vertex feature"]]:
    """The ``num_basis`` smallest Laplacian eigenpairs of an undirected graph.

    The combinatorial Laplacian L = D − A, or with ``normalized=True`` the
    symmetric normalised L = I − D^{−½} A D^{−½}, with D the diagonal of degrees.
    The adjacency is symmetrised and its diagonal ignored (no self-loops). This
    is a dense eigendecomposition on the host, O(V³) time and O(V²) memory,
    practical up to a few thousand nodes; large or structured graphs want a
    sparse solver (kernellib's ``laplacian_eigpairs`` takes sparse graphs and
    solves grid graphs by an exact Kronecker product, Lanczos or ARPACK).

    Args:
        adjacency: A, symmetric and non-negative, V × V.
        num_basis: M, the number of low-frequency eigenpairs, from 1 to V.
        normalized: Use the symmetric normalised Laplacian rather than D − A.

    Returns:
        ``(eigvals, eigvecs)``: the M eigenvalues ascending (round-off
        negatives clipped to 0) and the ℓ²-orthonormal eigenvectors as
        columns.

    Raises:
        ValidationError: If ``adjacency`` is not square, has a negative entry,
            or ``num_basis`` is not an integer in [1, V].

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.basis.eigen import graph_laplacian_eigpairs
        >>> path = jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        >>> eigvals, eigvecs = graph_laplacian_eigpairs(path, num_basis=2)
        >>> eigvals.shape, eigvecs.shape
        ((2,), (3, 2))
        >>> # D − A of the 3-node path has eigenvalues 0, 1 and 3
        >>> values, _ = graph_laplacian_eigpairs(path, 3, normalized=False)
        >>> bool(jnp.allclose(values, jnp.array([0.0, 1.0, 3.0]), atol=1e-12))
        True
    """
    weights = np.array(adjacency, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        msg = f"'adjacency' must be square, got shape {weights.shape}"
        raise ValidationError(msg)
    n_vertices = weights.shape[0]
    message = f"'num_basis' must be an integer in [1, {n_vertices}], got {num_basis!r}"
    try:
        count = operator.index(num_basis)
    except TypeError as err:
        raise ValidationError(message) from err
    if isinstance(num_basis, bool) or not 1 <= count <= n_vertices:
        raise ValidationError(message)
    if np.any(weights < 0):
        msg = "'adjacency' must have non-negative entries"
        raise ValidationError(msg)
    weights = 0.5 * (weights + einx.id("row col -> col row", weights))
    np.fill_diagonal(weights, 0.0)

    degree = weights.sum(axis=1)
    if normalized:
        inverse_sqrt_degree = np.where(degree > 0, degree ** (-0.5), 0.0)
        # I − D^{−½} A D^{−½}, the two-sided diagonal scaling written out
        scaled = einx.multiply(
            "row, row col, col -> row col",
            inverse_sqrt_degree,
            weights,
            inverse_sqrt_degree,
        )
        laplacian = np.eye(n_vertices) - scaled
    else:
        laplacian = np.diag(degree) - weights
    laplacian = 0.5 * (laplacian + einx.id("row col -> col row", laplacian))

    eigvals, eigvecs = np.linalg.eigh(laplacian)
    # clip the tiny negatives that round-off can leave
    eigvals = np.maximum(eigvals[:count], 0.0)
    return jnp.asarray(eigvals), jnp.asarray(eigvecs[:, :count])
