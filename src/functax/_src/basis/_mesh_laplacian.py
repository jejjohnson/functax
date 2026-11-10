"""The eigenbasis of the cotangent Laplace–Beltrami operator on a triangle mesh.

Unlike a combinatorial graph Laplacian, the cotangent Laplacian with a mass
matrix is a consistent discretisation of the Laplace–Beltrami operator, so its
eigenpairs converge to the surface's eigenfunctions as the mesh is refined.
Moved from geonnax ``_basis/_mesh`` at ``be5537d``.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import operator

import einx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float, Int

from functax._src.geometry._mesh import (
    _corner_geometry,
    _lumped_areas,
    _validate_mesh,
)
from functax.exceptions import GeometryError, ValidationError


__all__ = ["cotangent_laplacian_eigpairs"]


def cotangent_laplacian_eigpairs(
    vertices: Float[ArrayLike, "vertex 3"],
    triangles: Int[ArrayLike, "triangle 3"],
    num_basis: int,
) -> tuple[
    Float[Array, " feature"], Float[Array, "vertex feature"], Float[Array, " vertex"]
]:
    """The smallest eigenpairs of the cotangent Laplace–Beltrami operator.

    The cotangent Laplacian (Pinkall and Polthier 1993) weights each edge
    (i, j) by the cotangents of the two angles α_ij, β_ij opposite it (one for
    a boundary edge): L_ij = −½ (cot α_ij + cot β_ij) on an edge and
    L_ii = −∑_{j ≠ i} L_ij, a symmetric positive semidefinite stiffness matrix
    with the constants in its null space. With the lumped mass M = diag(a) of
    `mesh_vertex_areas`, the eigenproblem is L φ = λ M φ, solved as
    (M^{−½} L M^{−½}) ψ = λ ψ with φ = M^{−½} ψ, so the eigenvectors are
    M-orthonormal: Φᵀ M Φ = I. On a closed mesh λ₀ = 0 with
    φ₀ = 1/√(∑_v a_v); on the unit sphere λ → ℓ(ℓ + 1) with multiplicity
    2ℓ + 1 under refinement. A mesh with a boundary gets natural (Neumann)
    conditions. This is the eigenbasis behind NORM (Chen et al. 2024).

    Algorithm:
        1. Check the mesh; compute each triangle's area and corner cotangents.
        2. Assemble L, adding ½ cot θ to the edge opposite each corner.
        3. Diagonalise M^{−½} L M^{−½} densely on the host in float64, which
           is O(V³) time and O(V²) memory, practical up to a few thousand
           vertices.
        4. Keep the M smallest pairs, clip round-off negatives to 0, and map
           ψ back to φ = M^{−½} ψ.

    Args:
        vertices: The vertex positions.
        triangles: Vertex indices, three per triangle, in either orientation.
            Every vertex must belong to a triangle.
        num_basis: M, the number of eigenpairs, from 1 to V.

    Returns:
        ``(eigvals, eigvecs, areas)``: the M eigenvalues ascending, the
        M-orthonormal eigenvectors (column m is φ_m at the vertices), and the
        vertex areas a, the mass diagonal and quadrature weights. The analysis
        of a vertex field u is ``eigvecs.T @ (areas * u)`` and the synthesis
        ``eigvecs @ coefficients``.

    Raises:
        GeometryError: If a shape is wrong, an index is out of range, a vertex
            is not finite or in no triangle, or a triangle has zero area.
        ValidationError: If ``num_basis`` is not an integer in [1, V].

    References:
        Pinkall, U., Polthier, K. (1993). Computing discrete minimal surfaces
        and their conjugates. Experimental Mathematics 2(1), 15–36.
        Meyer, M., Desbrun, M., Schröder, P., Barr, A. H. (2003). Discrete
        differential-geometry operators for triangulated 2-manifolds.
        Visualization and Mathematics III, 35–57.
        Chen, G., et al. (2024). Learning neural operators on Riemannian
        manifolds. National Science Open; arXiv:2302.08166.

    Examples:
        >>> from functax.basis.eigen import cotangent_laplacian_eigpairs
        >>> from functax.geometry import icosphere
        >>> vertices, triangles = icosphere(3)
        >>> eigvals, eigvecs, areas = cotangent_laplacian_eigpairs(
        ...     vertices, triangles, num_basis=9
        ... )
        >>> eigvals.shape, eigvecs.shape, areas.shape
        ((9,), (642, 9), (642,))
        >>> [round(float(v), 1) for v in eigvals]
        [0.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0, 6.0]
    """
    points, corners = _validate_mesh(vertices, triangles)
    n_vertices = points.shape[0]
    try:
        count = operator.index(num_basis)
    except TypeError as err:
        msg = f"'num_basis' must be an integer in [1, {n_vertices}], got {num_basis!r}"
        raise ValidationError(msg) from err
    if isinstance(num_basis, bool) or not 1 <= count <= n_vertices:
        msg = f"'num_basis' must be an integer in [1, {n_vertices}], got {num_basis!r}"
        raise ValidationError(msg)
    double_area, cot = _corner_geometry(points, corners)
    areas = _lumped_areas(n_vertices, corners, double_area)
    if np.any(areas <= 0):
        msg = (
            f"{int(np.sum(areas <= 0))} vertex(es) belong to no triangle; "
            "remove unreferenced vertices first"
        )
        raise GeometryError(msg)

    # --- assemble L: corner k's cotangent weights the opposite edge ---
    head = np.roll(corners, -1, axis=1).ravel()
    tail = np.roll(corners, -2, axis=1).ravel()
    half_cot = 0.5 * cot.ravel()
    weights = np.zeros((n_vertices, n_vertices))
    np.add.at(weights, (head, tail), half_cot)
    np.add.at(weights, (tail, head), half_cot)
    laplacian = np.diag(weights.sum(axis=1)) - weights
    laplacian = 0.5 * (laplacian + einx.id("row col -> col row", laplacian))

    # --- solve the symmetric problem M^{−½} L M^{−½} ψ = λ ψ ---
    inverse_sqrt_mass = 1.0 / np.sqrt(areas)
    symmetric = einx.multiply(
        "row, row col, col -> row col", inverse_sqrt_mass, laplacian, inverse_sqrt_mass
    )
    symmetric = 0.5 * (symmetric + einx.id("row col -> col row", symmetric))
    eigvals, psi = np.linalg.eigh(symmetric)
    eigvals = np.maximum(eigvals[:count], 0.0)
    # φ = M^{−½} ψ
    eigvecs = einx.multiply(
        "vertex, vertex feature -> vertex feature", inverse_sqrt_mass, psi[:, :count]
    )
    return jnp.asarray(eigvals), jnp.asarray(eigvecs), jnp.asarray(areas)
