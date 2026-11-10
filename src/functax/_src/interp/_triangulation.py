"""Linear interpolation on a triangulation of scattered sources: barycentric stencils.

The sources are triangulated on the host (SciPy's Delaunay), or a given mesh is
used as it is; each target is located in its simplex, and its D + 1 barycentric
weights at the simplex's vertices form a `Stencil` that applies and transposes
in JAX like every local scheme.
"""

from __future__ import annotations

from functax._src._backend import import_scipy, is_tracer, require_jax


require_jax(__name__)

import operator
from typing import Any

import einx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from functax._src.geometry._types import Points
from functax._src.interp._fit import Identity
from functax._src.interp._locate import check_outside
from functax._src.interp._neighbourhood import (
    check_scattered_policies,
    scattered_targets,
)
from functax._src.interp._plan import Plan
from functax._src.interp._policies import Policies
from functax._src.interp._schemes import Scheme
from functax._src.interp._stencil import Stencil
from functax.exceptions import GeometryError, ValidationError


__all__ = [
    "Delaunay",
]


# The barycentric tolerance of the locate in a given mesh: a target counts as
# inside a simplex when every bₖ ≥ −10⁻¹², so targets on a shared edge are found
_INSIDE_TOLERANCE = 1e-12

# Targets × simplices above which the locate in a given mesh runs in chunks
_MAX_PAIRS = 2**22


def _as_simplices(value: Any) -> tuple[tuple[int, ...], ...] | None:
    """Simplices as a tuple of tuples of ints (hashable and strict JSON), or None."""
    if value is None:
        return None
    rows = np.asarray(value)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 3:
        msg = (
            "'simplices' must be a non-empty (simplex, vertex) array with at least "
            f"3 vertices per simplex, got shape {rows.shape}"
        )
        raise ValidationError(msg)
    if not np.issubdtype(rows.dtype, np.integer):
        msg = f"'simplices' must hold integer vertex indices, got {rows.dtype}"
        raise ValidationError(msg)
    if (rows < 0).any():
        msg = "'simplices' must hold non-negative vertex indices"
        raise ValidationError(msg)
    return tuple(tuple(operator.index(index) for index in row) for row in rows)


def _affine_maps(
    vertices: np.ndarray, simplices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """T_τ⁻¹ and r_τ of each simplex, in the layout of ``Delaunay.transform``.

    With r_τ the last vertex and T_τ the D × D matrix of edges vₖ − r_τ as
    columns, (b₁, …, b_D) = T_τ⁻¹ (x − r_τ).

    Raises:
        GeometryError: If a simplex is degenerate (zero volume).
    """
    corners = vertices[simplices]
    reference = corners[:, -1]
    # columns vₖ − r_τ, k = 1, …, D
    edges = np.swapaxes(corners[:, :-1] - reference[:, None], 1, 2)
    determinants = np.linalg.det(edges)
    scale = np.abs(edges).max(axis=(1, 2)) ** edges.shape[1]
    degenerate = np.abs(determinants) <= 1e-14 * scale
    if degenerate.any():
        first = int(np.flatnonzero(degenerate)[0])
        msg = f"'simplices' row {first} is degenerate (zero volume)"
        raise GeometryError(msg)
    # the host fit of a fixed mesh: one D × D inverse per simplex, as SciPy stores
    return np.linalg.inv(edges), reference


def _locate_in_mesh(
    targets: np.ndarray, inverses: np.ndarray, references: np.ndarray
) -> np.ndarray:
    """The first simplex holding each target, or −1, by brute force on the host."""
    n_targets = targets.shape[0]
    n_simplices = inverses.shape[0]
    found = np.full(n_targets, -1, dtype=np.int64)
    chunk = max(1, _MAX_PAIRS // n_simplices)
    for start in range(0, n_targets, chunk):
        block = targets[start : start + chunk]
        # (b₁, …, b_D) = T_τ⁻¹ (x − r_τ) for every target and simplex
        offsets = block[:, None, :] - references[None, :, :]
        leading = np.einsum("srd,tsd->tsr", inverses, offsets)
        last = 1.0 - leading.sum(axis=-1)
        inside = (leading >= -_INSIDE_TOLERANCE).all(axis=-1) & (
            last >= -_INSIDE_TOLERANCE
        )
        hit = inside.any(axis=1)
        found[start : start + chunk] = np.where(hit, inside.argmax(axis=1), -1)
    return found


def _barycentric_stencil(
    coordinates: Float[Array, "target dim"],
    simplex: Int[Array, " target"],
    simplices: Int[Array, "simplex vertex"],
    inverses: Float[Array, "simplex dim dim"],
    references: Float[Array, "simplex dim"],
    row_ok: Bool[Array, " target"],
    n_source: int,
) -> Stencil:
    """bₖ = T_τ⁻¹ (x − r_τ) for k ≤ D and b_{D+1} = 1 − ∑ₖ bₖ, at τ's vertices."""
    zero_index = jnp.zeros((), simplex.dtype)
    safe_simplex = jnp.where(row_ok, simplex, zero_index)
    zero = jnp.zeros((), coordinates.dtype)
    # x − r_τ, with a target outside read at the origin of simplex 0 and masked
    offsets = coordinates - references[safe_simplex]
    offsets = einx.where("target, target dim, -> target dim", row_ok, offsets, zero)
    # (b₁, …, b_D) = T_τ⁻¹ (x − r_τ)
    leading = einx.dot(
        "target row col, target col -> target row", inverses[safe_simplex], offsets
    )
    # b_{D+1} = 1 − ∑ₖ bₖ
    last = 1.0 - einx.sum("target [row]", leading)
    weights = einx.id("target row, target -> target (row + 1)", leading, last)
    weights = einx.where("target, target tap, -> target tap", row_ok, weights, zero)
    # an invalid row reads vertex 0 with weight 0
    indices = simplices[safe_simplex]
    indices = einx.where(
        "target, target tap, -> target tap", row_ok, indices, zero_index
    )
    return Stencil(indices, weights, n_source=n_source, valid=row_ok, check=False)


class Delaunay(Scheme):
    """Linear interpolation on a triangulation of the sources: D + 1 taps per target.

    For a target x in simplex τ with inverse affine map T_τ⁻¹ and reference
    vertex r_τ (as ``scipy.spatial.Delaunay.transform`` stores them), the
    barycentric weights are

        (b₁, …, b_D) = T_τ⁻¹ (x − r_τ)  and  b_{D+1} = 1 − ∑ₖ bₖ,

    at τ's D + 1 vertices, so f̂(x) = ∑ₖ bₖ y_{vₖ}. On a plane (D = 2) these
    are the three weights of the triangle, and the plan reproduces affine
    functions exactly inside the convex hull. It equals SciPy's
    ``LinearNDInterpolator``.

    Algorithm:
        1. Triangulate the real (not padding) sources with SciPy's Delaunay,
           or take ``simplices`` as they are (a mesh).
        2. Locate each target: ``find_simplex`` on a fresh triangulation; on a
           given mesh, the first simplex whose barycentric coordinates are all
           at least −10⁻¹², by brute force over chunks of targets on the host.
        3. Weigh in JAX in the dtype of the coordinates. A target outside the
           hull (or padding, or non-finite) is invalid: its weights are zero
           and its taps point at vertex 0, so it reads ``fill_value``.

    Building the plan runs on the host and needs concrete coordinates;
    applying it traces. A fresh triangulation needs the ``functax[scipy]``
    extra; a given mesh does not.

    Args:
        simplices: The simplices of a given mesh, one row of D + 1 source
            indices each (triangles on a plane), or ``None`` for the Delaunay
            triangulation of the sources. Stored as a tuple of tuples, so the
            scheme round-trips through strict JSON.

    Raises:
        ValidationError: If ``simplices`` is not a non-empty integer
            (simplex, vertex) array with non-negative indices and at least 3
            vertices per simplex.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Points
        >>> from functax.interp import Delaunay
        >>> square = Points(
        ...     jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        ... )
        >>> targets = Points(jnp.array([[0.25, 0.5], [2.0, 2.0]]))
        >>> plan = Delaunay().plan(square, targets)
        >>> # f = 1 + 2x + 3y is affine, so it is read exactly; outside is NaN
        >>> plan(jnp.array([1.0, 3.0, 4.0, 6.0])).tolist()
        [3.0, nan]
        >>> mesh = Delaunay(simplices=[[0, 1, 2], [1, 3, 2]]).plan(square, targets)
        >>> mesh.read.n_taps, mesh.valid.tolist()
        (3, [True, False])
    """

    simplices: Any = eqx.field(static=True, default=None, converter=_as_simplices)

    def plan(
        self, source: Any, target: Any, *, policies: Policies | None = None
    ) -> Plan:
        """H for scattered ``source`` data read at ``target`` by barycentric weights.

        Args:
            source: `Points` with the ``"euclidean"`` metric and D ≥ 2
                coordinates; padding points (``valid`` False) are left out of
                a fresh triangulation, and a given simplex that touches one is
                dropped.
            target: `Points`, or an `Axis`, a `RectilinearGrid` or an
                `AffineGrid` read at its nodes; a grid keeps its shape as
                ``target_shape``.
            policies: ``outside`` (``"nan"`` or ``"raise"``) acts on targets
                outside the hull; ``missing`` and ``fill_value`` as for any
                plan.

        Returns:
            The `Plan`, with an (M, D + 1) `Stencil` read-out and the identity
            fit.

        Raises:
            GeometryError: If the source is not euclidean `Points` with D ≥ 2,
                the target has another dimension, a given simplex has the
                wrong number of vertices or is degenerate.
            ValidationError: If the coordinates are traced, or a given simplex
                indexes past the sources.
            PolicyError: Under ``outside="clamp"`` or ``"extrapolate"``.
            OutsideDomainError: Under ``outside="raise"`` when a target is
                outside the hull.
            MissingExtraError: If SciPy (the ``functax[scipy]`` extra) is
                missing and no ``simplices`` are given.
        """
        policies = Policies() if policies is None else policies
        check_scattered_policies(policies, "Delaunay")
        if not isinstance(source, Points):
            msg = f"'source' must be Points, got {type(source).__name__}"
            raise GeometryError(msg)
        if source.metric != "euclidean":
            msg = (
                "Delaunay triangulates in the plane: 'source' must have the "
                f"'euclidean' metric, got {source.metric!r}"
            )
            raise GeometryError(msg)
        n_source, n_dims = source.xy.shape
        if n_dims < 2:
            msg = f"'source' must have at least 2 coordinates per point, got {n_dims}"
            raise GeometryError(msg)
        coordinates, target_shape, target_valid = scattered_targets(target)
        if coordinates.shape[1] != n_dims:
            msg = (
                f"'target' has {coordinates.shape[1]} coordinates per point, but "
                f"'source' has {n_dims}"
            )
            raise GeometryError(msg)
        leaves = (source.xy, source.valid, coordinates, target_valid)
        if any(is_tracer(leaf) for leaf in leaves):
            msg = (
                "Delaunay builds its plan on the host and needs concrete source "
                "and target coordinates; build it outside jit"
            )
            raise ValidationError(msg)

        # --- triangulate or take the mesh, on the host ---
        vertices = np.asarray(source.xy, dtype=np.float64)
        targets = np.asarray(coordinates, dtype=np.float64)
        real = (
            np.ones(n_source, dtype=bool)
            if source.valid is None
            else np.asarray(source.valid)
        )
        finite = np.asarray(np.isfinite(targets).all(axis=1))
        queries = np.where(finite[:, None], targets, 0.0)
        if self.simplices is None:
            spatial = import_scipy("spatial", needed_by="Delaunay.plan")
            kept = np.flatnonzero(real)
            triangulation = spatial.Delaunay(vertices[kept])
            simplices = kept[triangulation.simplices]
            inverses = triangulation.transform[:, :n_dims]
            references = triangulation.transform[:, n_dims]
            simplex = triangulation.find_simplex(queries)
        else:
            simplices = np.asarray(self.simplices)
            if simplices.shape[1] != n_dims + 1:
                msg = (
                    f"'simplices' must have {n_dims + 1} vertices per simplex for "
                    f"{n_dims}-D sources, got {simplices.shape[1]}"
                )
                raise GeometryError(msg)
            if simplices.max() >= n_source:
                msg = (
                    f"'simplices' must index the {n_source} sources, got index "
                    f"{int(simplices.max())}"
                )
                raise ValidationError(msg)
            simplices = simplices[real[simplices].all(axis=1)]
            inverses, references = _affine_maps(vertices, simplices)
            simplex = _locate_in_mesh(queries, inverses, references)

        # --- weigh in JAX, in the dtype of the coordinates ---
        row_ok = (simplex >= 0) & finite
        if target_valid is not None:
            row_ok = row_ok & np.asarray(target_valid)
        check_outside(row_ok, policies)
        dtype = coordinates.dtype
        stencil = _barycentric_stencil(
            coordinates,
            jnp.asarray(simplex, dtype=jnp.int32),
            jnp.asarray(simplices, dtype=jnp.int32),
            jnp.asarray(inverses, dtype=dtype),
            jnp.asarray(references, dtype=dtype),
            jnp.asarray(row_ok),
            n_source,
        )
        return Plan(Identity(), stencil, policies, None, target_shape)
