# Bases from data, graphs and meshes

Role: ① primitive (`role-ladder`, `references/primitive.md`).
Unit: basis.

Open this when no analytic basis fits the domain: the modes come from an ensemble of fields, from a graph, or from a triangle mesh.

## Names

All three live in `functax.basis.eigen`.

| Name | What it gives |
|---|---|
| `eof_basis(data, n_modes, *, center=True)` | The leading empirical orthogonal functions (EOFs, the PCA modes) of a `(sample, grid)` data matrix: `(Φ, singular_values)`, with Φ of shape `(grid, n_modes)` and orthonormal columns. |
| `graph_laplacian_eigpairs(adjacency, num_basis, *, normalized=True)` | The smallest eigenpairs of an undirected graph's Laplacian: `(eigenvalues, eigenvectors)`. |
| `cotangent_laplacian_eigpairs(vertices, triangles, num_basis)` | The smallest eigenpairs of the cotangent Laplace–Beltrami operator on a triangle mesh: `(eigenvalues, eigenvectors, vertex_areas)`. |

## Use it

```python
import jax.numpy as jnp
import jax.random as jr

from functax.basis.eigen import (
    cotangent_laplacian_eigpairs,
    eof_basis,
    graph_laplacian_eigpairs,
)
from functax.geometry import icosphere

# 40 snapshots of a field on 30 grid points, built from two patterns
grid = jnp.linspace(0.0, 1.0, 30)
patterns = jnp.stack([jnp.sin(jnp.pi * grid), jnp.sin(2 * jnp.pi * grid)])  # (2, 30)
amplitudes = jr.normal(jr.key(0), (40, 2))
snapshots = amplitudes @ patterns  # (40, 30)
Phi, singular_values = eof_basis(snapshots, n_modes=2)  # (30, 2), (2,)
assert jnp.allclose(Phi.T @ Phi, jnp.eye(2), atol=1e-4)
# two modes explain the centred data
centred = snapshots - snapshots.mean(axis=0)
assert jnp.allclose(centred @ Phi @ Phi.T, centred, atol=1e-3)

# the 3-node path graph: D − A has eigenvalues 0, 1 and 3
path = jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
values, vectors = graph_laplacian_eigpairs(path, 3, normalized=False)
assert jnp.allclose(values, jnp.array([0.0, 1.0, 3.0]), atol=1e-4)

# harmonics of a mesh: on the sphere the first eigenvalue is 0
vertices, triangles = icosphere(2)
eigenvalues, modes, areas = cotangent_laplacian_eigpairs(vertices, triangles, 4)
assert modes.shape == (vertices.shape[0], 4) and abs(float(eigenvalues[0])) < 1e-3
```

## Rules

- `eof_basis` takes one row per sample and one column per grid point. Flatten the spatial dimensions first.
- It removes the mean over samples by default (`center=True`), so add the mean back when you reconstruct a field.
- The EOF columns are orthonormal, so the coefficients of a new field are `Φ.T @ (field − mean)`; no solve is needed.
- Use the graph Laplacian for features on an irregular network and the cotangent Laplacian for a surface given as a mesh; both return the smooth modes first.
- The mesh eigenvectors are orthonormal in the vertex-area inner product, which is why the areas come back with them.
