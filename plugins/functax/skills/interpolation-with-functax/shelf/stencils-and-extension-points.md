# Stencils and extension points

Role: ① primitive (`role-ladder`, `references/primitive.md`).
Unit: interp.

Open this when you write a new scheme, or need the pieces a plan is made of: the sparse read-out, the per-axis bases, the fits and the functions that locate a target.

## Names

All of them live in `functax.interp`. A plan is a fit 𝒮 followed by a read-out H.

| Name | What it is |
|---|---|
| `Readout` | The base of a read-out: `apply(coefficients)`, `apply_adjoint(residual)`, `operator()`, and `valid`. |
| `Stencil(indices, weights, *, n_source, valid=None)` | The sparse read-out: row k reads S source values at flat indices. `to_dense()`, `Stencil.from_dense(matrix, max_taps=...)`, and `outer @ inner`. |
| `compose(outer, inner, *, merge=False)` | The product of two stencils as one stencil. |
| `tensor_stencil(source, target, bases, *, policies)` | H = H₁ ⊗ … ⊗ H_D from one basis per source axis: what every local scheme calls. |
| `FourierReadout` | The read-out of `Fourier`: band-limited resampling by two FFTs. |
| `Basis` | A local basis on one axis: its `taps`, its `support`, its `stencil(offsets)` and its `prefilter_order`. |
| `NearestBasis`, `LinearBasis`, `CubicBasis`, `BSplineBasis` | The four bases behind `Nearest`, `Linear`, `Cubic` and `BSpline`. |
| `bspline(values, order)` | The centred B-spline βᵏ at given offsets. |
| `Fit` | The base of a fit: `apply(data)`, `apply_adjoint(coefficients)`, `data_size`. |
| `Identity` | The fit of every local scheme, c = y. |
| `BandedPrefilter`, `spline_poles(order)` | The B-spline prefilter c = B⁻¹ y, per axis, and its poles. |
| `HermiteFit`, `ChebyshevFit` | The fits of the 1-D series schemes and of `Chebyshev`. |
| `ReadGeometry`, `LocalGeometry`, `HermiteGeometry` | What a plan's read-out was built from, kept so that `plan.derivative` can rebuild it. |
| `locate(axis, positions, *, extent="nodes")` | The cell, the offset in [0, 1) and the inside flag of each target on an axis. |
| `index_positions(axis, positions)` | The fractional index of each target, with node j at j. |
| `locate_curvilinear(grid, positions)` | The same on a `CurvilinearGrid`: a nearest-cell search, then Newton steps. |
| `remap_taps(taps, n_nodes, boundary, *, periodic=False)` | Tap indices brought back onto the axis by a boundary rule. |
| `check_outside(inside, policies)` | Enforces `outside="raise"` on the host. |
| `clenshaw(x, coefficients, *, interval=(-1.0, 1.0))` | A Chebyshev series evaluated by Clenshaw's recurrence. |

## Use it

```python
import jax.numpy as jnp

from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import (
    LinearBasis,
    NearestBasis,
    Policies,
    Stencil,
    clenshaw,
    compose,
    locate,
    tensor_stencil,
)

# a stencil by hand: the midpoints of a 4-sample signal
H = Stencil(jnp.array([[0, 1], [1, 2], [2, 3]]), jnp.full((3, 2), 0.5), n_source=4)
assert H.apply(jnp.array([0.0, 2.0, 4.0, 6.0])).tolist() == [1.0, 3.0, 5.0]
assert H.apply_adjoint(jnp.ones(3)).tolist() == [0.5, 1.0, 1.0, 0.5]

# midpoints of midpoints, as one stencil
outer = Stencil(jnp.array([[0, 1], [1, 2]]), jnp.full((2, 2), 0.5), n_source=3)
product = compose(outer, H, merge=True)
assert product.weights.tolist() == [[0.25, 0.5, 0.25], [0.25, 0.5, 0.25]]
assert jnp.allclose((outer @ H).to_dense(), product.to_dense())

# where a target falls on an axis
cell, offset, inside = locate(Axis(jnp.arange(4.0)), jnp.array([1.25, 3.0, 3.25]))
assert cell.tolist() == [1, 3, 3] and inside.tolist() == [True, True, False]

# the stencil of a scheme of your own: nearest on one axis, linear on the other
grid = RectilinearGrid((Axis(jnp.arange(3.0)), Axis(jnp.arange(4.0))))
target = Points(jnp.array([[1.2, 2.5]]))
mixed = tensor_stencil(
    grid, target, (NearestBasis(), LinearBasis()), policies=Policies()
)
assert mixed.n_taps == 2
assert jnp.allclose(
    mixed.apply(jnp.arange(12.0)), 6.5
)  # row 1, between columns 2 and 3

# T₀ + 2 T₁ at x = 0.5
assert jnp.allclose(clenshaw(jnp.array(0.5), jnp.array([1.0, 2.0])), 2.0)
```

## Rules

- Reach for these only when no scheme fits. Composing `Tensor`, `plan.derivative`, `plan[window]` and `outer @ inner` covers most needs without a new class.
- A new local scheme is a `LocalScheme` with a `bases(n_axes)` method that returns one `Basis` per source axis; the stencil, the policies and the adjoint then come for free.
- A new global scheme supplies a `Fit` and a `Readout`. Write `apply_adjoint` by hand for both, and check it with the dot-product test ⟨Hx, r⟩ = ⟨x, Hᵀr⟩.
- Build stencils on the host. `Stencil(..., check=True)` validates the indices there; functax's own builders pass `check=False` because the check would become a host callback under `jit`.
- `compose(..., merge=True)` needs concrete indices, so it cannot run on traced stencils.
- Keep a scheme free of data and geometry, with `get_config()` returning strict JSON, so that plans can be cached by `plan_key`.
