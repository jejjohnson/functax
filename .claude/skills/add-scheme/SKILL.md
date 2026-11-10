---
name: add-scheme
description: Add an interpolation scheme, a basis, a quadrature rule, a transform or a geometry type to functax — logic under functax/_src/<area>/, a re-export from the public area module, require_jax on import, an eqx.Module with __check_init__, an executable docstring, the SciPy oracle test, the dot-product test for a linear read-out, the gradient and transform-invariance tests, the API page and the capability index. Use when asked to add, port or implement anything in src/functax.
---

# Add a scheme, basis, rule, transform or geometry type

Read "The contracts", "JAX and Equinox" and "Boundaries" in `AGENTS.md` first; this is the step-by-step. Then read the live issue (`gh issue view <N>`): it names the oracle, the tolerance and the gate, and it is newer than the design page.

## 1. Make sure it does not exist yet

- Search `docs/api/capabilities.md` (functax and the upstream sections for JAX, Equinox, lineax and einx) and the "Reuse before you write" table in `AGENTS.md`.
- Code moving in from another stack library (geonnax, spectraldiffx, kernellib) is moved, not rewritten: keep its behaviour and add the parity test its wave's gate names (G1-parity: equal to the source at its pinned commit to 1e-12 relative in float64).

## 2. Pick its home

| It is… | Private home | Public module |
|---|---|---|
| An axis, grid, point set, the sphere, affine raster maths | `src/functax/_src/geometry/` | `functax.geometry` |
| A basis φⱼ(x; θ), Fourier features | `src/functax/_src/basis/` | `functax.basis` |
| A derivative of a continuous field | `src/functax/_src/diffops/` | `functax.diffops` |
| A stencil, locate, scheme or plan | `src/functax/_src/interp/` | `functax.interp` |
| A fit, projection or gap fill | `src/functax/_src/project/` | `functax.project` |
| A quadrature rule | `src/functax/_src/quad/` | `functax.quad` |
| A spectral or non-uniform transform | `src/functax/_src/transform/` | `functax.transform` |

One concept per file: a protocol and its implementations share a file. The first feature in an area creates the public module (`src/functax/<area>.py`, or a package when it has a NumPy-only submodule, as `geometry` does).

## 3. Write it

- **Import check.** A module that needs JAX starts with `from functax._src._backend import require_jax` and `require_jax(__name__)` before its first JAX import. Only the NumPy-only base (`geometry.affine` and what it imports) skips it, and it imports nothing from the extra.
- **Imports** under the fixed aliases: `import einx`, `import equinox as eqx`, `import jax`, `import jax.numpy as jnp`, `import lineax as lx`, `from jaxtyping import Array, Float, Int`.
- **State in an `eqx.Module`.** Arrays are leaves; configuration (an order, a boundary policy, a shape) is `eqx.field(static=True)`. Static checks in `__check_init__`, raising `ValidationError` that names the field; checks on traced values with `eqx.error_if`.
- **The apply traces.** No Python control flow on traced values, static output shapes, no host round-trip. A gather masks targets outside the source explicitly; a `jnp.where` guards its untaken branch. The fit may use NumPy on the host.
- **Precision is the caller's.** Preserve the input dtype; constants in the input's dtype.
- **einx** for axis work, **lineax** for solves. A linear read-out stores its rows and has a hand-written adjoint (a scatter-add), not `jax.linear_transpose` at apply time.
- **Docstring** in the `python-style` layout: one-line summary (the capability-index entry), the defining equation in Unicode, `Algorithm:` with numbered steps, `Args:` with meaning and symbol, `Returns:`, `Raises:`, two or three `Examples:` whose outputs are shapes, `allclose` booleans or reprs, and `References:` when a paper defines the method.
- Comments above the line, the Unicode equation above each mathematical line, descriptive names, one operation per line.

## 4. Export and document

- Re-export from the area module's sorted `__all__`. Nothing is added to the top-level `functax.__all__`.
- A **new area module** joins `SUBMODULES` (a package, `SUBPACKAGES`) in `tests/fast/test_public_api.py`, and gets `docs/api/<area>.md` with a `::: functax.<area>` block, a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`.
- A new module in the NumPy-only base joins `BASE_MODULES` in `tests/integration/test_install.py` and the import in the `base-install` job of `ci.yml`.
- `uv run poe capabilities`.

## 5. Tests

In `tests/fast/<area>/test_<name>.py`, and in `tests/slow/<area>/test_<name>.py` for any test that takes over a second (the oracle sweeps across orders and policies, gradient and trace checks; "Test tiers" in `AGENTS.md`), with x64 on (from `conftest.py`), using the helpers in `tests/_harness.py` ("Testing a scheme" in `docs/contributing.md`):

- **Oracle** (`assert_matches_oracle`): equal to the SciPy reference the issue names (`scipy.ndimage.map_coordinates`, `scipy.interpolate.RegularGridInterpolator`, `scipy.integrate`, …) at the stated tolerance, 1e-12 relative in float64 unless the issue says otherwise, on interior targets.
- **Properties** with hypothesis: reproduces constants (and linear fields, for linear and higher schemes), weights sum to one, a rule integrates its exact degree.
- **Dot-product test** (`assert_adjoint`, then `assert_transpose_matches`) for a linear read-out: ⟨Hx, r⟩ = ⟨x, Hᵀr⟩ to 1e-12 relative on random x and r from a fixed key.
- **Gradient check** (`assert_grad_matches_fd`): `jax.jacrev` against central differences, in the target positions and, for a basis, in θ.
- **Transform invariance** (`assert_traces`): eager, `jit` and `vmap` agree with a Python loop to 1e-12 relative.
- **Edges:** targets on the boundary and outside it (each outside policy), a single cell, a descending axis, float32 input keeps float32, each invalid argument raises `ValidationError` (`match=` on the field name).

## 6. Verify

```bash
uv run pytest --no-cov tests/fast/<area>/test_<x>.py tests/slow/<area>/test_<x>.py src/functax/_src/<area>/_<x>.py tests/fast/test_public_api.py tests/fast/test_capabilities.py -v
```

then the `pre-pr-check` skill.
