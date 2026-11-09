"""Shared scheme checks: oracle, adjoint, transpose, traces and gradient.

Each helper raises `AssertionError` with the worst index and the values there,
so a failing scheme test says where it fails, not just that it does. The
tolerances default to the policy in `docs/contributing.md` ("Testing a
scheme"): 1e-12 relative in float64 for exact schemes; a scheme with a solve
states its own.

The helpers need JAX's 64-bit mode, which `tests/conftest.py` turns on for the
whole suite, and they refuse to run without it rather than test in float32.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree


__all__ = [
    "assert_adjoint",
    "assert_grad_matches_fd",
    "assert_matches_oracle",
    "assert_traces",
    "assert_transpose_matches",
]


def _require_x64() -> None:
    """Fail when JAX's 64-bit mode is off, since every tolerance assumes float64."""
    if not jax.config.read("jax_enable_x64"):
        raise AssertionError(
            "the oracle harness needs float64: enable jax_enable_x64 "
            "(tests/conftest.py does this for the suite)"
        )


def assert_matches_oracle(
    actual: Any,
    expected: Any,
    *,
    rtol: float,
    atol: float = 0.0,
    where: Any = None,
) -> None:
    """Assert ``|actual − expected| ≤ atol + rtol·|expected|`` wherever ``where`` holds.

    NaNs match only NaNs at the same index, so a scheme that marks outside
    targets with NaN is compared on that too. Boolean and integer arrays (masks,
    indices) must match exactly, whatever the tolerances.

    Args:
        actual: The values under test.
        expected: The reference values (SciPy, a closed form), the same shape.
        rtol: The relative tolerance.
        atol: The absolute tolerance, for references that cross zero.
        where: A boolean mask of the same shape; only ``True`` entries are
            compared (interior targets, say). ``None`` compares every entry.

    Raises:
        AssertionError: If the shapes differ or any compared entry is out of
            tolerance. The message gives the number of failures, the worst
            index, both values there, the error and the tolerance.

    Examples:
        >>> import numpy as np
        >>> assert_matches_oracle(
        ...     np.array([1.0, 2.0]), np.array([1.0, 2.0]), rtol=1e-12
        ... )
        >>> assert_matches_oracle(
        ...     np.array([1.0, 9.0]),
        ...     np.array([1.0, 2.0]),
        ...     rtol=1e-12,
        ...     where=np.array([True, False]),
        ... )
    """
    actual_values = np.asarray(actual)
    expected_values = np.asarray(expected)
    if actual_values.shape != expected_values.shape:
        raise AssertionError(
            f"shape mismatch: actual {actual_values.shape}, "
            f"expected {expected_values.shape}"
        )
    if where is None:
        mask = np.ones(actual_values.shape, dtype=bool)
    else:
        mask = np.broadcast_to(np.asarray(where, dtype=bool), actual_values.shape)
    floating = np.issubdtype(actual_values.dtype, np.inexact) or np.issubdtype(
        expected_values.dtype, np.inexact
    )
    if not floating:
        # Booleans and integers (masks, indices) must match exactly.
        actual_values = actual_values.astype(np.float64)
        expected_values = expected_values.astype(np.float64)
        rtol, atol = 0.0, 0.0

    # e = |a − b|; a NaN on one side only counts as an infinite error
    error = np.abs(actual_values - expected_values)
    both_nan = np.isnan(actual_values) & np.isnan(expected_values)
    error = np.where(both_nan, 0.0, error)
    error = np.where(np.isnan(error), np.inf, error)
    # τ = atol + rtol·|b|; a NaN reference contributes no tolerance
    reference_scale = np.where(np.isnan(expected_values), 0.0, expected_values)
    tolerance = atol + rtol * np.abs(reference_scale)
    failing = mask & (error > tolerance)
    if not failing.any():
        return

    # The worst entry is the one furthest past its own tolerance.
    excess = np.where(failing, error - tolerance, -np.inf)
    worst = np.unravel_index(np.argmax(excess), excess.shape)
    raise AssertionError(
        f"{int(failing.sum())} of {int(mask.sum())} compared entries out of "
        f"tolerance; worst at index {tuple(int(i) for i in worst)}: "
        f"actual {actual_values[worst]!r}, expected {expected_values[worst]!r}, "
        f"error {error[worst]:.3e} > tolerance {tolerance[worst]:.3e} "
        f"(rtol={rtol}, atol={atol})"
    )


def assert_adjoint(
    apply: Callable[[jax.Array], jax.Array],
    apply_adjoint: Callable[[jax.Array], jax.Array],
    n_source: int,
    n_target: int,
    *,
    key: jax.Array,
    rtol: float = 1e-12,
) -> None:
    """Assert the dot-product test ⟨Hx, r⟩ = ⟨x, Gr⟩ on random x and r.

    Draws x ~ 𝒩(0, I_N) and r ~ 𝒩(0, I_M) in float64 and checks
    |⟨Hx, r⟩ − ⟨x, Gr⟩| ≤ rtol · max(‖Hx‖ ‖r‖, ‖x‖ ‖Gr‖). A correct adjoint
    gives about 1e-15; a wrong one gives O(1).

    Args:
        apply: H, a linear map from ℝᴺ to ℝᴹ.
        apply_adjoint: G, the claimed adjoint Hᵀ, from ℝᴹ to ℝᴺ.
        n_source: N, the length of x.
        n_target: M, the length of Hx.
        key: The PRNG key for x and r.
        rtol: The relative tolerance.

    Raises:
        AssertionError: If x64 is off, an output has the wrong shape, or the
            relative error exceeds ``rtol``; the message gives both inner
            products and the relative error.

    Examples:
        >>> import jax, jax.numpy as jnp
        >>> H = jnp.array([[1.0, 2.0, 0.0], [0.0, 1.0, -1.0]])
        >>> assert_adjoint(
        ...     lambda x: H @ x, lambda r: H.T @ r, 3, 2, key=jax.random.key(0)
        ... )
    """
    _require_x64()
    key_source, key_target = jax.random.split(key)
    # x ~ 𝒩(0, I_N), r ~ 𝒩(0, I_M)
    source = jax.random.normal(key_source, (n_source,), dtype=jnp.float64)
    residual = jax.random.normal(key_target, (n_target,), dtype=jnp.float64)

    forward = apply(source)
    backward = apply_adjoint(residual)
    if forward.shape != (n_target,):
        raise AssertionError(f"H x has shape {forward.shape}, expected ({n_target},)")
    if backward.shape != (n_source,):
        raise AssertionError(f"Hᵀ r has shape {backward.shape}, expected ({n_source},)")

    # ⟨Hx, r⟩ and ⟨x, Gr⟩
    lhs = float(jnp.vdot(forward, residual))
    rhs = float(jnp.vdot(source, backward))
    # the larger of ‖Hx‖ ‖r‖ and ‖x‖ ‖Gr‖, so a G that is too large also fails
    scale_forward = float(jnp.linalg.norm(forward) * jnp.linalg.norm(residual))
    scale_backward = float(jnp.linalg.norm(source) * jnp.linalg.norm(backward))
    scale = max(scale_forward, scale_backward)
    if scale == 0.0:
        return
    relative_error = abs(lhs - rhs) / scale
    if relative_error > rtol:
        raise AssertionError(
            f"dot-product test failed: <Hx, r> = {lhs!r}, <x, H^T r> = {rhs!r}, "
            f"relative error {relative_error:.3e} > rtol {rtol}"
        )


def assert_transpose_matches(
    apply: Callable[[jax.Array], jax.Array],
    apply_adjoint: Callable[[jax.Array], jax.Array],
    x_example: jax.Array,
    *,
    key: jax.Array | None = None,
    rtol: float = 1e-12,
) -> None:
    """Assert the hand-written adjoint equals ``jax.linear_transpose`` of H.

    Args:
        apply: H, a linear map.
        apply_adjoint: G, the claimed adjoint.
        x_example: An input of H's shape and dtype (its values are unused).
        key: The PRNG key for the residual r; a fixed key when ``None``.
        rtol: The relative tolerance, applied entrywise with an absolute floor
            of ``rtol·max|Hᵀr|`` so entries near zero do not fail on round-off.

    Raises:
        AssertionError: If x64 is off or G r differs from the transpose; the
            message gives the worst index.

    Examples:
        >>> import jax.numpy as jnp
        >>> H = jnp.array([[1.0, 2.0, 0.0], [0.0, 1.0, -1.0]])
        >>> assert_transpose_matches(
        ...     lambda x: H @ x, lambda r: H.T @ r, jnp.zeros(3)
        ... )
    """
    _require_x64()
    if key is None:
        key = jax.random.key(0)
    output_example = jax.eval_shape(apply, x_example)
    residual = jax.random.normal(key, output_example.shape, dtype=output_example.dtype)
    (transpose,) = jax.linear_transpose(apply, x_example)(residual)
    adjoint = apply_adjoint(residual)
    scale = float(jnp.max(jnp.abs(transpose))) if transpose.size else 0.0
    assert_matches_oracle(adjoint, transpose, rtol=rtol, atol=rtol * scale)


def _batch_size(args: tuple[Any, ...]) -> int:
    """The common leading length of every array leaf in ``args``."""
    sizes = {leaf.shape[0] for leaf in jax.tree.leaves(args) if np.ndim(leaf) > 0}
    if len(sizes) != 1:
        raise AssertionError(
            f"every array argument needs the same leading batch axis, got {sizes}"
        )
    return sizes.pop()


def _assert_trees_match(actual: Any, expected: Any, *, rtol: float, what: str) -> None:
    """Compare two pytrees leaf by leaf, naming the leaf and ``what`` on failure."""
    actual_leaves, actual_tree = jax.tree.flatten(actual)
    expected_leaves, expected_tree = jax.tree.flatten(expected)
    if actual_tree != expected_tree:
        raise AssertionError(f"{what}: tree structure {actual_tree} != {expected_tree}")
    for index, (got, want) in enumerate(
        zip(actual_leaves, expected_leaves, strict=True)
    ):
        try:
            assert_matches_oracle(got, want, rtol=rtol)
        except AssertionError as err:
            raise AssertionError(f"{what}, leaf {index}: {err}") from err


def assert_traces(fn: Callable[..., Any], *args: Any, rtol: float = 1e-12) -> None:
    """Assert ``jit(fn)``, ``vmap(fn)`` and an eager loop agree over a batch.

    Every array leaf of ``args`` carries a leading batch axis of the same
    length B. The reference is the eager loop ``[fn(*args[b]) for b in B]``,
    stacked; ``jit(fn)`` on each slice, ``vmap(fn)`` and ``jit(vmap(fn))`` on
    the batch must match it.

    Args:
        fn: The function under test; it may take and return pytrees.
        *args: The batched arguments.
        rtol: The relative tolerance.

    Raises:
        AssertionError: If x64 is off, the batch axes differ, or any variant
            disagrees with the eager loop; the message names the variant.

    Examples:
        >>> import jax.numpy as jnp
        >>> assert_traces(lambda x: jnp.sin(x) * 2.0, jnp.linspace(0.0, 1.0, 4))
    """
    _require_x64()
    batch = _batch_size(args)

    def slice_at(index: int) -> tuple[Any, ...]:
        return jax.tree.map(lambda leaf: leaf[index], args)

    eager = [fn(*slice_at(index)) for index in range(batch)]
    reference = jax.tree.map(lambda *leaves: jnp.stack(leaves), *eager)

    jitted_fn = jax.jit(fn)
    jitted = [jitted_fn(*slice_at(index)) for index in range(batch)]
    jitted_stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *jitted)
    _assert_trees_match(jitted_stacked, reference, rtol=rtol, what="jit(fn)")

    vmapped = jax.vmap(fn)(*args)
    _assert_trees_match(vmapped, reference, rtol=rtol, what="vmap(fn)")

    jit_vmapped = jax.jit(jax.vmap(fn))(*args)
    _assert_trees_match(jit_vmapped, reference, rtol=rtol, what="jit(vmap(fn))")


def assert_grad_matches_fd(
    fn: Callable[[Any], Any],
    x: Any,
    *,
    eps: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    """Assert ``jax.jacrev(fn)`` agrees with central finite differences at x.

    For each input entry k, (f(x + ε eₖ) − f(x − ε eₖ)) / 2ε approximates
    column k of the Jacobian. Inputs and outputs may be any pytree of arrays;
    both are flattened. The tolerance is relative to the largest entry of the
    finite-difference Jacobian, since single entries near zero carry the
    O(ε²) truncation error at full size.

    Args:
        fn: The differentiable function under test.
        x: The point, a float64 array or pytree of them.
        eps: The step ε.
        rtol: The relative tolerance.

    Raises:
        AssertionError: If x64 is off or the Jacobians differ; the message
            gives the worst (output, input) index.

    Examples:
        >>> import jax.numpy as jnp
        >>> assert_grad_matches_fd(
        ...     lambda x: jnp.sum(jnp.sin(x)), jnp.array([0.1, 0.7])
        ... )
    """
    _require_x64()
    flat_x, unravel = ravel_pytree(x)

    def flat_fn(flat_input: jax.Array) -> jax.Array:
        output, _ = ravel_pytree(fn(unravel(flat_input)))
        return output

    jacobian = jax.jacrev(flat_fn)(flat_x)

    def central_difference(step: jax.Array) -> jax.Array:
        # (f(x + ε eₖ) − f(x − ε eₖ)) / 2ε
        return (flat_fn(flat_x + step) - flat_fn(flat_x - step)) / (2.0 * eps)

    steps = eps * jnp.eye(flat_x.shape[0], dtype=flat_x.dtype)
    # one column per input entry: (input, output) → (output, input)
    finite_difference = jax.vmap(central_difference, out_axes=1)(steps)
    scale = float(jnp.max(jnp.abs(finite_difference))) if jacobian.size else 0.0
    assert_matches_oracle(jacobian, finite_difference, rtol=0.0, atol=rtol * scale)
