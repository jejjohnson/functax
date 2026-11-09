"""Content-hash keys for plans: one key per scheme, policies and pair of geometries.

A plan is a pure function of its scheme, its policies and its two geometries,
so the sha256 of those inputs names it: a cache keyed by `plan_key` builds
each plan once. The hash follows pipekit's ``stable_json`` and ``sha256_hex``
conventions without importing pipekit.
"""

from __future__ import annotations

from functax._src._backend import is_tracer, require_jax


require_jax(__name__)

import dataclasses
import hashlib
import json
from typing import Any

import equinox as eqx
import jax
import numpy as np

from functax import __version__
from functax._src.interp._config import encode_json_value
from functax._src.interp._policies import Policies
from functax._src.interp._schemes import Scheme
from functax.exceptions import ValidationError


__all__ = [
    "plan_key",
]


def _describe(value: Any, name: str, arrays: list[np.ndarray]) -> Any:
    """``value`` as strict JSON, with each array replaced by its dtype and shape.

    The array itself is appended to ``arrays`` (C-contiguous, little-endian),
    and its entry records its position there, so the hash can follow the
    description with the raw bytes.
    """
    if isinstance(value, jax.Array | np.ndarray):
        if is_tracer(value):
            msg = (
                f"'{name}' is traced; plan_key hashes concrete arrays on the "
                "host, so call it outside jit"
            )
            raise ValidationError(msg)
        host = np.asarray(value)
        little_endian = host.dtype.newbyteorder("<")
        contiguous = np.ascontiguousarray(host, dtype=little_endian)
        arrays.append(contiguous)
        return {
            "array": len(arrays) - 1,
            "dtype": host.dtype.name,
            "shape": list(host.shape),
        }
    if isinstance(value, eqx.Module):
        description: dict[str, Any] = {"class": type(value).__name__}
        for field in dataclasses.fields(value):
            field_value = getattr(value, field.name)
            description[field.name] = _describe(
                field_value, f"{name}.{field.name}", arrays
            )
        return description
    if isinstance(value, tuple | list):
        return [
            _describe(item, f"{name}[{index}]", arrays)
            for index, item in enumerate(value)
        ]
    return encode_json_value(value, name)


def plan_key(
    scheme: Scheme,
    source: Any,
    target: Any,
    *,
    policies: Policies | None = None,
) -> str:
    """sha256 hex of the scheme, the policies and both geometries' bytes.

    key = sha256(J ‖ 0x00 ‖ b₁ ‖ … ‖ b_K), with J the stable JSON (sorted
    keys, floats by ``repr``) of the functax version, the scheme's and the
    policies' configs, and each geometry's class and fields, every array
    there written as its dtype and shape; b₁, …, b_K are those arrays'
    C-contiguous little-endian bytes in the same order. Equal inputs give
    the same key in any process; a change of one ULP in any coordinate, of
    a dtype, a shape, a static field (period, epoch, metric) or the functax
    version gives another. Runs on the host.

    Args:
        scheme: The scheme the plan is built with.
        source: The source geometry (an `Axis`, a grid).
        target: The target geometry (`Points`, a grid).
        policies: The policies the plan is built with; ``Policies()`` when
            ``None``, as in `Scheme.plan`.

    Returns:
        The 64-character lowercase hex digest.

    Raises:
        ValidationError: If ``scheme`` is not a `Scheme`, ``policies`` not
            `Policies`, a geometry not an Equinox module, or an array in one
            is traced (inside ``jit``).

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points
        >>> from functax.interp import Linear, Nearest, plan_key
        >>> axis, points = Axis(jnp.arange(4.0)), Points(jnp.array([[0.5]]))
        >>> key = plan_key(Linear(), axis, points)
        >>> len(key), key == plan_key(Linear(), axis, points)
        (64, True)
        >>> key == plan_key(Nearest(), axis, points)
        False
        >>> moved = Points(jnp.nextafter(points.xy, 1.0))
        >>> key == plan_key(Linear(), axis, moved)
        False
    """
    if not isinstance(scheme, Scheme):
        msg = f"'scheme' must be a Scheme, got {type(scheme).__name__}"
        raise ValidationError(msg)
    policies = Policies() if policies is None else policies
    if not isinstance(policies, Policies):
        msg = f"'policies' must be Policies, got {type(policies).__name__}"
        raise ValidationError(msg)
    for name, geometry in (("source", source), ("target", target)):
        if not isinstance(geometry, eqx.Module):
            msg = (
                f"'{name}' must be a geometry (an Axis, a grid or Points), got "
                f"{type(geometry).__name__}"
            )
            raise ValidationError(msg)
    arrays: list[np.ndarray] = []
    description = {
        "functax": __version__,
        "scheme": scheme.get_config(),
        "policies": policies.get_config(),
        "source": _describe(source, "source", arrays),
        "target": _describe(target, "target", arrays),
    }
    # pipekit's stable_json, made strict: sorted keys, floats by repr, no NaN
    header = json.dumps(description, sort_keys=True, allow_nan=False)
    digest = hashlib.sha256(header.encode("utf-8"))
    # the header fixes every array's dtype and shape, so plain concatenation
    # of their bytes after one separator is unambiguous
    digest.update(b"\x00")
    for array in arrays:
        digest.update(array.tobytes())
    return digest.hexdigest()
