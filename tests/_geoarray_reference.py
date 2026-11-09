"""geoarray's affine maths, copied verbatim as the reference for functax's kernels.

Pinned from jejjohnson/geotoolz at 1e7308b, files
``packages/geotoolz-array/src/geoarray/_src/sampling.py`` (``to_pixel``) and
``packages/geotoolz-array/src/geoarray/_src/grid.py`` (``_snap``, ``_invert``
and the body of ``Grid.window``). Copied, not imported, so functax's tests do
not depend on geotoolz. ``namespace`` is reduced to NumPy, the only backend
the comparison uses, and ``window`` returns geoarray's ``PixelWindow`` fields
as a tuple ``(col_off, row_off, width, height)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def to_pixel(transform: Any, xy: Any) -> Any:
    xp = np
    a, b, c, d, e, f = (transform[k] for k in range(6))
    det = a * e - b * d
    dx = xy[..., 0] - c
    dy = xy[..., 1] - f
    # [[a, b], [d, e]]⁻¹ = [[e, −b], [−d, a]] / det
    col = (e * dx - b * dy) / det - 0.5
    row = (-d * dx + a * dy) / det - 0.5
    return xp.stack([row, col], axis=-1)


def _snap(value: float, atol: float) -> float:
    nearest = round(value)
    return float(nearest) if abs(value - nearest) <= atol else value


def _invert(transform: Sequence[float]) -> tuple[float, ...]:
    a, b, c, d, e, f = transform
    det = a * e - b * d
    if det == 0.0:
        raise ValueError(f"a singular transform cannot be inverted: {transform}")
    return (
        e / det,
        -b / det,
        (b * f - c * e) / det,
        -d / det,
        a / det,
        (c * d - a * f) / det,
    )


def window(
    transform: Sequence[float], bounds: Sequence[float], *, atol: float = 1e-6
) -> tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = (float(v) for v in bounds)
    corners = ((x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max))
    ia, ib, ic, id_, ie, if_ = _invert(transform)
    cols = [ia * x + ib * y + ic for x, y in corners]
    rows = [id_ * x + ie * y + if_ for x, y in corners]
    r0 = math.floor(_snap(min(rows), atol))
    r1 = math.ceil(_snap(max(rows), atol))
    c0 = math.floor(_snap(min(cols), atol))
    c1 = math.ceil(_snap(max(cols), atol))
    return (c0, r0, c1 - c0, r1 - r0)
