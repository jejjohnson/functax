"""Coercion of affine transforms to six float64 numbers, on NumPy alone.

The transform T = (a, b, c, d, e, f) maps a pixel corner at column j and row i
to map coordinates x = a·j + b·i + c, y = d·j + e·i + f. This is the order of
``affine.Affine``, rasterio and geoarray, not GDAL's (c, a, b, f, d, e).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from functax.exceptions import GeometryError


__all__ = [
    "as_transform",
]


_AFFINE_ATTRIBUTES = ("a", "b", "c", "d", "e", "f")


def as_transform(transform: Any) -> np.ndarray:
    """Coerce a transform to a float64 array of (a, b, c, d, e, f).

    Args:
        transform: Anything with attributes ``a`` to ``f`` (an
            ``affine.Affine`` or rasterio ``Affine``), six numbers, the nine
            numbers of a 3 × 3 affine matrix whose last row is (0, 0, 1), or a
            ``(2, 3)`` or ``(3, 3)`` array.

    Returns:
        A ``(6,)`` float64 NumPy array.

    Raises:
        GeometryError: If the input does not describe an affine transform.

    Examples:
        >>> as_transform([20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0]).tolist()
        [20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0]
        >>> as_transform([[1.0, 0.0, 5.0], [0.0, -1.0, 7.0], [0.0, 0.0, 1.0]]).shape
        (6,)
    """
    if all(hasattr(transform, name) for name in _AFFINE_ATTRIBUTES):
        values = [getattr(transform, name) for name in _AFFINE_ATTRIBUTES]
        return np.asarray(values, dtype=np.float64)
    try:
        array = np.asarray(transform, dtype=np.float64)
    except (TypeError, ValueError) as err:
        msg = f"'transform' must describe an affine transform, got {transform!r}"
        raise GeometryError(msg) from err
    flat = array.ravel()
    if flat.size == 9:
        if not np.array_equal(flat[6:], [0.0, 0.0, 1.0]):
            msg = (
                "'transform' as a 3 × 3 matrix must have last row (0, 0, 1), "
                f"got {flat[6:].tolist()}"
            )
            raise GeometryError(msg)
        flat = flat[:6]
    if flat.size != 6:
        msg = (
            "'transform' must have 6 numbers (a, b, c, d, e, f), "
            f"got shape {array.shape}"
        )
        raise GeometryError(msg)
    return flat
