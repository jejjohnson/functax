"""Shared type aliases and structural protocols.

The aliases are written with [PEP 695](https://peps.python.org/pep-0695/)
``type`` statements (Python 3.12+), and
the protocols are *structural*: any object with the right methods satisfies
them, no inheritance or registration required. That is what lets
[`Pipeline`][mypackage.Pipeline] accept a user-defined transform it has never
heard of.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


__all__ = [
    "Fittable",
    "Series",
    "Transform",
]


type Series = Sequence[float]
"""Any read-only sequence of numbers accepted as input by this package.

A `list`, `tuple`, `range`, or anything else implementing
`collections.abc.Sequence` works. Inputs are coerced to `list[float]` on the
way in and never mutated.
"""


@runtime_checkable
class Transform(Protocol):
    """Structural protocol for anything that maps a series to a new series.

    Implementations must not mutate their input and must return a fresh
    `list[float]`.

    Examples:
        >>> from mypackage import Clip, Transform
        >>> isinstance(Clip(lower=0.0), Transform)
        True
        >>> isinstance("not a transform", Transform)
        False
    """

    def apply(self, series: Series) -> list[float]:
        """Return the transformed series."""
        ...


@runtime_checkable
class Fittable(Protocol):
    """Structural protocol for transforms that learn state from data.

    Examples:
        >>> from mypackage import Fittable, Standardize
        >>> isinstance(Standardize(), Fittable)
        True
    """

    def fit(self, series: Series) -> Fittable:
        """Learn any required state from ``series`` and return ``self``."""
        ...

    def apply(self, series: Series) -> list[float]:
        """Return the transformed series."""
        ...
