"""The named rules for targets off the domain, missing samples and edge stencils.

The stack had three outside conventions before functax: NaN with ``valid``
(geoops ``interp_at``), the fill value or 0 (the geom samplers), and NaN
(geopatcher ``PointDomain``); vardax clamped without flagging. `Policies`
names each one, so switching a sampler to functax changes no output.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import math

import equinox as eqx

from functax.exceptions import PolicyError


__all__ = [
    "BOUNDARY",
    "EXTENT",
    "MISSING",
    "OUTSIDE",
    "Policies",
]


OUTSIDE: tuple[str, ...] = ("nan", "clamp", "extrapolate", "raise")
"""What happens at a target outside the domain."""

MISSING: tuple[str, ...] = ("propagate", "renormalise", "fill")
"""What happens at a non-finite source sample."""

BOUNDARY: tuple[str, ...] = ("mirror", "wrap", "edge", "constant")
"""How a tap that falls off the end of an axis is read."""

EXTENT: tuple[str, ...] = ("nodes", "cells")
"""Where the domain ends: at the end nodes, or half a cell beyond them."""


def _canonical_fill(value: float | int) -> float | int:
    """Every NaN as the one ``math.nan`` object.

    Static fields sit in the pytree structure, which JAX compares with ``==``
    inside tuples; two NaN objects are unequal there unless they are the same
    object, which would make equal policies look different to ``jit``.
    """
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    return value


class Policies(eqx.Module):
    """Outside the domain, at missing samples, and at the edges.

    The three outside conventions of the stack map onto these values:
    geoops ``interp_at`` is ``outside="nan"``, ``extent="cells"``,
    ``boundary="edge"``; a sampler that writes its fill value is
    ``outside="nan"`` with that ``fill_value``; vardax's clamp is
    ``outside="clamp"``.

    Args:
        outside: At a target off the domain. ``"nan"`` marks the row invalid
            (zero weights) and the plan writes ``fill_value`` there;
            ``"clamp"`` moves the target onto the nearest boundary;
            ``"extrapolate"`` keeps the scheme's weights there; ``"raise"``
            raises `OutsideDomainError` when the plan is built (host only).
        missing: At a non-finite source sample. ``"propagate"`` lets NaN
            through; ``"renormalise"`` drops those taps and divides each row
            by its remaining weight, marking a row with no weight left
            invalid; ``"fill"`` replaces them by ``fill_value`` first.
        boundary: At a tap past the end of an axis of N samples.
            ``"mirror"`` reflects about the end sample (j ↦ −j and
            j ↦ 2(N − 1) − j); ``"wrap"`` is j mod N; ``"edge"`` clips to
            [0, N − 1]; ``"constant"`` drops the tap (weight 0). A periodic
            axis always wraps.
        extent: ``"nodes"``, the domain is [x₀, x_{N−1}]; ``"cells"``, the
            union of pixel areas, half a cell wider at each end.
        fill_value: Written at invalid rows (and at missing samples under
            ``missing="fill"``); NaN by default.

    Raises:
        PolicyError: If a field names an option that does not exist, or
            ``fill_value`` is not a number.

    Examples:
        >>> from functax.interp import Policies
        >>> geoops = Policies(extent="cells", boundary="edge")
        >>> geoops.outside, geoops.missing
        ('nan', 'propagate')
        >>> Policies(outside="wrap")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        functax.exceptions.PolicyError: 'outside' must be one of ...
    """

    # str rather than Literal, so a bad value reaches the PolicyError below
    # (Literal is not checked at run time, and the type hook would preempt it)
    outside: str = eqx.field(static=True, default="nan")
    missing: str = eqx.field(static=True, default="propagate")
    boundary: str = eqx.field(static=True, default="mirror")
    extent: str = eqx.field(static=True, default="nodes")
    fill_value: float | int = eqx.field(
        static=True, default=math.nan, converter=_canonical_fill
    )

    def __check_init__(self) -> None:
        for name, options in (
            ("outside", OUTSIDE),
            ("missing", MISSING),
            ("boundary", BOUNDARY),
            ("extent", EXTENT),
        ):
            value = getattr(self, name)
            if value not in options:
                msg = f"'{name}' must be one of {list(options)}, got {value!r}"
                raise PolicyError(msg)
        if isinstance(self.fill_value, bool) or not isinstance(
            self.fill_value, int | float
        ):
            msg = f"'fill_value' must be a number, got {self.fill_value!r}"
            raise PolicyError(msg)
