"""Tests for `Cubic`, `BSpline` and the `BandedPrefilter`. The fast tier: behaviour,
validation and small closed-form checks, each well under a second.
"""

from __future__ import annotations

import pytest

from functax import ValidationError
from functax.interp import (
    BandedPrefilter,
    BSpline,
    BSplineBasis,
    spline_poles,
)


class TestValidation:
    @pytest.mark.parametrize("order", [-1, 6])
    def test_rejects_orders_outside_0_to_5(self, order: int) -> None:
        with pytest.raises(
            ValidationError, match=r"'order' must be an integer in \[0, 5\]"
        ):
            BSpline(order)
        with pytest.raises(
            ValidationError, match=r"'order' must be an integer in \[0, 5\]"
        ):
            spline_poles(order)

    def test_basis_rejects_a_bool_order(self) -> None:
        basis = object.__new__(BSplineBasis)
        object.__setattr__(basis, "order", True)
        with pytest.raises(ValidationError, match="'order' must be an integer"):
            basis.__check_init__()

    def test_prefilter_rejects_mismatched_axes(self) -> None:
        with pytest.raises(ValidationError, match="one entry per axis"):
            BandedPrefilter((3,), (5, 6), (False, False))
