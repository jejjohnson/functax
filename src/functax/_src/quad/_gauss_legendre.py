"""Gauss–Legendre nodes and weights on [−1, 1], built on the host."""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import operator

import numpy as np

from functax.exceptions import ValidationError


__all__ = ["gauss_legendre"]


def gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss–Legendre nodes and weights on [−1, 1].

    ∫₋₁¹ p(t) dt = ∑ₖ wₖ p(tₖ) exactly for every polynomial p of degree at most
    2n − 1. The rule is a host-side constant (NumPy's ``leggauss``), as in
    geonnax, so it can size static arrays.

    Args:
        n: The number of nodes.

    Returns:
        ``(nodes, weights)``, two float64 NumPy arrays of length n, the nodes
        ascending.

    Raises:
        ValidationError: If ``n`` is not an integer of at least 1.

    Examples:
        >>> from functax.quad import gauss_legendre
        >>> nodes, weights = gauss_legendre(3)
        >>> nodes.shape, bool(abs(weights.sum() - 2.0) < 1e-14)
        ((3,), True)
        >>> # exact up to degree 5: ∫ t³ + t² dt = 2/3
        >>> bool(abs(weights @ (nodes**3 + nodes**2) - 2 / 3) < 1e-14)
        True
    """
    if isinstance(n, bool):
        msg = f"'n' must be an integer of at least 1, got {n!r}"
        raise ValidationError(msg)
    try:
        count = operator.index(n)
    except TypeError as err:
        msg = f"'n' must be an integer of at least 1, got {n!r}"
        raise ValidationError(msg) from err
    if count < 1:
        msg = f"'n' must be an integer of at least 1, got {n!r}"
        raise ValidationError(msg)
    return np.polynomial.legendre.leggauss(count)
