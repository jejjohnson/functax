"""Descriptive statistics for one-dimensional series.

Two complementary styles live side by side:

- **Batch, pure functions** — [`summarize`][mypackage.summarize],
  [`quantile`][mypackage.quantile], [`zscores`][mypackage.zscores] take a whole
  series and return a value. No hidden state, no mutation.
- **Streaming, stateful** — [`RunningStats`][mypackage.RunningStats] consumes
  observations one at a time in constant memory using Welford's algorithm.

For a series x₁, …, x_N the sample mean x̄ and variance s² are

x̄ = (1/N) ∑ᵢ xᵢ,

s² = ∑ᵢ (xᵢ − x̄)² / (N − δ),

where δ is ``ddof`` (1 for the unbiased sample variance, 0 for the
population variance).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from mypackage._typing import Series
from mypackage.exceptions import ValidationError
from mypackage.utils import require_non_empty


__all__ = [
    "RunningStats",
    "Summary",
    "quantile",
    "summarize",
    "zscores",
]


@dataclass(frozen=True, slots=True)
class Summary:
    """An immutable snapshot of a series' descriptive statistics.

    Instances are frozen, so a summary can be cached, compared, or used as a
    fixture without any risk of being mutated downstream.

    Attributes:
        count: Number of observations.
        mean: Arithmetic mean.
        variance: Sample variance with the ``ddof`` used at construction.
        minimum: Smallest observation.
        maximum: Largest observation.
        median: 50th percentile, linearly interpolated.
        ddof: Delta degrees of freedom used for ``variance``.

    Examples:
        >>> summary = summarize([1.0, 2.0, 3.0, 4.0])
        >>> summary.count, summary.mean, summary.median
        (4, 2.5, 2.5)
        >>> round(summary.std, 4)
        1.291
        >>> summary.spread
        3.0
    """

    count: int
    mean: float
    variance: float
    minimum: float
    maximum: float
    median: float
    ddof: int = 1

    @property
    def std(self) -> float:
        """Standard deviation, i.e. the square root of ``variance``."""
        return math.sqrt(self.variance)

    @property
    def spread(self) -> float:
        """Difference between ``maximum`` and ``minimum``."""
        return self.maximum - self.minimum

    def to_dict(self) -> dict[str, float]:
        """Return the summary as a plain dictionary.

        Derived properties (``std``, ``spread``) are included so the result is
        directly serialisable to JSON or renderable as a table.

        Returns:
            A dictionary of every field plus ``std`` and ``spread``.

        Examples:
        """
        data: dict[str, float] = dict(asdict(self))
        data["std"] = self.std
        data["spread"] = self.spread
        return data


def _mean(values: list[float]) -> float:
    """Arithmetic mean of a non-empty list that survives an overflowing sum.

    ``math.fsum`` is exact, but raises ``OverflowError`` when the running
    total leaves the float range even though the mean itself is finite — for
    example ``[1e308, 1e308]``. In that case every value is rescaled by a power
    of two that bounds the largest magnitude, which is exact, summed, and the
    scale reapplied. `math.ldexp` is used throughout because ``2.0 ** 1024``
    itself overflows.
    """
    try:
        return math.fsum(values) / len(values)
    except OverflowError:
        _, exponent = math.frexp(max(abs(value) for value in values))
        scaled = [math.ldexp(value, -exponent) for value in values]
        return math.ldexp(math.fsum(scaled) / len(values), exponent)


def summarize(series: Series, *, ddof: int = 1) -> Summary:
    """Compute descriptive statistics for ``series`` in a single pass.

    Args:
        series: The observations to summarise.
        ddof: Delta degrees of freedom for the variance. Use ``1`` for the
            unbiased sample variance (the default) or ``0`` for the population
            variance.

    Returns:
        A frozen [`Summary`][mypackage.Summary].

    Raises:
        EmptySeriesError: If ``series`` is empty.
        ValidationError: If ``ddof`` is negative or not smaller than the
            number of observations.

    Examples:
        >>> summary = summarize([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], ddof=0)
        >>> summary.count, summary.mean, summary.variance
        (8, 5.0, 4.0)
        >>> summary.minimum, summary.median, summary.maximum
        (2.0, 4.5, 9.0)

        A single observation has no unbiased variance, so use ``ddof=0``:

        >>> summarize([3.0], ddof=0).variance
        0.0

        The mean survives inputs whose sum would overflow:

        >>> summarize([1e308, 1e308], ddof=0).mean
        1e+308
    """
    values = require_non_empty(series)
    count = len(values)
    if ddof < 0:
        msg = f"'ddof' must be non-negative, got {ddof}"
        raise ValidationError(msg)
    if ddof >= count:
        msg = (
            f"'ddof' must be smaller than the number of observations, "
            f"got ddof={ddof} with {count} value(s)"
        )
        raise ValidationError(msg)

    mean = _mean(values)
    # `d * d` rather than `d ** 2`: float `**` raises OverflowError, whereas
    # multiplication overflows to inf, which is the honest IEEE answer when
    # the variance itself is not representable.
    variance = math.fsum(d * d for d in (value - mean for value in values)) / (
        count - ddof
    )
    return Summary(
        count=count,
        mean=mean,
        variance=variance,
        minimum=min(values),
        maximum=max(values),
        median=quantile(values, 0.5),
        ddof=ddof,
    )


def quantile(series: Series, q: float) -> float:
    """Return the ``q``-th quantile using linear interpolation.

    With sorted observations x_(1) ≤ … ≤ x_(N) and the fractional position
    h = q (N − 1), the result is

    Q(q) = x_(j+1) + (h − j) (x_(j+2) − x_(j+1)),  with j = ⌊h⌋.

    This matches NumPy's default ``method="linear"``.

    Args:
        series: The observations.
        q: Quantile level in ``[0, 1]``.

    Returns:
        The interpolated quantile.

    Raises:
        EmptySeriesError: If ``series`` is empty.
        ValidationError: If ``q`` lies outside ``[0, 1]``.

    Examples:
        >>> data = [1.0, 2.0, 3.0, 4.0]
        >>> quantile(data, 0.0), quantile(data, 0.5), quantile(data, 1.0)
        (1.0, 2.5, 4.0)
        >>> round(quantile(data, 0.25), 4)
        1.75
    """
    values = sorted(require_non_empty(series))
    if not 0.0 <= q <= 1.0:
        msg = f"'q' must lie in [0, 1], got {q!r}"
        raise ValidationError(msg)

    if len(values) == 1:
        return values[0]

    position = q * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def zscores(series: Series, *, ddof: int = 1) -> list[float]:
    """Standardise ``series`` to zero mean and unit variance.

    zᵢ = (xᵢ − x̄) / s,  with x̄ the mean and s the standard deviation.

    Args:
        series: The observations to standardise.
        ddof: Delta degrees of freedom for the standard deviation.

    Returns:
        A new list of standardised values.

    Raises:
        EmptySeriesError: If ``series`` is empty.
        ValidationError: If the series is constant, so ``s == 0``.

    Examples:
        >>> [round(z, 4) for z in zscores([1.0, 2.0, 3.0])]
        [-1.0, 0.0, 1.0]
        >>> try:
        ...     zscores([2.0, 2.0, 2.0])
        ... except ValidationError as err:
        ...     print(err)
        cannot standardise a constant series (std == 0)
    """
    values = require_non_empty(series)
    summary = summarize(values, ddof=ddof)
    std = summary.std
    if std == 0.0:
        msg = "cannot standardise a constant series (std == 0)"
        raise ValidationError(msg)
    return [(value - summary.mean) / std for value in values]


@dataclass(slots=True)
class RunningStats:
    """Streaming mean and variance via Welford's online algorithm.

    Memory is constant regardless of how many observations are consumed, and
    the update is numerically stable — unlike the naive "sum of squares minus
    square of sums" formulation, which loses precision catastrophically for
    large means.

    Each observation xₙ updates the running mean x̄ₙ and the sum of squared
    deviations M₂,ₙ:

    x̄ₙ = x̄ₙ₋₁ + (xₙ − x̄ₙ₋₁) / n,

    M₂,ₙ = M₂,ₙ₋₁ + (xₙ − x̄ₙ₋₁) (xₙ − x̄ₙ).

    Attributes:
        ddof: Delta degrees of freedom used by ``variance`` and ``std``. Must
            be a non-negative integer.

    Raises:
        ValidationError: If ``ddof`` is negative or not an integer.

    Examples:
        >>> stats = RunningStats()
        >>> stats.update_many([1.0, 2.0, 3.0, 4.0]).mean
        2.5
        >>> round(stats.std, 4)
        1.291
        >>> stats.count, stats.minimum, stats.maximum
        (4, 1.0, 4.0)

        The streaming result agrees with the batch one:

        >>> data = [0.5, -1.25, 3.0, 7.75]
        >>> RunningStats().update_many(data).mean == summarize(data).mean
        True
    """

    ddof: int = 1
    _count: int = field(default=0, repr=False)
    _mean: float = field(default=0.0, repr=False)
    _m2: float = field(default=0.0, repr=False)
    _minimum: float = field(default=math.inf, repr=False)
    _maximum: float = field(default=-math.inf, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.ddof, bool) or not isinstance(self.ddof, int):
            msg = f"'ddof' must be an integer, got {self.ddof!r}"
            raise ValidationError(msg)
        if self.ddof < 0:
            msg = f"'ddof' must be non-negative, got {self.ddof}"
            raise ValidationError(msg)

    def update(self, value: float) -> RunningStats:
        """Consume a single observation.

        Args:
            value: The observation to fold into the running statistics.

        Returns:
            ``self``, so that updates can be chained.

        Raises:
            ValidationError: If ``value`` is not a number.
        """
        try:
            observation = float(value)
        except (TypeError, ValueError) as err:
            msg = f"'value' must be a number, got {value!r}"
            raise ValidationError(msg) from err

        self._count += 1
        delta = observation - self._mean
        self._mean += delta / self._count
        self._m2 += delta * (observation - self._mean)
        self._minimum = min(self._minimum, observation)
        self._maximum = max(self._maximum, observation)
        return self

    def update_many(self, series: Series) -> RunningStats:
        """Consume every observation in ``series``.

        Args:
            series: The observations to fold in, in order.

        Returns:
            ``self``, so that updates can be chained.
        """
        for value in series:
            self.update(value)
        return self

    @property
    def count(self) -> int:
        """Number of observations consumed so far."""
        return self._count

    @property
    def mean(self) -> float:
        """Running arithmetic mean. Zero before any observation."""
        return self._mean

    @property
    def variance(self) -> float:
        """Running variance. Zero until more than ``ddof`` observations."""
        if self._count <= self.ddof:
            return 0.0
        return self._m2 / (self._count - self.ddof)

    @property
    def std(self) -> float:
        """Running standard deviation."""
        return math.sqrt(self.variance)

    @property
    def minimum(self) -> float:
        """Smallest observation seen so far.

        Raises:
            EmptySeriesError: If no observation has been consumed.
        """
        self._require_data()
        return self._minimum

    @property
    def maximum(self) -> float:
        """Largest observation seen so far.

        Raises:
            EmptySeriesError: If no observation has been consumed.
        """
        self._require_data()
        return self._maximum

    def _require_data(self) -> None:
        if self._count == 0:
            require_non_empty([], name="observations")
