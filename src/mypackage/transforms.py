"""Composable series transforms with a fit / apply lifecycle.

Every transform here satisfies the [`Transform`][mypackage.Transform]
protocol — a single `apply` method. Transforms that learn state from data
(`Standardize`, `MinMaxScale`) additionally satisfy
[`Fittable`][mypackage.Fittable] and raise
[`NotFittedError`][mypackage.NotFittedError] if used before `fit`.

Because the protocols are structural, [`Pipeline`][mypackage.Pipeline] happily
composes your own transforms alongside the built-in ones — no base class to
inherit from, no registry to join:

```python
from dataclasses import dataclass

from mypackage import Series, chain, Standardize


@dataclass(frozen=True, slots=True)
class Square:
    def apply(self, series: Series) -> list[float]:
        return [value**2 for value in series]


pipeline = chain(Standardize(), Square())
```
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Self

from mypackage._typing import Series, Transform
from mypackage.exceptions import NotFittedError, ValidationError
from mypackage.smoothing import Padding, moving_average
from mypackage.stats import summarize
from mypackage.utils import as_floats, require_non_empty, require_positive


__all__ = [
    "Clip",
    "MinMaxScale",
    "MovingAverage",
    "Pipeline",
    "Standardize",
    "chain",
]


@dataclass(slots=True)
class Standardize:
    """Centre a series on zero and scale it to unit standard deviation.

    zᵢ = (xᵢ − x̄) / s,  with x̄ the mean and s the standard deviation.

    The mean and standard deviation are learned by `fit` and reused by every
    later `apply`, which is what makes this safe to fit on a training split and
    apply to a held-out one.

    Attributes:
        ddof: Delta degrees of freedom for the standard deviation.

    Examples:
        >>> scaler = Standardize().fit([10.0, 12.0, 14.0])
        >>> [round(value, 4) for value in scaler.apply([10.0, 12.0, 14.0])]
        [-1.0, 0.0, 1.0]

        The learned statistics carry over to unseen data:

        >>> round(scaler.apply([16.0])[0], 4)
        2.0

        `inverse` round-trips:

        >>> [round(v, 6) for v in scaler.inverse(scaler.apply([11.0, 13.0]))]
        [11.0, 13.0]
    """

    ddof: int = 1
    mean_: float | None = field(default=None, init=False)
    std_: float | None = field(default=None, init=False)

    def fit(self, series: Series) -> Self:
        """Learn the mean and standard deviation of ``series``.

        Args:
            series: The reference observations.

        Returns:
            ``self``, so ``fit`` chains into ``apply``.

        Raises:
            EmptySeriesError: If ``series`` is empty.
            ValidationError: If ``series`` is constant, leaving ``std == 0``.
        """
        summary = summarize(series, ddof=self.ddof)
        if summary.std == 0.0:
            msg = "cannot standardise a constant series (std == 0)"
            raise ValidationError(msg)
        self.mean_ = summary.mean
        self.std_ = summary.std
        return self

    def apply(self, series: Series) -> list[float]:
        """Standardise ``series`` with the learned statistics.

        Args:
            series: The observations to transform.

        Returns:
            A new list of standardised values.

        Raises:
            NotFittedError: If `fit` has not been called.
        """
        mean, std = self._check_fitted()
        return [(value - mean) / std for value in as_floats(series)]

    def inverse(self, series: Series) -> list[float]:
        """Map standardised values back to the original scale.

        Args:
            series: Standardised observations.

        Returns:
            A new list on the original scale.

        Raises:
            NotFittedError: If `fit` has not been called.
        """
        mean, std = self._check_fitted()
        return [value * std + mean for value in as_floats(series)]

    def fit_apply(self, series: Series) -> list[float]:
        """Fit on ``series`` and immediately transform it."""
        return self.fit(series).apply(series)

    def _check_fitted(self) -> tuple[float, float]:
        if self.mean_ is None or self.std_ is None:
            msg = "Standardize is not fitted; call .fit(series) first"
            raise NotFittedError(msg)
        return self.mean_, self.std_


@dataclass(slots=True)
class MinMaxScale:
    """Linearly rescale a series into ``feature_range``.

    yᵢ = a + (b − a) (xᵢ − min x) / (max x − min x),

    with (a, b) the ``feature_range``.

    Attributes:
        feature_range: Target ``(low, high)`` interval.

    Examples:
        >>> MinMaxScale().fit_apply([0.0, 5.0, 10.0])
        [0.0, 0.5, 1.0]
        >>> MinMaxScale(feature_range=(-1.0, 1.0)).fit_apply([0.0, 5.0, 10.0])
        [-1.0, 0.0, 1.0]
    """

    feature_range: tuple[float, float] = (0.0, 1.0)
    minimum_: float | None = field(default=None, init=False)
    maximum_: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        low, high = self.feature_range
        if not low < high:
            msg = f"'feature_range' must satisfy low < high, got {self.feature_range!r}"
            raise ValidationError(msg)

    def fit(self, series: Series) -> Self:
        """Learn the minimum and maximum of ``series``.

        Args:
            series: The reference observations.

        Returns:
            ``self``.

        Raises:
            EmptySeriesError: If ``series`` is empty.
            ValidationError: If ``series`` is constant.
        """
        values = require_non_empty(series)
        minimum, maximum = min(values), max(values)
        if minimum == maximum:
            msg = "cannot rescale a constant series (max == min)"
            raise ValidationError(msg)
        self.minimum_ = minimum
        self.maximum_ = maximum
        return self

    def apply(self, series: Series) -> list[float]:
        """Rescale ``series`` with the learned bounds.

        Args:
            series: The observations to transform.

        Returns:
            A new list of rescaled values.

        Raises:
            NotFittedError: If `fit` has not been called.
        """
        if self.minimum_ is None or self.maximum_ is None:
            msg = "MinMaxScale is not fitted; call .fit(series) first"
            raise NotFittedError(msg)
        low, high = self.feature_range
        span = self.maximum_ - self.minimum_
        return [
            low + (high - low) * (value - self.minimum_) / span
            for value in as_floats(series)
        ]

    def fit_apply(self, series: Series) -> list[float]:
        """Fit on ``series`` and immediately transform it."""
        return self.fit(series).apply(series)


@dataclass(frozen=True, slots=True)
class Clip:
    """Clamp every value into ``[lower, upper]``.

    Stateless: `fit` is a no-op that returns ``self``, so `Clip` drops into a
    [`Pipeline`][mypackage.Pipeline] next to fitted transforms without special
    casing.

    Attributes:
        lower: Lower bound, or `None` for unbounded below.
        upper: Upper bound, or `None` for unbounded above.

    Examples:
        >>> Clip(lower=0.0).apply([-2.0, 0.5, 3.0])
        [0.0, 0.5, 3.0]
        >>> Clip(lower=0.0, upper=1.0).apply([-2.0, 0.5, 3.0])
        [0.0, 0.5, 1.0]
    """

    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        both_set = self.lower is not None and self.upper is not None
        if both_set and self.lower > self.upper:
            msg = (
                f"'lower' must not exceed 'upper', "
                f"got lower={self.lower!r}, upper={self.upper!r}"
            )
            raise ValidationError(msg)

    def fit(self, series: Series) -> Clip:
        """Return ``self`` — `Clip` learns nothing from data."""
        return self

    def apply(self, series: Series) -> list[float]:
        """Clamp every value of ``series`` into the configured bounds.

        Args:
            series: The observations to clamp.

        Returns:
            A new list of clamped values.
        """
        values = as_floats(series)
        if self.lower is not None:
            values = [max(self.lower, value) for value in values]
        if self.upper is not None:
            values = [min(self.upper, value) for value in values]
        return values

    def fit_apply(self, series: Series) -> list[float]:
        """Fit on ``series`` and immediately transform it."""
        return self.apply(series)


@dataclass(frozen=True, slots=True)
class MovingAverage:
    """Transform adapter around [`moving_average`][mypackage.moving_average].

    Stateless, so it composes freely inside a
    [`Pipeline`][mypackage.Pipeline].

    Attributes:
        window: Number of samples averaged per output point.
        padding: Boundary strategy, see [`Padding`][mypackage.Padding].

    Examples:
        >>> MovingAverage(window=3).apply([1.0, 2.0, 3.0, 4.0, 5.0])[1:4]
        [2.0, 3.0, 4.0]
    """

    window: int
    padding: Padding | str = Padding.EDGE

    def __post_init__(self) -> None:
        require_positive(self.window, name="window")

    def fit(self, series: Series) -> MovingAverage:
        """Return ``self`` — `MovingAverage` learns nothing from data."""
        return self

    def apply(self, series: Series) -> list[float]:
        """Smooth ``series`` with the configured window.

        Args:
            series: The observations to smooth.

        Returns:
            A new list of smoothed values.
        """
        return moving_average(series, self.window, padding=self.padding)

    def fit_apply(self, series: Series) -> list[float]:
        """Fit on ``series`` and immediately transform it."""
        return self.apply(series)


@dataclass(slots=True)
class Pipeline:
    """Apply a sequence of transforms left to right.

    Fitting is sequential: each step is fitted on the output of the previous
    one, exactly as it will see the data at apply time.

    Attributes:
        steps: The transforms to apply, in order.

    Examples:
        >>> pipeline = Pipeline([Standardize(), Clip(lower=-1.0, upper=1.0)])
        >>> [round(v, 4) for v in pipeline.fit_apply([10.0, 12.0, 14.0, 40.0])]
        [-0.6385, -0.4966, -0.3547, 1.0]

        Pipelines compose with ``|``, and nest:

        >>> combined = pipeline | MovingAverage(window=3)
        >>> len(combined)
        3
        >>> [round(v, 4) for v in combined.fit_apply([10.0, 12.0, 14.0, 40.0])]
        [-0.5912, -0.4966, 0.0495, 0.5484]

        Empty pipelines are the identity:

        >>> Pipeline([]).fit_apply([1.0, 2.0])
        [1.0, 2.0]
    """

    steps: list[Transform] = field(default_factory=list)

    def __post_init__(self) -> None:
        for index, step in enumerate(self.steps):
            # A runtime-checkable protocol only checks that `apply` *exists*,
            # so `class Bad: apply = 7` would pass `isinstance` and then fail
            # at apply time. Require it to be callable.
            if not callable(getattr(step, "apply", None)):
                msg = (
                    f"pipeline step {index} does not implement "
                    f"Transform.apply, got {step!r}"
                )
                raise ValidationError(msg)

    def fit(self, series: Series) -> Self:
        """Fit every step in turn on the output of the previous step.

        Args:
            series: The reference observations.

        Returns:
            ``self``.
        """
        current = as_floats(series)
        for step in self.steps:
            fit = getattr(step, "fit", None)
            if callable(fit):
                fit(current)
            current = step.apply(current)
        return self

    def apply(self, series: Series) -> list[float]:
        """Apply every step in turn.

        Args:
            series: The observations to transform.

        Returns:
            A new list holding the output of the final step.

        Raises:
            NotFittedError: If any stateful step has not been fitted.
        """
        current = as_floats(series)
        for step in self.steps:
            current = step.apply(current)
        return current

    def fit_apply(self, series: Series) -> list[float]:
        """Fit every step and return the fully transformed series."""
        return self.fit(series).apply(series)

    def __or__(self, other: Transform | Pipeline) -> Pipeline:
        """Return a new pipeline with ``other`` appended."""
        extra = other.steps if isinstance(other, Pipeline) else [other]
        return Pipeline([*self.steps, *extra])

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Transform]:
        return iter(self.steps)

    def __getitem__(self, index: int) -> Transform:
        return self.steps[index]


def chain(*transforms: Transform) -> Pipeline:
    """Build a [`Pipeline`][mypackage.Pipeline] from positional transforms.

    A small convenience over ``Pipeline([...])`` that reads better inline.

    Args:
        *transforms: Transforms to apply, in order.

    Returns:
        A new pipeline.

    Raises:
        ValidationError: If any argument does not implement ``apply``.

    Examples:
        >>> pipeline = chain(MinMaxScale(), MovingAverage(window=3))
        >>> [round(v, 4) for v in pipeline.fit_apply([0.0, 5.0, 10.0])]
        [0.1667, 0.5, 0.8333]
    """
    return Pipeline(list(transforms))
