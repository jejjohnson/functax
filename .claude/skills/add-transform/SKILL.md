---
name: add-transform
description: Add a transform to the package — a dataclass that satisfies the runtime-checkable Transform protocol (apply returns a new list), stateless and frozen or fitted with trailing-underscore state, NotFittedError before fit, fit returning self, fit_apply, validation in __post_init__, composition in Pipeline, the protocol tests, docs and the capability index. Use when asked to add a scaler, a filter adapter, an encoder or any step that goes into a Pipeline in src/mypackage/transforms.py.
---

<!-- template-example: this recipe is written for the example package's Transform protocol; rewrite it for your package's extension points. -->

# Add a transform

Read "2. Transforms" under "The contracts" in `AGENTS.md` first; this is the step-by-step.

## 1. Make sure it does not exist yet

- Search `docs/api/capabilities.md`: `Standardize` (with `inverse`), `MinMaxScale`, `Clip`, `MovingAverage`, and `Pipeline` / `chain` for combinations. A composition of existing transforms is a pipeline, not a new class.
- A transform that wraps a function (as `MovingAverage` wraps `moving_average`) needs the function first: if it is missing, add it with `add-function`, then adapt it here.

## 2. Pick the kind (`src/mypackage/transforms.py`)

| It… | Kind | Exemplar |
|---|---|---|
| Learns nothing from data | stateless: `@dataclass(frozen=True, slots=True)` | `Clip`, `MovingAverage` |
| Learns parameters in `fit` | fitted: `@dataclass(slots=True)` | `Standardize`, `MinMaxScale` |

## 3. Write it

- `apply(self, series: Series) -> list[float]` returns a **new** list (`as_floats(series)` coerces); never mutate the input or `self` in `apply`.
- Stateless: `fit(self, series) -> <Name>` returns `self` unchanged; `fit_apply` returns `self.apply(series)`.
- Fitted: learned state in `init=False` fields with a trailing underscore, defaulting to `None` (`mean_: float | None = field(default=None, init=False)`); `fit(self, series) -> Self` validates (`require_non_empty`, a constant series raises `ValidationError`), sets them and returns `self`; `apply` (and `inverse`, if any) first checks them and raises `NotFittedError("<Name> is not fitted; call .fit(series) first")` — never a silent identity; `fit_apply` is `self.fit(series).apply(series)`.
- Constructor arguments validated in `__post_init__` with `ValidationError` (`MinMaxScale` checks `low < high`, `MovingAverage` the window).
- No base class and no registry: the `Transform` / `Fittable` protocols are structural, and `Pipeline` accepts any object with a callable `apply`.
- Docstring: the defining equation in Unicode (`z = (x − x̄) / s`, `eman-style-math`, never `$…$`), `Attributes:` for the fields with each one's symbol, and an executable `Examples:` block (`fit_apply` on a short series, rounded output).

## 4. Export and document

- `__all__` in `transforms.py` and in `src/mypackage/__init__.py` (import and `__all__`, sorted).
- The API page `docs/api/transforms.md` picks it up from `::: mypackage.transforms`; add the name to the Transforms row of `docs/api/index.md` and, if it changes the picture, the "Stateless versus fitted" table in `docs/guide/transforms.md`.
- `uv run poe capabilities`.

## 5. Tests (`tests/test_transforms.py`)

- Add an instance to the `transform` parameter list of `TestProtocols.test_transform_protocol`, and a fitted one to `test_fittable_protocol`.
- A `Test<Name>` class: values on a hand-computed series; learned state transfers to new data; `apply` before `fit` raises `NotFittedError` (`match=r"call \.fit\(series\) first"`); invalid constructor arguments raise `ValidationError`; a stateless one's `fit` returns `self`; a frozen one raises `AttributeError` on assignment; it works inside a `Pipeline` (fitted on the previous step's output).

## 6. Verify

```bash
uv run pytest --no-cov tests/test_transforms.py src/mypackage/transforms.py tests/test_public_api.py tests/test_capabilities.py -v
uv run poe test-cov
```

then the `pre-pr-check` skill.
