# Exceptions

Every error raised by the package derives from
[`FunctaxError`][functax.FunctaxError] *and* from the closest
standard-library exception.

```mermaid
graph TD
    E[Exception] --> M[FunctaxError]
    V[ValueError] --> VE[ValidationError]
    M --> VE
    VE --> ES[EmptySeriesError]
    R[RuntimeError] --> NF[NotFittedError]
    M --> NF
```

::: functax.exceptions
