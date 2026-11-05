# Exceptions

Every error raised by the package derives from
[`MypackageError`][mypackage.MypackageError] *and* from the closest
standard-library exception.

```mermaid
graph TD
    E[Exception] --> M[MypackageError]
    V[ValueError] --> VE[ValidationError]
    M --> VE
    VE --> ES[EmptySeriesError]
    R[RuntimeError] --> NF[NotFittedError]
    M --> NF
```

::: mypackage.exceptions
