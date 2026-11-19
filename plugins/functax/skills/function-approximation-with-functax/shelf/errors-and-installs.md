# Errors and the two installs

Role: ⓪ structure (`role-ladder`, `references/structure.md`).
Unit: errors.

Open this when you catch an error from functax, raise one from code built on it, or decide which install a dependent needs.

## Names

The exceptions are defined in `functax.exceptions` and exported again from `functax`, with `__version__`.

| Name | Raised when | Also a |
|---|---|---|
| `FunctaxError` | The base class of every exception functax raises. | `Exception` |
| `ValidationError` | An argument fails validation. | `ValueError` |
| `GeometryError` | An axis, a grid or a point set is malformed. | `ValueError` |
| `PolicyError` | An interpolation policy names an option that does not exist. | `ValueError` |
| `OutsideDomainError` | A target lies off the domain under `outside="raise"`. | `ValueError` |
| `NonlinearSchemeError` | A linear-only operation meets a plan whose fit is nonlinear in the data. | `TypeError` |
| `MissingExtraError` | A module or a plan needs an optional extra that is not installed. | `ImportError` |

`__version__` is the installed version as a string.

## Use it

```python
import functax
from functax.exceptions import (
    FunctaxError,
    GeometryError,
    MissingExtraError,
    NonlinearSchemeError,
    OutsideDomainError,
    PolicyError,
    ValidationError,
)

assert functax.FunctaxError is FunctaxError
assert isinstance(functax.__version__, str)

# every error is also the closest standard exception
assert issubclass(GeometryError, ValidationError)
assert issubclass(ValidationError, ValueError)
assert issubclass(PolicyError, ValueError)
assert issubclass(OutsideDomainError, ValueError)
assert issubclass(NonlinearSchemeError, TypeError)
assert issubclass(MissingExtraError, ImportError)

from functax.interp import Policies

try:
    Policies(outside="bounce")
except PolicyError as err:
    assert isinstance(err, ValueError)
```

## Rules

- Catch `FunctaxError` to catch everything functax raises; catch a subclass to handle one cause.
- Existing `except ValueError` and `except ImportError` blocks keep working, because each error is also the standard exception.
- The base install, `functax`, depends on NumPy alone and gives `functax` and `functax.geometry.affine`.
- Every other module needs `functax[jax]`, which brings JAX, Equinox, lineax, einx and jaxtyping. Importing one without it raises `MissingExtraError`, whose message names the extra.
- Importing `functax` itself never imports JAX, so a NumPy-only library can depend on it.
- `OutsideDomainError` is raised when the plan is built, on the host, never inside a traced function.
