# Exceptions

Every error raised by the package derives from [`FunctaxError`][functax.FunctaxError] *and* from the closest standard-library exception.

![functax's seven exceptions: FunctaxError under Exception; ValidationError, NonlinearSchemeError and MissingExtraError under FunctaxError and ValueError, TypeError and ImportError; GeometryError, PolicyError and OutsideDomainError under ValidationError; each card says when it is raised and in which modules](https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/functax_exceptions/functax_exceptions-light.svg)

The schematic is kept in [geostack](https://github.com/jejjohnson/geostack/tree/main/docs/images/libraries/functax_exceptions) with the stack's other library schematics.

::: functax.exceptions
