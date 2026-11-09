"""Strict-JSON configs for modules made of static fields: schemes and policies.

A config is a dict of JSON values (str, int, float, bool, None, list, dict)
with the class name under ``"class"`` and one entry per constructor field, so
``json.dumps(config, allow_nan=False)`` always succeeds. Classes are found by
name among the loaded subclasses of a root class, so a new scheme needs no
entry in any table: defining it is registering it.
"""

from __future__ import annotations

from functax._src._backend import require_jax


require_jax(__name__)

import dataclasses
import inspect
import math
from typing import Any

import equinox as eqx

from functax.exceptions import ValidationError


__all__ = [
    "build_from_config",
    "config_of",
    "encode_json_value",
]


# The non-finite floats strict JSON cannot hold, written as {"float": text}
_NON_FINITE = ("nan", "inf", "-inf")


def encode_json_value(value: Any, name: str) -> Any:
    """``value`` as strict JSON: nested schemes as configs, tuples as lists.

    A non-finite float becomes ``{"float": "nan"}``, ``{"float": "inf"}`` or
    ``{"float": "-inf"}``, the one tagged object a config holds besides a
    nested config.

    Raises:
        ValidationError: If ``value`` (named ``name`` in the message) has no
            strict-JSON form, such as an array or a callable.
    """
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"float": repr(value)}
    if isinstance(value, tuple | list):
        return [
            encode_json_value(item, f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    get_config = getattr(value, "get_config", None)
    if isinstance(value, eqx.Module) and callable(get_config):
        return get_config()
    msg = f"'{name}' has no strict-JSON form, got {type(value).__name__}"
    raise ValidationError(msg)


def _decode(value: Any, name: str, root: type[eqx.Module]) -> Any:
    """The inverse of `encode_json_value`, with lists back as tuples."""
    if isinstance(value, list):
        return tuple(
            _decode(item, f"{name}[{index}]", root) for index, item in enumerate(value)
        )
    if isinstance(value, dict):
        if "class" in value:
            return build_from_config(value, root=root, expected=root)
        if set(value) == {"float"} and value["float"] in _NON_FINITE:
            return float(value["float"])
        msg = (
            f"'{name}' must be a config with a 'class' or a non-finite float "
            f'such as {{"float": "nan"}}, got {value!r}'
        )
        raise ValidationError(msg)
    if value is None or isinstance(value, bool | int | float | str):
        return value
    msg = f"'{name}' must be a JSON value, got {type(value).__name__}"
    raise ValidationError(msg)


def config_of(module: eqx.Module) -> dict[str, Any]:
    """``{"class": name, field: value, …}`` for every constructor field of ``module``.

    Raises:
        ValidationError: If a field has no strict-JSON form.
    """
    config: dict[str, Any] = {"class": type(module).__name__}
    for field in dataclasses.fields(module):
        if field.init:
            value = getattr(module, field.name)
            config[field.name] = encode_json_value(value, field.name)
    return config


def _concrete_classes(root: type[eqx.Module], name: str) -> list[type[eqx.Module]]:
    """The non-abstract classes called ``name`` among ``root`` and its subclasses."""
    # a set, so a class reached twice through multiple inheritance counts once
    seen: set[type[eqx.Module]] = {root}
    pending = [root]
    while pending:
        candidate = pending.pop()
        for subclass in candidate.__subclasses__():
            if subclass not in seen:
                seen.add(subclass)
                pending.append(subclass)
    return [cls for cls in seen if cls.__name__ == name and not inspect.isabstract(cls)]


def build_from_config(
    config: Any, *, root: type[eqx.Module], expected: type[eqx.Module]
) -> Any:
    """The module a `config_of` config describes, found among ``root``'s subclasses.

    Nested configs (a `Tensor`'s schemes) are resolved among the same ``root``.

    Args:
        config: The decoded config.
        root: The class whose loaded subclasses may be named.
        expected: The class the result must be an instance of.

    Raises:
        ValidationError: If ``config`` is not a dict with a ``"class"`` name,
            names no loaded class, or more than one, or a class other than
            ``expected``; if a key is unknown or a required field is missing.
    """
    if not isinstance(config, dict) or not isinstance(config.get("class"), str):
        msg = f"'config' must be a dict with a 'class' name, got {config!r}"
        raise ValidationError(msg)
    name = config["class"]
    classes = _concrete_classes(root, name)
    if not classes:
        msg = f"'config' names {name!r}, which is not a loaded {root.__name__}"
        raise ValidationError(msg)
    if len(classes) > 1:
        modules = sorted(f"{cls.__module__}.{cls.__qualname__}" for cls in classes)
        msg = f"'config' names {name!r}, which is ambiguous: {modules}"
        raise ValidationError(msg)
    (cls,) = classes
    if not issubclass(cls, expected):
        msg = f"'config' names {name!r}, which is not a {expected.__name__}"
        raise ValidationError(msg)
    fields = {field.name: field for field in dataclasses.fields(cls) if field.init}
    unknown = sorted(set(config) - {"class"} - set(fields))
    if unknown:
        msg = f"'config' has keys {unknown} that {name} does not take"
        raise ValidationError(msg)
    missing = sorted(
        field_name
        for field_name, field in fields.items()
        if field_name not in config
        and field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
    )
    if missing:
        msg = f"'config' is missing {missing}, which {name} needs"
        raise ValidationError(msg)
    arguments = {
        field_name: _decode(config[field_name], field_name, root)
        for field_name in fields
        if field_name in config
    }
    return cls(**arguments)
