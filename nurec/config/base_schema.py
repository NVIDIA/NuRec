# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Iterable, Mapping, TypeVar, Union, cast

from omegaconf import DictConfig, OmegaConf
from pydantic import (
    BaseModel as PydanticBaseModel,
)
from pydantic import (
    Field,  # reexport
)


__all__ = [
    "BaseConfigSchema",
    "Field",
    "config_to_primitive",
    "PerSensor",
    "resolve_per_sensor",
    "validate_per_sensor_coverage",
]

_T = TypeVar("_T")

# A per-sensor-configurable value: either a single value that applies to ALL
# active sensors (broadcast), or a dict keyed by sensor id (e.g. lidar_id) that
# gives a value per sensor. When a dict is used it must list EVERY active sensor
# id (enforced by validate_per_sensor_coverage at config time) -- there is no
# fallback key; use a single value to apply the same setting to all sensors.
#
# NOTE: the scalar-vs-dict distinction is made structurally (``isinstance`` of a
# mapping), so ``PerSensor[T]`` is only unambiguous when ``T`` is itself NOT a
# mapping type. Do not use ``PerSensor[dict[...]]`` (or ``PerSensor[DictConfig]``):
# a broadcast scalar of a mapping type would be misread as a per-sensor dict.
PerSensor = Union[_T, dict[str, _T]]


# Sentinel distinguishing "no default supplied" (-> raise on miss) from an explicit
# default value (which may legitimately be None for optional fields).
_MISSING: Any = object()

# Mapping types treated as a per-sensor dict. ``omegaconf.DictConfig`` does NOT
# subclass ``dict``, so an OmegaConf-loaded per-sensor field would otherwise be
# misread as a broadcast scalar; include it explicitly.
_PER_SENSOR_DICT_TYPES = (dict, DictConfig)


# The parameter is typed ``_T | Mapping[str, _T]`` rather than ``PerSensor[_T]``
# because ``dict`` is INVARIANT in its value type while ``Mapping`` is COVARIANT.
# Most PerSensor fields are ``Literal``-typed (lidar_ray_mode, box_method, the
# raydrop virtual-ray mode, ...), and ``dict[str, Literal["ut", "pbf", "hybrid"]]``
# is not assignable to ``dict[str, str]``, so solving ``_T`` against
# ``PerSensor[_T]`` fails for every one of them. ``Mapping[str, Literal[...]]``
# IS assignable to ``Mapping[str, str]``, so the covariant spelling accepts them.
#
# This widens only what the FUNCTION accepts. ``PerSensor`` itself stays
# ``Union[_T, dict[str, _T]]``, so the field annotations -- and therefore what
# pydantic will actually construct and hand to this function -- still admit only
# ``dict``/``DictConfig``, which is what ``_PER_SENSOR_DICT_TYPES`` below tests
# for. A non-dict ``Mapping`` cannot reach here through a validated config.
def resolve_per_sensor(value: _T | Mapping[str, _T], sensor_id: str | None, *, default: Any = _MISSING) -> _T:
    """Resolve a ``PerSensor`` value for a specific sensor id.

    Variants:
    - A non-dict ``value`` broadcasts to every sensor (so ``sensor_id`` may be
      ``None`` when the caller only has a scalar value -- e.g. a renderer before it
      can map a frame to a lidar sensor id).
    - A dict (plain ``dict`` or ``omegaconf.DictConfig``) is looked up by
      ``sensor_id``. On a miss:
      - if a ``default`` value was supplied, return it;
      - otherwise raise (so a misconfigured required dict fails loudly rather than
        silently using a wrong value). In particular, passing ``sensor_id=None``
        against a dict is a miss (there is no ``None`` key), so it returns
        ``default`` if given, else raises.

    ``default`` is typed ``Any`` so passing it does not over-constrain ``_T`` (e.g.
    ``default=None`` on a ``PerSensor[Optional[list[float]]]`` field).

    Required fields are coverage-checked up front by ``validate_per_sensor_coverage``,
    so the miss-raise here is only a defensive backstop.
    """

    # broadcast value to all sensors
    if not isinstance(value, _PER_SENSOR_DICT_TYPES):
        return cast(_T, value)

    # dict value: look up by sensor_id
    per_sensor = cast(dict[str, _T], value)
    if sensor_id in per_sensor:
        return per_sensor[sensor_id]

    # default fallback on a miss, or raise if no default was supplied
    if default is not _MISSING:
        return default

    raise ValueError(
        f"per-sensor value has no entry for sensor '{sensor_id}'; got keys "
        f"{sorted(per_sensor)}. List every active sensor id, or use a single value "
        f"to apply the same setting to all sensors."
    )


def validate_per_sensor_coverage(
    value: PerSensor[Any], active_ids: Iterable[str], *, field_name: str = "value"
) -> None:
    """Validate a ``PerSensor`` dict matches the active sensor ids exactly.

    A non-dict value (broadcast) is always valid. A dict (plain ``dict`` or
    ``omegaconf.DictConfig``) is valid if its keys are EXACTLY ``active_ids``:
      - every id in ``active_ids`` must have an entry (no silent miss at runtime);
      - no extra/unknown keys are allowed (a typo'd or decommissioned sensor id
        would otherwise pass validation and never be applied).
    Raises otherwise.
    """
    if not isinstance(value, _PER_SENSOR_DICT_TYPES):
        return

    active = list(active_ids)
    active_set = set(active)
    keys = list(value)
    missing = [sid for sid in active if sid not in value]
    extra = [k for k in keys if k not in active_set]
    if missing or extra:
        problems = []
        if missing:
            problems.append(f"missing entries for {missing}")
        if extra:
            problems.append(f"unknown sensor keys {extra}")
        raise ValueError(
            f"{field_name}: per-sensor dict must match the active sensor ids exactly "
            f"(active={sorted(active_set)}, keys={sorted(keys)}): {'; '.join(problems)}. "
            f"List every active sensor id (and only those), or use a single value to "
            f"apply the same setting to all sensors."
        )


def config_to_primitive(config, resolve=True):
    """Convert OmegaConf/DictConfig or Pydantic model to plain Python dict/list.

    Safe to import from nurec.models etc. (no nurec.config.nurec).
    """
    if config is None:
        return None
    if isinstance(config, BaseConfigSchema):
        return config.model_dump(exclude_none=True)
    if isinstance(config, PydanticBaseModel):
        return config.model_dump()
    if isinstance(config, dict):
        return dict(config)
    if isinstance(config, list):
        return list(config)
    return OmegaConf.to_container(config, resolve=resolve)


class BaseConfigSchema(PydanticBaseModel):
    """
    The base class for NuRec config structs, acts very similar to python's dataclasses.
    We extend it with an extra method which converts a typed config into an untyped one
    for use with modules which have not yet been reworked.
    """

    def to_dictconfig(self) -> DictConfig:
        # TODO: remove once the entire repo has transitioned to strongly typed config.
        return DictConfig(self.model_dump())

    def __hash__(self) -> int:
        return hash((type(self), self.__repr__()))
