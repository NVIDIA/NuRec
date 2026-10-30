# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load pre-rename pickles that reference `nre.*` modules.

Lightning checkpoints store `hyper_parameters` with Enum instances whose
`__module__` was `nre.config.*` before the package was renamed to `nurec`.
`install_torch_load_module_remap` gives `torch.load` a default `pickle_module`
whose `Unpickler.find_class` rewrites that module prefix, so a checkpoint saved
by the old package still unpickles. Callers that pass their own `pickle_module`
or that resolve to `weights_only=True` keep their own policy.
"""

import functools
import io
import os
import pickle
import types

from typing import Any

import torch


_OLD_ROOT = "nre"
_NEW_ROOT = "nurec"
_TRUE_VALUES = ("1", "y", "yes", "true")


def remap_module(module: str) -> str:
    """Return `module` with a leading `nre` package root replaced by `nurec`."""
    if module == _OLD_ROOT or module.startswith(_OLD_ROOT + "."):
        return _NEW_ROOT + module[len(_OLD_ROOT) :]
    return module


class LegacyModuleUnpickler(pickle.Unpickler):
    """`pickle.Unpickler` that resolves `nre.*` globals from `nurec.*`."""

    def find_class(self, module: str, name: str) -> Any:
        return super().find_class(remap_module(module), name)


def _legacy_load(file, **kwargs):
    return LegacyModuleUnpickler(file, **kwargs).load()


def _legacy_loads(data, **kwargs):
    return LegacyModuleUnpickler(io.BytesIO(data), **kwargs).load()


# torch.load inspects `pickle_module.__name__`, so this has to be a real module object
# exposing the subset of the `pickle` interface that torch.load uses.
LEGACY_PICKLE_MODULE = types.ModuleType(__name__ + ".pickle_module")
LEGACY_PICKLE_MODULE.Unpickler = LegacyModuleUnpickler  # type: ignore[attr-defined]
LEGACY_PICKLE_MODULE.UnpicklingError = pickle.UnpicklingError  # type: ignore[attr-defined]
LEGACY_PICKLE_MODULE.load = _legacy_load  # type: ignore[attr-defined]
LEGACY_PICKLE_MODULE.loads = _legacy_loads  # type: ignore[attr-defined]


def _resolves_to_full_unpickle(pickle_module: Any, weights_only: bool | None) -> bool:
    """Mirror `torch.load`'s policy resolution: True only when it would fall back to the `pickle` module."""
    if pickle_module is not None:
        return False
    if os.getenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0") in _TRUE_VALUES:
        return False
    if weights_only is None:
        return os.getenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "0") in _TRUE_VALUES
    return not weights_only


def install_torch_load_module_remap() -> None:
    """Make `torch.load` unpickle `nre.*` globals as `nurec.*` whenever it would use the plain `pickle` module.

    Idempotent. Only the default policy changes: an explicit `pickle_module`,
    an explicit `weights_only=True`, or `TORCH_FORCE_WEIGHTS_ONLY_LOAD` leave
    `torch.load` untouched.
    """
    original = torch.load
    if getattr(original, "__nurec_legacy_module_remap__", False):
        return

    @functools.wraps(original)
    def load(f, map_location=None, pickle_module=None, *, weights_only=None, **kwargs):
        if _resolves_to_full_unpickle(pickle_module, weights_only):
            pickle_module, weights_only = LEGACY_PICKLE_MODULE, False
        return original(f, map_location, pickle_module, weights_only=weights_only, **kwargs)

    load.__nurec_legacy_module_remap__ = True  # type: ignore[attr-defined]
    torch.load = load
