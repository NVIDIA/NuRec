# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os

from pathlib import Path

from omegaconf import OmegaConf
from upath import UPath

from nurec.ncore_backends import DatasetRequest, resolve_dataset


def _resolve_sam3_aux_dir(
    dataset_path: str, mask_dir: str | None = None, mask_manifest_path: str | None = None
) -> str | None:
    """Find a local SAM3 artifact using the dataset's sequence identity.

    NDAS merges aux output under ``<metadata-dir>/<sequence-id>/sam3``;
    standalone aux generation commonly places it next to the data stores.
    Explicit mask inputs bypass discovery. Return None if absent, reject
    ambiguous roots, and leave mask/manifest validation to the initializer.
    The resolved value is saved in configs, including an absent artifact.
    """
    if mask_dir is not None or mask_manifest_path is not None:
        return None
    metadata_path = Path(dataset_path)
    if not metadata_path.is_file():
        return None
    metadata = resolve_dataset(DatasetRequest(metadata_path=UPath(metadata_path))).metadata
    candidates = {
        metadata_path.parent / metadata.sequence_id / "sam3",
        metadata_path.parent / "sam3",
        *(Path(str(path)).parent / "sam3" for path in metadata.dataset_paths if path.protocol in ("", "file")),
    }
    roots = sorted({path.resolve() for path in candidates if path.exists()})
    if len(roots) > 1:
        raise ValueError(f"Ambiguous SAM3 aux directories for {dataset_path}: {roots}; set sam3_aux_dir explicitly")
    return str(roots[0]) if roots else None


# Returns the *single* non-None value (it's an error if none or both values are set)
def xor(a: str | None, b: str | None) -> str:
    assert (a is None) != (b is None), f"error resolving xor value, both values are either set or missing [{a} / {b}]"
    match a:
        case None:
            assert isinstance(b, str), "[xor] either a or b (but not both) need to be of type str"
            return b
        case str():
            return a


# Importing this module registers the common OmegaConf resolvers. Keep this
# dependency independent of either top-level typed schema so NuRecConfig can embed
# InstantNuRecConfig without creating a parser import cycle.
OmegaConf.register_new_resolver("sam3_aux_dir", _resolve_sam3_aux_dir)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("calc_exp_lr_decay_rate", lambda factor, n: factor ** (1.0 / n))
OmegaConf.register_new_resolver("add", lambda a, b: a + b)
OmegaConf.register_new_resolver("sub", lambda a, b: a - b)
OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
OmegaConf.register_new_resolver("div", lambda a, b: a / b)
OmegaConf.register_new_resolver("divceil", lambda a, b: int(math.ceil(a / b)))
OmegaConf.register_new_resolver("power", lambda a, b: a**b)
OmegaConf.register_new_resolver("sqrt", lambda a: float(math.sqrt(a)))
OmegaConf.register_new_resolver("min", lambda a, b: min(int(a), int(b)))
OmegaConf.register_new_resolver("dirname", lambda p: os.path.dirname(p))
OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p))
OmegaConf.register_new_resolver("join_path", lambda a, b: os.path.join(a, b))
OmegaConf.register_new_resolver("sum_values", lambda d: sum([v for v in d.values()]))
OmegaConf.register_new_resolver("int", lambda a: int(a))
OmegaConf.register_new_resolver("load_json", lambda path: OmegaConf.create(json.load(open(path))))
OmegaConf.register_new_resolver("xor", lambda a, b: xor(a, b))
OmegaConf.register_new_resolver("and_then", lambda condition, then, or_else: then if condition else or_else)
OmegaConf.register_new_resolver("len", lambda p: len(p))
OmegaConf.register_new_resolver("invalid_key", lambda p, k: k not in p)
