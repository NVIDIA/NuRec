# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from pathlib import Path
from typing import Any, Optional


# Sentinel value for `predict.export_ply.save_to_dir` that resolves to the NCORE dataset directory.
DATASET_DIRECTORY = "dataset_directory"


def resolve_export_ply_base_dir(
    save_to_dir: Optional[str],
    *,
    out_dir: str,
    run_id: str,
    meta: dict[str, Any] | None = None,
) -> str:
    if save_to_dir is None:
        return os.path.join(out_dir, run_id)

    if save_to_dir == DATASET_DIRECTORY:
        if meta is None or "ncore_json_path" not in meta:
            raise ValueError(f"export_ply.save_to_dir={DATASET_DIRECTORY!r} requires ncore_json_path in batch metadata")
        dataset_dir = Path(str(meta["ncore_json_path"])).parent
        return os.path.join(str(dataset_dir), "instant_nurec")

    return save_to_dir
