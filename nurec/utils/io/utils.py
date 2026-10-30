# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

from typing import List, Tuple, TypeAlias

from pxr import Gf, Usd, UsdGeom

from nurec.utils.types import NamedUSDStage


# Each entry represents either a reference to a USD stage, or to a specific prim within that stage.
USDReferences: TypeAlias = List[Tuple[NamedUSDStage, str]]


def initialize_usd_stage():
    stage = Usd.Stage.CreateInMemory()
    stage.SetMetadata("metersPerUnit", 1)
    stage.SetMetadata("upAxis", "Z")

    # Define xform containing everything.
    world_path = "/World"
    UsdGeom.Xform.Define(stage, world_path)
    stage.SetMetadata("defaultPrim", world_path[1:])

    return stage


def sanitize_usd_path(path: str) -> str:
    # Remove any characters that are not alphanumeric, underscore, or forward slash
    sanitized = re.sub(r"[^a-zA-Z0-9_/]", "", path)

    # Replace multiple consecutive forward slashes with a single one
    sanitized = re.sub(r"/+", "/", sanitized)

    # Ensure the path doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized

    return sanitized


def nurec_tf_to_usd_tf(T) -> Gf.Matrix4d:
    return Gf.Matrix4d(T).GetTranspose()
