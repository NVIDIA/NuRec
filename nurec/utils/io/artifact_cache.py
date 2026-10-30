# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import zipfile

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import List
from uuid import uuid4

from nurec.config.checkpoint import ArtifactConfig
from nurec.utils.io.usd_default_layer import serialize_usd_default_layer
from nurec.utils.io.utils import USDReferences
from nurec.utils.misc import dataclass_items
from nurec.utils.types import ArtifactContents, NamedSerialized, NamedUSDStage


logger = logging.getLogger(__name__)


def publish_existing_usdz(source_path: Path, destination_path: Path) -> None:
    """Atomically publish an immutable numbered USDZ as ``last.usdz``.

    ``ArtifactCache.write_to_usdz`` replaces numbered artifacts atomically, so
    subsequent retries install a new inode. A hard link therefore avoids a
    full artifact copy without exposing partial or retried ZIP contents through
    an already-published ``last.usdz``.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_name(f".{destination_path.name}.{uuid4().hex}.tmp")
    try:
        os.link(source_path, temporary_path)
        os.replace(temporary_path, destination_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    logger.info("File %s hard-linked from %s.", destination_path, source_path)


@dataclass
class ArtifactCache:
    artifact_config: ArtifactConfig

    checkpoint: NamedSerialized | None = None
    parsed_config: NamedSerialized | None = None

    data_info: NamedSerialized | None = None
    metadata: NamedSerialized | None = None
    datasource_summary: NamedSerialized | None = None

    meshes: ArtifactContents | None = None
    # Proxy meshes are all USD meshes in `meshes`, augmented with USD-specific information
    proxy_meshes: USDReferences | None = None

    rig_trajectories: ArtifactContents | None = None
    firing_patterns: ArtifactContents | None = None
    sequence_tracks: ArtifactContents | None = None
    track_split_metadata: NamedSerialized | None = None
    # USD references of the sequence tracks to be associated to the NeRF prim
    usd_sequence_tracks: USDReferences | None = None

    nrend_checkpoint: ArtifactContents | None = None

    # Ordered class names for semantic segmentation (index i -> name); optional
    semantic_class_names: NamedSerialized | None = None
    # Per-class RGB (index i -> [r,g,b]) from NCORE aux stuff_colors; optional
    semantic_class_colors: NamedSerialized | None = None

    def archive_members(self) -> list[NamedSerialized | NamedUSDStage]:
        def is_artifact_contents(obj) -> bool:
            if not isinstance(obj, Sequence):
                return False
            return all(isinstance(item, (NamedSerialized, NamedUSDStage)) for item in obj)

        # filter from the root stage the meshes and sequences_tracks which are included under the NeRF XForm
        def include_usd_stage(element: NamedUSDStage):
            return (
                isinstance(element, NamedUSDStage)
                and element not in [stage for stage, _ in self.proxy_meshes or []]
                and element not in [stage for stage, _ in self.usd_sequence_tracks or []]
            )

        # Discover USD files in dataclass items.
        # Assume that all files are stored in the same directory, hence the path is equal to the filename.
        refs: List[NamedUSDStage] = []
        for _, member in dataclass_items(self):
            if include_usd_stage(member):
                refs.append(member)
            elif is_artifact_contents(member):
                for member_list_elem in member:
                    if include_usd_stage(member_list_elem):
                        refs.append(member_list_elem)

        usd_default_layer = NamedUSDStage(filename="default.usda", stage=serialize_usd_default_layer(refs))
        members: list[NamedSerialized | NamedUSDStage] = [usd_default_layer]

        for _, member in dataclass_items(self):
            if member is not None:
                if isinstance(member, NamedSerialized) or isinstance(member, NamedUSDStage):
                    members.append(member)
                elif is_artifact_contents(member):
                    members.extend(member)

        return members

    def write_to_usdz(self, file_path: Path) -> None:
        # Make sure path to usdz-file exists, as zipfile.Zipfile(...) can only create files, not directories
        file_path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = file_path.with_name(f".{file_path.name}.{uuid4().hex}.tmp")
        try:
            with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_STORED) as zip_file:
                for member in self.archive_members():
                    member.save_to_zip(zip_file)
            os.replace(temporary_path, file_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        logger.info(f"File {file_path} created successfully.")
