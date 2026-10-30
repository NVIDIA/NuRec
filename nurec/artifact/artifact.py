# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import copy
import glob
import hashlib
import json
import logging
import zipfile

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Dict, FrozenSet, Generator, List, Optional

import numpy as np
import yaml


log = logging.getLogger(__name__)


# Filenames of the per-class semantic metadata side files inside the USDZ artifact.
# Centralized here so the system-side exporter, the artifact-side loader, and any tests
# reference the same string.
SEMANTIC_CLASS_NAMES_FILENAME = "semantic_class_names.json"
SEMANTIC_CLASS_COLORS_FILENAME = "semantic_class_colors.json"
TRACK_SPLIT_METADATA_FILENAME = "track_split_metadata.json"
_CHECKPOINT_NAME = "checkpoint.ckpt"
# Keep this local so the artifact reader does not depend on compression.
_COMPACT_PAYLOAD_NAME = "compact.safetensors"
# The directory of measured firing patterns, written by nurec.utils.io.firing_patterns.
FIRING_PATTERNS_DIR = "firing_patterns"


@dataclass
class Artifact:
    """
    Represents an NuRec artifact: a .usdz file which includes the state dict.

    The artifact fields are lazy-loaded and cached. Use `with artifact.temporary_cache(): ...` to automatically clear the cache after the block.
    """

    source: Path

    # for caching
    _checkpoint: BytesIO | None = field(default=None, init=False, repr=False)
    _data_info: Dict[str, Any] | None = field(default=None, init=False, repr=False)
    _datasource_summary: Dict[str, Any] | None = field(default=None, init=False, repr=False)
    _mesh_ply: bytes | None = field(default=None, init=False, repr=False)
    _ground_mesh_ply: bytes | None = field(default=None, init=False, repr=False)
    _metadata: Dict[str, Any] | None = field(default=None, init=False, repr=False)
    _nrend_checkpoint: bytes | None = field(default=None, init=False, repr=False)
    _parsed_config: Dict[str, Any] | None = field(default=None, init=False, repr=False)
    _rig_trajectories: Dict[str, Any] | None = field(default=None, init=False, repr=False)
    _custom_rig_trajectories: Dict[str, Any] | None = field(default=None, init=False, repr=False)
    _sequence_tracks: Dict[str, Any] | None = field(default=None, init=False, repr=False)
    _track_split_metadata: Dict[str, Any] | None = field(default=None, init=False, repr=False)
    _track_split_metadata_loaded: bool = field(default=False, init=False, repr=False)
    _external_assets: Dict[str, Dict[str, Any]] | None = field(default=None, init=False, repr=False)
    _firing_patterns: Dict[str, bytes] | None = field(default=None, init=False, repr=False)
    _semantic_class_names: Dict[int, str] | None = field(default=None, init=False, repr=False)
    _semantic_class_names_loaded: bool = field(default=False, init=False, repr=False)
    _semantic_class_colors: np.ndarray | None = field(default=None, init=False, repr=False)
    _semantic_class_colors_loaded: bool = field(default=False, init=False, repr=False)
    _zip_namelist: FrozenSet[str] | None = field(default=None, init=False, repr=False)
    _sha256_footprint: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        assert self.source.suffix == ".usdz"

    @staticmethod
    def discover_from_glob(glob_query: str, recursive: bool = True) -> List[Artifact]:
        """
        A factory method to create artifact instances
        """
        assert glob_query.endswith(".usdz"), (
            f"glob query needs to end in .usdz to find valid artifacts (got {glob_query=})."
        )
        return [Artifact(Path(path)) for path in glob.glob(glob_query, recursive=recursive)]

    def clear_cache(self) -> None:
        """
        Clears the cache for the artifact. Use `with artifact.temporary_cache(): ...` to automatically clear the cache after the block.
        """
        self._checkpoint = None
        self._data_info = None
        self._datasource_summary = None
        self._mesh_ply = None
        self._ground_mesh_ply = None
        self._metadata = None
        self._nrend_checkpoint = None
        self._parsed_config = None
        self._rig_trajectories = None
        self._custom_rig_trajectories = None
        self._sequence_tracks = None
        self._track_split_metadata = None
        self._track_split_metadata_loaded = False
        self._external_assets = None
        self._firing_patterns = None
        self._semantic_class_names = None
        self._semantic_class_names_loaded = False
        self._semantic_class_colors = None
        self._semantic_class_colors_loaded = False
        self._zip_namelist = None
        self._sha256_footprint = None

    @contextlib.contextmanager
    def temporary_cache(self) -> Generator[Artifact, None, None]:
        """
        Enters a context after which the artifact cache is automatically cleared.
        """

        try:
            yield self
        finally:
            self.clear_cache()

    @property
    def checkpoint(self) -> BytesIO:
        if self._checkpoint is None:
            if self.is_compact:
                raise ValueError("compact USDZ artifacts do not contain checkpoint.ckpt; use the compact loader")
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._checkpoint = BytesIO(zip_file.read(_CHECKPOINT_NAME))

        self._checkpoint.seek(0)  # reset the stream to the beginning if reusing the cache
        return self._checkpoint

    @property
    def is_compact(self) -> bool:
        """Return true if this USDZ stores compact tensors instead of checkpoint.ckpt."""
        return _COMPACT_PAYLOAD_NAME in self._get_zip_namelist()

    @property
    def data_info(self) -> Dict[str, Any]:
        if self._data_info is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._data_info = json.load(zip_file.open("data_info.json"))

        return self._data_info

    @property
    def datasource_summary(self) -> Dict[str, Any]:
        if self._datasource_summary is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._datasource_summary = json.load(zip_file.open("datasource_summary.json"))

        return self._datasource_summary

    @property
    def mesh_ply(self) -> bytes:
        if self._mesh_ply is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._mesh_ply = zip_file.read("mesh.ply")

        return self._mesh_ply

    @property
    def ground_mesh_ply(self) -> bytes:
        if self._ground_mesh_ply is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._ground_mesh_ply = zip_file.read("mesh_ground.ply")

        return self._ground_mesh_ply

    @property
    def metadata(self) -> Dict[str, Any]:
        if self._metadata is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._metadata = yaml.safe_load(zip_file.open("metadata.yaml"))

        return self._metadata

    @property
    def nrend_checkpoint(self) -> bytes:
        if self._nrend_checkpoint is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._nrend_checkpoint = zip_file.read("volume.nurec")

        return self._nrend_checkpoint

    @property
    def parsed_config(self) -> Dict[str, Any]:
        # It happens frequently that an old artifact is used, so the config returned here is a parsed, versioned config
        # from an earlier version of the program. Once obtained, you need to upgrade this config to the current version
        # by applying upgrade_config() to it. We do not apply this prior to returning the config because you may also
        # want to apply upgrade_model() which requires the pre-upgrade config returned here.
        if self._parsed_config is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._parsed_config = yaml.safe_load(zip_file.open("parsed_config.yaml"))

        return self._parsed_config

    @property
    def rig_trajectories(self) -> Dict[str, Any]:
        if self._rig_trajectories is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._rig_trajectories = json.load(zip_file.open("rig_trajectories.json"))

        return self._rig_trajectories

    def load_custom_rig_trajectories(self, custom_rig_trajectories_path: str) -> None:
        """
        Load custom rig trajectories from an external file without overwriting the original artifact trajectories.

        Args:
            custom_rig_trajectories_path: Path to a JSON file containing custom rig trajectories
        """
        with open(custom_rig_trajectories_path, "r") as file:
            self._custom_rig_trajectories = json.load(file)

    @property
    def custom_rig_trajectories(self) -> Dict[str, Any]:
        """
        Get the loaded custom rig trajectories. Must call load_custom_rig_trajectories() first.

        Returns:
            The custom rig trajectories as a dictionary

        Raises:
            ValueError: If custom rig trajectories haven't been loaded yet
        """
        if self._custom_rig_trajectories is None:
            raise ValueError("Custom rig trajectories not loaded. Call load_custom_rig_trajectories(path) first.")
        return self._custom_rig_trajectories

    @property
    def sequence_tracks(self) -> Dict[str, Any]:
        if self._sequence_tracks is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._sequence_tracks = json.load(zip_file.open("sequence_tracks.json"))

        return self._sequence_tracks

    @property
    def track_split_metadata(self) -> Optional[Dict[str, Any]]:
        if self._track_split_metadata_loaded:
            return copy.deepcopy(self._track_split_metadata)

        if TRACK_SPLIT_METADATA_FILENAME in self._get_zip_namelist():
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._track_split_metadata = json.load(zip_file.open(TRACK_SPLIT_METADATA_FILENAME))

        self._track_split_metadata_loaded = True
        return copy.deepcopy(self._track_split_metadata)

    @property
    def external_assets(self) -> Dict[str, Dict[str, Any]]:
        """
        Lazy-loaded external assets mapping track_id to PLY bytes.
        Expected structure in artifact: external_assets/{track_id}/{track_id}.ply
        """
        if self._external_assets is None:
            self._external_assets = {}
            with zipfile.ZipFile(self.source, "r") as zip_file:
                metadata_dict = {}
                if "external_assets/metadata.yaml" in zip_file.namelist():
                    metadata_content = yaml.safe_load(zip_file.open("external_assets/metadata.yaml"))
                    if "assets" in metadata_content:
                        for asset_info in metadata_content["assets"]:
                            track_id = asset_info["track_id"]
                            metadata_dict[track_id] = {
                                "label_class": asset_info.get("label_class"),
                                "cuboids_dims": asset_info.get("cuboids_dims"),
                            }

                for file_name in zip_file.namelist():
                    path = PurePosixPath(file_name)

                    # Expected pattern: external_assets / {track_id} / {track_id}.ply
                    if path.parts[0] != "external_assets":
                        continue
                    if len(path.parts) != 3:
                        log.warning(f"Skipping external_assets zip entry with unexpected depth: '{file_name}'")
                        continue

                    track_id = path.parts[1]

                    if path.parts[2] != f"{track_id}.ply":
                        log.warning(
                            f"Skipping zip entry with unexpected filename: '{file_name}' (expected '{track_id}.ply')"
                        )
                        continue

                    # Read the PLY file bytes
                    self._external_assets[track_id] = {
                        "ply_bytes": zip_file.read(file_name),
                        "metadata": metadata_dict.get(track_id, {}),
                    }

        return self._external_assets

    @property
    def firing_patterns(self) -> Dict[str, bytes]:
        """Lazy-loaded measured firing patterns, mapping logical lidar id to ``.npz`` bytes.

        Expected structure in artifact: firing_patterns/{logical_lidar_id}.npz. A lidar with a
        parametric sensor model does not need one, so the mapping is normally empty or partial.
        """
        if self._firing_patterns is None:
            self._firing_patterns = {}
            with zipfile.ZipFile(self.source, "r") as zip_file:
                for file_name in zip_file.namelist():
                    path = PurePosixPath(file_name)
                    if path.parts[0] != FIRING_PATTERNS_DIR:
                        continue
                    if len(path.parts) != 2 or path.suffix != ".npz":
                        log.warning(f"Skipping unexpected firing_patterns zip entry: '{file_name}'")
                        continue
                    self._firing_patterns[path.stem] = zip_file.read(file_name)

        return self._firing_patterns

    def _get_zip_namelist(self) -> FrozenSet[str]:
        """Return the set of entry names in the zip, cached after the first read."""
        if self._zip_namelist is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._zip_namelist = frozenset(zip_file.namelist())
        return self._zip_namelist

    @property
    def semantic_class_names(self) -> Optional[Dict[int, str]]:
        """
        ``{class_id: class_name}`` for semantics rendered by this artifact, if packaged.

        Written at export when the datasource has camera semantic segmentation aux data. IDs need
        not be contiguous — sparse mappings are preserved so renderers can label each class id
        independently of its position. When :attr:`semantic_class_colors` is also present,
        rendering can match validation ``stuff_colors``.
        """
        if self._semantic_class_names_loaded:
            return dict(self._semantic_class_names) if self._semantic_class_names else None

        # Accepted on-disk shapes (any other shape is warned and treated as "file present but
        # unparseable", so a malformed/empty file does not silently disable downstream class-color
        # matching):
        #   - ``{"<id>": "name", ...}`` — current format, JSON object keys are str-encoded ids.
        #   - ``["name0", "name1", ...]`` — legacy: positional list, class id = list index.
        #   - ``{"class_names": ["name0", "name1", ...]}`` — legacy: same as above under a key.
        # For older artifacts or those trained without semantics, it is expected the file is
        # absent, so we skip loading silently.
        names: Optional[Dict[int, str]] = None
        if SEMANTIC_CLASS_NAMES_FILENAME in self._get_zip_namelist():
            with zipfile.ZipFile(self.source, "r") as zip_file:
                data = json.load(zip_file.open(SEMANTIC_CLASS_NAMES_FILENAME))
            if isinstance(data, list):
                if data and all(isinstance(v, str) for v in data):
                    names = {i: v for i, v in enumerate(data)}
            elif isinstance(data, dict):
                raw_names = data.get("class_names")
                if isinstance(raw_names, (list, tuple)) and raw_names and all(isinstance(v, str) for v in raw_names):
                    names = {i: v for i, v in enumerate(raw_names)}
                elif (
                    "class_names" not in data
                    and data
                    and all(
                        isinstance(k, str) and k.lstrip("-").isdigit() and isinstance(v, str) for k, v in data.items()
                    )
                ):
                    names = {int(k): v for k, v in data.items()}
            if names is None:
                log.warning(
                    "%s present but no class names parsed (root type=%s); ignoring",
                    SEMANTIC_CLASS_NAMES_FILENAME,
                    type(data).__name__,
                )

        # Set the value field first, then mark loaded — concurrent or post-exception readers
        # never observe ``_loaded=True`` with the field still in its default state.
        self._semantic_class_names = names
        self._semantic_class_names_loaded = True
        return dict(self._semantic_class_names) if self._semantic_class_names else None

    @property
    def semantic_class_colors(self) -> Optional[np.ndarray]:
        """
        (N, 3) uint8 RGB per class id (same indexing as :attr:`semantic_class_names`), if packaged.

        Written at export from NCORE camera aux ``stuff_colors`` so semantics rendering matches
        validation coloring. JSON is either a list of ``[r, g, b]`` or ``{"colors": [...]}``.
        """
        if self._semantic_class_colors_loaded:
            return self._semantic_class_colors.copy() if self._semantic_class_colors is not None else None

        # Accept a top-level non-empty list of [r, g, b] triples or {"colors": non-empty list}.
        # Anything else — empty list, wrong root type, dict missing the key, or correct list
        # shape but wrong array shape — is treated as "file present but unparseable" and warned
        # exactly once so a malformed file does not silently disable downstream class-color
        # matching. For older artifacts or those trained without semantics, it is
        # expected semantics files will not be present, so we skip silently.
        colors: Optional[np.ndarray] = None
        if SEMANTIC_CLASS_COLORS_FILENAME in self._get_zip_namelist():
            with zipfile.ZipFile(self.source, "r") as zip_file:
                raw = json.load(zip_file.open(SEMANTIC_CLASS_COLORS_FILENAME))
            rows = raw["colors"] if isinstance(raw, dict) and "colors" in raw else raw
            if isinstance(rows, list) and rows:
                arr = np.asarray(rows, dtype=np.uint8)
                if arr.ndim == 2 and arr.shape[1] == 3:
                    colors = arr
                else:
                    log.warning(
                        "%s has invalid shape %s; expected (N, 3)",
                        SEMANTIC_CLASS_COLORS_FILENAME,
                        arr.shape,
                    )
            else:
                log.warning(
                    "%s present but no colors parsed (root type=%s); ignoring",
                    SEMANTIC_CLASS_COLORS_FILENAME,
                    type(raw).__name__,
                )

        # Set the value field first, then mark loaded — concurrent or post-exception readers
        # never observe ``_loaded=True`` with the field still in its default state.
        self._semantic_class_colors = colors
        self._semantic_class_colors_loaded = True
        return self._semantic_class_colors.copy() if self._semantic_class_colors is not None else None

    @property
    def scene_id(self) -> str:
        """
        The name used to identify the scene via `scene_id=` when requesting
        """
        return self.metadata["scene_id"]

    @property
    def sha256_footprint(self) -> str:
        """
        SHA256 hex digest of the artifact's ``source`` file, computed lazily on first access
        (read in chunks) and cached thereafter.
        """
        if self._sha256_footprint is None:
            digest = hashlib.sha256()
            chunk_size = 1 << 20
            with open(self.source, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    digest.update(chunk)
            self._sha256_footprint = digest.hexdigest()

        return self._sha256_footprint
