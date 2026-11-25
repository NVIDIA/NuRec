# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict SAM3-mask-backed NCore point clouds for vehicle-light initialization."""

from __future__ import annotations

import hashlib
import json
import logging
import zipfile
import zlib

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Iterator, NamedTuple

import cv2
import numpy as np
import torch

import ncore.sensors

from nurec.datasets.tracks import CuboidTracks
from nurec.libs.vren.interface import image_points_to_world_rays_shutter_pose
from nurec.utils.geometry import se3_matrix_to_tquat
from nurec.utils.misc import to_torch
from nurec.utils.ncore_utils import VEHICLE_LIGHT_CLASS_NAME
from nurec.utils.types import PointCloud


if TYPE_CHECKING:
    from nurec.datasets.ncore import NCOREDataSource


logger = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True)
class MaskedTrackPointCloud:
    track_id: str
    point_cloud: PointCloud
    track_pose_index: int


def _count_tracks_overlapping_interval(
    track_start_timestamps_us: torch.Tensor,
    track_end_timestamps_us: torch.Tensor,
    interval_start_us: int,
    interval_end_us: int,
) -> int:
    """Count tracks that could be active during an inclusive timestamp interval.

    Starts/ends are int64 ``[N_tracks]`` microsecond tensors on the same CPU
    or CUDA device. Returns a Python int (zero for empty inputs); CUDA inputs
    synchronize when reading the count. Raises ValueError for unequal shapes.
    """

    if track_start_timestamps_us.shape != track_end_timestamps_us.shape:
        raise ValueError(
            "track_start_timestamps_us and track_end_timestamps_us must have matching shapes, "
            f"got {tuple(track_start_timestamps_us.shape)} and {tuple(track_end_timestamps_us.shape)}"
        )
    overlaps = torch.logical_and(
        track_start_timestamps_us <= interval_end_us,
        track_end_timestamps_us >= interval_start_us,
    )
    return int(torch.count_nonzero(overlaps).item())


def _flatten_ray_intersections(
    intersections_cnt: torch.Tensor,
    intersections_tracks_idx: torch.Tensor,
    intersections_ts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten dense timed intersections into valid distance/index rows.

    Inputs share a device: int32 counts ``[N]``, int32 track indices ``[N, K]``,
    and floating entry/exit distances ``[N, K, 2]`` from the CUDA binding.
    Counts must not exceed capacity K. Returns distances ``[M, 2]`` and
    ``[ray, track]`` indices ``[M, 2]`` with input dtype/device preserved;
    M is the sum of counts. Raises ValueError for incompatible input shapes.
    """

    if intersections_cnt.ndim != 1:
        raise ValueError(f"intersections_cnt must have shape [N], got {tuple(intersections_cnt.shape)}")
    if intersections_tracks_idx.ndim != 2 or intersections_tracks_idx.shape[0] != intersections_cnt.shape[0]:
        raise ValueError(
            "intersections_tracks_idx must have shape [N, K] matching intersections_cnt, "
            f"got {tuple(intersections_tracks_idx.shape)}"
        )
    expected_ts_shape = (*intersections_tracks_idx.shape, 2)
    if intersections_ts.shape != expected_ts_shape:
        raise ValueError(f"intersections_ts must have shape {expected_ts_shape}, got {tuple(intersections_ts.shape)}")

    slot_idxs = torch.arange(intersections_tracks_idx.shape[1], device=intersections_cnt.device)
    valid_slots = slot_idxs.unsqueeze(0) < intersections_cnt.unsqueeze(1)
    ray_idxs, slot_idxs = valid_slots.nonzero(as_tuple=True)
    track_idxs = intersections_tracks_idx[ray_idxs, slot_idxs]
    intersection_idxs = torch.stack((ray_idxs, track_idxs.long()), dim=1).to(intersections_tracks_idx.dtype)
    return intersections_ts[ray_idxs, slot_idxs], intersection_idxs


def _select_frontmost_intersection_rows(
    intersection_ts: torch.Tensor,
    intersection_idxs: torch.Tensor,
) -> torch.Tensor:
    """Return one frontmost cuboid-intersection row for every intersecting ray.

    Inputs are floating entry/exit distances and integer ``[ray, track]``
    indices, both ``[M, 2]`` on the same device with one row per ray/track pair.
    Entry distances are clamped to zero; ties choose the lower track index.
    Returns int64 ``[N_hit_rays]`` row indices on the input device, including
    an empty tensor for no hits. Raises ValueError for incompatible shapes.
    """

    if intersection_ts.ndim != 2 or intersection_ts.shape[1] != 2:
        raise ValueError(f"intersection_ts must have shape [N, 2], got {tuple(intersection_ts.shape)}")
    if intersection_idxs.ndim != 2 or intersection_idxs.shape != intersection_ts.shape:
        raise ValueError(
            "intersection_idxs must have the same [N, 2] shape as intersection_ts, "
            f"got {tuple(intersection_idxs.shape)}"
        )
    if intersection_ts.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=intersection_ts.device)

    ray_idxs = intersection_idxs[:, 0].long()
    track_idxs = intersection_idxs[:, 1].long()
    entry_ts = torch.clamp_min(intersection_ts[:, 0], 0)
    n_hit_rays = int(ray_idxs.max().item()) + 1

    frontmost_ts = torch.full(
        (n_hit_rays,),
        float("inf"),
        dtype=entry_ts.dtype,
        device=entry_ts.device,
    )
    frontmost_ts.scatter_reduce_(0, ray_idxs, entry_ts, reduce="amin", include_self=True)
    is_frontmost = entry_ts == frontmost_ts[ray_idxs]

    # A boundary ray can enter two touching cuboids at the same distance. Keep
    # the lower track index so every ray still has exactly one deterministic owner.
    sentinel = torch.iinfo(track_idxs.dtype).max
    frontmost_track = torch.full((n_hit_rays,), sentinel, dtype=track_idxs.dtype, device=track_idxs.device)
    frontmost_track.scatter_reduce_(
        0,
        ray_idxs,
        torch.where(is_frontmost, track_idxs, sentinel),
        reduce="amin",
        include_self=True,
    )
    selected = torch.logical_and(is_frontmost, track_idxs == frontmost_track[ray_idxs])
    return selected.nonzero(as_tuple=False).flatten()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_identity_sha256(
    container: dict,
    *,
    field_owner: str,
    sampled_frames: list,
) -> str:
    identity_sha256 = container.get("sampled_frame_identity_sha256")
    if (
        not isinstance(identity_sha256, str)
        or len(identity_sha256) != 64
        or any(character not in "0123456789abcdef" for character in identity_sha256)
    ):
        raise ValueError(f"SAM3 mask manifest has invalid {field_owner} sampled_frame_identity_sha256")
    expected = _canonical_sha256(sampled_frames)
    if identity_sha256 != expected:
        raise ValueError(
            f"SAM3 mask manifest {field_owner} sampled_frame_identity_sha256 mismatch: "
            f"expected={expected} actual={identity_sha256}"
        )
    return identity_sha256


def _load_class_pixel_mask(
    mask_path: Path,
    *,
    frame_width: int,
    frame_height: int,
    camera_id: str,
    label_id: int,
) -> np.ndarray:
    """Read a class map and select one label after nearest-neighbor resizing.

    Args:
        mask_path: Lossless single-channel uint8 PNG.
        frame_width: Required source-frame width.
        frame_height: Required source-frame height.
        camera_id: Camera identifier used in shape errors.
        label_id: Semantic class ID to select.

    Returns:
        Boolean [frame_height, frame_width] mask for the requested class.

    Raises:
        FileNotFoundError: If the PNG cannot be read.
        ValueError: If storage dtype, channels or aspect ratio violate the class-map contract.
    """
    mask_image = _read_class_mask_png(mask_path)
    if mask_image.shape != (frame_height, frame_width):
        mask_image = _resize_mask_image(
            mask_image,
            mask_path=mask_path,
            frame_width=frame_width,
            frame_height=frame_height,
            camera_id=camera_id,
        )
    return mask_image == label_id


def _read_class_mask_png(mask_path: Path) -> np.ndarray:
    """Read a class-map PNG at its stored resolution as a uint8 [H, W] image."""

    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_image is None:
        raise FileNotFoundError(f"Required pixel mask is missing: {mask_path}")
    if mask_image.ndim != 2:
        raise ValueError(f"Mask {mask_path} must be single-channel, got shape {mask_image.shape}")
    if mask_image.dtype != np.uint8:
        raise ValueError(f"Mask {mask_path} must use uint8 storage, got {mask_image.dtype}")
    return mask_image


def _resize_mask_image(
    mask_image: np.ndarray,
    *,
    mask_path: Path,
    frame_width: int,
    frame_height: int,
    camera_id: str,
) -> np.ndarray:
    mask_height, mask_width = mask_image.shape
    if mask_width * frame_height != frame_width * mask_height:
        raise ValueError(
            f"Mask {mask_path} has incompatible aspect ratio {mask_image.shape}; "
            f"expected {(frame_height, frame_width)} for {camera_id}"
        )
    return cv2.resize(mask_image, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)


# label_schema field naming the per-frame mask file format, a missing field means PNG.
_MASK_FORMAT_FIELD = "mask_format"
_MASK_FILE_SUFFIXES = {"png": ".png", "rle": ".rle.npz"}
_RLE_MASK_SUFFIX = _MASK_FILE_SUFFIXES["rle"]
_RLE_MASK_ARRAYS = frozenset({"shape", "runs", "values"})
# OpenCV's default decode limit (CV_IO_MAX_IMAGE_PIXELS), so an RLE mask can't be larger than a PNG mask.
_MAX_RLE_MASK_PIXELS = 1 << 30


def encode_pixel_mask_rle(mask_image: np.ndarray) -> dict[str, np.ndarray]:
    """Losslessly run-length encode a single-channel uint8 class map into the arrays of one ``.rle.npz`` file.

    ``shape`` is uint32 ``[2]`` holding (height, width).
    ``runs`` is uint32 ``[R, 2]`` holding (start, length) of each run of equal nonzero pixels, sorted by start.
    ``values`` is uint8 ``[R]`` holding each run's label ID.
    A start is the row-major flat pixel index ``y * width + x``, so a run may continue onto the next row.
    Zero pixels are unassigned and are not stored, an all-zero image has ``R == 0``.
    """

    if mask_image.ndim != 2 or mask_image.dtype != np.uint8:
        raise ValueError(f"RLE encoding needs a 2D uint8 image, got {mask_image.dtype} with shape {mask_image.shape}")
    height, width = mask_image.shape
    if height == 0 or width == 0 or height * width > _MAX_RLE_MASK_PIXELS:
        raise ValueError(f"RLE mask shape {mask_image.shape} is empty or over {_MAX_RLE_MASK_PIXELS} pixels")
    flat = mask_image.reshape(-1)
    # Segments of equal value start at pixel 0 and wherever the value changes, each ends where the next starts.
    bounds = np.concatenate(([0], np.flatnonzero(flat[1:] != flat[:-1]) + 1, [flat.size]))
    starts = bounds[:-1]
    values = flat[starts]
    labeled = values != 0
    runs = np.stack((starts[labeled], np.diff(bounds)[labeled]), axis=1).astype(np.uint32)
    return {"shape": np.array([height, width], dtype=np.uint32), "runs": runs, "values": values[labeled]}


def save_pixel_mask_rle(mask_path: Path, mask_image: np.ndarray, *, overwrite: bool = False) -> None:
    """Atomically write one uint8 class map as an uncompressed ``.rle.npz`` file."""

    if not mask_path.name.endswith(_RLE_MASK_SUFFIX):
        raise ValueError(f"RLE mask path must end with {_RLE_MASK_SUFFIX}: {mask_path}")
    if mask_path.exists() and not overwrite:
        raise FileExistsError(f"RLE mask already exists: {mask_path}")
    arrays = encode_pixel_mask_rle(mask_image)
    temporary = mask_path.with_name(f".{mask_path.name}.tmp")
    try:
        # np.savez appends .npz to path names that lack it, so it writes through an open handle.
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        temporary.replace(mask_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class _PixelMaskRle(NamedTuple):
    """One validated ``.rle.npz`` mask."""

    shape: tuple[int, int]
    starts: np.ndarray
    lengths: np.ndarray
    values: np.ndarray


def _read_pixel_mask_rle(mask_path: Path) -> _PixelMaskRle:
    """Read and validate one ``.rle.npz`` mask in O(runs).

    Returns (height, width), the int64 run starts and lengths, and the uint8 run label IDs.
    """

    try:
        archive = np.load(mask_path, allow_pickle=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Required pixel mask is missing: {mask_path}") from None
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError(f"RLE mask {mask_path} is not a readable .npz archive: {error}") from error
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError(f"RLE mask {mask_path} must be an .npz archive, got a bare {type(archive).__name__}")
    with archive:
        if set(archive.files) != _RLE_MASK_ARRAYS:
            raise ValueError(f"RLE mask {mask_path} must hold exactly {sorted(_RLE_MASK_ARRAYS)}, got {archive.files}")
        try:
            shape = archive["shape"]
            runs = archive["runs"]
            values = archive["values"]
        except (OSError, ValueError, EOFError, zipfile.BadZipFile, zlib.error) as error:
            raise ValueError(f"RLE mask {mask_path} has unreadable arrays: {error}") from error
    # NpzFile returns raw bytes for a member without the .npy header.
    for name, array in (("shape", shape), ("runs", runs), ("values", values)):
        if not isinstance(array, np.ndarray):
            raise ValueError(f"RLE mask {mask_path} {name} is not an .npy array")
    if shape.dtype != np.uint32 or shape.shape != (2,):
        raise ValueError(f"RLE mask {mask_path} shape must be uint32 [2], got {shape.dtype} {shape.shape}")
    if runs.dtype != np.uint32 or runs.ndim != 2 or runs.shape[1] != 2:
        raise ValueError(f"RLE mask {mask_path} runs must be uint32 [R, 2], got {runs.dtype} {runs.shape}")
    if values.dtype != np.uint8 or values.shape != runs.shape[:1]:
        raise ValueError(
            f"RLE mask {mask_path} values must be uint8 [{runs.shape[0]}], got {values.dtype} {values.shape}"
        )
    height, width = int(shape[0]), int(shape[1])
    if height == 0 or width == 0:
        raise ValueError(f"RLE mask {mask_path} has an empty shape {(height, width)}")
    if height * width > _MAX_RLE_MASK_PIXELS:
        raise ValueError(f"RLE mask {mask_path} shape {(height, width)} is over {_MAX_RLE_MASK_PIXELS} pixels")
    starts = runs[:, 0].astype(np.int64)
    lengths = runs[:, 1].astype(np.int64)
    if runs.shape[0] > 0:
        ends = starts + lengths
        if lengths.min() == 0:
            raise ValueError(f"RLE mask {mask_path} has zero-length runs")
        if np.any(starts[1:] < ends[:-1]):
            raise ValueError(f"RLE mask {mask_path} runs must be sorted by start and must not overlap")
        if ends[-1] > height * width:
            raise ValueError(f"RLE mask {mask_path} runs end at pixel {ends[-1]}, outside its {height}x{width} shape")
        if values.min() == 0:
            raise ValueError(f"RLE mask {mask_path} stores unassigned runs, value 0 must be left implicit")
    return _PixelMaskRle((height, width), starts, lengths, values)


def _rle_runs_to_flat_idxs(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Expand validated runs into sorted int64 row-major flat pixel indices."""

    # Each output position is its rank among all expanded pixels plus the gap skipped before its run.
    run_offsets = np.cumsum(lengths) - lengths
    return np.arange(int(lengths.sum()), dtype=np.int64) + np.repeat(starts - run_offsets, lengths)


def _rle_to_mask_image(rle: _PixelMaskRle) -> np.ndarray:
    """Rebuild the dense uint8 class map an RLE mask was encoded from."""

    mask_image = np.zeros(rle.shape[0] * rle.shape[1], dtype=np.uint8)
    mask_image[_rle_runs_to_flat_idxs(rle.starts, rle.lengths)] = np.repeat(rle.values, rle.lengths)
    return mask_image.reshape(rle.shape)


def _load_rle_class_pixel_idxs(
    mask_path: Path,
    *,
    frame_width: int,
    frame_height: int,
    camera_id: str,
    label_id: int,
) -> np.ndarray:
    """Load one label of an RLE class map as int64 flat indices into the frame's row-major pixel grid."""

    rle = _read_pixel_mask_rle(mask_path)
    if rle.shape == (frame_height, frame_width):
        selected = rle.values == label_id
        return _rle_runs_to_flat_idxs(rle.starts[selected], rle.lengths[selected])
    # Other resolutions go through the dense PNG resize, so both formats pick the same pixels.
    mask_image = _resize_mask_image(
        _rle_to_mask_image(rle),
        mask_path=mask_path,
        frame_width=frame_width,
        frame_height=frame_height,
        camera_id=camera_id,
    )
    return np.flatnonzero(mask_image == label_id)


def _load_masked_pixel_idxs(
    mask_path: Path,
    *,
    frame_width: int,
    frame_height: int,
    camera_id: str,
    label_id: int,
    device: torch.device,
) -> torch.Tensor:
    """Return one label's pixels in a frame as sorted int64 row-major flat indices on ``device``."""

    if mask_path.name.endswith(_RLE_MASK_SUFFIX):
        # RLE decodes straight to indices, which skips the dense bool upload and the device nonzero.
        return to_torch(
            _load_rle_class_pixel_idxs(
                mask_path,
                frame_width=frame_width,
                frame_height=frame_height,
                camera_id=camera_id,
                label_id=label_id,
            ),
            device=device,
        )
    pixel_mask = to_torch(
        _load_class_pixel_mask(
            mask_path,
            frame_width=frame_width,
            frame_height=frame_height,
            camera_id=camera_id,
            label_id=label_id,
        ).reshape(-1),
        device=device,
    )
    return pixel_mask.nonzero(as_tuple=False).flatten()


def _camera_pixel_mask_root(mask_root: Path, camera_id: str, camera_count: int) -> Path:
    camera_root = mask_root / camera_id
    if camera_root.is_dir():
        return camera_root
    if camera_count > 1:
        raise FileNotFoundError(f"Multi-camera pixel masks require a subdirectory for {camera_id}: {camera_root}")
    return mask_root


def _vehicle_light_label_id(manifest: dict) -> int:
    """Resolve the semantic vehicle-light ID from the unified SAM3 schema.

    Args:
        manifest: Parsed SAM3 manifest with class definitions.

    Returns:
        The configured vehicle-light label ID.

    Raises:
        ValueError: If the schema is unsupported or lacks one semantic vehicle-light class.
    """
    if manifest.get("schema_version") != 4:
        raise ValueError(f"SAM3 mask manifest schema_version must be 4, got {manifest.get('schema_version')!r}")
    lights = [
        label
        for label in manifest.get("label_schema", {}).get("classes", [])
        if label.get("name") == VEHICLE_LIGHT_CLASS_NAME and "semantic" in label.get("outputs", ())
    ]
    if len(lights) != 1 or type(lights[0].get("label_id")) is not int or not 1 <= lights[0]["label_id"] <= 255:
        raise ValueError(f"SAM3 manifest requires one semantic {VEHICLE_LIGHT_CLASS_NAME} class")
    return lights[0]["label_id"]


def _mask_file_suffix(manifest: dict) -> str:
    # Optional and outside the sampled-frame identity hash, so PNG-only manifests stay valid unchanged.
    mask_format = manifest["label_schema"].get(_MASK_FORMAT_FIELD, "png")
    if not isinstance(mask_format, str) or mask_format not in _MASK_FILE_SUFFIXES:
        raise ValueError(
            f"SAM3 mask manifest label_schema.{_MASK_FORMAT_FIELD} must be one of {sorted(_MASK_FILE_SUFFIXES)}, "
            f"got {mask_format!r}"
        )
    return _MASK_FILE_SUFFIXES[mask_format]


def load_source_indexed_pixel_mask_paths(
    mask_root: Path,
    manifest_path: Path,
    camera_ids: list[str],
    *,
    source_dataset_path: Path | None = None,
) -> dict[str, dict[int, Path]]:
    """Validate sampled-frame provenance and return light masks by source index.

    Args:
        mask_root: Directory holding per-camera semantic masks.
        manifest_path: Schema-4 SAM3 manifest describing the sampled frames.
        camera_ids: Requested cameras, including cameras without sampled frames.
        source_dataset_path: Optional source dataset to verify against provenance.

    Returns:
        Mask paths indexed by source frame for each requested camera. Cameras
        with no sampled vehicle-light frames have an empty mapping. Paths end
        in .png or .rle.npz, following the manifest's label_schema.mask_format.

    Raises:
        ValueError: If manifest, camera or sampled-frame provenance is inconsistent.
        FileNotFoundError: If the manifest or a sampled camera's masks are missing.
    """

    paths_by_camera, light_frames = _load_semantic_mask_paths(
        mask_root,
        manifest_path,
        camera_ids,
        source_dataset_path=source_dataset_path,
    )
    return {
        camera: {index: path for index, path in paths.items() if index in light_frames[camera]}
        for camera, paths in paths_by_camera.items()
    }


def _load_semantic_mask_paths(
    mask_root: Path,
    manifest_path: Path,
    camera_ids: list[str],
    *,
    source_dataset_path: Path | None = None,
) -> tuple[dict[str, dict[int, Path]], dict[str, set[int]]]:
    """Validate the manifest and every semantic mask file of the requested cameras.

    Returns the mask paths of all semantic frames by camera and source index, and per camera the source indices
    whose frame has the vehicle-light label active.
    """

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"SAM3 mask manifest does not exist: {manifest_path}") from None
    if not isinstance(manifest, dict):
        raise ValueError(f"SAM3 mask manifest root must be an object: {manifest_path}")
    light_id = _vehicle_light_label_id(manifest)
    mask_suffix = _mask_file_suffix(manifest)
    semantic_ids = {
        label["label_id"] for label in manifest["label_schema"]["classes"] if "semantic" in label["outputs"]
    }
    source_provenance = manifest.get("source_provenance")
    if not isinstance(source_provenance, dict):
        raise ValueError(f"SAM3 mask manifest is missing source_provenance: {manifest_path}")
    sampled_frames = source_provenance.get("sampled_frames")
    if not isinstance(sampled_frames, list):
        raise ValueError(f"SAM3 mask manifest has no sampled_frames: {manifest_path}")
    provenance_count = source_provenance.get("sampled_frame_identity_count")
    if (
        isinstance(provenance_count, bool)
        or not isinstance(provenance_count, int)
        or provenance_count != len(sampled_frames)
    ):
        raise ValueError(
            "SAM3 mask manifest source_provenance sampled-frame count mismatch: "
            f"identity_count={provenance_count!r} sampled_frames={len(sampled_frames)}"
        )
    _validate_identity_sha256(
        source_provenance,
        field_owner="source_provenance",
        sampled_frames=sampled_frames,
    )

    provenance_camera_ids = source_provenance.get("camera_ids")
    if (
        not isinstance(provenance_camera_ids, list)
        or any(not isinstance(camera_id, str) for camera_id in provenance_camera_ids)
        or len(set(provenance_camera_ids)) != len(provenance_camera_ids)
    ):
        raise ValueError("SAM3 mask manifest source_provenance.camera_ids must be a unique string list")
    provenance_cameras = set(provenance_camera_ids)
    sampled_camera_ids = set()
    for item in sampled_frames:
        if not isinstance(item, dict) or not isinstance(item.get("camera_id"), str):
            raise ValueError(f"Invalid SAM3 sampled-frame camera identity: {item!r}")
        sampled_camera_ids.add(item["camera_id"])
    if set(provenance_camera_ids) != sampled_camera_ids:
        raise ValueError("SAM3 source_provenance.camera_ids must match sampled-frame cameras")

    if source_dataset_path is not None:
        dataset_stat = source_dataset_path.stat()
        dataset_provenance = source_provenance.get("dataset")
        if not isinstance(dataset_provenance, dict):
            raise ValueError("SAM3 mask manifest source_provenance.dataset must be an object")
        expected_filename = source_dataset_path.name
        expected_size = dataset_stat.st_size
        expected_sha256 = _file_sha256(source_dataset_path)
        if (
            dataset_provenance.get("filename") != expected_filename
            or dataset_provenance.get("size_bytes") != expected_size
            or dataset_provenance.get("sha256") != expected_sha256
        ):
            raise ValueError(
                "SAM3 mask manifest dataset provenance mismatch: "
                f"expected filename={expected_filename!r} size_bytes={expected_size} "
                f"sha256={expected_sha256}, "
                f"got {dataset_provenance!r}"
            )

    requested = set(camera_ids)
    # The artifact layout depends on SAM3 coverage, not the RGB supervision camera set.
    sam3_camera_count = len(provenance_cameras)
    paths_by_camera: dict[str, dict[int, Path]] = {camera_id: {} for camera_id in camera_ids}
    light_frames: dict[str, set[int]] = {camera: set() for camera in camera_ids}
    for item in sampled_frames:
        if not isinstance(item, dict):
            raise ValueError(f"SAM3 sampled frame must be an object: {item!r}")
        camera_id = item.get("camera_id")
        if not isinstance(camera_id, str) or camera_id not in provenance_cameras:
            raise ValueError(f"SAM3 sampled frame camera is not declared in source_provenance.camera_ids: {item!r}")
        active_ids = item["active_label_ids"]
        if not semantic_ids.intersection(active_ids):
            continue
        if camera_id not in requested:
            continue
        source_index = item.get("source_frame_index")
        source_file = item.get("source_frame_file")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
            or not isinstance(source_file, str)
            or Path(source_file).name != source_file
        ):
            raise ValueError(f"Invalid SAM3 sampled-frame identity: {item!r}")
        source_stem = Path(source_file).stem
        if not source_stem.isdecimal() or int(source_stem) != source_index:
            raise ValueError(
                "SAM3 source-frame filename/index mismatch: "
                f"camera={camera_id} file={source_file!r} index={source_index}"
            )
        camera_paths = paths_by_camera[str(camera_id)]
        if source_index in camera_paths:
            raise ValueError(f"Duplicate SAM3 sampled frame for {camera_id}: {source_index}")
        camera_root = _camera_pixel_mask_root(mask_root, str(camera_id), sam3_camera_count)
        camera_paths[source_index] = camera_root / f"{source_stem}{mask_suffix}"
        if light_id in active_ids:
            light_frames[str(camera_id)].add(source_index)

    for camera_id, indexed_paths in paths_by_camera.items():
        if camera_id not in provenance_cameras:
            continue
        if not indexed_paths:
            continue
        camera_root = _camera_pixel_mask_root(mask_root, camera_id, sam3_camera_count)
        expected_names = {path.name for path in indexed_paths.values()}
        actual_names = {path.name for path in camera_root.glob(f"*{mask_suffix}") if path.is_file()}
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        if missing or extra:
            raise FileNotFoundError(
                f"SAM3 source-index mask set mismatch for {camera_id}: missing={missing[:5]} extra={extra[:5]}"
            )
    unsampled_cameras = sorted(requested - provenance_cameras)
    if unsampled_cameras:
        logger.warning(
            "No SAM3 samples were generated for cameras %s; skipping them for vehicle-light initialization only. "
            "RGB supervision is unaffected.",
            unsampled_cameras,
        )
    return paths_by_camera, light_frames


def convert_pixel_masks_to_rle(mask_root: Path, manifest_path: Path, *, overwrite: bool = False) -> int:
    """Write an RLE mask next to every semantic PNG mask of a SAM3 manifest, then mark the manifest as RLE.

    The PNGs stay and ``path_template`` keeps naming them, so readers that predate ``mask_format`` still load them.
    Every RLE file is read back and compared with its PNG before the manifest changes.
    Returns the number of converted masks.
    """

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"SAM3 mask manifest root must be an object: {manifest_path}")
    _vehicle_light_label_id(manifest)
    if _mask_file_suffix(manifest) != _MASK_FILE_SUFFIXES["png"]:
        raise ValueError(f"Only PNG SAM3 mask manifests can be converted to RLE: {manifest_path}")
    source_provenance = manifest.get("source_provenance")
    camera_ids = source_provenance.get("camera_ids") if isinstance(source_provenance, dict) else None
    if not isinstance(camera_ids, list):
        raise ValueError(f"SAM3 mask manifest is missing source_provenance.camera_ids: {manifest_path}")
    # Every semantic frame, not only vehicle-light ones, because the loader checks the full file set.
    png_paths, _ = _load_semantic_mask_paths(mask_root, manifest_path, camera_ids)

    converted = 0
    for indexed_paths in png_paths.values():
        for png_path in indexed_paths.values():
            mask_image = _read_class_mask_png(png_path)
            rle_path = png_path.with_name(f"{png_path.stem}{_RLE_MASK_SUFFIX}")
            save_pixel_mask_rle(rle_path, mask_image, overwrite=overwrite)
            rle = _read_pixel_mask_rle(rle_path)
            if rle.shape != mask_image.shape or not np.array_equal(_rle_to_mask_image(rle), mask_image):
                raise RuntimeError(f"RLE mask {rle_path} does not reproduce {png_path}")
            converted += 1

    manifest["label_schema"][_MASK_FORMAT_FIELD] = "rle"
    # Same layout as the producer's manifest writer, through a same-directory replacement.
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        # Check the RLE file set the way training will, so stray .rle.npz files fail here instead of at load time.
        _load_semantic_mask_paths(mask_root, temporary, camera_ids)
        temporary.replace(manifest_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return converted


class _CameraKey(NamedTuple):
    """Identifies the per-camera arrays, which are device tensors."""

    datasource_id: int
    camera_id: str
    device: str


class _FrameKey(NamedTuple):
    """Everything a cached per-frame product depends on."""

    datasource_id: int
    camera_id: str
    camera_frame_index: int
    mask_path: str
    device: str
    label_id: int = 255


class _RaysKey(NamedTuple):
    frame: _FrameKey
    return_color: bool


@dataclass(slots=True, kw_only=True)
class _CameraArrays:
    """Per-camera inputs that do not change between vehicle-light layers."""

    camera_model: ncore.sensors.CameraModel
    all_pixels: torch.Tensor
    all_footprints: torch.Tensor


@dataclass(slots=True, kw_only=True)
class _CameraFrameRays:
    """Backprojected rays for vehicle light pixels in one camera frame, independent of the cuboid tracks."""

    origins: torch.Tensor
    directions: torch.Tensor
    rays_timestamps_us: torch.Tensor
    footprints: torch.Tensor
    color: torch.Tensor | None


@dataclass(slots=True)
class _CameraFrameCache:
    """Memoize the per-camera and per-frame half of the vehicle-light frame loop.

    Both layers walk the same frames and differ only in their cuboid tracks, so the mask, color and ray work repeats.
    """

    # None means the budget is measured from free VRAM at the first store, not at construction.
    byte_budget: int | None
    # Holding each keyed datasource keeps its id() from being reused while entries keyed on it exist.
    datasources: dict[int, NCOREDataSource] = field(default_factory=dict)
    cameras: dict[_CameraKey, _CameraArrays] = field(default_factory=dict)
    pixel_idxs: dict[_FrameKey, torch.Tensor] = field(default_factory=dict)
    frame_rays: dict[_RaysKey, _CameraFrameRays] = field(default_factory=dict)
    cached_bytes: int = 0
    mask_hits: int = 0
    mask_misses: int = 0
    rays_hits: int = 0
    rays_misses: int = 0
    over_budget: int = 0

    def get_camera_arrays(self, key: _CameraKey) -> _CameraArrays | None:
        return self.cameras.get(key)

    def put_camera_arrays(self, key: _CameraKey, value: _CameraArrays) -> None:
        # Only the grid tensors are counted, the camera model holds intrinsics and no per-pixel buffers.
        if self._reserve(_tensor_bytes(value.all_pixels) + _tensor_bytes(value.all_footprints)):
            self.cameras[key] = value

    def get_pixel_idxs(self, key: _FrameKey) -> torch.Tensor | None:
        idxs = self.pixel_idxs.get(key)
        if idxs is None:
            self.mask_misses += 1
        else:
            self.mask_hits += 1
        return idxs

    def put_pixel_idxs(self, key: _FrameKey, value: torch.Tensor) -> None:
        if self._reserve(_tensor_bytes(value)):
            self.pixel_idxs[key] = value

    def get_frame_rays(self, key: _RaysKey) -> _CameraFrameRays | None:
        rays = self.frame_rays.get(key)
        if rays is None:
            self.rays_misses += 1
        else:
            self.rays_hits += 1
        return rays

    def put_frame_rays(self, key: _RaysKey, value: _CameraFrameRays) -> None:
        size = (
            _tensor_bytes(value.origins)
            + _tensor_bytes(value.directions)
            + _tensor_bytes(value.rays_timestamps_us)
            + _tensor_bytes(value.footprints)
            + _tensor_bytes(value.color)
        )
        if self._reserve(size):
            self.frame_rays[key] = value

    def _reserve(self, size: int) -> bool:
        if self.byte_budget is None:
            self.byte_budget = _default_frame_cache_budget()
        if self.byte_budget == 0 or self.cached_bytes + size > self.byte_budget:
            self.over_budget += 1
            return False
        self.cached_bytes += size
        return True

    def clear(self) -> None:
        self.datasources.clear()
        self.cameras.clear()
        self.pixel_idxs.clear()
        self.frame_rays.clear()
        self.cached_bytes = 0

    def log_summary(self) -> None:
        if not self.cached_bytes and not self.over_budget:
            return
        logger.info(
            "Vehicle-light frame cache: masks %d/%d hit, rays %d/%d hit, %.1f of %.1f MiB held, %d entries over budget",
            self.mask_hits,
            self.mask_hits + self.mask_misses,
            self.rays_hits,
            self.rays_hits + self.rays_misses,
            self.cached_bytes / (1024 * 1024),
            (self.byte_budget or 0) / (1024 * 1024),
            self.over_budget,
        )


def _device_key(device: torch.device) -> str:
    """Name a device unambiguously for cache keys."""

    if device.type == "cuda" and device.index is None:
        return f"cuda:{torch.cuda.current_device()}"
    return str(device)


def _tensor_bytes(tensor: torch.Tensor | None) -> int:
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


# Use 10% of the free memory as the cache budget.
_FRAME_CACHE_FREE_MEMORY_FRACTION = 0.1
# Ceiling for devices with big VRAM.
_FRAME_CACHE_MAX_BYTES = 8 * 1024**3
_active_frame_cache: ContextVar[_CameraFrameCache | None] = ContextVar("vehicle_light_frame_cache", default=None)


def _default_frame_cache_budget() -> int:
    if not torch.cuda.is_available():
        return 0
    free_bytes, _ = torch.cuda.mem_get_info()
    return min(int(free_bytes * _FRAME_CACHE_FREE_MEMORY_FRACTION), _FRAME_CACHE_MAX_BYTES)


@contextmanager
def shared_vehicle_light_frame_cache(byte_budget: int | None = None) -> Iterator[None]:
    """Share per-frame mask, color and ray work across the vehicle-light layers initialized inside this scope.

    Cached tensors stay on the device until the outermost scope exits.
    Outside any scope, vehicle-light initialization runs uncached.

    Args:
        byte_budget: Cap on cached bytes, 0 caches nothing.
            None measures min(10% of free VRAM, 8 GiB) at the first store, or 0 without CUDA.
            A nested scope ignores it and reuses the outer cache.
    """

    active = _active_frame_cache.get()
    if active is not None:
        if byte_budget is not None and byte_budget != active.byte_budget:
            logger.warning(
                "Nested vehicle-light frame cache scope ignores byte_budget=%d, the outer scope's %s stays in force",
                byte_budget,
                active.byte_budget,
            )
        yield
        return
    cache = _CameraFrameCache(byte_budget=byte_budget)
    token = _active_frame_cache.set(cache)
    try:
        yield
    finally:
        _active_frame_cache.reset(token)
        cache.log_summary()
        cache.clear()
        # A generator that outlives the scope still holds this cache, so it must not store device tensors again.
        cache.byte_budget = 0


def _current_frame_cache() -> _CameraFrameCache:
    # With no scope open the zero budget refuses every entry, which is exactly the old uncached path.
    cache = _active_frame_cache.get()
    return cache if cache is not None else _CameraFrameCache(byte_budget=0)


def get_masked_camera_track_point_clouds(
    datasource: NCOREDataSource,
    camera_ids: list[str],
    cuboid_tracks: CuboidTracks,
    *,
    pixel_mask_dir: str,
    pixel_mask_manifest_path: str,
    cuboid_dim_scale_factor: float = 1.0,
    return_color: bool = True,
    keep_all_track_poses: bool = False,
    device: torch.device = torch.device("cuda"),
    cuboid_clamp_distance_m: float = 0.15,
) -> Generator[MaskedTrackPointCloud, None, None]:
    """Backproject SAM3 pixels and assign each point to one nearest eligible cuboid.

    The VREN cuboid-ray intersection used by this initializer requires CUDA.
    """

    if device.type != "cuda":
        raise RuntimeError(f"Vehicle-light mask backprojection requires a CUDA device, got {device}")
    if cuboid_clamp_distance_m < 0:
        raise ValueError("cuboid_clamp_distance_m must be non-negative")
    mask_root = Path(pixel_mask_dir)
    label_id = _vehicle_light_label_id(json.loads(Path(pixel_mask_manifest_path).read_text()))
    indexed_masks = load_source_indexed_pixel_mask_paths(
        mask_root,
        Path(pixel_mask_manifest_path),
        camera_ids,
        source_dataset_path=Path(str(datasource.get_path())),
    )
    # Limit only this initializer; do not change the datasource's RGB supervision cameras.
    camera_ids = [camera_id for camera_id in camera_ids if indexed_masks[camera_id]]
    if not camera_ids:
        return

    if not any(indexed_masks.values()):
        return
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Pixel-mask directory does not exist: {mask_root}")
    frame_cache = _current_frame_cache()
    # Rays depend on this datasource's poses, so two datasources in one scope must not share entries.
    datasource_id = id(datasource)
    frame_cache.datasources.setdefault(datasource_id, datasource)
    datasource._maybe_init_worker()
    if keep_all_track_poses:
        logger.warning("keep_all_track_poses has no effect for timestamp-interpolated vehicle-light ray assignment")

    cuboid_tracks = cuboid_tracks.to_device(device)
    cuboid_dims = cuboid_tracks.cuboids_dims.to(device)
    scaled_cuboid_dims = cuboid_dims * cuboid_dim_scale_factor
    # ray_intersection adds padding to the full cuboid dimensions, while
    # cuboid_clamp_distance_m is a per-side tolerance.
    cuboid_dims_padding = scaled_cuboid_dims - cuboid_dims + 2 * cuboid_clamp_distance_m
    track_pose_starts = cuboid_tracks.tracks_packinfo[:, 0].long()
    track_pose_ends = track_pose_starts + cuboid_tracks.tracks_packinfo[:, 1].long() - 1
    track_start_timestamps_us = cuboid_tracks.tracks_timestamps_us[track_pose_starts]
    track_end_timestamps_us = cuboid_tracks.tracks_timestamps_us[track_pose_ends]
    total_mask_pixels = 0
    total_assigned_pixels = 0
    assigned_track_ids: set[str] = set()
    requested_unique_ids = {camera_id: datasource.camera_unique_ids[camera_id].id for camera_id in camera_ids}
    for unique_camera_id in dict.fromkeys(requested_unique_ids.values()):
        for camera_id in camera_ids:
            if requested_unique_ids[camera_id] != unique_camera_id:
                continue
            camera_sensor = datasource.camera_sensors[camera_id]
            camera_params = camera_sensor.model_parameters
            frame_width, frame_height = camera_params.resolution
            camera_key = _CameraKey(datasource_id=datasource_id, camera_id=camera_id, device=_device_key(device))
            camera_arrays = frame_cache.get_camera_arrays(camera_key)
            if camera_arrays is None:
                camera_arrays = _CameraArrays(
                    camera_model=ncore.sensors.camera_model_from_parameters(camera_params, device=str(device)),
                    all_pixels=to_torch(datasource.cameras_all_pixels[camera_id], device=device),
                    all_footprints=to_torch(
                        datasource.cameras_all_footprints[camera_id].reshape(-1),
                        device=device,
                    ),
                )
                frame_cache.put_camera_arrays(camera_key, camera_arrays)
            camera_model = camera_arrays.camera_model
            all_pixels = camera_arrays.all_pixels
            all_camera_footprints = camera_arrays.all_footprints

            camera_masks = indexed_masks[camera_id]
            available_indices = {int(index) for index in datasource.camera_frame_ranges[camera_id]}
            unavailable = sorted(set(camera_masks) - available_indices)
            if unavailable:
                raise ValueError(f"SAM3 sampled frames are unavailable in NCore camera {camera_id}: {unavailable[:5]}")

            for camera_frame_index in sorted(camera_masks):
                camera_frame_timestamp_start, camera_frame_timestamp_end = camera_sensor.frames_timestamps_us[
                    camera_frame_index
                ]
                max_intersections_per_ray = _count_tracks_overlapping_interval(
                    track_start_timestamps_us,
                    track_end_timestamps_us,
                    int(camera_frame_timestamp_start),
                    int(camera_frame_timestamp_end),
                )
                mask_path = camera_masks[camera_frame_index]
                frame_key = _FrameKey(
                    datasource_id=datasource_id,
                    camera_id=camera_id,
                    camera_frame_index=camera_frame_index,
                    mask_path=str(mask_path),
                    device=_device_key(device),
                    label_id=label_id,
                )
                masked_pixel_idxs = frame_cache.get_pixel_idxs(frame_key)
                if masked_pixel_idxs is None:
                    masked_pixel_idxs = _load_masked_pixel_idxs(
                        mask_path,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        camera_id=camera_id,
                        label_id=label_id,
                        device=device,
                    )
                    frame_cache.put_pixel_idxs(frame_key, masked_pixel_idxs)
                if masked_pixel_idxs.numel() == 0:
                    continue

                total_mask_pixels += masked_pixel_idxs.numel()
                if max_intersections_per_ray == 0:
                    continue

                rays_key = _RaysKey(frame=frame_key, return_color=return_color)
                frame_rays = frame_cache.get_frame_rays(rays_key)
                if frame_rays is None:
                    pixels = all_pixels[masked_pixel_idxs]
                    footprints = all_camera_footprints[masked_pixel_idxs]
                    color = None
                    if return_color:
                        color = to_torch(
                            camera_sensor.get_frame_image_array(camera_frame_index).reshape((-1, 3)),
                            device=device,
                        )[masked_pixel_idxs]

                    sensor_to_nurec_startend = datasource.world_to_nurec.transform_poses(
                        camera_sensor.get_frames_T_sensor_target(
                            target_node="world",
                            frame_indices=camera_frame_index,
                            frame_timepoint=None,
                        )
                    )
                    image_points = camera_model.pixels_to_image_points(pixels)
                    world_rays = image_points_to_world_rays_shutter_pose(
                        camera_params,
                        image_points,
                        se3_matrix_to_tquat(sensor_to_nurec_startend),
                        torch.LongTensor([camera_frame_timestamp_start, camera_frame_timestamp_end]),
                    ).world_rays

                    relative_frame_times = camera_model.image_points_relative_frame_times(image_points)
                    rays_timestamps_us = int(camera_frame_timestamp_start) + (
                        relative_frame_times
                        * float(int(camera_frame_timestamp_end) - int(camera_frame_timestamp_start))
                    ).to(torch.int64)
                    frame_rays = _CameraFrameRays(
                        origins=world_rays[:, :3].contiguous(),
                        directions=world_rays[:, 3:].contiguous(),
                        rays_timestamps_us=rays_timestamps_us,
                        footprints=footprints,
                        color=color,
                    )
                    frame_cache.put_frame_rays(rays_key, frame_rays)

                ray_origins = frame_rays.origins
                ray_directions = frame_rays.directions
                rays_timestamps_us = frame_rays.rays_timestamps_us
                footprints = frame_rays.footprints
                color = frame_rays.color

                intersection_result = cuboid_tracks.ray_intersection(
                    ray_origins,
                    ray_directions,
                    rays_timestamps_us,
                    cuboids_dims_padding=cuboid_dims_padding,
                    max_intersections_per_ray=max_intersections_per_ray,
                    with_intersections_ts=True,
                )
                assert intersection_result.intersections_ts is not None
                intersection_ts, intersection_idxs = _flatten_ray_intersections(
                    intersection_result.intersections_cnt,
                    intersection_result.intersections_tracks_idx,
                    intersection_result.intersections_ts,
                )
                # The flattened outputs own their storage; release dense buffers
                # before selecting hits and before the next frame's allocation.
                del intersection_result
                selected_rows = _select_frontmost_intersection_rows(intersection_ts, intersection_idxs)
                if selected_rows.numel() == 0:
                    continue

                selected_ray_idxs = intersection_idxs[selected_rows, 0].long()
                # The interpolation binding requires indices matching int32 packinfo.
                selected_track_idxs = intersection_idxs[selected_rows, 1]
                selected_entry_ts = torch.clamp_min(intersection_ts[selected_rows, 0], 0)
                selected_world_origins = ray_origins[selected_ray_idxs]
                selected_world_entries = selected_world_origins + ray_directions[selected_ray_idxs] * (
                    selected_entry_ts + 0.01
                ).unsqueeze(-1)
                selected_track_poses = cuboid_tracks.interpolate_tracks_poses(
                    rays_timestamps_us[selected_ray_idxs],
                    selected_track_idxs,
                )
                selected_world_to_local = selected_track_poses.inv()
                selected_local_origins = selected_world_to_local * selected_world_origins
                selected_local_entries = selected_world_to_local * selected_world_entries
                total_assigned_pixels += int(selected_rows.numel())

                for track_index_tensor in torch.unique(selected_track_idxs):
                    track_index = int(track_index_tensor.item())
                    point_mask = selected_track_idxs == track_index
                    ray_idxs = selected_ray_idxs[point_mask]
                    local_start = selected_local_origins[point_mask]
                    entry_ts = selected_entry_ts[point_mask]
                    local_end = selected_local_entries[point_mask]

                    half_extent = scaled_cuboid_dims[track_index, :3] / 2
                    inward_epsilon = torch.minimum(torch.full_like(half_extent, 0.01), half_extent * 0.01)
                    inner_half_extent = torch.clamp(half_extent - inward_epsilon, min=1.0e-4)
                    local_end = torch.maximum(torch.minimum(local_end, inner_half_extent), -inner_half_extent)

                    track_start = int(cuboid_tracks.tracks_packinfo[track_index, 0].item())
                    track_count = int(cuboid_tracks.tracks_packinfo[track_index, 1].item())
                    track_timestamps_us = cuboid_tracks.tracks_timestamps_us[track_start : track_start + track_count]
                    representative_timestamp_us = rays_timestamps_us[ray_idxs].median()
                    track_pose_index = int(
                        torch.argmin(torch.abs(track_timestamps_us - representative_timestamp_us)).item()
                    )
                    track_id = cuboid_tracks.tracks_id[track_index]
                    assigned_track_ids.add(track_id)
                    yield MaskedTrackPointCloud(
                        track_id=track_id,
                        point_cloud=PointCloud(
                            xyz_start=local_start,
                            xyz_end=local_end,
                            flags=None,
                            color=color[ray_idxs] if color is not None else None,
                            semantic_class_id=None,
                            camera_footprint_scale=footprints[ray_idxs] * entry_ts,
                        ),
                        track_pose_index=track_pose_index,
                    )

    if total_mask_pixels > 0:
        logger.info(
            "Assigned %d/%d SAM3 vehicle-light pixels to %d timestamp-interpolated cuboid tracks",
            total_assigned_pixels,
            total_mask_pixels,
            len(assigned_track_ids),
        )
    if total_mask_pixels > 0 and total_assigned_pixels == 0:
        logger.warning("SAM3 vehicle-light masks contained pixels, but none intersected a cuboid track")
