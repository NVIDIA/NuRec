# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import time

from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, AsyncIterator, Callable, Iterator, List, Optional, Sequence, Tuple
from uuid import uuid4

import aiofiles
import click
import lietorch as lt
import numpy as np

from scipy.spatial.transform import Rotation as R

import ncore.impl.common.transformations as ncore_transformations
import nurec.utils.cli as cli

from nurec.artifact import Artifact
from nurec.grpc.client_wait import wait_for_grpc_server
from nurec.grpc.protos.common_pb2 import AABB
from nurec.grpc.protos.common_pb2 import Empty as EmptyRequest
from nurec.grpc.protos.sensorsim_pb2 import (
    AvailableCamerasRequest,
    AvailableCamerasReturn,
    AvailableEgoMasksReturn,
    BboxColor,
    CameraRenderOutput,
    CameraRenderProduct,
    CameraRenderRequest,
    CameraRenderReturn,
    CameraSpec,
    DynamicObject,
    DynamicObjectTrack,
    EditAssetsRequest,
    EditAssetsResponse,
    EgoMaskId,
    ExternalAssetObjectsRequest,
    ExternalAssetObjectsReturn,
    HarmonizerSessionId,
    ImageFormat,
    LidarDeviceType,
    LidarFiringPatternRequest,
    LidarFiringPatternReturn,
    LidarFloatingFilterSource,
    LidarOutputMode,
    LidarRenderFilter,
    LidarRenderRequest,
    LidarRenderReturn,
    LidarReturnType,
    LidarSpec,
    OverlayBoundingBoxSource,
    PosePair,
    ReplaceAssetAction,
    RestoreModelParametersRequest,
)
from nurec.grpc.protos.sensorsim_pb2_grpc import SensorsimServiceStub
from nurec.grpc.serve import (
    actor_tracks_to_grpc,
    grpc,
    grpc_pose_to_torch_se3,
    se3_to_grpc_pose,
    tquat_to_grpc_pose,
)
from nurec.render.actors import ActorTracks, clip_window_to_cuboid_track
from nurec.utils.cli import SettingsCollector
from nurec.utils.geometry import pose_offsets_to_se3, se3_matrix_to_se3, se3_matrix_to_tquat
from nurec.utils.io.export.render_helpers import CameraOutputSubdir
from nurec.utils.io.export.render_lidar_ply import save_lidar_ply
from nurec.utils.types import RigTrajectories


log = logging.getLogger(__name__)

"""
This script demonstrates controlling dynamic actors via gRPC ``CameraRenderRequest`` and selecting the camera
render product with ``--output-type``: color (default), distance, or semantics. Use ``--overlay-bboxes model``
or ``artifact`` to draw cuboid wireframes on color outputs via ``render_camera``.
Camera images are written as ``<output_dir>/<output_type>/<camera_id>/<frame_index>.<ext>`` where the first-level
folder is ``color``, ``distance_colorized``, or ``semantics_colorized``. With ``--repeat-count N`` greater than one,
each repetition is rooted under ``<output_dir>/repeat-NNN``. LiDAR outputs otherwise remain flat under ``--output-dir``.

It requires a pretrained `GaussiansComposite` model with an NuRec artifact, to train a suitable model make sure you train a dynamic reconstruction and enable saving `rig_trajectories` and `sequence_tracks`.

For example:
    ```
    python run.py --config-name apps/AV/NV/3dgut_dynamic.yaml out_dir=OUTDIR mode=train dataset.path=DATAPATH \
            checkpoint.artifact.enabled=true \
            checkpoint.artifact.rig_trajectories.enabled=true \
            checkpoint.artifact.sequence_tracks.enabled=true
    ```

Before running the script, make sure to start the grpc server, using the same `port` and `artifact-glob` that includes the artifact path. For example:

```
python run.py serve-grpc --artifact-glob ARTIFACT_GLOB
```
"""


def _output_dir_for_repeat(output_dir: str, repeat_index: int, repeat_count: int) -> str:
    """Return the output directory for one repetition without changing the default layout."""
    if repeat_count == 1:
        return output_dir
    return os.path.join(output_dir, f"repeat-{repeat_index:03d}")


async def _maybe_create_temporal_harmonizer_session(
    client_service: SensorsimServiceStub,
    requests: list[CameraRenderRequest],
    repeat_count: int,
) -> str | None:
    """Create and attach a temporal session when the target server has one enabled.

    This helper is called only for camera renders that request the colour product. It queries the
    target server and returns ``None`` unless temporal Harmonizer is enabled. For a temporal server,
    it creates one session, attaches its ID to every trajectory frame, and returns that ID.
    """
    server_config = await client_service.get_server_config(EmptyRequest())
    temporal_harmonizer_enabled = server_config.server_config.get("enable_harmonizer_temporal", "").lower() == "true"
    if not temporal_harmonizer_enabled:
        return None

    if repeat_count > 1:
        raise ValueError("repeat_count greater than 1 is not supported with a temporal harmonizer session")

    session = await client_service.create_harmonizer_session(EmptyRequest())
    session_id = session.session_id
    log.info(f"Created temporal harmonizer session {session_id}")
    for request in requests:
        request.harmonizer_session_id = session_id
    return session_id


def _force_sequential_for_temporal_session(harmonizer_session_id: str | None, sequential: bool) -> bool:
    """Require ordered rendering when a temporal Harmonizer session is active."""
    if harmonizer_session_id is not None and not sequential:
        log.info("Temporal Harmonizer is enabled; forcing sequential in-order rendering.")
        return True
    return sequential


async def _repeat_request_sequence(
    method: Callable,
    requests: list[Any],
    sequential: bool,
    repeat_count: int,
    limiter: Optional["_AdmissionLimiter"] = None,
) -> AsyncIterator[tuple[int, list[Any]]]:
    """Execute the same in-memory request objects repeatedly.

    One ``limiter`` is shared across every repetition so the adaptive serialize-on-shed clamp
    (or an explicit ``--max-concurrent-renders`` bound) stays in effect for the whole run.
    """
    if limiter is None:
        limiter = _AdmissionLimiter()
    for repeat_index in range(repeat_count):
        if repeat_count > 1:
            log.info("Rendering request sequence repetition %d/%d", repeat_index + 1, repeat_count)
        yield repeat_index, await _execute_renders(method, requests, sequential, limiter)


def dynamic_poses_to_dynamic_objects(dynamic_poses: list) -> List[DynamicObject]:
    """Convert internal dynamic_poses to gRPC DynamicObject format."""
    return [
        DynamicObject(
            track_id=dynamic_pose["track_id"],
            pose_pair=PosePair(
                start_pose=tquat_to_grpc_pose(
                    dynamic_pose["frame_start_actor_to_world"].data,
                ),
                end_pose=tquat_to_grpc_pose(
                    dynamic_pose["frame_end_actor_to_world"].data,
                ),
            ),
        )
        for dynamic_pose in dynamic_poses
    ]


def _build_dynamic_actor_poses(
    cuboid_tracks_list: list,
    interpolator_list: list,
    remove: set,
    frame_start_us: int,
    frame_end_us: int,
    frame_idx: int,
    *,
    demo_actor_transform: bool,
    delta_matrices: "np.ndarray",
) -> list[dict[str, Any]]:
    """Build per-actor start/end world poses for the actor-editing render path.

    A track is included when its keyframe span overlaps ``[frame_start_us, frame_end_us]`` and its
    interpolation timestamps are clamped to the track's ``[min, max]`` span (see
    :func:`clip_window_to_cuboid_track`). This is deliberately an overlap+clamp gate rather than the
    stricter full-containment gate: a partially-overlapping actor is emitted with edge-clamped poses
    instead of being dropped, while the clamp still guarantees the interpolator never extrapolates
    beyond the track's keyframes (the constraint the caller must preserve).
    """
    dynamic_poses: list[dict[str, Any]] = []
    for cuboid_track, interpolator in zip(cuboid_tracks_list, interpolator_list):
        track_id = cuboid_track.tracks_id[0]

        # remove defined as a list of track_ids(str) in edit-assets.json
        if track_id in remove:
            continue

        clipped = clip_window_to_cuboid_track(cuboid_track, frame_start_us, frame_end_us)
        if clipped is None:
            continue
        ts_start, ts_end = clipped

        interpolated_world_poses = interpolator.interpolate_to_timestamps([ts_start, ts_end])
        assert interpolated_world_poses.shape == (2, 4, 4)
        assert interpolated_world_poses.dtype == np.float32

        frame_start_actor_to_world = lt.SE3(se3_matrix_to_tquat(interpolated_world_poses[0]))
        frame_end_actor_to_world = lt.SE3(se3_matrix_to_tquat(interpolated_world_poses[1]))

        if demo_actor_transform:
            delta_matrix = se3_matrix_to_se3(delta_matrices[frame_idx])
            frame_start_actor_to_world = frame_start_actor_to_world * delta_matrix
            frame_end_actor_to_world = frame_end_actor_to_world * delta_matrix

        dynamic_poses.append(
            {
                "track_id": track_id,
                "frame_start_actor_to_world": frame_start_actor_to_world,
                "frame_end_actor_to_world": frame_end_actor_to_world,
            }
        )
    return dynamic_poses


def generate_request(
    scene_id: str,
    camera_spec: CameraSpec,
    frame_start_camera_to_world: np.ndarray,  # 4x4 SE3 matrix
    frame_end_camera_to_world: np.ndarray,  # 4x4 SE3 matrix
    frame_start_timestamp_us: int,
    frame_end_timestamp_us: int,  # Pass frame-end timestamp +1 here (due to the #@$! half-closed-interval convention)
    height: int,
    dynamic_poses: list,
    format: ImageFormat,
    ego_mask_id: EgoMaskId | None,
    render_products: Optional[list[CameraRenderProduct]] = None,
    overlay_bounding_boxes_source: OverlayBoundingBoxSource = OverlayBoundingBoxSource.OVERLAY_BOUNDING_BOX_SOURCE_UNSPECIFIED,
    distance_colormap_vmax_m: Optional[float] = None,
    bbox_color: Optional[Tuple[int, int, int]] = None,
    apply_harmonizer: bool = False,
) -> CameraRenderRequest:
    # gRPC uses half-closed-interval convention, so start and end timestamp can not be equal
    assert frame_start_timestamp_us < frame_end_timestamp_us
    request = CameraRenderRequest(
        scene_id=scene_id,
        resolution_h=height,
        resolution_w=1,
        camera_intrinsics=camera_spec,
        frame_start_us=frame_start_timestamp_us,
        frame_end_us=frame_end_timestamp_us,
        sensor_pose=PosePair(
            start_pose=se3_to_grpc_pose(frame_start_camera_to_world),
            end_pose=se3_to_grpc_pose(frame_end_camera_to_world),
        ),
        dynamic_objects=dynamic_poses_to_dynamic_objects(dynamic_poses),
        image_format=format,
        image_quality=95,
        ego_mask_id=ego_mask_id,
        insert_ego_mask=ego_mask_id is not None,
        render_products=render_products or [CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR],
        overlay_bounding_boxes=overlay_bounding_boxes_source,
        apply_harmonizer=apply_harmonizer,
    )
    if distance_colormap_vmax_m is not None:
        request.distance_colormap_vmax_m = distance_colormap_vmax_m
    if bbox_color is not None:
        request.bbox_color.CopyFrom(BboxColor(r=bbox_color[0], g=bbox_color[1], b=bbox_color[2]))
    return request


def _grpc_output_type_dir(product: int) -> CameraOutputSubdir:
    if product == CameraRenderProduct.CAMERA_RENDER_PRODUCT_SEMANTICS_COLORIZED:
        return CameraOutputSubdir.SEMANTICS_COLORIZED
    if product == CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED:
        return CameraOutputSubdir.DISTANCE_COLORIZED
    return CameraOutputSubdir.COLOR


def _grpc_output_dir_for_camera_item(out: CameraRenderOutput) -> str:
    return _grpc_output_type_dir(out.render_product)


def _extract_distance_colormap_vmax_m(response: CameraRenderReturn) -> Optional[float]:
    """Pull the server-resolved distance colormap max out of a distance-colorized output.

    Reads the value the server actually used to render distance colorization, rather than
    the CLI argument: the server falls back to its own default when the CLI omits
    ``--colorization-max-distance``, and that resolved value isn't otherwise recoverable.
    """
    for out in response.camera_render_outputs:
        if out.render_product == CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED and out.HasField(
            "distance_colormap_vmax_m"
        ):
            return out.distance_colormap_vmax_m
    return None


def _write_distance_colormap_metadata(
    responses: list[CameraRenderReturn],
    render_products: list[CameraRenderProduct],
    output_dir: str,
    camera_id: str,
) -> None:
    if not responses or CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED not in render_products:
        return
    distance_vmax_m = _extract_distance_colormap_vmax_m(responses[0])
    if distance_vmax_m is None:
        return
    dist_root = os.path.join(output_dir, CameraOutputSubdir.DISTANCE_COLORIZED, camera_id)
    os.makedirs(dist_root, exist_ok=True)
    dist_path = os.path.join(dist_root, "distance_colormap_vmax_m.json")
    with open(dist_path, "w") as df:
        json.dump({"distance_colormap_vmax_m": distance_vmax_m}, df, indent=2)
    log.info("Wrote %s (vmax_m=%.3f)", dist_path, distance_vmax_m)


def select_ego_mask_id(
    available_ego_masks: AvailableEgoMasksReturn,
    rig_name: str | None,
    camera_id: str,
) -> EgoMaskId | None:
    """Select the exact per-rig/per-camera ego mask requested by the CLI."""
    if rig_name is None:
        return None

    available_ego_mask_lookup: dict[tuple[str, str], EgoMaskId] = {
        (m.ego_mask_id.rig_config_id, m.ego_mask_id.camera_logical_id): m.ego_mask_id
        for m in available_ego_masks.ego_mask_metadata
    }
    available_rig_ids = sorted({rig_id for rig_id, _ in available_ego_mask_lookup})
    if rig_name not in available_rig_ids:
        raise KeyError(f"{rig_name=} not found in available rigs {available_rig_ids}")

    ego_mask_id = available_ego_mask_lookup.get((rig_name, camera_id))
    if ego_mask_id is None:
        available_cameras_for_rig = sorted(cam for rig, cam in available_ego_mask_lookup if rig == rig_name)
        raise KeyError(
            f"No ego mask for {rig_name=}, {camera_id=}. Available cameras for this rig: {available_cameras_for_rig}"
        )

    return ego_mask_id


def compose_sensor_to_world(
    poses_rig_to_world: np.ndarray,  # 2x4x4, frame-start and frame-end rig poses
    rig_offset_se3: np.ndarray,  # 4x4, identity unless --rig-*-offset was given
    pose_sensor_to_rig: np.ndarray,  # 4x4 extrinsics
) -> np.ndarray:  # 2x4x4, frame-start and frame-end sensor poses
    """Compose the frame-start/end sensor-to-world poses that the render requests expect.

    Both RGB and LiDAR requests carry a sensor-to-world ``sensor_pose``, so the mount
    extrinsics (and any rig offset) have to be composed onto the rig trajectory. Sending
    the bare rig pose renders the sensor at the rig origin instead of where it is mounted.
    """
    assert poses_rig_to_world.shape == (2, 4, 4)
    assert rig_offset_se3.shape == (4, 4)
    assert pose_sensor_to_rig.shape == (4, 4)
    return poses_rig_to_world @ rig_offset_se3 @ pose_sensor_to_rig


def generate_lidar_request(
    scene_id: str,
    lidar_spec: LidarSpec | None,
    poses,
    frame_start_timestamp_us: int,
    frame_end_timestamp_us: int,
    dynamic_poses: list,
    render_filter: LidarRenderFilter | None,
    lidar_id: str = "",
    output_mode: LidarOutputMode | None = None,
    firing_pattern_id: str = "",
    return_types: Sequence[LidarReturnType] = (),
) -> LidarRenderRequest:
    """Construct a lidar RPC request from trajectory and output settings.

    Args:
        scene_id: Scene identifier known to the rendering service.
        lidar_spec: Optional modeled-lidar configuration; ``None`` omits it.
        poses: Pair of ``(4, 4)`` lidar-to-world transforms at frame start and end.
        frame_start_timestamp_us: Physical frame-start timestamp in microseconds.
        frame_end_timestamp_us: End of the requested pose interval in microseconds.
        dynamic_poses: Actor records with ``track_id`` and start/end actor-to-world
            poses, converted to protobuf dynamic-object entries.
        render_filter: Optional filter overrides; ``None`` leaves server defaults in effect.
        lidar_id: Logical sensor ID, or an empty string for a single-lidar artifact.
        output_mode: Requested output representation; ``None`` leaves the field unset
            for the server's point-cloud default. Per-ray output must be explicit.
        firing_pattern_id: Optional exact pattern ID, overriding phase selection by time.
        return_types: Requested per-ray return layers in output order. Empty selects
            strongest in per-ray mode and is required for point-cloud mode.

    Returns:
        A populated ``LidarRenderRequest``. Semantic compatibility is checked by the server.

    Raises:
        KeyError: If an actor record lacks a required pose or track field.
        IndexError: If the pose pair does not contain both start and end transforms.
        TypeError: If a value cannot be assigned to its protobuf field.
        ValueError: If a numeric value is outside its protobuf field's range.
    """
    return LidarRenderRequest(
        scene_id=scene_id,
        lidar_config=lidar_spec,
        lidar_id=lidar_id,
        output_mode=output_mode,
        firing_pattern_id=firing_pattern_id,
        return_types=return_types,
        frame_start_us=frame_start_timestamp_us,
        frame_end_us=frame_end_timestamp_us,
        sensor_pose=PosePair(
            start_pose=se3_to_grpc_pose(poses[0]),
            end_pose=se3_to_grpc_pose(poses[1]),
        ),
        dynamic_objects=dynamic_poses_to_dynamic_objects(dynamic_poses),
        render_filter=render_filter,
    )


async def save_request(fname: str, image_bytes: bytes) -> None:
    async with aiofiles.open(fname, "wb") as output_file:
        await output_file.write(image_bytes)


def firing_pattern_arrays(pattern: LidarFiringPatternReturn) -> dict[str, np.ndarray]:
    """Decode a firing pattern into the arrays that index every per-ray frame.

    Args:
        pattern: Protobuf pattern response with ray directions, firing times,
            and optional packet boundaries, grid dimensions, and content ID.

    Returns:
        Named NumPy arrays for offline storage: float32 ``(n_rays, 3)`` directions
        and int64 ``(n_rays,)`` firing offsets in microseconds, plus available
        metadata. Exact per-packet times are expanded to per-ray offsets.
        Arrays decoded with ``frombuffer`` may be read-only views of message bytes.

    Raises:
        ValueError: If buffer lengths disagree with declared sizes, packet offsets
            are invalid, or per-ray and per-packet timing is missing or ambiguous.
    """
    directions = np.frombuffer(pattern.directions_sensor, dtype="<f4")
    expected_direction_values = pattern.num_rays * 3
    if directions.shape != (expected_direction_values,):
        raise ValueError(
            f"firing pattern declares {pattern.num_rays} rays but carries {len(directions)} direction values"
        )
    arrays: dict[str, np.ndarray] = {
        "directions_sensor": directions.reshape(pattern.num_rays, 3),
    }
    ray_timestamp_offsets = np.frombuffer(pattern.timestamp_offsets_us, dtype="<i8")
    packet_ray_offsets = None
    packet_timestamp_offsets = None
    # Zero means the sensor exposes no (column, line) addressing, so write neither key
    # rather than a grid of zeros a reader would take at face value.
    if pattern.num_columns or pattern.num_rows:
        if not (pattern.num_columns and pattern.num_rows) or pattern.num_columns * pattern.num_rows != pattern.num_rays:
            raise ValueError("firing pattern grid dimensions must both be set and cover every ray")
        arrays["n_columns"] = np.array(pattern.num_columns, dtype=np.int64)
        arrays["n_rows"] = np.array(pattern.num_rows, dtype=np.int64)
    if pattern.firing_pattern_id:
        arrays["firing_pattern_id"] = np.asarray(pattern.firing_pattern_id)
    if pattern.frame_period_us:
        arrays["frame_period_us"] = np.asarray(pattern.frame_period_us, dtype=np.int64)
    if pattern.HasField("packet_layout"):
        layout = pattern.packet_layout
        packet_ray_offsets = np.frombuffer(layout.ray_offsets_u32_le, dtype="<u4")
        if packet_ray_offsets.shape != (layout.num_packets + 1,):
            raise ValueError(
                f"packet layout declares {layout.num_packets} packets but carries "
                f"{len(packet_ray_offsets)} prefix offsets"
            )
        if (
            len(packet_ray_offsets) == 0
            or packet_ray_offsets[0] != 0
            or packet_ray_offsets[-1] != pattern.num_rays
            or np.any(packet_ray_offsets[1:] <= packet_ray_offsets[:-1])
        ):
            raise ValueError("packet layout prefix offsets must start at zero, increase, and cover every ray")
        arrays["packet_ray_offsets"] = packet_ray_offsets
        if layout.timestamp_offsets_us:
            packet_timestamp_offsets = np.frombuffer(layout.timestamp_offsets_us, dtype="<i8")
            if packet_timestamp_offsets.shape != (layout.num_packets,):
                raise ValueError(
                    f"packet layout declares {layout.num_packets} packets but carries "
                    f"{len(packet_timestamp_offsets)} timestamp offsets"
                )
            arrays["packet_timestamp_offsets_us"] = packet_timestamp_offsets
        if layout.starts_new_scan_mask_packed:
            expected_bytes = (layout.num_packets + 7) // 8
            if len(layout.starts_new_scan_mask_packed) != expected_bytes:
                raise ValueError(f"packet scan-start mask must contain exactly {expected_bytes} bytes")
            arrays["packet_starts_new_scan"] = np.unpackbits(
                np.frombuffer(layout.starts_new_scan_mask_packed, dtype=np.uint8),
                count=layout.num_packets,
                bitorder="little",
            ).view(bool)

    if len(ray_timestamp_offsets) and packet_timestamp_offsets is not None:
        raise ValueError("firing pattern must use either per-ray or per-packet timestamps, not both")
    if len(ray_timestamp_offsets):
        if ray_timestamp_offsets.shape != (pattern.num_rays,):
            raise ValueError(
                f"firing pattern declares {pattern.num_rays} rays but carries "
                f"{len(ray_timestamp_offsets)} timestamp offsets"
            )
        arrays["timestamp_offsets_us"] = ray_timestamp_offsets
    elif packet_timestamp_offsets is not None:
        assert packet_ray_offsets is not None
        arrays["timestamp_offsets_us"] = np.repeat(packet_timestamp_offsets, np.diff(packet_ray_offsets))
    else:
        raise ValueError("firing pattern carries neither per-ray nor per-packet timestamps")
    return arrays


@contextmanager
def _atomic_output(path: str) -> Iterator[IO[bytes]]:
    """Atomically publish a completed file with ordinary umask-based permissions."""
    temporary_path = os.path.join(os.path.dirname(path) or ".", f".lidar-{uuid4().hex}")
    # Exclusive creation respects the umask (unlike NamedTemporaryFile's fixed 0600)
    # without changing process-wide permissions while other export threads are writing.
    with open(temporary_path, "xb") as output:
        try:
            yield output
            output.flush()
            os.fsync(output.fileno())
            output.close()
            os.replace(temporary_path, path)
        finally:
            output.close()
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)


def _atomic_save_npz(path: str, arrays: dict[str, np.ndarray]) -> None:
    """Compress an NPZ without exposing a partially written destination."""
    with _atomic_output(path if path.endswith(".npz") else f"{path}.npz") as output:
        np.savez_compressed(output, **arrays)


def _claim_per_ray_output(output_dir: str) -> None:
    """Reserve one run's pattern directory, refusing existing or legacy exports."""
    os.makedirs(output_dir, exist_ok=True)
    if any(os.path.lexists(os.path.join(output_dir, name)) for name in ("firing_patterns.json", "firing_pattern.npz")):
        raise ValueError(
            "Per-ray export requires a fresh output directory; an existing firing-pattern export was found"
        )
    try:
        os.mkdir(os.path.join(output_dir, "firing_patterns"))
    except FileExistsError as exc:
        raise ValueError("Per-ray export requires a fresh output directory; firing_patterns already exists") from exc


def _save_firing_pattern_catalog(
    output_dir: str,
    patterns_by_id: dict[str, LidarFiringPatternReturn],
    catalog: dict[str, str] | None = None,
) -> dict[str, str]:
    """Atomically publish newly observed patterns, then their output-root-relative catalog."""
    if not patterns_by_id:
        raise ValueError("a per-ray render sequence must select at least one firing pattern")

    catalog_path = os.path.join(output_dir, "firing_patterns.json")
    if catalog is None and os.path.exists(catalog_path):
        raise ValueError("Refusing to replace another run's firing-pattern catalog")
    updated = dict(catalog or {})
    missing = {key: pattern for key, pattern in patterns_by_id.items() if key not in updated}
    if not missing:
        return updated
    os.makedirs(os.path.join(output_dir, "firing_patterns"), exist_ok=True)
    for pattern_id, pattern in missing.items():
        relative_path = f"firing_patterns/pattern_{len(updated):03d}.npz"
        _atomic_save_npz(os.path.join(output_dir, relative_path), firing_pattern_arrays(pattern))
        updated[pattern_id] = relative_path
    with _atomic_output(catalog_path) as output:
        output.write(json.dumps(updated, indent=2).encode("utf-8"))
    log.info(f"wrote {catalog_path}: {len(updated)} firing patterns")
    return updated


async def _cache_response_firing_patterns(
    client_service,
    *,
    scene_id: str,
    lidar_id: str,
    requests: list[LidarRenderRequest],
    responses: list[LidarRenderReturn],
    patterns_by_id: dict[str, LidarFiringPatternReturn],
) -> None:
    """Cache and validate exact geometry for response IDs without changing requests."""
    if len(requests) != len(responses):
        raise ValueError(f"render returned {len(responses)} lidar frames for {len(requests)} requests")

    missing_ids: dict[str, int] = {}
    for request, response in zip(requests, responses):
        pattern_id = response.firing_pattern_id
        if not pattern_id:
            raise ValueError("server returned a per-ray frame without a firing pattern ID")
        if request.firing_pattern_id and request.firing_pattern_id != pattern_id:
            raise ValueError(
                f"server returned firing pattern {pattern_id!r} for a request pinned to {request.firing_pattern_id!r}"
            )

        if response.HasField("firing_pattern"):
            pattern = response.firing_pattern
            if pattern.firing_pattern_id != pattern_id:
                raise ValueError(
                    f"embedded firing pattern {pattern.firing_pattern_id!r} does not match response ID {pattern_id!r}"
                )
            firing_pattern_arrays(pattern)
            if pattern.num_rays != response.num_rays:
                raise ValueError(
                    f"firing pattern {pattern_id!r} has {pattern.num_rays} rays but its response "
                    f"has {response.num_rays}"
                )
            patterns_by_id.setdefault(pattern_id, pattern)
        elif pattern_id not in patterns_by_id:
            missing_ids.setdefault(pattern_id, response.num_rays)

    for pattern_id, expected_rays in missing_ids.items():
        # A later response in this batch may have supplied the geometry.
        if pattern_id in patterns_by_id:
            continue
        try:
            pattern = await _call_with_admission_retry(
                client_service.get_lidar_firing_pattern,
                LidarFiringPatternRequest(
                    scene_id=scene_id,
                    lidar_id=lidar_id,
                    firing_pattern_id=pattern_id,
                ),
            )
        except grpc.aio.AioRpcError as exc:
            if exc.code() not in (grpc.StatusCode.INVALID_ARGUMENT, grpc.StatusCode.NOT_FOUND):
                raise
            raise ValueError(
                f"server selected firing pattern {pattern_id!r} but its exact geometry is unavailable: {exc.details()}"
            ) from exc
        if pattern.firing_pattern_id != pattern_id:
            raise ValueError(f"requested firing pattern {pattern_id!r}, got {pattern.firing_pattern_id!r}")
        firing_pattern_arrays(pattern)
        if pattern.num_rays != expected_rays:
            raise ValueError(
                f"firing pattern {pattern_id!r} has {pattern.num_rays} rays but its response has {expected_rays}"
            )
        patterns_by_id[pattern_id] = pattern

    for request, response in zip(requests, responses):
        pattern = patterns_by_id[response.firing_pattern_id]
        if pattern.num_rays != response.num_rays:
            raise ValueError(
                f"firing pattern {response.firing_pattern_id!r} has {pattern.num_rays} rays but "
                f"a response has {response.num_rays}"
            )


def _pin_requests_to_response_patterns(requests: list[LidarRenderRequest], responses: list[LidarRenderReturn]) -> None:
    """Explicitly reuse first-pass geometry on later export repetitions."""
    for request, response in zip(requests, responses, strict=True):
        request.firing_pattern_id = response.firing_pattern_id


def per_ray_arrays(response: LidarRenderReturn) -> dict[str, np.ndarray]:
    """Decode a per-ray response into layer-major arrays indexed by return then ray.

    Each packed validity layer has its own trailing padding bits, so masks must be split at
    ``ceil(num_rays / 8)`` before unpacking. Treating the concatenation as one bit stream
    would shift every layer after the first whenever num_rays is not byte-aligned.

    Args:
        response: Protobuf per-ray render result declaring ray count, return layers,
            distances in metres, validity bits, and optional intensities and pattern ID.

    Returns:
        Arrays named ``return_types``, ``return_type_names``, ``distance_m``, and ``valid``, plus optional
        ``intensity`` and ``firing_pattern_id``. Ray data has shape
        ``(num_return_layers, num_rays)``; invalid distances and intensities must be
        ignored using ``valid``. Buffer-backed distance and intensity arrays may be read-only.

    Raises:
        ValueError: If there are no return layers, layer types are unspecified or unsupported,
            layer types disagree with the declared count, or any distance, validity, or
            intensity buffer has the wrong size.
    """
    n = response.num_rays
    num_layers = response.num_return_layers
    if num_layers == 0:
        raise ValueError("per-ray response declares zero return layers")
    if len(response.return_types) != num_layers:
        raise ValueError(
            f"per-ray response declares {num_layers} return layers but carries "
            f"{len(response.return_types)} return types"
        )
    if any(
        value not in (LidarReturnType.LIDAR_RETURN_FIRST, LidarReturnType.LIDAR_RETURN_STRONGEST)
        for value in response.return_types
    ):
        raise ValueError("per-ray response contains an unspecified or unsupported return type")

    expected_values = num_layers * n
    distances = np.frombuffer(response.ray_distances_m_buffer, dtype="<f4")
    if distances.size != expected_values:
        raise ValueError(
            f"per-ray distance buffer has {distances.size} entries but the response declares "
            f"{num_layers} layers x {n} rays"
        )

    mask_stride = (n + 7) // 8
    packed_valid = np.frombuffer(response.ray_valid_masks_packed, dtype=np.uint8)
    if packed_valid.size != num_layers * mask_stride:
        raise ValueError(
            f"per-ray validity buffer has {packed_valid.size} bytes but the response declares "
            f"{num_layers} layers x {n} rays"
        )

    arrays: dict[str, np.ndarray] = {
        "return_types": np.asarray(response.return_types, dtype=np.int32),
        "return_type_names": np.asarray([LidarReturnType.Name(value) for value in response.return_types]),
        "distance_m": distances.reshape(num_layers, n),
        "valid": np.unpackbits(packed_valid.reshape(num_layers, mask_stride), axis=1, bitorder="little")[:, :n].view(
            bool
        ),
    }
    if response.ray_intensities_buffer:
        intensities = np.frombuffer(response.ray_intensities_buffer, dtype="<f4")
        if intensities.size != expected_values:
            raise ValueError(
                f"per-ray intensity buffer has {intensities.size} entries but the response declares "
                f"{num_layers} layers x {n} rays"
            )
        arrays["intensity"] = intensities.reshape(num_layers, n)
    if response.firing_pattern_id:
        arrays["firing_pattern_id"] = np.asarray(response.firing_pattern_id)
    return arrays


def _lidar_request_arrays(request: LidarRenderRequest) -> dict[str, np.ndarray]:
    """Record the actual wire poses and interval without reducing their precision."""
    arrays = {
        "frame_start_us": np.asarray(request.frame_start_us, dtype=np.int64),
        "frame_end_us": np.asarray(request.frame_end_us, dtype=np.int64),
        "scene_id": np.asarray(request.scene_id),
        "lidar_id": np.asarray(request.lidar_id),
    }
    for name, pose in (("start", request.sensor_pose.start_pose), ("end", request.sensor_pose.end_pose)):
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = R.from_quat([pose.quat.x, pose.quat.y, pose.quat.z, pose.quat.w]).as_matrix()
        transform[:3, 3] = [pose.vec.x, pose.vec.y, pose.vec.z]
        arrays[f"T_lidar_to_world_{name}"] = transform
    return arrays


async def save_lidar_response(
    fname: str,
    response: LidarRenderReturn,
    format: str = "bin",
    request: LidarRenderRequest | None = None,
) -> None:
    """Save a lidar response as raw binary, PLY, or per-ray NPZ arrays.

    Args:
        fname: Output path. Callers supply the suffix; NumPy appends ``.npz`` for
            per-ray output when it is missing.
        response: Render response matching the requested storage representation.
        format: ``per-ray`` stores all emitted rays in NPZ, including an embedded
            pattern when supplied; ``ply`` stores the point cloud with intensity
            colors; ``bin`` stores raw point and intensity buffers.
        request: Optional originating request. Per-ray exports include its exact
            interval, sensor poses, and scene/sensor IDs when supplied. The CLI always
            supplies it; point-cloud exports ignore it.

    Returns:
        ``None`` after the output file has been written.

    Raises:
        ValueError: If per-ray data is absent or malformed, an embedded pattern is
            invalid, or point-cloud buffers cannot be reshaped for PLY output.
        OSError: If the destination cannot be opened or written.
    """
    if format == "per-ray":
        if not response.num_rays:
            raise ValueError(
                "The server returned no per-ray data. Was the request sent with "
                "output_mode=LIDAR_OUTPUT_PER_RAY against a lidar served from a firing pattern?"
            )
        arrays = per_ray_arrays(response)
        # Only present when the request asked the server to embed it. The run-level catalog
        # maps every selected ID to its sidecar, including singleton runs.
        if response.HasField("firing_pattern"):
            if response.firing_pattern.firing_pattern_id != response.firing_pattern_id:
                raise ValueError("embedded firing pattern ID does not match the response ID")
            if response.firing_pattern.num_rays != response.num_rays:
                raise ValueError("embedded firing pattern ray count does not match the response")
            arrays.update(firing_pattern_arrays(response.firing_pattern))
        if request is not None:
            arrays.update(_lidar_request_arrays(request))
        # CPU-bound on ~3.9 MB a frame, and the caller gathers every frame, so doing it inline
        # would serialise that gather and stall the loop's in-flight reads.
        await asyncio.to_thread(_atomic_save_npz, fname, arrays)
        return

    if format == "ply":
        # Parse xyz and intensity from response
        if response.point_xyzs_buffer:
            xyz = np.frombuffer(response.point_xyzs_buffer, dtype="<f4").reshape(-1, 3)
        else:
            xyz = np.array(response.point_xyzs, dtype=np.float32).reshape(-1, 3)

        intensity = None
        if response.point_intensities_buffer:
            intensity = np.frombuffer(response.point_intensities_buffer, dtype="<f4")
        elif response.point_intensities:
            intensity = np.array(response.point_intensities, dtype=np.float32)

        # Saving stays synchronous; both render commands use this same PLY representation.
        save_lidar_ply(fname, xyz, intensity)
    else:
        # Raw binary format
        async with aiofiles.open(fname, "wb") as f:
            if response.point_xyzs_buffer:
                await f.write(response.point_xyzs_buffer)
            if response.point_intensities_buffer:
                await f.write(response.point_intensities_buffer)


# --- Admission-tolerant render fan-out -------------------------------------------------------
#
# A server started with `--multiproc-max-in-flight N` bounds how many concurrent renders it admits;
# at the default `--multiproc-admission-timeout-s 0` it *sheds* every excess concurrent render as
# grpc RESOURCE_EXHAUSTED rather than queuing it. This client fans its per-frame renders out
# concurrently (asyncio.gather), so without tolerance the very first shed aborts the whole run.
#
# Tolerance has two parts:
#   1. Retry ONLY RESOURCE_EXHAUSTED with capped exponential backoff (bounded total attempts).
#   2. An ADAPTIVE concurrency clamp (`_AdmissionLimiter`). By default the fan-out is unbounded --
#      against a non-admission server nothing ever sheds, so the clamp never engages and the
#      historical parallelism is preserved (Trap 5). But retry-with-backoff alone is a lottery
#      against a small admission cap (N contenders race for one slot; some caller loses every
#      retry), so on the FIRST observed shed the limiter installs a shared serializing semaphore
#      that all subsequent and retried calls pass through -- collapsing the contention so retries
#      drain one at a time and exhaustion becomes practically impossible. An explicit
#      `--max-concurrent-renders` bound takes precedence and disables the adaptive override (the
#      user already picked a bound); shed-retry still applies.
_RESOURCE_EXHAUSTED_MAX_ATTEMPTS = 12
_RESOURCE_EXHAUSTED_BASE_BACKOFF_S = 0.05
_RESOURCE_EXHAUSTED_MAX_BACKOFF_S = 2.0


class _AdmissionLimiter:
    """Shared, adaptively-installed concurrency clamp for one render fan-out.

    - With an explicit ``limit`` (``--max-concurrent-renders``): a fixed semaphore guards every
      call from the start and no adaptive behavior is needed.
    - Without one: starts unbounded (``active`` is ``None``); the first ``note_shed()`` installs a
      shared ``Semaphore(1)`` so all further/retried calls serialize through a single slot.
    """

    def __init__(self, limit: Optional[int] = None) -> None:
        if limit is not None and limit <= 0:
            raise ValueError(f"--max-concurrent-renders must be a positive integer, got {limit}")
        self._explicit: Optional[asyncio.Semaphore] = asyncio.Semaphore(limit) if limit is not None else None
        # A user-chosen bound disables the adaptive serialize-on-shed override.
        self._adaptive_disabled: bool = limit is not None
        self._adaptive: Optional[asyncio.Semaphore] = None

    @property
    def active(self) -> Optional[asyncio.Semaphore]:
        """The semaphore currently gating calls (explicit bound, else the adaptive clamp if installed)."""
        return self._explicit if self._explicit is not None else self._adaptive

    def note_shed(self) -> None:
        """Engage the adaptive serializer on the first admission shed (idempotent; no-op if bounded)."""
        # Creating a Semaphore is synchronous, so this check-and-set is atomic within one event-loop
        # step -- no lock needed on the single-threaded asyncio loop.
        if not self._adaptive_disabled and self._adaptive is None:
            log.info("render admission shed observed; serializing subsequent renders (Semaphore(1)) to drain")
            self._adaptive = asyncio.Semaphore(1)


async def _call_with_admission_retry(
    method: Callable,
    request: Any,
    *,
    limiter: Optional[_AdmissionLimiter] = None,
    max_attempts: int = _RESOURCE_EXHAUSTED_MAX_ATTEMPTS,
    base_backoff_s: float = _RESOURCE_EXHAUSTED_BASE_BACKOFF_S,
    max_backoff_s: float = _RESOURCE_EXHAUSTED_MAX_BACKOFF_S,
) -> Any:
    """Invoke a unary render or pattern-lookup RPC, tolerating capacity backpressure.

    Retries ONLY grpc RESOURCE_EXHAUSTED (the admission-shed status) with capped exponential
    backoff, up to ``max_attempts`` total tries; every other status -- and a genuine failure --
    propagates unchanged on its first occurrence. Each attempt is gated by ``limiter.active`` (which
    may transition from unbounded to a serializing semaphore mid-fan-out; see ``_AdmissionLimiter``),
    and each shed calls ``limiter.note_shed()`` so the first shed engages the adaptive clamp.
    """
    attempt = 0
    while True:
        active = limiter.active if limiter is not None else None
        try:
            if active is not None:
                async with active:
                    return await method(request)
            return await method(request)
        except grpc.aio.AioRpcError as error:
            if error.code() is not grpc.StatusCode.RESOURCE_EXHAUSTED:
                raise
            if limiter is not None:
                limiter.note_shed()
            attempt += 1
            if attempt >= max_attempts:
                log.error(
                    "RPC still reports RESOURCE_EXHAUSTED after %d attempts; giving up. "
                    "Check server capacity and concurrent load.",
                    max_attempts,
                )
                raise
            backoff_s = min(base_backoff_s * (2 ** (attempt - 1)), max_backoff_s)
            log.warning(
                "RPC capacity exhausted (RESOURCE_EXHAUSTED); retrying in %.3fs (attempt %d/%d)",
                backoff_s,
                attempt,
                max_attempts,
            )
            await asyncio.sleep(backoff_s)


async def _execute_renders(
    method: Callable,
    reqs: list[Any],
    sequential_exec: bool,
    limiter: Optional[_AdmissionLimiter] = None,
) -> list[Any]:
    """Execute requests with warmup: first request solo, rest either sequentially or in parallel.

    Every render tolerates admission backpressure via ``_call_with_admission_retry``, sharing one
    ``limiter`` across the whole fan-out so the adaptive serialize-on-shed clamp (or an explicit
    bound) applies to every call. Results are returned in request order (``asyncio.gather``
    preserves ordering).
    """
    if len(reqs) == 0:
        return []
    if limiter is None:
        limiter = _AdmissionLimiter()
    warmup, *rest = reqs
    results = [await _call_with_admission_retry(method, warmup, limiter=limiter)]
    if rest:
        n_requests = len(rest)
        t0 = time.perf_counter()
        if sequential_exec:
            results.extend([await _call_with_admission_retry(method, r, limiter=limiter) for r in rest])
        else:
            results.extend(
                await asyncio.gather(*[_call_with_admission_retry(method, r, limiter=limiter) for r in rest])
            )
        elapsed = time.perf_counter() - t0
        s_per_request = elapsed / n_requests
        print(f"Served {n_requests=} in {elapsed:.3f}s, {s_per_request=:.3f}s")
    return results


@click.command("render-grpc")
@cli.scopedtimer_cli_options(print_func=log.info)
@click.option(
    "--artifact-path",
    type=str,
    help="Path to the NuRec artifact `last.usdz`",
    default=None,
    required=True,
)
@click.option(
    "--output-dir",
    type=str,
    help="Path to the output rendered image",
    required=True,
    default=None,
)
@click.option(
    "--host",
    type=str,
    help="GRPC server host",
    default="localhost",
    required=False,
)
@click.option(
    "--port",
    type=int,
    help="Port to run the gRPC server on",
    default=8080,
)
@click.option(
    "--height",
    type=int,
    help="Height of the image",
    default=300,
)
@click.option(
    "--camera-id",
    type=str,
    help="Camera ID",
    default="camera_front_wide_120fov",
)
@click.option(
    "--image-format",
    type=click.Choice(["png", "jpeg"]),
    help="png or jpeg",
    default="jpeg",
)
@click.option(
    "--output-type",
    type=click.Choice(["color", "distance-colorized", "semantics-colorized"]),
    multiple=True,
    default=("color",),
    help="Camera render product(s). Can be specified multiple times to render several output types in one pass: "
    "color (default), distance-colorized (turbo colormap), semantics-colorized (class colors). "
    "Ignored when --lidar is set. Prefer --image-format png for semantics.",
)
@click.option(
    "--overlay-bboxes",
    type=click.Choice(["none", "model", "artifact"]),
    default="none",
    help="CameraRenderRequest.overlay_bounding_boxes: track source for cuboid wireframes drawn on every requested output (none, model checkpoint tracks, or USDZ sequence_tracks).",
)
@click.option(
    "--bbox-color",
    type=(click.IntRange(0, 255), click.IntRange(0, 255), click.IntRange(0, 255)),
    default=None,
    help="RGB color (0-255) for bounding box edges when --overlay-bboxes is model or artifact. Server default is (0, 255, 0).",
)
@click.option(
    "--colorization-max-distance",
    type=float,
    help="Upper bound of the distance colormap in meters. Applies when --output-type is distance-colorized. Uses server default when not set.",
    default=None,
)
@click.option(
    "--lidar",
    is_flag=True,
    help="If set, call render_lidar instead of render_camera and save point clouds.",
    default=False,
)
@click.option(
    "--lidar-id",
    type=str,
    help="Lidar ID to use for frame timestamps when --lidar is set.",
    default=None,
)
@click.option(
    "--lidar-format",
    type=click.Choice(["bin", "ply", "per-ray"]),
    help="Output format for lidar: 'bin' (raw binary), 'ply' (with intensity colors, like validation), "
    "or 'per-ray' (one uncompensated .npz per frame plus firing_patterns.json mapping IDs to "
    "firing_patterns/ sidecars). Per-ray runs require a fresh output directory. 'per-ray' also selects "
    "the server's per-ray output mode, which needs a lidar served from a firing pattern.",
    default="bin",
)
@click.option(
    "--frame-step",
    type=int,
    help="Step size in frames",
    default=1,
)
@click.option(
    "--repeat-count",
    type=click.IntRange(min=1),
    help="Render the same request sequence this many times. Repeated outputs are written under repeat-NNN.",
    default=1,
)
@click.option(
    "--disable-rolling-shutter",
    is_flag=True,
    help="Disable rolling shutter by applying the frame-end timestamps to full frames (useful for debugging)",
    default=False,
)
@click.option(
    "--enable-editing-actors",
    is_flag=True,
    help="Enable sending dynamic actor updates in render requests.",
    default=False,
)
@click.option(
    "--demo-actor-transform",
    is_flag=True,
    help="Demo: apply a precomputed transformation to actor poses. Requires --enable-editing-actors.",
    default=False,
)
@click.option(
    "--wait-server",
    type=int,
    default=None,
    help="Wait up to N seconds for the gRPC server to come up before rendering "
    "(0 = wait indefinitely). If unset, fail fast on the first RPC.",
)
@click.option(
    "--shutdown-server-on-completion",
    is_flag=True,
    help="Shutdown the server on completion",
    default=False,
)
@click.option(
    "--frame-naming",
    type=click.Choice(["frame-end-timestamp", "contiguous-output-index"]),
    help=(
        "File naming scheme for exported frames: "
        "'frame-end-timestamp' - global absolute frame-end timestamp in microseconds, "
        "'contiguous-output-index' - frames always indexed 0,1,2,... per sensor, irrespective of --frame-step."
    ),
    default="contiguous-output-index",
)
@click.option(
    "--rig-name",
    type=str,
    default=None,
    help="Rig name for the inpainted ego hood (e.g. hyperion8.0 or hyperion8.1). Set to None to disable inpainting.",
)
@click.option(
    "--rig-translation-offset",
    nargs=3,
    type=float,
    help="Translation offsets (tx,ty,tz) in meters in rig space to be applied to the rig prior to rendering.",
    default=(0.0, 0.0, 0.0),
)
@click.option(
    "--rig-rotation-offset",
    nargs=3,
    type=float,
    # TODO: update this to (roll, pitch, yaw) once the axis permutation hack is removed from pose_offsets_to_se3().
    help="Rotation offsets (yaw, -roll, -pitch) in degrees in rig space to be applied to the rig prior to rendering.",
    default=(0.0, 0.0, 0.0),
)
@click.option(
    "--edit-assets",
    type=str,
    help="Path to JSON file containing edit assets configuration, see nurec/utils/io/export/external_assets.py for more details + example",
    default=None,
    required=False,
)
@click.option(
    "--sequential",
    is_flag=True,
    help="Send gRPC requests one-by-one (no parallel batch after warmup)",
    default=False,
)
@click.option(
    "--max-concurrent-renders",
    type=int,
    default=None,
    help=(
        "Cap the number of in-flight render RPCs during the concurrent fan-out. Unset = unbounded "
        "(the historical behavior; no throughput change against a single-process server). Set this "
        "to your server's --multiproc-max-in-flight to avoid shedding renders when the server runs "
        "with the default --multiproc-admission-timeout-s 0. RESOURCE_EXHAUSTED sheds are retried "
        "with backoff regardless of this value."
    ),
)
@click.option(
    "--temporal-harmonizer-session",
    is_flag=True,
    help=(
        "Deprecated and has no effect. Use --apply-harmonizer; render-grpc manages a session automatically for an "
        "applied colour trajectory targeting a temporal server."
    ),
    default=False,
)
@click.option(
    "--apply-harmonizer",
    is_flag=True,
    help=(
        "Request Harmonizer postprocessing for every camera colour frame. On a temporal target, "
        "render-grpc creates and manages one session automatically."
    ),
    default=False,
)
@click.option(
    "--lidar-raydrop-threshold",
    type=float,
    help="Raydrop probability threshold [0-1]. Rays with raydrop > threshold are dropped.",
    default=0.5,
)
@click.option(
    "--lidar-opacity-threshold",
    type=float,
    help="Opacity threshold [0-1]. Rays with opacity <= threshold are dropped. Set to 0.0 to disable.",
    default=0.0,
)
@click.option(
    "--lidar-distance-filter/--no-lidar-distance-filter",
    "lidar_enable_distance_filter",
    help="Enable/disable distance-based edge filtering to remove floating points. (default: use artifact config)",
    default=None,
)
@click.option(
    "--lidar-distance-filter-threshold",
    type=float,
    help="Distance filter threshold [0-1]. Higher = fewer points filtered. (default: use artifact config)",
    default=None,
)
@click.option(
    "--lidar-distance-filter-neighbor-source",
    type=click.Choice(["rangeview", "angle"]),
    help="Ray-neighbour source for the distance filter: 'rangeview' needs the sensor's (rows, columns) "
    "grid; 'angle' works from ray directions alone and is the one a model-free lidar can use. "
    "(default: use artifact config)",
    default=None,
)
@click.option(
    "--lidar-distance-filter-elevation-resolution-deg",
    type=float,
    help="Angle source only: elevation row (scanline) spacing in degrees. Sensors whose rows interleave "
    "within a frame (e.g. ET30X) need this set explicitly (~FOV_el/n_beams). (default: use artifact config)",
    default=None,
)
@click.option(
    "--lidar-distance-filter-azimuth-max-gap-deg",
    type=float,
    help="Angle source only: max azimuth gap in degrees for two rays to count as neighbours, which skips "
    "scan holes. (default: use artifact config)",
    default=None,
)
@click.pass_context
def render_grpc(
    ctx: click.Context,
    artifact_path: str,
    output_dir: str,
    host: str,
    port: int,
    height: int,
    camera_id: str,
    image_format: str,
    output_type: tuple[str, ...],
    overlay_bboxes: str,
    bbox_color: Optional[Tuple[int, int, int]],
    colorization_max_distance: Optional[float],
    frame_step: int,
    repeat_count: int,
    disable_rolling_shutter: bool,
    enable_editing_actors: bool,
    demo_actor_transform: bool,
    wait_server: Optional[int],
    shutdown_server_on_completion: bool,
    rig_name: Optional[str],
    rig_translation_offset: Tuple[float, float, float],
    rig_rotation_offset: Tuple[float, float, float],
    edit_assets: Optional[str],
    lidar: bool,
    lidar_id: Optional[str],
    lidar_format: str,
    sequential: bool,
    max_concurrent_renders: Optional[int],
    temporal_harmonizer_session: bool,
    apply_harmonizer: bool,
    lidar_raydrop_threshold: float,
    lidar_opacity_threshold: float,
    lidar_enable_distance_filter: Optional[bool],
    lidar_distance_filter_threshold: Optional[float],
    lidar_distance_filter_neighbor_source: Optional[str],
    lidar_distance_filter_elevation_resolution_deg: Optional[float],
    lidar_distance_filter_azimuth_max_gap_deg: Optional[float],
    frame_naming: str,
) -> None:
    """Render camera or LiDAR outputs over the artifact trajectory.

    Args:
        ctx: Click invocation context used to capture and save effective CLI settings.
        artifact_path: Local artifact path supplying trajectories and sensor calibration.
        output_dir: Directory for rendered frames, settings, and optional pattern sidecars.
        host: Rendering service hostname or IP address.
        port: Rendering service TCP port.
        height: Requested camera image height in pixels.
        camera_id: Logical camera to render when ``lidar`` is false.
        image_format: Camera image encoding, ``png`` or ``jpeg``.
        output_type: Camera products to render, selected from ``color``,
            ``distance-colorized``, and ``semantics-colorized`` without duplicates.
        overlay_bboxes: Bounding-box source: ``none``, ``model``, or ``artifact``.
        bbox_color: Optional RGB byte values overriding the bounding-box color.
        colorization_max_distance: Optional distance-colormap maximum in metres.
        frame_step: Stride through trajectory frames when generating requests.
        repeat_count: Positive number of repetitions of the same request sequence;
            multiple runs write separate ``repeat-NNN`` directories.
        disable_rolling_shutter: Whether camera requests use a single exposure pose.
        enable_editing_actors: Whether requests may update actor poses.
        demo_actor_transform: Apply demonstration actor transforms; requires actor editing.
        wait_server: Connection-wait limit in seconds, zero for indefinite waiting,
            or ``None`` to fail on the first unsuccessful RPC.
        shutdown_server_on_completion: Whether to shut down the service after rendering.
        rig_name: Optional ego-hood rig identifier for camera inpainting; requires color output.
        rig_translation_offset: ``(tx, ty, tz)`` offset in metres in rig coordinates.
        rig_rotation_offset: ``(yaw, -roll, -pitch)`` offset in degrees in rig coordinates.
        edit_assets: Optional path to a JSON configuration for asset replacement or insertion.
        lidar: Whether to render lidar instead of cameras.
        lidar_id: Optional logical lidar ID; required when selection would be ambiguous.
        lidar_format: ``bin`` or ``ply`` for point clouds, or ``per-ray`` for NPZ
            ray results and an ID-to-pattern catalog; per-ray mode requires stored patterns.
        sequential: Whether to issue requests one at a time after warmup.
        max_concurrent_renders: Optional positive bound on concurrent render RPCs.
        temporal_harmonizer_session: Deprecated flag; temporal sessions are now
            detected and managed automatically and force ordered camera rendering.
        lidar_raydrop_threshold: Drop rays with predicted raydrop above this [0, 1] threshold.
        lidar_opacity_threshold: Drop rays at or below this [0, 1] opacity threshold;
            zero disables this filter.
        lidar_enable_distance_filter: Optional edge-filter switch; ``None`` uses artifact defaults.
        lidar_distance_filter_threshold: Optional [0, 1] edge threshold; higher
            values filter fewer points, and ``None`` uses artifact defaults.
        lidar_distance_filter_neighbor_source: Optional ``rangeview`` or ``angle``
            neighbor source; ``None`` uses artifact defaults. Model-free rays need ``angle``.
        lidar_distance_filter_elevation_resolution_deg: Optional angular row spacing
            in degrees for the angle filter; ``None`` uses artifact defaults.
        lidar_distance_filter_azimuth_max_gap_deg: Optional maximum neighbor gap in
            degrees for the angle filter; ``None`` uses artifact defaults.
        frame_naming: ``frame-end-timestamp`` for absolute microsecond timestamps,
            or ``contiguous-output-index`` for zero-based output indices per sensor.

    Returns:
        ``None``; writes rendered outputs and associated metadata to ``output_dir``.

    Raises:
        click.UsageError: If a per-ray output directory is nonempty or claimed by another run.
        ValueError: If option combinations, sensor selection, concurrency or repetition
            limits, trajectory data, or per-ray responses are invalid.
        RuntimeError: If requested asset edits fail or the async runner cannot start.
        grpc.RpcError: If an RPC fails after applicable admission retries.
        OSError: If artifact, configuration, or output files cannot be read or written.
    """

    if lidar and lidar_format == "per-ray" and os.path.isdir(output_dir) and any(Path(output_dir).iterdir()):
        raise click.UsageError("Per-ray export requires a fresh, empty --output-dir for each run and lidar", ctx)
    os.makedirs(output_dir, exist_ok=True)

    if demo_actor_transform and not enable_editing_actors:
        raise ValueError("--demo-actor-transform requires --enable-editing-actors")

    if temporal_harmonizer_session:
        log.warning(
            "--temporal-harmonizer-session is deprecated and has no effect. Use --apply-harmonizer; render-grpc "
            "manages a session automatically for an applied colour trajectory targeting a temporal server."
        )

    if apply_harmonizer and lidar:
        raise ValueError("--apply-harmonizer is not valid with --lidar")

    if apply_harmonizer and "color" not in output_type:
        raise ValueError("--apply-harmonizer requires the color output type")

    if lidar and output_type != ("color",):
        raise ValueError("--output-type applies only to camera rendering; with --lidar use the default color.")

    if len(output_type) != len(set(output_type)):
        seen: list[str] = []
        dups = []
        for v in output_type:
            if v in seen and v not in dups:
                dups.append(v)
            else:
                seen.append(v)
        raise ValueError(f"--output-type cannot be specified more than once; duplicate(s): {', '.join(dups)}.")

    per_ray_output_claimed = lidar and lidar_format == "per-ray"
    if per_ray_output_claimed:
        try:
            _claim_per_ray_output(output_dir)
        except ValueError as exc:
            raise click.UsageError(str(exc), ctx) from exc

    # Capture, log, and save CLI settings
    collector = SettingsCollector.from_click_context(ctx, "render-grpc")
    collector.log_settings(log)
    collector.save_json(Path(output_dir) / "render_grpc_cli_args.json")
    if repeat_count > 1:
        for repeat_index in range(repeat_count):
            repeat_output_dir = Path(_output_dir_for_repeat(output_dir, repeat_index, repeat_count))
            repeat_output_dir.mkdir(parents=True, exist_ok=True)
            collector.save_json(repeat_output_dir / "render_grpc_cli_args.json")

    format = {"png": ImageFormat.PNG, "jpeg": ImageFormat.JPEG}[image_format]

    _output_type_map = {
        "color": CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR,
        "distance-colorized": CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED,
        "semantics-colorized": CameraRenderProduct.CAMERA_RENDER_PRODUCT_SEMANTICS_COLORIZED,
    }
    render_products = [_output_type_map[ot] for ot in output_type]

    if rig_name is not None and "color" not in output_type:
        raise ValueError(
            "--rig-name requires the color output type; ego mask inpainting only applies to the color product."
        )

    overlay_bboxes_source = {
        "none": OverlayBoundingBoxSource.OVERLAY_BOUNDING_BOX_SOURCE_UNSPECIFIED,
        "model": OverlayBoundingBoxSource.OVERLAY_BOUNDING_BOX_SOURCE_MODEL,
        "artifact": OverlayBoundingBoxSource.OVERLAY_BOUNDING_BOX_SOURCE_ARTIFACT,
    }[overlay_bboxes]

    if "semantics-colorized" in output_type and image_format == "jpeg":
        log.warning("JPEG is lossy; consider --image-format png for semantics outputs.")

    asyncio.run(
        _render_grpc(
            artifact_path,
            output_dir,
            host,
            port,
            height,
            camera_id,
            format,
            render_products,
            overlay_bboxes_source,
            colorization_max_distance,
            frame_step,
            disable_rolling_shutter,
            enable_editing_actors,
            demo_actor_transform,
            wait_server,
            shutdown_server_on_completion,
            rig_name,
            rig_translation_offset,
            rig_rotation_offset,
            edit_assets,
            lidar,
            lidar_id,
            lidar_format,
            sequential,
            max_concurrent_renders,
            lidar_raydrop_threshold,
            lidar_opacity_threshold,
            lidar_enable_distance_filter,
            lidar_distance_filter_threshold,
            lidar_distance_filter_neighbor_source,
            lidar_distance_filter_elevation_resolution_deg,
            lidar_distance_filter_azimuth_max_gap_deg,
            frame_naming,
            bbox_color,
            repeat_count,
            apply_harmonizer,
            per_ray_output_claimed=per_ray_output_claimed,
        )
    )


def _build_lidar_render_filter(
    *,
    raydrop_threshold: float,
    opacity_threshold: float,
    enable_distance_filter: Optional[bool],
    distance_filter_threshold: Optional[float],
    neighbor_source: Optional[str],
    elevation_resolution_deg: Optional[float],
    azimuth_max_gap_deg: Optional[float],
) -> LidarRenderFilter:
    """The request's filter settings, leaving anything the caller omitted unset.

    An unset optional field is how the request says "use the artifact config": the server only
    overrides its own default where ``HasField`` is true, so omitting a flag must not write one.
    """
    render_filter = LidarRenderFilter(
        raydrop_threshold=raydrop_threshold,
        opacity_threshold=opacity_threshold,
    )
    if enable_distance_filter is not None:
        render_filter.enable_distance_filter = enable_distance_filter
    if distance_filter_threshold is not None:
        render_filter.distance_filter_threshold = distance_filter_threshold
    if neighbor_source is not None:
        render_filter.neighbor_source = (
            LidarFloatingFilterSource.LIDAR_FLOATING_FILTER_SOURCE_ANGLE
            if neighbor_source == "angle"
            else LidarFloatingFilterSource.LIDAR_FLOATING_FILTER_SOURCE_RANGEVIEW
        )
    if elevation_resolution_deg is not None:
        render_filter.elevation_resolution_deg = elevation_resolution_deg
    if azimuth_max_gap_deg is not None:
        render_filter.azimuth_max_gap_deg = azimuth_max_gap_deg
    return render_filter


async def _render_grpc(
    artifact_path: str,
    output_dir: str,
    host: str,
    port: int,
    height: int,
    camera_id: str,
    format: ImageFormat,
    render_products: list[CameraRenderProduct],
    overlay_bounding_boxes: OverlayBoundingBoxSource,
    colorization_max_distance: Optional[float],
    frame_step: int,
    disable_rolling_shutter: bool,
    enable_editing_actors: bool,
    demo_actor_transform: bool,
    wait_server: Optional[int],
    shutdown_server_on_completion: bool,
    rig_name: Optional[str],
    rig_translation_offset: Tuple[float, float, float],
    rig_rotation_offset: Tuple[float, float, float],
    edit_assets: Optional[str],
    lidar: bool,
    lidar_id: Optional[str],
    lidar_format: str,
    sequential: bool,
    max_concurrent_renders: Optional[int],
    lidar_raydrop_threshold: float,
    lidar_opacity_threshold: float,
    lidar_enable_distance_filter: Optional[bool],
    lidar_distance_filter_threshold: Optional[float],
    lidar_distance_filter_neighbor_source: Optional[str] = None,
    lidar_distance_filter_elevation_resolution_deg: Optional[float] = None,
    lidar_distance_filter_azimuth_max_gap_deg: Optional[float] = None,
    frame_naming: str = "contiguous-output-index",
    bbox_color: Optional[Tuple[int, int, int]] = None,
    repeat_count: int = 1,
    apply_harmonizer: bool = False,
    *,
    per_ray_output_claimed: bool = False,
) -> None:
    """Render outputs, claiming per-ray storage unless the CLI already did so."""
    if repeat_count < 1:
        raise ValueError("repeat_count must be a positive integer")
    if lidar and lidar_format == "per-ray" and not per_ray_output_claimed:
        _claim_per_ray_output(output_dir)

    # Concurrent render fan-out clamp. Unset `--max-concurrent-renders` = unbounded with adaptive
    # serialize-on-shed (see _AdmissionLimiter): against a non-admission server nothing sheds so the
    # previous parallelism is preserved exactly; against a bounded-admission server the first shed
    # engages a serializing clamp so the fan-out drains instead of exhausting retries. An explicit
    # bound takes precedence (and is validated > 0 by _AdmissionLimiter).
    render_limiter = _AdmissionLimiter(max_concurrent_renders)

    artifacts_list = Artifact.discover_from_glob(artifact_path)
    assert len(artifacts_list) == 1
    format_ext = {ImageFormat.PNG: "png", ImageFormat.JPEG: "jpeg"}[format]
    artifact = artifacts_list[0]

    # Get test trajectory and poses
    scene_id = artifact.scene_id
    rig_trajectories = RigTrajectories.from_dict(artifact.rig_trajectories)

    # Assume we only have a single trajectory
    assert len(rig_trajectories.rig_trajectories) == 1, (
        f"_render_grpc: expected a single rig_trajectory, got {len(rig_trajectories.rig_trajectories)}"
    )
    trajectory = rig_trajectories.rig_trajectories[0]

    rig_pose_interpolator = ncore_transformations.PoseInterpolator(
        poses=trajectory.T_rig_worlds.cpu(), timestamps=trajectory.T_rig_world_timestamps_us.cpu()
    )

    # Grab cuboid tracks, associated pose interpolators and store them in lists
    assert len(artifact.sequence_tracks) == 1  # Assumes only one sequence
    sequence_tracks = next(iter(artifact.sequence_tracks.values()))
    artifact_track_ids = list(sequence_tracks["tracks_data"]["tracks_id"])
    controllable_actor_tracks = ActorTracks.from_dict_cpu(sequence_tracks)
    cuboid_tracks_list = list(controllable_actor_tracks._cuboid_tracks_list)
    interpolator_list = list(controllable_actor_tracks._interpolator_list)

    # Configure channel with larger message size limits since lidar output can be large
    # Retry the connection roughly every second so the wait honors --wait-server within ~1s
    # (gRPC's default backoff grows toward ~120s, which would overshoot the requested timeout)
    channel_options = [
        ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50MB
        ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50MB
        ("grpc.max_reconnect_backoff_ms", 1000),
    ]
    async with grpc.aio.insecure_channel(f"{host}:{port}", options=channel_options) as channel:
        if wait_server is not None:
            await wait_for_grpc_server(channel, wait_server)

        client_service = SensorsimServiceStub(channel)

        available_cameras: AvailableCamerasReturn = await client_service.get_available_cameras(
            AvailableCamerasRequest(scene_id=scene_id)
        )

        # Process edit assets configuration if provided
        remove = set()
        if edit_assets:
            with open(edit_assets, "r") as f:
                edit_assets_config = json.load(f)

            # Get available external assets from GRPC AssetBank
            external_assets_response: ExternalAssetObjectsReturn = await client_service.get_external_asset_objects(
                ExternalAssetObjectsRequest(scene_id=scene_id)
            )
            available_scene_track_ids = set(external_assets_response.track_ids)

            log.info(f"Available external asset track IDs: {available_scene_track_ids}")

            # track_ids to remove, used later when building CameraRenderRequest
            remove.update(edit_assets_config.get("remove", []))

            # Process tracks to replace
            replace_actions = []
            replace_list = edit_assets_config.get("replace", [])

            id_to_dims: dict = {}
            if replace_list:
                for asset_meta in edit_assets_config.get("metadata", {}).get("external_assets_metadata", []):
                    track_id = asset_meta.get("track_id")
                    cuboid_dims = asset_meta.get("cuboid_dims")
                    if track_id is not None and cuboid_dims is not None:
                        id_to_dims[track_id] = cuboid_dims

            for replacement_spec in replace_list:
                original_id = replacement_spec.get("original_id")
                replacement_id = replacement_spec.get("replacement_id")
                if not original_id or not replacement_id:
                    continue

                # Get specified dimensions or fallback to dimensions packaged in edit-assets.json
                object_size_list = replacement_spec.get("object_size", id_to_dims.get(replacement_id))
                if object_size_list is not None and len(object_size_list) == 0:
                    object_size_list = id_to_dims.get(replacement_id)

                assert object_size_list is not None and len(object_size_list) == 3, (
                    f"object_size must be list of 3 floats for replacing {original_id} --> {replacement_id}, got: {object_size_list}"
                )

                action = ReplaceAssetAction(
                    original_id=original_id,
                    replacement_id=replacement_id,
                    object_size=AABB(
                        size_x=float(object_size_list[0]),
                        size_y=float(object_size_list[1]),
                        size_z=float(object_size_list[2]),
                    ),
                )
                replace_actions.append(action)

            # Process tracks to insert
            insert: List[DynamicObjectTrack] = []
            actor_tracks = None
            track_to_asset = None
            if edit_assets_config["insert"]["data"]:
                inserted_track_ids = edit_assets_config["insert"]["data"]["tracks_data"]["tracks_id"]

                # Read asset_ids to decouple track identity from asset selection
                asset_ids = edit_assets_config["insert"]["asset_ids"]

                # Validate correspondence between tracks and assets
                assert len(asset_ids) == len(inserted_track_ids), (
                    f"asset_ids length mismatch {len(asset_ids)} with inserted track_ids {len(inserted_track_ids)}"
                )

                # Validate all required assets exist in AssetBank or on filesystem
                path_ids = {a for a in asset_ids if a not in available_scene_track_ids}

                missing_paths = {p for p in path_ids if not os.path.isfile(p)}
                assert not missing_paths, (
                    f"Missing assets (not in AssetBank or on disk): {missing_paths}. "
                    f"Available AssetBank assets: {sorted(available_scene_track_ids)}"
                )

                # Check for track ID conflicts with existing artifact tracks
                conflicting_ids = set(inserted_track_ids) & set(artifact_track_ids)
                assert not conflicting_ids, (
                    f"Conflicting track IDs between inserted and existing tracks: {sorted(conflicting_ids)}"
                )

                track_to_asset = dict(zip(inserted_track_ids, asset_ids))

                # Convert insert data to gRPC format
                actor_tracks = ActorTracks.from_dict_cpu(edit_assets_config["insert"]["data"])
                insert = actor_tracks_to_grpc(actor_tracks, track_to_asset=track_to_asset)

            if len(replace_actions) != 0 or len(insert) != 0:
                log.info(f"Sending EditAssetsRequest: {len(replace_actions)} replacements, {len(insert)} insertions")

                edit_request = EditAssetsRequest(
                    scene_id=scene_id,
                    replace=replace_actions,
                    insert=insert,
                )

                edit_response: EditAssetsResponse = await client_service.edit_assets(edit_request)

                if not edit_response.success:
                    raise RuntimeError(f"{edit_response.message}")

                # Add inserted cuboid_tracks to the artifact's cuboid_tracks
                if len(insert) != 0 and actor_tracks is not None:
                    cuboid_tracks_list.extend(actor_tracks._cuboid_tracks_list)
                    interpolator_list.extend(actor_tracks._interpolator_list)

        ego_mask_id: EgoMaskId | None = None
        if not lidar:
            available_ego_masks: AvailableEgoMasksReturn = await client_service.get_available_ego_masks(EmptyRequest())
            available_rig_ids = sorted({m.ego_mask_id.rig_config_id for m in available_ego_masks.ego_mask_metadata})
            print(f"Available rigs: {available_rig_ids}, choosing {rig_name=}.")
            ego_mask_id = select_ego_mask_id(available_ego_masks, rig_name, camera_id)

        # Validate camera and get timestamps based on rendering mode
        camera = None
        pose_lidar_to_rig: np.ndarray | None = None
        if lidar:
            # For lidar-only rendering, use lidar timestamps
            available_lidar_ids = list(trajectory.lidars_frame_timestamps_us.keys())
            if not available_lidar_ids:
                raise ValueError("No lidar timestamps available in artifact")

            # Use specified lidar_id or first available
            if lidar_id is not None:
                unique_lidar_id = lidar_id if "@" in lidar_id else f"{lidar_id}@{scene_id}"
                if unique_lidar_id not in trajectory.lidars_frame_timestamps_us:
                    raise ValueError(
                        f"{lidar_id=} not found in available lidars. "
                        f"Available: {[lid.split('@')[0] for lid in available_lidar_ids]}"
                    )
            else:
                unique_lidar_id = available_lidar_ids[0]
                log.info(f"Using first available lidar: {unique_lidar_id.split('@')[0]}")

            timestamps_us = trajectory.lidars_frame_timestamps_us[unique_lidar_id].cpu().numpy()
            num_frames = len(timestamps_us)
            log.info(f"Test trajectory contains poses for {num_frames} lidar frames")

            # Mount extrinsics for the sensor pose. Unlike the camera path — which reads them
            # off the service via get_available_cameras — there is no get_available_lidars RPC,
            # so these come from the locally loaded artifact. That makes --artifact-path and the
            # server's --artifact-glob load-bearing: point them at the same artifact or the two
            # sides disagree on where the lidar sits.
            if unique_lidar_id not in rig_trajectories.lidar_calibrations:
                raise ValueError(f"Missing lidar calibration for {unique_lidar_id=} in the artifact")
            pose_lidar_to_rig = (
                rig_trajectories.lidar_calibrations[unique_lidar_id].T_sensor_rig.cpu().numpy().astype(np.float64)
            )
            assert pose_lidar_to_rig.shape == (4, 4)
            log.info(f"Lidar-to-rig translation: {pose_lidar_to_rig[:3, 3]}")
        else:
            assert camera_id in [
                available_camera.logical_id for available_camera in available_cameras.available_cameras
            ]
            for available_camera in available_cameras.available_cameras:
                if available_camera.logical_id == camera_id:
                    camera = available_camera
                    break
            if camera is None:
                raise ValueError(f"{camera_id=} not found in available cameras")

            unique_camera_id = camera_id + f"@{scene_id}"
            timestamps_us = trajectory.cameras_frame_timestamps_us[unique_camera_id].cpu().numpy()
            num_frames = len(timestamps_us)
            log.info(f"Test trajectory contains poses for {num_frames} video frames")

        assert timestamps_us.shape == (num_frames, 2)
        assert timestamps_us.dtype == np.int64
        frame_start_timestamps_us = timestamps_us[:, 0]
        frame_end_timestamps_us = timestamps_us[:, 1]

        # Make sure no frame-end timestamp is missing.
        if (num_missing_timestamps := np.sum(frame_end_timestamps_us < 0)) > 0:
            raise ValueError(
                f"Frame-end timestamps are required but {num_missing_timestamps} out of {num_frames} are missing"
            )

        if (num_missing_timestamps := np.sum(frame_start_timestamps_us < 0)) > 0:
            # Handle case when frame start timestamps stored inside the artifact are -1.
            # Happened when saving rig_trajectories with obsolete datasource.get_rig_trajectories(end_frame_timestamps_only=True) argument.
            log.warning(
                f"Rolling shutter effect disabled due to {num_missing_timestamps} "
                f"out of {num_frames} missing frame-start timestamps"
            )
            frame_start_timestamps_us = frame_end_timestamps_us
        elif disable_rolling_shutter:
            log.warning("Rolling shutter effect disabled from command-line. Only using frame-end timestamps.")
            frame_start_timestamps_us = frame_end_timestamps_us

        # spin objects to highlight controllability
        euler_angles = np.zeros((num_frames, 3))
        euler_angles[:, 2] = 40.0 * np.sin(np.linspace(0, 2 * np.pi, euler_angles.shape[0]))
        delta_matrices = np.eye(4)[None].repeat(euler_angles.shape[0], axis=0).astype(np.float32)
        delta_matrices[:, :3, :3] = R.from_euler("xyz", euler_angles, degrees=True).as_matrix()

        # Common: output frame timestamps
        output_frame_timestamps: list[dict[str, int | str]] = []
        frame_count = 0

        def _build_dynamic_poses(frame_start_us: int, frame_end_us: int, frame_idx: int) -> list[dict[str, Any]]:
            return _build_dynamic_actor_poses(
                cuboid_tracks_list,
                interpolator_list,
                remove,
                frame_start_us,
                frame_end_us,
                frame_idx,
                demo_actor_transform=demo_actor_transform,
                delta_matrices=delta_matrices,
            )

        lidar_requests: list[LidarRenderRequest] = []
        lidar_spec = LidarSpec(lidar_type=LidarDeviceType.PANDAR128) if lidar else None
        lidar_id_for_request = unique_lidar_id.split("@")[0] if lidar else ""
        per_ray = lidar_format == "per-ray"
        lidar_file_suffix = "npz" if per_ray else lidar_format
        lidar_output_mode = (
            LidarOutputMode.LIDAR_OUTPUT_PER_RAY if per_ray else LidarOutputMode.LIDAR_OUTPUT_POINT_CLOUD
        )

        # Build lidar render filter from CLI parameters
        lidar_render_filter = (
            _build_lidar_render_filter(
                raydrop_threshold=lidar_raydrop_threshold,
                opacity_threshold=lidar_opacity_threshold,
                enable_distance_filter=lidar_enable_distance_filter,
                distance_filter_threshold=lidar_distance_filter_threshold,
                neighbor_source=lidar_distance_filter_neighbor_source,
                elevation_resolution_deg=lidar_distance_filter_elevation_resolution_deg,
                azimuth_max_gap_deg=lidar_distance_filter_azimuth_max_gap_deg,
            )
            if lidar
            else None
        )

        if not lidar:
            assert camera is not None
            rig_offset_se3 = pose_offsets_to_se3(rig_translation_offset, rig_rotation_offset)
            pose_camera_to_rig = grpc_pose_to_torch_se3(camera.rig_to_camera).cpu().numpy()

            frame_data: list[dict] = []
            for frame_idx in range(0, num_frames, frame_step):
                frame_start_us = int(frame_start_timestamps_us[frame_idx])
                frame_end_us = int(frame_end_timestamps_us[frame_idx])
                frame_end_us_grpc = frame_end_us + 1 if frame_start_us == frame_end_us else frame_end_us

                dynamic_poses = (
                    _build_dynamic_poses(frame_start_us, frame_end_us, frame_idx) if enable_editing_actors else []
                )

                pose_rig_to_world = rig_pose_interpolator.interpolate_to_timestamps([frame_start_us, frame_end_us])
                frame_start_camera_to_world_se3, frame_end_camera_to_world_se3 = compose_sensor_to_world(
                    pose_rig_to_world, rig_offset_se3, pose_camera_to_rig
                )

                match frame_naming:
                    case "frame-end-timestamp":
                        # Zero-pad to 20 digits (full int64 microsecond range) so a lexical filename
                        # sort stays numeric, matching the val export in GaussiansSystem.
                        file_name = f"{frame_end_us:020d}.{format_ext}"
                    case "contiguous-output-index":
                        file_name = f"{frame_count:06d}.{format_ext}"
                    case _:
                        raise ValueError(f"Invalid frame naming scheme: {frame_naming}")

                output_frame_timestamps.append(
                    {
                        "file_name": file_name,
                        "render_frame_idx": frame_count,
                        "frame_start_timestamp_us": frame_start_us,
                        "frame_end_timestamp_us": frame_end_us,
                    }
                )
                frame_data.append(
                    {
                        "frame_start_us": frame_start_us,
                        "frame_end_us_grpc": frame_end_us_grpc,
                        "dynamic_poses": dynamic_poses,
                        "frame_start_camera_to_world": frame_start_camera_to_world_se3,
                        "frame_end_camera_to_world": frame_end_camera_to_world_se3,
                    }
                )
                frame_count += 1

            requests = [
                generate_request(
                    scene_id=scene_id,
                    camera_spec=camera.intrinsics,
                    frame_start_camera_to_world=fd["frame_start_camera_to_world"],
                    frame_end_camera_to_world=fd["frame_end_camera_to_world"],
                    frame_start_timestamp_us=fd["frame_start_us"],
                    frame_end_timestamp_us=fd["frame_end_us_grpc"],
                    height=height,
                    dynamic_poses=fd["dynamic_poses"],
                    format=format,
                    ego_mask_id=ego_mask_id,
                    render_products=render_products,
                    overlay_bounding_boxes_source=overlay_bounding_boxes,
                    distance_colormap_vmax_m=colorization_max_distance,
                    bbox_color=bbox_color,
                    apply_harmonizer=apply_harmonizer,
                )
                for fd in frame_data
            ]

            harmonizer_session_id: str | None = None
            if apply_harmonizer and CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR in render_products:
                harmonizer_session_id = await _maybe_create_temporal_harmonizer_session(
                    client_service, requests, repeat_count
                )
                sequential = _force_sequential_for_temporal_session(harmonizer_session_id, sequential)
            try:
                async for repeat_index, responses in _repeat_request_sequence(
                    client_service.render_camera, requests, sequential, repeat_count, render_limiter
                ):
                    if output_dir:
                        repeat_output_dir = _output_dir_for_repeat(output_dir, repeat_index, repeat_count)
                        save_tasks = []
                        for contiguous_frame_idx, response in enumerate(responses):
                            fname = str(output_frame_timestamps[contiguous_frame_idx]["file_name"])
                            for out in response.camera_render_outputs:
                                type_dir = _grpc_output_dir_for_camera_item(out)
                                cam_out = os.path.join(repeat_output_dir, type_dir, camera_id)
                                os.makedirs(cam_out, exist_ok=True)
                                save_tasks.append(save_request(os.path.join(cam_out, fname), out.image_bytes))
                        await asyncio.gather(*save_tasks)
                        for product in render_products:
                            cam_out = os.path.join(repeat_output_dir, _grpc_output_type_dir(product), camera_id)
                            os.makedirs(cam_out, exist_ok=True)
                            with open(os.path.join(cam_out, "timestamps.json"), "w") as tf:
                                json.dump(output_frame_timestamps, tf, indent=2)
                        if (
                            responses
                            and CameraRenderProduct.CAMERA_RENDER_PRODUCT_SEMANTICS_COLORIZED in render_products
                        ):
                            seg_entries = None
                            for out in responses[0].camera_render_outputs:
                                if (
                                    out.render_product == CameraRenderProduct.CAMERA_RENDER_PRODUCT_SEMANTICS_COLORIZED
                                    and out.semantics_class_mapping
                                ):
                                    seg_entries = out.semantics_class_mapping
                                    break
                            if seg_entries:
                                seg_map = [
                                    {
                                        "class_index": int(entry.class_index),
                                        "class_name": entry.class_name,
                                        "color_rgb": [int(entry.color_r), int(entry.color_g), int(entry.color_b)],
                                    }
                                    for entry in seg_entries
                                ]
                                # Per-camera mapping file so a single output_dir shared across camera
                                # invocations doesn't have later writers clobber earlier ones.
                                seg_root = os.path.join(
                                    repeat_output_dir,
                                    CameraOutputSubdir.SEMANTICS_COLORIZED,
                                    camera_id,
                                )
                                os.makedirs(seg_root, exist_ok=True)
                                seg_path = os.path.join(seg_root, "semantics_class_mapping.json")
                                with open(seg_path, "w") as mf:
                                    json.dump(seg_map, mf, indent=2)
                                log.info("Wrote %s (%d classes)", seg_path, len(seg_map))
                        _write_distance_colormap_metadata(responses, render_products, repeat_output_dir, camera_id)
            finally:
                if harmonizer_session_id is not None:
                    await client_service.end_harmonizer_session(HarmonizerSessionId(session_id=harmonizer_session_id))
                    log.info(f"Ended temporal harmonizer session {harmonizer_session_id}")
        else:
            # Applies to every sensor in the rig frame, lidar included; identity unless the
            # caller passed --rig-translation-offset / --rig-rotation-offset.
            rig_offset_se3 = pose_offsets_to_se3(rig_translation_offset, rig_rotation_offset)
            for frame_idx in range(0, num_frames, frame_step):
                frame_start_us = int(frame_start_timestamps_us[frame_idx])
                frame_end_us = int(frame_end_timestamps_us[frame_idx])
                frame_end_us_grpc = frame_end_us + 1 if frame_start_us == frame_end_us else frame_end_us

                dynamic_poses = (
                    _build_dynamic_poses(frame_start_us, frame_end_us, frame_idx) if enable_editing_actors else []
                )

                assert pose_lidar_to_rig is not None  # Set in the if branch above when lidar=True
                poses_rig_to_world = rig_pose_interpolator.interpolate_to_timestamps([frame_start_us, frame_end_us])
                poses = compose_sensor_to_world(poses_rig_to_world, rig_offset_se3, pose_lidar_to_rig)
                lidar_requests.append(
                    generate_lidar_request(
                        scene_id=scene_id,
                        lidar_spec=lidar_spec,
                        lidar_id=lidar_id_for_request,
                        poses=poses,
                        frame_start_timestamp_us=frame_start_us,
                        frame_end_timestamp_us=frame_end_us_grpc,
                        dynamic_poses=dynamic_poses,
                        render_filter=lidar_render_filter,
                        output_mode=lidar_output_mode,
                    )
                )

                match frame_naming:
                    case "frame-end-timestamp":
                        # Zero-pad to 20 digits to match the camera RGB naming above and the val
                        # export in GaussiansSystem, so a lexical filename sort stays numeric.
                        lidar_file_name = f"{frame_end_us:020d}.{lidar_file_suffix}"
                    case "contiguous-output-index":
                        lidar_file_name = f"{frame_count:06d}.{lidar_file_suffix}"
                    case _:
                        raise ValueError(f"Invalid frame naming scheme: {frame_naming}")
                output_frame_timestamps.append(
                    {
                        "file_name": lidar_file_name,
                        "render_frame_idx": frame_count,
                        "frame_start_timestamp_us": frame_start_us,
                        "frame_end_timestamp_us": frame_end_us,
                    }
                )
                frame_count += 1

            # Per-ray frames are indexed by firing-pattern ray order. Learn IDs from the
            # ordered render responses and fetch exact geometry only for cache misses.
            patterns_by_id: dict[str, LidarFiringPatternReturn] = {}
            pattern_catalog: dict[str, str] | None = None

            async for repeat_index, lidar_responses in _repeat_request_sequence(
                client_service.render_lidar, lidar_requests, sequential, repeat_count, render_limiter
            ):
                if output_dir:
                    if per_ray:
                        await _cache_response_firing_patterns(
                            client_service,
                            scene_id=scene_id,
                            lidar_id=lidar_id_for_request,
                            requests=lidar_requests,
                            responses=lidar_responses,
                            patterns_by_id=patterns_by_id,
                        )
                        # Repeats intentionally pin the first response's exact geometry;
                        # they benchmark replay, not repeated timestamp-based selection.
                        _pin_requests_to_response_patterns(lidar_requests, lidar_responses)
                        pattern_catalog = await asyncio.to_thread(
                            _save_firing_pattern_catalog, output_dir, patterns_by_id, pattern_catalog
                        )
                    repeat_output_dir = _output_dir_for_repeat(output_dir, repeat_index, repeat_count)
                    os.makedirs(repeat_output_dir, exist_ok=True)
                    await asyncio.gather(
                        *[
                            save_lidar_response(
                                os.path.join(
                                    repeat_output_dir,
                                    str(output_frame_timestamps[idx]["file_name"]),
                                ),
                                response,
                                format=lidar_format,
                                request=lidar_requests[idx],
                            )
                            for idx, response in enumerate(lidar_responses)
                        ]
                    )
                    with open(os.path.join(repeat_output_dir, "timestamps.json"), "w") as f:
                        json.dump(output_frame_timestamps, f, indent=2)

        # Restore model parameters if assets were edited
        if edit_assets:
            restore_request = RestoreModelParametersRequest(scene_id=scene_id)
            await client_service.restore_model_parameters(restore_request)
            print(f"Restored model parameters for scene: {scene_id}")

        if shutdown_server_on_completion:
            await client_service.shut_down(EmptyRequest())
