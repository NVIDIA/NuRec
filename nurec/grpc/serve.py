# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import functools
import gc
import io
import logging
import math
import os
import tempfile
import threading
import time
import traceback
import uuid

from collections import OrderedDict
from concurrent import futures
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, NoReturn, Optional, Tuple
from urllib.parse import unquote, urlparse

import click
import grpc
import numpy as np
import torch

from einops import rearrange
from grpc_health.v1 import health, health_pb2, health_pb2_grpc  # type: ignore
from omegaconf import OmegaConf
from PIL import Image

import ncore.data
import ncore.impl.common.transformations as ncore_transformations
import nurec.grpc.protos.common_pb2 as grpc_types
import nurec.grpc.protos.ipc_pb2 as ipc_types
import nurec.grpc.protos.sensorsim_pb2 as sensorsim_types
import nurec.utils.cli as cli
import nurec.utils.profiling as profile

from nurec.artifact import Artifact
from nurec.asset_edits.runtime import (
    AspectOverrideRuntime,
    AspectScheduleRegistry,
    AspectSwitch,
    TrackSchedule,
    load_manifest_from_usdz,
    load_schedule_from_path,
    validate_aspect_gain,
)
from nurec.config.model import RendererBackend
from nurec.config.version import get_version
from nurec.datasets.summary import DataSourceSummary
from nurec.datasets.tracks import CuboidTracks, TrackFlags
from nurec.grpc.cache_keys import (
    SCENE_ID_HEADER,
    SCENE_URL_HEADER,
    BackendCacheKey,
    SceneCacheKey,
    require_scene_id,
)
from nurec.grpc.downloader import Downloader, check_safe_scene_id
from nurec.grpc.ego_hood import EgocarRigBank
from nurec.grpc.grpc_server_config import (
    HARMONIZER_TEMPORAL_COSMOS_BASE_MODEL_URL,
    HARMONIZER_TEMPORAL_COSMOS_TOKENIZER_URL,
    HARMONIZER_TEMPORAL_MODEL_FILENAME,
    HARMONIZER_TEMPORAL_MODEL_URL,
    HARMONIZER_TEMPORAL_RESOLUTION,
    GrpcServerConfig,
    allowed_image_format_names,
    grpc_cli_options,
)
from nurec.grpc.nvjpeg_batch_encoder import NvjpegImageEncodingError

# Pose helpers live in nurec.grpc.pose (a focused, GPU-stack-free target for clients and the
# bench harness); re-exported here for compatibility with existing importers.
from nurec.grpc.pose import grpc_pose_to_torch_se3 as grpc_pose_to_torch_se3
from nurec.grpc.pose import grpc_pose_to_tquat as grpc_pose_to_tquat
from nurec.grpc.pose import se3_to_grpc_pose as se3_to_grpc_pose
from nurec.grpc.pose import tquat_to_grpc_pose as tquat_to_grpc_pose
from nurec.grpc.protos.sensorsim_pb2 import (
    AvailableCamerasRequest,
    AvailableCamerasReturn,
    AvailableDynamicObjectsRequest,
    AvailableDynamicObjectsReturn,
    AvailableEgoMasksReturn,
    AvailableTrajectoriesRequest,
    AvailableTrajectoriesReturn,
    BatchCameraRenderRequest,
    BatchCameraRenderRequestItem,
    BatchCameraRenderReturn,
    BatchCameraRenderReturnItem,
    BatchRGBRenderRequest,
    BatchRGBRenderRequestItem,
    BatchRGBRenderReturn,
    BatchRGBRenderReturnItem,
    BivariateWindshieldModelParameters,
    CameraRenderOutput,
    CameraRenderProduct,
    CameraRenderRequest,
    CameraRenderReturn,
    CameraSpec,
    EditAssetsRequest,
    EditAssetsResponse,
    ExternalAssetObjectsRequest,
    ExternalAssetObjectsReturn,
    FthetaCameraParam,
    ImageFormat,
    LidarRenderFilter,
    LidarRenderRequest,
    LidarRenderReturn,
    OpenCVFisheyeCameraParam,
    OpenCVPinholeCameraParam,
    OverlayBoundingBoxSource,
    RestoreModelParametersRequest,
    RGBRenderRequest,
    RGBRenderReturn,
    SemanticsClassEntry,
)
from nurec.grpc.protos.sensorsim_pb2_grpc import SensorsimServiceServicer, add_SensorsimServiceServicer_to_server
from nurec.grpc.scene_cache import SceneCache, SceneCacheLease
from nurec.grpc.trajectory import QVec, Trajectory
from nurec.harmonizer.model import HarmonizerModelFactory, HarmonizerSessionClosedError, HarmonizerState
from nurec.models.gaussians.utils import Asset
from nurec.render import (
    ActorsSnapshot,
    CameraFrame,
    CameraRenderOptions,
    LidarRayBundle,
    MissingRenderOutputError,
    PoseRange,
    RayBundle,
    RenderableModel,
    overlay_bounding_boxes,
    transform_intrinsics_to_resolution,
)
from nurec.render.actors import ActorTracks
from nurec.render.lidar_firing_pattern import LidarFiringPattern
from nurec.utils.cli import SettingsCollector
from nurec.utils.lidar_model import LidarModelBundle
from nurec.utils.metrics import MetricsCollector, create_metric_sample
from nurec.utils.misc import unpack_optional
from nurec.utils.ncore_types import require_concrete_external_distortion_parameters
from nurec.utils.profiling import ScopedTimer
from nurec.utils.types import (
    FrameConversion,
    HalfClosedInterval,
    RigTrajectories,
)
from nurec.utils.visualize import (
    SEMANTICS_COLOR_LUT_DEFAULT_SIZE,
    build_semantics_class_colors,
    scalar2img,
    sem2img,
)


log = logging.getLogger("nurec.grpc.serve")

# CUDA IPC transport is always built in (//nurec/libs/cuda_ipc_transport:pylib is an
# unconditional dep of :pylib), but kept behind a defensive try/except: if the
# import ever fails, IpcManager stays None and the use_cuda_ipc path is disabled
# rather than crashing the whole server (callers gate on `IpcManager is not None`).
try:
    import importlib

    IpcManager = importlib.import_module("nurec.libs.cuda_ipc_transport").IpcManager  # type: ignore
except (ImportError, AttributeError):
    IpcManager = None  # type: ignore
_CUDA_IPC_AVAILABLE = IpcManager is not None


@contextlib.contextmanager
def _nvtx_range(name: str) -> Generator[None, None, None]:
    """Emit a CUDA NVTX range for nsys when a CUDA device is available."""
    if torch.cuda.is_available():
        with torch.cuda.nvtx.range(name):
            yield
    else:
        yield


# Cap on how long a render waits for the deferred Harmonizer load before failing loudly, and
# how often that wait rechecks RPC liveness.
_HARMONIZER_LOAD_TIMEOUT_S = 1800
_HARMONIZER_POLL_S = 1.0


def actors_snapshot_from_render_request(
    request: RGBRenderRequest | CameraRenderRequest | LidarRenderRequest,
) -> ActorsSnapshot:
    track_ids = []
    track_poses = []

    for dynamic_object in request.dynamic_objects:
        if not dynamic_object.HasField("pose_pair"):
            raise ValueError(
                f"DynamicObject {dynamic_object.track_id!r} has no pose_pair. Editing-actors "
                "requests re-pose every listed actor, so a pose-less entry (e.g. one sent only "
                "to carry aspect_state) would silently re-pose the actor to the origin with a "
                "degenerate rotation. Send pose_pair on every listed actor."
            )
        track_ids.append(DataSourceSummary._clean_track_id_str(dynamic_object.track_id))

        track_poses.append(
            torch.stack(
                [
                    grpc_pose_to_tquat(dynamic_object.pose_pair.start_pose),
                    grpc_pose_to_tquat(dynamic_object.pose_pair.end_pose),
                ]
            )
        )

    if len(track_ids) == 0:
        return ActorsSnapshot.empty()

    return ActorsSnapshot(actor_ids=track_ids, actor_poses=torch.stack(track_poses, dim=0))


def actor_tracks_from_grpc(grpc_tracks: List[sensorsim_types.DynamicObjectTrack]) -> ActorTracks:
    """Convert gRPC DynamicObjectTrack list to ActorTracks."""
    if not grpc_tracks:
        return ActorTracks()

    cuboid_tracks_list = []
    interpolator_list = []

    for grpc_track in grpc_tracks:
        trajectory = Trajectory.from_grpc(grpc_track.trajectory)
        poses_se3 = trajectory.poses.as_se3()  # [N_poses, 4, 4] np.ndarray
        timestamps_int64 = trajectory.timestamps_us.astype(np.int64)
        object_dims = np.array(
            [grpc_track.object_size.size_x, grpc_track.object_size.size_y, grpc_track.object_size.size_z],
            dtype=np.float32,
        )

        cuboid_track = CuboidTracks.Factory.from_numpy(
            tracks_id=[grpc_track.id],
            tracks_poses=[poses_se3.astype(np.float32)],
            tracks_timestamps_us=[timestamps_int64],
            tracks_label_class=[grpc_track.semantic_class],
            tracks_flags=[TrackFlags.DYNAMIC | TrackFlags.CONTROLLABLE],
            cuboids_dims=[object_dims],
        )

        interpolator = ncore_transformations.PoseInterpolator(
            poses=cuboid_track.tracks_poses.matrix().cpu(), timestamps=cuboid_track.tracks_timestamps_us.cpu()
        )

        cuboid_tracks_list.append(cuboid_track)
        interpolator_list.append(interpolator)

    return ActorTracks(_cuboid_tracks_list=cuboid_tracks_list, _interpolator_list=interpolator_list)


def actor_tracks_to_grpc(
    actor_tracks: ActorTracks, track_to_asset: Optional[Dict[str, str]] = None
) -> List[sensorsim_types.DynamicObjectTrack]:
    """Convert ActorTracks to gRPC DynamicObjectTrack format.

    Args:
        actor_tracks: ActorTracks to convert
        track_to_asset: Optional mapping from track_id to asset_id.
                        The track_id is derived from the actor_tracks, the asset_id is derived from the assets in the AssetBank.
                        Use get_external_asset_objects() to get the list of available asset_ids.
    """
    grpc_objects = []

    for cuboid_track in actor_tracks._cuboid_tracks_list:
        (track_id,) = cuboid_track.tracks_id  # Validates exactly one track
        (track_label_class,) = cuboid_track.tracks_label_class  # Validates exactly one class
        poses_se3 = cuboid_track.tracks_poses
        timestamps_us = cuboid_track.tracks_timestamps_us

        trajectory = Trajectory(
            timestamps_us=timestamps_us.cpu().numpy().astype(np.uint64),
            poses=QVec.from_se3(poses_se3.matrix().cpu().numpy()),
        )

        if len(cuboid_track.cuboids_dims) == 0:
            raise ValueError(f"No cuboid dimensions found for track {track_id}")
        track_dim = cuboid_track.cuboids_dims[0].cpu().numpy().astype(np.float32)  # [3] array

        # Decouple track identity from asset selection
        asset_id = track_to_asset[track_id] if track_to_asset else "none"

        grpc_object = sensorsim_types.DynamicObjectTrack(
            id=track_id,
            semantic_class=track_label_class,
            trajectory=trajectory.to_grpc(),
            object_size=grpc_types.AABB(size_x=track_dim[0], size_y=track_dim[1], size_z=track_dim[2]),
            asset_id=asset_id,
        )
        grpc_objects.append(grpc_object)

    return grpc_objects


@dataclass
class TorchCamera:
    camera_model_parameters: ncore.data.ConcreteCameraModelParametersUnion
    logical_camera_id: str
    unique_sensor_idx: int
    time_range_us: HalfClosedInterval
    world_to_nurec: FrameConversion
    T_camera_rig: torch.Tensor

    def __repr__(self):
        return (
            f"TorchCamera(logical_camera_id={self.logical_camera_id}, "
            f"camera_model_parameters={self.camera_model_parameters}, "
            f"time_range_us={self.time_range_us})"
        )

    @ScopedTimer()
    def build_ray_bundle(self, request: RGBRenderRequest | CameraRenderRequest) -> RayBundle:
        start_timestamp_us = request.frame_start_us
        end_timestamp_us = request.frame_end_us
        resolution_hw = (request.resolution_h, request.resolution_w)
        start_pose = grpc_pose_to_tquat(request.sensor_pose.start_pose)
        end_pose = grpc_pose_to_tquat(request.sensor_pose.end_pose)

        request_interval = HalfClosedInterval(start_timestamp_us, end_timestamp_us)

        if request_interval not in self.time_range_us:
            raise TimestampOutOfRangeError(
                f"Requested time range {request_interval} is outside of scene time range {self.time_range_us}."
            )

        original_w, original_h = self.camera_model_parameters.resolution.tolist()
        aspect_ratio = original_w / original_h

        req_resolution_h, req_resolution_w = resolution_hw
        if (req_resolution_w / original_w) > (req_resolution_h / original_h):
            new_w = req_resolution_w
            new_h = int(new_w / aspect_ratio)
            image_domain_scale = new_w / original_w
        else:
            new_h = req_resolution_h
            new_w = int(new_h * aspect_ratio)
            image_domain_scale = new_h / original_h

        with ScopedTimer("camera_model_parameters.transform"):
            camera_params = self.camera_model_parameters.transform(
                image_domain_scale=image_domain_scale,
                new_resolution=(new_w, new_h),
            )

        # TODO: remove this half-closed-interval convention from gRPC API because it is rather unintuitive to add 1 us
        # to the exact timestamp of the end of frame.
        if end_timestamp_us <= start_timestamp_us:
            raise TimestampOutOfRangeError(
                f"Render time range [{start_timestamp_us=}, {end_timestamp_us=}) is empty. "
                "To render at an instant t, use [t, t+1) as the interval."
            )

        return RayBundle.build(
            camera_model_parameters=camera_params,
            camera_to_world=PoseRange(start_pose, end_pose, start_timestamp_us, end_timestamp_us),
            world_to_nurec=self.world_to_nurec,
            unique_sensor_idx=self.unique_sensor_idx,
        )


def _rig_trajectory_to_grpc_trajectory(trajectory: RigTrajectories.RigTrajectory) -> grpc_types.Trajectory:
    poses = QVec.from_se3(trajectory.T_rig_worlds.numpy())
    timestamps_us = trajectory.T_rig_world_timestamps_us.numpy().astype(np.uint64)

    return Trajectory(
        timestamps_us=timestamps_us,
        poses=poses,
    ).to_grpc()


DEFAULT_BACKEND_CACHE_WAIT_S = 120.0
BACKEND_CACHE_WAIT_POLL_S = 1.0


class CacheFullError(Exception):
    """Raised when the cache is full and cannot accept new backends."""

    pass


class CacheWaitAbortedReason(Enum):
    CANCELLED = auto()
    DEADLINE_EXCEEDED = auto()
    SERVER_TIMEOUT = auto()


class CacheWaitAbortedError(CacheFullError):
    """Raised when waiting for backend cache capacity is aborted or times out."""

    def __init__(self, message: str, reason: CacheWaitAbortedReason):
        super().__init__(message)
        self.reason = reason


class NoSpareBackendsError(Exception):
    """Raised when OOM occurs and no spare backends are available to evict."""

    pass


class BackendInUseError(Exception):
    """Raised by ``BackendCache.remove_by_key`` when the resident backend for a key is currently checked out
    (in the ``in_use`` pool).

    A by-key eviction (the multiproc worker's coordinated scene detach) must NEVER unlink a backend a render
    still holds: its CUDA-IPC-aliased tensors are live, so removing it and dropping the owner refcount would
    let ``release_if_idle`` ``cudaFree`` storage the in-flight render aliases (use-after-free), and the
    render's later ``checkin`` would trip its ``backend in self.in_use`` assert. Refusing the removal is the
    last-line guard: it flows into the worker scene table's existing detach-uncertainty path (the owner
    retains the refcount + attachment and retries the eviction once the render releases the backend)."""

    pass


class SceneNotFoundError(KeyError):
    """Raised when a requested scene ID is not available on the server."""

    pass


@dataclass(frozen=True)
class BackendReservation:
    """Owns one logical cache slot while a backend is constructed outside the cache lock.

    The opaque token ties the reservation to its backend cache key and, when the cache is
    full, to the spare backend that the new backend is intended to replace after
    construction succeeds.
    """

    key: str
    token: object
    scene_id: str


@dataclass(frozen=True)
class ArtifactRegistration:
    """Associates an artifact with its stable source and source-lifetime policy.

    Downloaded sources are owned by ``SceneCache`` and disappear from the
    registry when that cache evicts them. Static and explicitly registered local
    sources are not owned by the download cache and remain registered.
    """

    artifact: Artifact
    downloaded: bool
    scene_key: SceneCacheKey
    # A scene whose backend is already resident and owned EXTERNALLY (the multi-process workers, whose
    # backends alias the owner's IPC weight storage). `get_backend` resolves and serves the resident
    # backend, but a cache miss must NOT rebuild it: a rebuild would load a private, non-aliased copy
    # that silently diverges from the owner's shared weights, so `create_backend` fails loud instead.
    resident_only: bool = False

    @property
    def backend_cache_key(self) -> BackendCacheKey:
        """Return the loaded-backend key bound to the artifact's exact bytes."""
        return BackendCacheKey(self.scene_key, self.artifact.sha256_footprint)


@dataclass(frozen=True)
class BackendCacheWaitPolicy:
    """Controls whether and how long a backend acquisition may wait for capacity."""

    should_continue: Callable[[], bool] | None = None
    abort_reason: Callable[[], CacheWaitAbortedReason] | None = None
    wait_deadline: float | None = None


class TimestampOutOfRangeError(ValueError):
    """Raised when a render request's time range is outside the scene range."""

    pass


class ModelIncompatibilityError(RuntimeError):
    """Raised when a model/artifact is incompatible with the server."""

    pass


class HarmonizerNotEnabledError(RuntimeError):
    """Raised when a request asks for Harmonizer but the server has not loaded one."""

    pass


def _render_exception_message(exc: Exception) -> str:
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc)


def _classify_render_exception(exc: Exception) -> tuple[grpc.StatusCode, str]:
    if isinstance(exc, HarmonizerSessionClosedError):
        return grpc.StatusCode.ABORTED, _render_exception_message(exc)
    if isinstance(exc, TimestampOutOfRangeError):
        return grpc.StatusCode.OUT_OF_RANGE, _render_exception_message(exc)
    if isinstance(exc, (ModelIncompatibilityError, MissingRenderOutputError, HarmonizerNotEnabledError)):
        # MissingRenderOutputError subclasses ValueError, so this must precede the generic
        # ValueError -> INVALID_ARGUMENT mapping below. A model lacking a requested head is a
        # precondition failure, not a malformed request.
        return grpc.StatusCode.FAILED_PRECONDITION, _render_exception_message(exc)
    if isinstance(exc, CacheWaitAbortedError):
        if exc.reason == CacheWaitAbortedReason.CANCELLED:
            return grpc.StatusCode.CANCELLED, _render_exception_message(exc)
        return grpc.StatusCode.DEADLINE_EXCEEDED, _render_exception_message(exc)
    if isinstance(exc, (CacheFullError, NoSpareBackendsError, torch.OutOfMemoryError, torch.cuda.OutOfMemoryError)):
        message = _render_exception_message(exc)
        if isinstance(exc, (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError)):
            extra = "This can occur when running an unconverged checkpoint, or when volumetric effects are present (e.g fog)."
            message = f"{message}\n{extra}" if message else extra
        return grpc.StatusCode.RESOURCE_EXHAUSTED, message
    if isinstance(exc, KeyError):
        return grpc.StatusCode.NOT_FOUND, _render_exception_message(exc)
    if isinstance(exc, (ValueError, TypeError)):
        return grpc.StatusCode.INVALID_ARGUMENT, _render_exception_message(exc)
    if isinstance(exc, RuntimeError):
        ipc_status = _classify_ipc_runtime_error(exc)
        if ipc_status is not None:
            return ipc_status, _render_exception_message(exc)
    return grpc.StatusCode.UNKNOWN, _render_exception_message(exc)


def _classify_ipc_runtime_error(exc: RuntimeError) -> Optional[grpc.StatusCode]:
    """Classify the plain RuntimeErrors raised by the CUDA IPC path: both across the
    pybind11 boundary by nurec.libs.cuda_ipc_transport's native IpcManager (pybind11
    surfaces a C++ std::runtime_error as an untyped Python RuntimeError, so there is
    no exception subclass to isinstance-check against) and the Python-side checks in
    _ipc_write_rgb — these are distinguished by the fixed message prefixes each site
    raises with. Returns None for messages this doesn't recognize (e.g. a genuine
    cudaMalloc/CUDA failure), which fall through to UNKNOWN so real server-side bugs
    stay loud rather than being misclassified as client errors.
    """
    message = str(exc)
    if message.startswith(
        ("requested_width", "requested_buffer_count", "released_buffer_id out of range", "buffer_id out of range")
    ):
        return grpc.StatusCode.INVALID_ARGUMENT
    if message.startswith("session not found"):
        return grpc.StatusCode.NOT_FOUND
    if message.startswith(("no free CUDA IPC buffer available", "render size (", "render output is on CUDA device")):
        return grpc.StatusCode.FAILED_PRECONDITION
    if message.startswith("insufficient free CUDA IPC buffers"):
        return grpc.StatusCode.RESOURCE_EXHAUSTED
    return None


def allowed_image_format_values(value: Optional[str]) -> Optional[frozenset[int]]:
    """Translate a configured name allow-list into protobuf enum values."""
    names = allowed_image_format_names(value)
    return None if names is None else frozenset(ImageFormat.Value(name) for name in names)


def require_allowed_image_format(
    request: RGBRenderRequest | CameraRenderRequest,
    allowed_formats: Optional[frozenset[int]],
    context: grpc.ServicerContext,
    *,
    request_description: str = "Request",
) -> None:
    """Abort before rendering when a camera request violates the server allow-list."""
    if allowed_formats is None or request.image_format in allowed_formats:
        return

    try:
        requested_name = ImageFormat.Name(request.image_format)
    except ValueError:
        requested_name = f"UNKNOWN({request.image_format})"
    allowed_names = ", ".join(ImageFormat.Name(value) for value in sorted(allowed_formats))
    message = (
        f"{request_description} requested image format {requested_name}, which is not allowed by this server. "
        f"Allowed image formats: {allowed_names}."
    )
    context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)
    raise  # unreachable, but needed for mypy


def require_cuda_ipc_output_format(
    request: RGBRenderRequest | CameraRenderRequest, context: grpc.ServicerContext
) -> None:
    """Abort before rendering when a use_cuda_ipc=true request declares an image_format other
    than RGB_UINT8_PLANAR. The CUDA IPC path always fills the session's buffers with
    rgb_uint8_planar data (see _ipc_write_rgb / _ipc_write_camera_color) regardless of what
    image_format says -- e.g. a declared JPEG would otherwise pass a JPEG-only
    --allowed-image-formats admission check (require_allowed_image_format validates the
    *declared* format, not what IPC actually produces) while silently returning raw planar
    bytes instead."""
    if not request.use_cuda_ipc or request.image_format == ImageFormat.RGB_UINT8_PLANAR:
        return
    try:
        requested_name = ImageFormat.Name(request.image_format)
    except ValueError:
        requested_name = f"UNKNOWN({request.image_format})"
    context.abort(
        grpc.StatusCode.INVALID_ARGUMENT,
        f"use_cuda_ipc=true requires image_format=RGB_UINT8_PLANAR; got {requested_name}. The CUDA IPC "
        "path always delivers rgb_uint8_planar data into the session's buffers regardless of the "
        "requested format.",
    )
    raise  # unreachable, but needed for mypy


def require_cuda_ipc_camera_request(request: CameraRenderRequest, context: grpc.ServicerContext) -> None:
    """Abort before rendering when a use_cuda_ipc=true CameraRenderRequest asks for something the
    CUDA IPC path can't deliver. IPC always produces exactly one RGB_UINT8_PLANAR COLOR output,
    written by the fused pack kernel directly from the float32 raster -- it can't also produce
    DISTANCE_COLORIZED/SEMANTICS_COLORIZED outputs (different dtypes/layouts, no pack kernel for
    them) or run the numpy-based bounding-box overlay (which needs a host-visible uint8 image the
    IPC path never materializes)."""
    if not request.use_cuda_ipc:
        return
    products = list(request.render_products) or [CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR]
    if products != [CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR]:
        product_names = ", ".join(CameraRenderProduct.Name(p) for p in products)
        context.abort(
            grpc.StatusCode.INVALID_ARGUMENT,
            "use_cuda_ipc=true only supports a single CAMERA_RENDER_PRODUCT_COLOR render_products "
            f"entry (or an empty render_products list, which defaults to COLOR); got [{product_names}].",
        )
        raise  # unreachable, but needed for mypy
    if request.overlay_bounding_boxes != OverlayBoundingBoxSource.OVERLAY_BOUNDING_BOX_SOURCE_UNSPECIFIED:
        context.abort(
            grpc.StatusCode.INVALID_ARGUMENT,
            "use_cuda_ipc=true does not support overlay_bounding_boxes: the CUDA IPC path writes "
            "directly into the client's mapped buffer, with no host-visible image to draw onto.",
        )
        raise  # unreachable, but needed for mypy
    require_cuda_ipc_output_format(request, context)


def _abort_with_classified_exception(context: grpc.ServicerContext, exc: Exception, rpc_name: str) -> None:
    status, message = _classify_render_exception(exc)
    if status == grpc.StatusCode.UNKNOWN:
        log.error("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        raise exc
    log.warning(f"{rpc_name} failed with {status.name}: {message}")
    context.abort(status, message)


def _grpc_backend_cache_wait_policy(context: grpc.ServicerContext | None) -> BackendCacheWaitPolicy | None:
    if context is None:
        return None

    def should_continue() -> bool:
        if not context.is_active():
            return False
        remaining = context.time_remaining()
        return remaining is None or remaining > 0

    def abort_reason() -> CacheWaitAbortedReason:
        remaining = context.time_remaining()
        if remaining is not None and remaining <= 0:
            return CacheWaitAbortedReason.DEADLINE_EXCEEDED
        if not context.is_active():
            return CacheWaitAbortedReason.CANCELLED
        return CacheWaitAbortedReason.DEADLINE_EXCEEDED

    server_cap = time.monotonic() + DEFAULT_BACKEND_CACHE_WAIT_S
    remaining = context.time_remaining()
    wait_deadline = server_cap if remaining is None else min(server_cap, time.monotonic() + remaining)

    return BackendCacheWaitPolicy(
        should_continue=should_continue,
        abort_reason=abort_reason,
        wait_deadline=wait_deadline,
    )


class BackendCache:
    """Thread-safe pool of loaded rendering backends keyed by artifact source and content SHA.

    A backend is either in use by a request, spare and reusable, or represented by
    a pending reservation while construction runs. ``maxsize`` bounds these logical
    cache slots globally; ``max_per_key`` optionally bounds them for each scene.
    Callers may pass a scene ID separately when multiple backend lookup keys
    should share that existing limit.

    A reservation made at global capacity claims the oldest eligible spare. The
    claimed spare stays available for checkout while its replacement is built, so
    a request for its key still reuses the warm backend. A successful commit
    evicts the claimed spare if it is still idle, or defers eviction until its
    checkout returns; a failed or cancelled construction releases the claim,
    leaving the original spare cached. Explicit GPU-OOM recovery may evict a
    claimed spare early to reclaim memory.

    ``put_with_retries`` is the production acquisition path. It reuses a spare when
    possible, waits on configured per-scene saturation, and evicts spare backends
    one at a time when construction encounters GPU OOM. Checked-out backends are
    never evicted and must be returned with ``checkin``.
    """

    def __init__(self, maxsize: int = 10, max_per_key: int | None = None):
        """
        Initialize the BackendCache with count-based LRU eviction and OOM retry capability.

        Args:
            maxsize: Maximum number of logical backend slots across all keys. When this limit
                is reached, a new reservation claims the oldest eligible spare for replacement.
            max_per_key: Optional maximum number of backend slots for one scene,
                counting in-use, spare, and pending backends. Saturated requests
                wait for capacity or replace a spare backend for that scene.
        """
        if maxsize <= 0:
            raise ValueError(f"BackendCache maxsize must be positive, got {maxsize}")
        if max_per_key is not None:
            if max_per_key <= 0:
                raise ValueError(f"BackendCache max_per_key must be positive when set, got {max_per_key}")
            if max_per_key > maxsize:
                raise ValueError(f"BackendCache max_per_key must be <= maxsize, got {max_per_key} > {maxsize}")
        self.lock = threading.RLock()
        self.capacity_changed = threading.Condition(self.lock)

        # List of spare backends (LRU order - oldest first)
        self.spares: List[Backend] = list()

        # Set of backends that are currently in use
        self.in_use: set[Backend] = set()

        self.maxsize = maxsize
        self.max_per_key = max_per_key
        self.pending_by_key: dict[str, int] = {}
        self.pending_reservations: dict[object, BackendReservation] = {}
        # Reservation token -> spare provisionally claimed for replacement.
        self._claimed_spares_by_reservation: dict[object, Backend] = {}
        # Superseded backends checked out during construction: evict once returned.
        self._evict_on_checkin: set[Backend] = set()
        log.info(
            f"BackendCache initialized with LRU eviction: maxsize={maxsize}, "
            f"max_per_key={max_per_key}, OOM retry enabled"
        )

    @property
    def pending_global(self) -> int:
        return len(self.pending_reservations)

    def _total_count_locked(self) -> int:
        # A claimed spare (pending replacement) or a superseded backend awaiting checkin
        # shares its replacement's logical slot, so both are deducted.
        return (
            len(self.in_use)
            + len(self.spares)
            + self.pending_global
            - len(self._claimed_spares_by_reservation)
            - len(self._evict_on_checkin)
        )

    def _key_count_locked(self, scene_id: str) -> int:
        in_use_count = sum(1 for backend in self.in_use if backend.scene_id == scene_id)
        spare_count = sum(1 for backend in self.spares if backend.scene_id == scene_id)
        return in_use_count + spare_count + self.pending_by_key.get(scene_id, 0)

    def _claimed_spare_ids(self) -> set[int]:
        """Identities of spares provisionally claimed by a pending reservation."""
        return {id(backend) for backend in self._claimed_spares_by_reservation.values()}

    def _oldest_spare_locked(
        self,
        excluded_keys: set[str] | None = None,
        *,
        include_claimed_spares: bool = False,
        required_scene_id: str | None = None,
    ) -> Backend | None:
        """Return the oldest eligible spare, normally hiding provisionally claimed spares."""

        excluded = set(excluded_keys or ())
        claimed_ids = set() if include_claimed_spares else self._claimed_spare_ids()
        for backend in self.spares:
            if backend.cache_key in excluded:
                continue
            if required_scene_id is not None and backend.scene_id != required_scene_id:
                continue
            if id(backend) in claimed_ids:
                continue
            return backend
        return None

    def _evict_spare_locked(self, excluded_keys: set[str] | None = None) -> Backend | None:
        backend = self._oldest_spare_locked(excluded_keys, include_claimed_spares=True)
        if backend is None:
            return None

        self.spares.remove(backend)
        reservation_tokens = [
            token for token, claimed_spare in self._claimed_spares_by_reservation.items() if claimed_spare is backend
        ]
        for token in reservation_tokens:
            del self._claimed_spares_by_reservation[token]
        return backend

    @staticmethod
    def _cleanup_evicted_backends(evicted_backends: list[Backend]) -> None:
        if not evicted_backends:
            return
        evicted_backends.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _raise_cache_full(self, key: str, reason: str) -> NoReturn:
        raise CacheFullError(f"Cannot add backend '{key}': {reason}")

    def _increment_pending_locked(self, reservation: BackendReservation) -> None:
        scene_id = reservation.scene_id
        self.pending_by_key[scene_id] = self.pending_by_key.get(scene_id, 0) + 1
        self.pending_reservations[reservation.token] = reservation

    def _consume_reservation_locked(self, reservation: BackendReservation) -> None:
        active_reservation = self.pending_reservations.get(reservation.token)
        if active_reservation is None:
            raise ValueError(f"Unknown or already consumed backend reservation for '{reservation.key}'")
        active_key = active_reservation.key
        active_scene_id = active_reservation.scene_id
        if active_key != reservation.key:
            raise ValueError(
                f"Backend reservation key mismatch: token belongs to '{active_key}', not '{reservation.key}'"
            )
        if active_scene_id != reservation.scene_id:
            raise ValueError(
                "Backend reservation scene-ID mismatch: "
                f"token belongs to '{active_scene_id}', not '{reservation.scene_id}'"
            )

        del self.pending_reservations[reservation.token]
        self.pending_by_key[active_scene_id] -= 1
        if self.pending_by_key[active_scene_id] == 0:
            del self.pending_by_key[active_scene_id]

    def _checkout_locked(self, key: str) -> Backend | None:
        for backend in self.spares:
            if backend.cache_key == key:
                self.spares.remove(backend)
                self.in_use.add(backend)
                return backend
        return None

    @staticmethod
    def _raise_wait_aborted(key: str, reason: CacheWaitAbortedReason) -> NoReturn:
        raise CacheWaitAbortedError(f"Stopped waiting for backend '{key}'", reason)

    def _abort_reason_from_policy(self, wait_policy: BackendCacheWaitPolicy | None) -> CacheWaitAbortedReason:
        if wait_policy is not None and wait_policy.abort_reason is not None:
            return wait_policy.abort_reason()
        return CacheWaitAbortedReason.CANCELLED

    def _ensure_wait_policy_continues(self, key: str, wait_policy: BackendCacheWaitPolicy | None) -> None:
        if wait_policy is None or wait_policy.should_continue is None or wait_policy.should_continue():
            return
        self._raise_wait_aborted(key, self._abort_reason_from_policy(wait_policy))

    def _wait_for_capacity(self, key: str, *, wait_policy: BackendCacheWaitPolicy | None = None) -> None:
        self._ensure_wait_policy_continues(key, wait_policy)
        wait_deadline = wait_policy.wait_deadline if wait_policy is not None else None

        timeout = BACKEND_CACHE_WAIT_POLL_S
        if wait_deadline is not None:
            timeout = min(timeout, max(0.0, wait_deadline - time.monotonic()))
            if timeout == 0.0:
                self._raise_wait_aborted(key, CacheWaitAbortedReason.SERVER_TIMEOUT)

        self.capacity_changed.wait(timeout=timeout)

        self._ensure_wait_policy_continues(key, wait_policy)
        if wait_deadline is not None and time.monotonic() >= wait_deadline:
            self._raise_wait_aborted(key, CacheWaitAbortedReason.SERVER_TIMEOUT)

    def _reserve_locked(self, key: str, scene_id: str | None = None) -> BackendReservation:
        """Reserve a logical slot without discarding a working spare before commit.

        The cache lock must already be held. At global capacity, the reservation
        claims an eligible spare from another key. ``put_reserved`` commits that
        replacement, while ``release_reservation`` restores any claim that remains.
        """

        scene_id = key if scene_id is None else scene_id
        key_count = self._key_count_locked(scene_id)
        claimed_spare = None
        if self.max_per_key is not None and key_count >= self.max_per_key:
            # A different content generation for the same scene can replace an
            # idle backend without exceeding the scene limit. An exact-key
            # spare is handled by checkout before reservation.
            claimed_spare = self._oldest_spare_locked(
                excluded_keys={key},
                required_scene_id=scene_id,
            )
            if claimed_spare is None:
                self._raise_cache_full(
                    key,
                    f"per-key limit {self.max_per_key} is full ({key_count} backends already reserved or cached)",
                )

        if claimed_spare is None and self._total_count_locked() >= self.maxsize:
            claimed_spare = self._oldest_spare_locked(excluded_keys={key})
            if claimed_spare is None:
                self._raise_cache_full(
                    key,
                    f"all {self.maxsize} cache slots are in-use or otherwise unavailable. "
                    "No eligible spare backends available to evict. "
                    "Consider increasing --cache-size or waiting for requests to complete.",
                )

        reservation = BackendReservation(key=key, token=object(), scene_id=scene_id)
        self._increment_pending_locked(reservation)
        if claimed_spare is not None:
            self._claimed_spares_by_reservation[reservation.token] = claimed_spare
            log.info(f"Provisionally replacing spare backend '{claimed_spare.cache_key}' with '{key}'")
        return reservation

    def reserve(self, key: str, scene_id: str | None = None) -> BackendReservation:
        """
        Reserve capacity for a backend before expensive construction starts.

        The returned reservation must be consumed by either put_reserved() or
        release_reservation().
        """

        with self.capacity_changed:
            return self._reserve_locked(key, scene_id)

    def checkout_or_reserve(
        self,
        key: str,
        *,
        scene_id: str | None = None,
        wait_policy: BackendCacheWaitPolicy | None = None,
        cache_miss_validator: Callable[[], None] | None = None,
    ) -> Backend | BackendReservation:
        """
        Return an existing backend for key, reserve capacity for a new one, or wait.

        A checked-in backend for the same scene still counts toward the cap,
        so waiters must re-run checkout before attempting a new reservation.
        ``cache_miss_validator`` runs only when no reusable backend exists; it
        prevents a missing scene from waiting for unrelated cache capacity while
        still allowing an already-loaded backend to outlive its source file.
        The validator is invoked without the cache lock held, so a blocking
        validator cannot stall unrelated cache operations. Competing waiters
        have no admission-order guarantee.
        """

        while True:
            with self.capacity_changed:
                backend = self._checkout_locked(key)
                if backend is not None:
                    return backend

            # The validator may block on unrelated locks (e.g. a scene download
            # publishing), so it must run with the cache lock released.
            if cache_miss_validator is not None:
                cache_miss_validator()

            with self.capacity_changed:
                # State may have changed while the lock was released: re-check
                # checkout before attempting a new reservation.
                backend = self._checkout_locked(key)
                if backend is not None:
                    return backend

                try:
                    return self._reserve_locked(key, scene_id)
                except CacheFullError as exc:
                    if self.max_per_key is None:
                        raise
                    log.debug(f"Waiting for backend '{key}': {exc}")
                    self._wait_for_capacity(key, wait_policy=wait_policy)

    def put_reserved(self, reservation: BackendReservation, backend: Backend) -> None:
        """Commit a constructed backend and evict any spare still claimed by its reservation."""

        assert backend.cache_key is None
        if backend.scene_id != reservation.scene_id:
            raise ValueError(
                f"Backend scene-ID mismatch: reservation is for {reservation.scene_id!r}, "
                f"but backend is for {backend.scene_id!r}"
            )
        evicted_spares: list[Backend] = []
        with self.capacity_changed:
            self._consume_reservation_locked(reservation)
            claimed_spare = self._claimed_spares_by_reservation.pop(reservation.token, None)
            if claimed_spare is not None:
                if claimed_spare in self.spares:
                    self.spares.remove(claimed_spare)
                    evicted_spares.append(claimed_spare)
                else:
                    # Reused during construction; evict once its checkout returns it.
                    self._evict_on_checkin.add(claimed_spare)
            backend.cache_key = reservation.key
            self.in_use.add(backend)
            self.capacity_changed.notify_all()

        # The cleanup helper must own the last evicted-spare reference before empty_cache().
        claimed_spare = None
        self._cleanup_evicted_backends(evicted_spares)

    def release_reservation(self, reservation: BackendReservation) -> None:
        """Cancel a reservation, making its claimed spare reusable if it remains cached."""

        with self.capacity_changed:
            self._consume_reservation_locked(reservation)
            self._claimed_spares_by_reservation.pop(reservation.token, None)
            self.capacity_changed.notify_all()

    def checkout(self, key: str) -> Backend | None:
        """
        Get an available backend for a key and mark as in-use.
        If no available backend is found, return None to indicate caller should create one.
        """

        with self.lock:
            return self._checkout_locked(key)

    def checkin(self, backend: Backend):
        """Return a backend to the pool, marking it as available."""

        assert backend.cache_key is not None

        evicted_backends: list[Backend] = []
        with self.capacity_changed:
            assert backend in self.in_use
            self.in_use.remove(backend)
            if backend in self._evict_on_checkin:
                self._evict_on_checkin.remove(backend)
                evicted_backends.append(backend)
            else:
                self.spares.append(backend)
            self.capacity_changed.notify_all()

        # Mirror evict_one_spare: do CUDA/GC cleanup outside the lock.
        del backend
        self._cleanup_evicted_backends(evicted_backends)

    def put(self, key: str, backend: Backend):
        """
        Add a new backend for the key, initially marked as in-use.
        Automatically evicts spare backends if cache size limit is reached.

        Raises:
            CacheFullError: If maxsize would be exceeded because all backends are in-use
        """

        # The backend must be a brand new instance
        assert backend.cache_key is None

        reservation = self.reserve(key, scene_id=backend.scene_id)
        try:
            self.put_reserved(reservation, backend)
        except Exception:
            self.release_reservation(reservation)
            raise

    @staticmethod
    def _log_acquisition(key: str, backend: Backend, created: bool) -> None:
        log.info(
            "Scene backend cache %s: scene_id=%s cache_key=%s sha256=%s",
            "miss" if created else "hit",
            backend.scene_id,
            key,
            backend.backend_cache_key.content_sha256,
        )

    def put_with_retries(
        self,
        key: str,
        backend_factory: Callable[[], Backend],
        max_retries: int = 5,
        enable_eviction: bool = True,
        *,
        scene_id: str | None = None,
        wait_policy: BackendCacheWaitPolicy | None = None,
        cache_miss_validator: Callable[[], None] | None = None,
    ) -> Backend:
        """
        Reuse or construct a backend, with policy-controlled waiting and OOM recovery.

        ``backend_factory`` runs only after this call owns a reservation. If GPU
        OOM occurs during construction, the reservation is released. A same-key
        backend that became spare is reused; otherwise, one eligible spare is
        evicted and construction is retried. When a scene is saturated and
        capacity is temporarily unavailable, the caller waits until capacity
        changes or ``wait_policy`` aborts the acquisition.
        """
        attempt = 0

        while attempt < max_retries:
            try:
                backend_or_reservation = self.checkout_or_reserve(
                    key,
                    scene_id=scene_id,
                    wait_policy=wait_policy,
                    cache_miss_validator=cache_miss_validator,
                )
                if not isinstance(backend_or_reservation, BackendReservation):
                    self._log_acquisition(key, backend_or_reservation, created=False)
                    return backend_or_reservation

                reservation = backend_or_reservation
                consumed = False
                constructed_backend: Backend | None = None
                try:
                    self._ensure_wait_policy_continues(key, wait_policy)
                    constructed_backend = backend_factory()
                    # Once construction succeeds the expensive work is paid for, so
                    # always commit: committing warms the cache for the aborting
                    # client's retry (or another client) instead of livelocking.
                    self.put_reserved(reservation, constructed_backend)
                    consumed = True
                    backend = constructed_backend
                    constructed_backend = None
                    self._log_acquisition(key, backend, created=True)
                    return backend
                finally:
                    if not consumed:
                        self.release_reservation(reservation)
                        if constructed_backend is not None:
                            backends_to_clean = [constructed_backend]
                            constructed_backend = None
                            self._cleanup_evicted_backends(backends_to_clean)

            except torch.cuda.OutOfMemoryError as e:
                self._ensure_wait_policy_continues(key, wait_policy)
                attempt += 1
                log.warning(
                    f"OOM error loading '{key}' (attempt {attempt}/{max_retries}). "
                    f"{'Trying to evict a spare backend...' if enable_eviction else 'Eviction disabled, will retry.'}"
                )

                checked_in_backend = self.checkout(key)
                if checked_in_backend is not None:
                    log.info(f"Reusing backend for '{key}' that became spare after OOM")
                    self._log_acquisition(key, checked_in_backend, created=False)
                    return checked_in_backend

                if enable_eviction:
                    # Try to evict a spare backend to free memory
                    if self.evict_one_spare(excluded_keys={key}):
                        log.info("Evicted a spare backend, retrying...")
                        continue
                    else:
                        # No more spares to evict
                        log.exception(
                            f"Cannot load '{key}': out of memory and no spare backends to evict. "
                            f"All {len(self.in_use)} cached backends are currently in use."
                        )
                        raise NoSpareBackendsError(
                            f"Out of GPU memory loading '{key}'. No spare backends available to evict."
                        ) from e
                else:
                    # Eviction disabled, just retry
                    if attempt >= max_retries:
                        log.exception(f"OOM loading '{key}' after {max_retries} attempts (eviction disabled)")
                        raise
                    continue

        raise torch.cuda.OutOfMemoryError(f"Failed to load '{key}' after {max_retries} attempts")

    def evict_one_spare(self, excluded_keys: set[str] | None = None) -> bool:
        """
        Evict the oldest spare backend to free up memory.

        Returns:
            True if a backend was evicted, False if no spares available.
        """
        # Extract backend info while holding lock
        evicted_backend = None
        with self.capacity_changed:
            if not self.spares:
                return False

            evicted_backend = self._evict_spare_locked(excluded_keys=excluded_keys)
            if evicted_backend is None:
                return False
            cache_key = evicted_backend.cache_key

            log.info(f"Evicting spare backend '{cache_key}' to free GPU memory")
            self.capacity_changed.notify_all()

        # Do expensive CUDA/GC cleanup outside the lock to avoid blocking other cache operations.
        # Drop the local reference first so the backend is actually freed before empty_cache().
        backends_to_clean = [evicted_backend]
        evicted_backend = None
        self._cleanup_evicted_backends(backends_to_clean)

        return True

    def loaded_scene_counts(self) -> tuple[int, list[tuple[str, int, int]]]:
        """Return capacity and per-scene loaded/reusable counts from the cache.

        Every spare is reusable, including one provisionally claimed for
        replacement: checkout may still hand it out while its replacement builds.
        """
        with self.lock:
            loaded_counts: dict[str, int] = {}
            reusable_counts: dict[str, int] = {}

            for backend in self.in_use:
                assert backend.cache_key is not None
                scene_id = backend.scene_id
                loaded_counts[scene_id] = loaded_counts.get(scene_id, 0) + 1

            for backend in self.spares:
                assert backend.cache_key is not None
                scene_id = backend.scene_id
                loaded_counts[scene_id] = loaded_counts.get(scene_id, 0) + 1
                reusable_counts[scene_id] = reusable_counts.get(scene_id, 0) + 1

            entries = [
                (scene_id, loaded_counts[scene_id], reusable_counts.get(scene_id, 0))
                for scene_id in sorted(loaded_counts)
            ]
            return self.maxsize, entries

    # -----------------------------------------------------------------------------------------------
    # Multiproc worker scene table API.
    #
    # The worker `_WorkerSceneTable` installs and removes pre-built backends by key and (with
    # coordinator eviction disabled) grows the cache in step with the attach stream. These two methods
    # give it a first-class, lock-and-notify-correct surface for those operations so it does not reach
    # into `lock`/`in_use`/`spares`/`maxsize`/`cache_key` directly (which skipped `capacity_changed`).
    # -----------------------------------------------------------------------------------------------

    def remove_by_key(self, key: str) -> Backend | None:
        """Evict the resident **spare** backend for ``key`` and wake capacity waiters.

        Finds the backend whose ``cache_key == key`` among the spare pool, unlinks it, clears its
        ``cache_key``, and notifies ``capacity_changed`` so waiters re-evaluate occupancy. Returns the
        removed backend, or ``None`` when no backend for ``key`` is resident. Unlike ``evict_one_spare``
        this runs no CUDA/GC cleanup: the caller owns the returned backend.

        **Refuses an in-use backend.** If the resident backend for ``key`` is currently checked out (a render
        is in flight against it), this raises `BackendInUseError` instead of unlinking it. Removing a live
        backend would drop the owner refcount and let ``release_if_idle`` free the CUDA-IPC tensors the
        in-flight render still aliases (use-after-free), and the render's later ``checkin`` would trip its
        ``backend in self.in_use`` assert. The multiproc worker scene table propagates this into its existing
        detach-uncertainty path, so the owner retains the refcount + attachment and the coordinator retries
        the eviction once the render releases the backend (`checkin` makes it a spare, removable cleanly).

        A pending reservation for ``key`` would race this removal (its commit expects the slot), so the
        caller must not hold one; the assert guards that invariant. The worker attaches and detaches a
        scene under the coordinator's per-scene serialization, so no reservation for ``key`` is ever
        outstanding when this is called. A spare claimed by another key's reservation is de-claimed on
        removal so ``_total_count_locked`` stays consistent.
        """

        with self.capacity_changed:
            assert self.pending_by_key.get(key, 0) == 0, (
                f"remove_by_key({key!r}) called while a reservation for it is pending"
            )
            if any(backend.cache_key == key for backend in self.in_use):
                raise BackendInUseError(
                    f"remove_by_key({key!r}) refused: a backend for {key!r} is in use (a render is in flight); "
                    "retry the eviction after the render checks it back in as a spare"
                )
            removed = next((backend for backend in self.spares if backend.cache_key == key), None)
            if removed is not None:
                self.spares.remove(removed)
                for token in [
                    token
                    for token, claimed_spare in self._claimed_spares_by_reservation.items()
                    if claimed_spare is removed
                ]:
                    del self._claimed_spares_by_reservation[token]
            if removed is not None:
                removed.cache_key = None
                self.capacity_changed.notify_all()
        return removed

    def grow_maxsize(self, n: int) -> None:
        """Grow the global slot budget by ``n`` (>= 1) and wake capacity waiters.

        Unlike a bare ``maxsize`` assignment this takes the cache lock and notifies ``capacity_changed``
        so a request blocked at capacity re-checks now that another slot exists. Grow-only: the worker
        scene table sizes the cache up in step with the attach stream and never shrinks it.
        """

        if n < 1:
            raise ValueError(f"grow_maxsize requires n >= 1, got {n}")
        with self.capacity_changed:
            self.maxsize += n
            self.capacity_changed.notify_all()


class CameraBank:
    def __init__(self, artifact: Artifact, world_to_nurec: FrameConversion):
        super().__init__()

        rig_trajectories = RigTrajectories.from_dict(artifact.rig_trajectories)

        # TODO can this be the following? self.world_to_nurec = rig_trajectories.world_to_nurec
        self.world_to_nurec = world_to_nurec

        self.cameras: dict[str, TorchCamera] = {}
        self._logged_missing_reconstruction_cameras: set[str] = set()
        self._logged_missing_reconstruction_cameras_lock = threading.Lock()

        unique_camera_id_to_camera_id = {
            uci: rig_trajectories.camera_calibrations[uci].logical_sensor_name
            for uci in rig_trajectories.camera_calibrations.keys()
        }

        # Assumes there is only a single rig_trajectory
        assert len(rig_trajectories.rig_trajectories) == 1, (
            f"{self.__class__.__name__}: expected a single rig_trajectory, got {len(rig_trajectories.rig_trajectories)}"
        )
        rig_trajectory = rig_trajectories.rig_trajectories[0]
        self.trajectory = _rig_trajectory_to_grpc_trajectory(rig_trajectory)

        rig_timestamps_us = rig_trajectory.T_rig_world_timestamps_us
        time_range_us = HalfClosedInterval(
            int(rig_timestamps_us[0].item()),
            int(rig_timestamps_us[-1].item()) + 1,  # +1 to make the interval half-closed
        )

        for unique_camera_id in rig_trajectory.cameras_frame_timestamps_us.keys():
            camera_calibration = rig_trajectories.camera_calibrations[unique_camera_id]
            logical_camera_id = unique_camera_id_to_camera_id[unique_camera_id]

            camera = TorchCamera(
                time_range_us=time_range_us,
                camera_model_parameters=camera_calibration.camera_model_parameters,
                logical_camera_id=logical_camera_id,
                unique_sensor_idx=camera_calibration.unique_sensor_idx,
                world_to_nurec=self.world_to_nurec,
                T_camera_rig=camera_calibration.T_sensor_rig,
            )

            self.cameras[logical_camera_id] = camera

    @staticmethod
    def _get_camera_model_parameters(
        camera_intrinsics: CameraSpec,
    ) -> ncore.data.ConcreteCameraModelParametersUnion:
        """Get the camera model parameters from the camera intrinsics."""
        # Get the external distortion parameters from the camera intrinsics if provided
        external_distortion_parameters: Optional[ncore.data.ConcreteExternalDistortionParametersUnion] = None
        if camera_intrinsics.WhichOneof("external_distortion") == "bivariate_windshield_model_param":
            windshield_param = camera_intrinsics.bivariate_windshield_model_param

            # Secure workaround for mismatch between integer enum values assigned in the
            # enum ncore.data.ReferencePolynomial (FORWARD:1, BACKWARD:2) (using auto()) and the
            # enum in the .proto file declaration of ReferencePolynomial (FORWARD:0, BACKWARD:1)
            reference_poly: ncore.data.ReferencePolynomial
            match windshield_param.reference_poly:
                case BivariateWindshieldModelParameters.ReferencePolynomial.FORWARD:
                    reference_poly = ncore.data.ReferencePolynomial.FORWARD
                case BivariateWindshieldModelParameters.ReferencePolynomial.BACKWARD:
                    reference_poly = ncore.data.ReferencePolynomial.BACKWARD
                case _:
                    raise ValueError(f"Unsupported reference polynomial type: {windshield_param.reference_poly}")

            external_distortion_parameters = ncore.data.BivariateWindshieldModelParameters(
                reference_poly=reference_poly,
                horizontal_poly=np.array(windshield_param.horizontal_poly, dtype=np.float32),
                vertical_poly=np.array(windshield_param.vertical_poly, dtype=np.float32),
                horizontal_poly_inverse=np.array(windshield_param.horizontal_poly_inverse, dtype=np.float32),
                vertical_poly_inverse=np.array(windshield_param.vertical_poly_inverse, dtype=np.float32),
            )
        elif camera_intrinsics.HasField("external_distortion"):  # Allows for providing no external distortion
            raise ValueError(
                f"Unsupported external distortion type {camera_intrinsics.WhichOneof('external_distortion')}"
            )

        camera_model_parameters: Optional[ncore.data.ConcreteCameraModelParametersUnion] = None
        # For now, only FThetaCameraModelParameters are supported
        if camera_intrinsics.WhichOneof("camera_param") == "ftheta_param":
            ftheta_param = camera_intrinsics.ftheta_param
            camera_model_parameters = ncore.data.FThetaCameraModelParameters(
                resolution=np.array([camera_intrinsics.resolution_w, camera_intrinsics.resolution_h], dtype=np.uint64),
                # shutter type enums are synchronized manually between gRPC and NCORE
                shutter_type=camera_intrinsics.shutter_type,  # type: ignore
                principal_point=np.array(
                    [ftheta_param.principal_point_x, ftheta_param.principal_point_y], dtype=np.float32
                ),
                # PolynomialType enums are synchronized manually between gRPC and NCORE
                reference_poly=ftheta_param.reference_poly,  # type: ignore
                pixeldist_to_angle_poly=np.array(ftheta_param.pixeldist_to_angle_poly, dtype=np.float32),
                angle_to_pixeldist_poly=np.array(ftheta_param.angle_to_pixeldist_poly, dtype=np.float32),
                max_angle=ftheta_param.max_angle,
                external_distortion_parameters=external_distortion_parameters,
                linear_cde=np.array(
                    [
                        ftheta_param.linear_cde.linear_c,
                        ftheta_param.linear_cde.linear_d,
                        ftheta_param.linear_cde.linear_e,
                    ],
                    dtype=np.float32,
                ),
            )
        elif camera_intrinsics.WhichOneof("camera_param") == "opencv_fisheye_param":
            opencv_fisheye_param = camera_intrinsics.opencv_fisheye_param
            camera_model_parameters = ncore.data.OpenCVFisheyeCameraModelParameters(
                resolution=np.array([camera_intrinsics.resolution_w, camera_intrinsics.resolution_h], dtype=np.uint64),
                principal_point=np.array(
                    [opencv_fisheye_param.principal_point_x, opencv_fisheye_param.principal_point_y],
                    dtype=np.float32,
                ),
                focal_length=np.array(
                    [opencv_fisheye_param.focal_length_x, opencv_fisheye_param.focal_length_y], dtype=np.float32
                ),
                radial_coeffs=np.array(opencv_fisheye_param.radial_coeffs, dtype=np.float32),
                max_angle=opencv_fisheye_param.max_angle,
                shutter_type=camera_intrinsics.shutter_type,  # type: ignore
                external_distortion_parameters=external_distortion_parameters,
            )
        elif camera_intrinsics.WhichOneof("camera_param") == "opencv_pinhole_param":
            opencv_pinhole_param = camera_intrinsics.opencv_pinhole_param
            camera_model_parameters = ncore.data.OpenCVPinholeCameraModelParameters(
                resolution=np.array([camera_intrinsics.resolution_w, camera_intrinsics.resolution_h], dtype=np.uint64),
                principal_point=np.array(
                    [opencv_pinhole_param.principal_point_x, opencv_pinhole_param.principal_point_y],
                    dtype=np.float32,
                ),
                focal_length=np.array(
                    [opencv_pinhole_param.focal_length_x, opencv_pinhole_param.focal_length_y], dtype=np.float32
                ),
                radial_coeffs=np.array(opencv_pinhole_param.radial_coeffs, dtype=np.float32),
                tangential_coeffs=np.array(opencv_pinhole_param.tangential_coeffs, dtype=np.float32),
                thin_prism_coeffs=np.array(opencv_pinhole_param.thin_prism_coeffs, dtype=np.float32),
                shutter_type=camera_intrinsics.shutter_type,  # type: ignore
                external_distortion_parameters=external_distortion_parameters,
            )
        # TODO: Add support for other camera model parameters

        if camera_model_parameters is None:
            raise TypeError("Unsupported camera model parameters type.")

        return camera_model_parameters

    def _get_camera(self, camera_intrinsics: CameraSpec) -> TorchCamera:
        """Get an existing TorchCamera from CameraBank or construct a new TorchCamera if parameters are provided.

        When explicit camera_param is provided in the request, render requests are accepted
        even if the logical_id does not match a camera used in reconstruction. This supports
        rendering from arbitrary camera positions/rigs (e.g. different car lines) as long as
        intrinsics and sensor_pose are fully specified in the request.
        """
        key = camera_intrinsics.logical_id
        # If camera_intrinsics does not have camera_param, return the camera model from the camera bank
        if camera_intrinsics.WhichOneof("camera_param") is None:
            return self.cameras[key]
        # If the request has an assigned camera param, it indicates that user customized the camera parameters
        else:
            with ScopedTimer("get_camera_model_parameters"):
                camera_model_parameters = self._get_camera_model_parameters(camera_intrinsics)
            with ScopedTimer("TorchCamera"):
                if key in self.cameras:
                    ref_camera = self.cameras[key]
                else:
                    # Camera not in reconstruction — use scene-level defaults from any available camera.
                    # This is valid because time_range_us and world_to_nurec are scene-level properties
                    # (same for all cameras), and the render request provides sensor_pose explicitly.
                    ref_camera = next(iter(self.cameras.values()))
                    self._log_missing_reconstruction_camera_once(key, ref_camera)
                return TorchCamera(
                    time_range_us=ref_camera.time_range_us,
                    camera_model_parameters=camera_model_parameters,
                    logical_camera_id=key,
                    unique_sensor_idx=ref_camera.unique_sensor_idx,
                    world_to_nurec=ref_camera.world_to_nurec,
                    T_camera_rig=ref_camera.T_camera_rig,
                )

    def _log_missing_reconstruction_camera_once(self, key: str, ref_camera: TorchCamera) -> None:
        with self._logged_missing_reconstruction_cameras_lock:
            if key in self._logged_missing_reconstruction_cameras:
                return
            self._logged_missing_reconstruction_cameras.add(key)

        log.info(
            f"Camera '{key}' not in reconstruction cameras "
            f"({list(self.cameras.keys())}). "
            f"Using scene-level defaults from '{ref_camera.logical_camera_id}'."
        )

    def _populate_ftheta_camera_param(
        self, camera_model_parameters: ncore.data.FThetaCameraModelParameters
    ) -> FthetaCameraParam:
        fthetaCameraParam = FthetaCameraParam()
        fthetaCameraParam.principal_point_x = camera_model_parameters.principal_point[0]
        fthetaCameraParam.principal_point_y = camera_model_parameters.principal_point[1]
        fthetaCameraParam.reference_poly = camera_model_parameters.reference_poly.value  # type: ignore
        fthetaCameraParam.pixeldist_to_angle_poly.extend(camera_model_parameters.pixeldist_to_angle_poly.tolist())
        fthetaCameraParam.angle_to_pixeldist_poly.extend(camera_model_parameters.angle_to_pixeldist_poly.tolist())
        fthetaCameraParam.max_angle = camera_model_parameters.max_angle
        fthetaCameraParam.linear_cde.linear_c = camera_model_parameters.linear_cde[0]
        fthetaCameraParam.linear_cde.linear_d = camera_model_parameters.linear_cde[1]
        fthetaCameraParam.linear_cde.linear_e = camera_model_parameters.linear_cde[2]
        return fthetaCameraParam

    def _populate_opencv_pinhole_camera_param(
        self, camera_model_parameters: ncore.data.OpenCVPinholeCameraModelParameters
    ) -> OpenCVPinholeCameraParam:
        opencvPinholeCameraParam = OpenCVPinholeCameraParam()
        opencvPinholeCameraParam.principal_point_x = camera_model_parameters.principal_point[0]
        opencvPinholeCameraParam.principal_point_y = camera_model_parameters.principal_point[1]
        opencvPinholeCameraParam.focal_length_x = camera_model_parameters.focal_length[0]
        opencvPinholeCameraParam.focal_length_y = camera_model_parameters.focal_length[1]
        opencvPinholeCameraParam.radial_coeffs.extend(camera_model_parameters.radial_coeffs.tolist())
        opencvPinholeCameraParam.tangential_coeffs.extend(camera_model_parameters.tangential_coeffs.tolist())
        opencvPinholeCameraParam.thin_prism_coeffs.extend(camera_model_parameters.thin_prism_coeffs.tolist())
        return opencvPinholeCameraParam

    def _populate_opencv_fisheye_camera_param(
        self, camera_model_parameters: ncore.data.OpenCVFisheyeCameraModelParameters
    ) -> OpenCVFisheyeCameraParam:
        opencvFisheyeCameraParam = OpenCVFisheyeCameraParam()
        opencvFisheyeCameraParam.principal_point_x = camera_model_parameters.principal_point[0]
        opencvFisheyeCameraParam.principal_point_y = camera_model_parameters.principal_point[1]
        opencvFisheyeCameraParam.focal_length_x = camera_model_parameters.focal_length[0]
        opencvFisheyeCameraParam.focal_length_y = camera_model_parameters.focal_length[1]
        opencvFisheyeCameraParam.radial_coeffs.extend(camera_model_parameters.radial_coeffs.tolist())
        opencvFisheyeCameraParam.max_angle = camera_model_parameters.max_angle
        return opencvFisheyeCameraParam

    @staticmethod
    def _populate_external_distortion_parameters(
        intrinsics_proto: CameraSpec,
        distortion_params: Optional[ncore.data.ConcreteExternalDistortionParametersUnion],
    ) -> None:
        """Populates the right external distortion parameters field in a CameraSpec proto class
        according to the concrete type of the external distortion parameters passed as argument
        """
        if distortion_params is None:
            return
        if isinstance(distortion_params, ncore.data.BivariateWindshieldModelParameters):
            distortion_params_proto = BivariateWindshieldModelParameters()
            # Secure workaround for mismatch between integer enum values assigned in the
            # enum ncore.data.ReferencePolynomial (FORWARD:1, BACKWARD:2) (using auto()) and the
            # enum in the .proto file declaration of ReferencePolynomial (FORWARD:0, BACKWARD:1)
            match distortion_params.reference_poly:
                case ncore.data.ReferencePolynomial.FORWARD:
                    distortion_params_proto.reference_poly = (
                        BivariateWindshieldModelParameters.ReferencePolynomial.FORWARD
                    )
                case ncore.data.ReferencePolynomial.BACKWARD:
                    distortion_params_proto.reference_poly = (
                        BivariateWindshieldModelParameters.ReferencePolynomial.BACKWARD
                    )
                case _:
                    raise ValueError(f"Unsupported reference polynomial type: {distortion_params.reference_poly}")
            distortion_params_proto.horizontal_poly.extend(distortion_params.horizontal_poly.tolist())
            distortion_params_proto.vertical_poly.extend(distortion_params.vertical_poly.tolist())
            distortion_params_proto.horizontal_poly_inverse.extend(distortion_params.horizontal_poly_inverse.tolist())
            distortion_params_proto.vertical_poly_inverse.extend(distortion_params.vertical_poly_inverse.tolist())
            intrinsics_proto.bivariate_windshield_model_param.CopyFrom(distortion_params_proto)
        else:
            raise TypeError(f"Unsupported external distortion parameters type {type(distortion_params).__name__}.")

    def get_available_cameras(self) -> Generator[AvailableCamerasReturn.AvailableCamera, None, None]:
        for logical_cam_id, camera in self.cameras.items():
            camera_model_parameters = camera.camera_model_parameters

            # For now, only FThetaCameraModelParameters and OpenCVPinholeCameraModelParameters are supported
            if isinstance(camera_model_parameters, ncore.data.FThetaCameraModelParameters):
                intrinsics_proto = CameraSpec(
                    logical_id=logical_cam_id,
                    ftheta_param=self._populate_ftheta_camera_param(camera_model_parameters),
                    resolution_w=camera_model_parameters.resolution[0],
                    resolution_h=camera_model_parameters.resolution[1],
                    shutter_type=camera_model_parameters.shutter_type.value,  # type: ignore
                )
                CameraBank._populate_external_distortion_parameters(
                    intrinsics_proto,
                    require_concrete_external_distortion_parameters(
                        camera_model_parameters.external_distortion_parameters
                    ),
                )
                yield AvailableCamerasReturn.AvailableCamera(
                    intrinsics=intrinsics_proto,
                    logical_id=logical_cam_id,
                    rig_to_camera=se3_to_grpc_pose(camera.T_camera_rig.numpy()),
                )
            elif isinstance(camera_model_parameters, ncore.data.OpenCVPinholeCameraModelParameters):
                intrinsics_proto = CameraSpec(
                    logical_id=logical_cam_id,
                    opencv_pinhole_param=self._populate_opencv_pinhole_camera_param(camera_model_parameters),
                    resolution_w=camera_model_parameters.resolution[0],
                    resolution_h=camera_model_parameters.resolution[1],
                    shutter_type=camera_model_parameters.shutter_type.value,  # type: ignore
                )
                CameraBank._populate_external_distortion_parameters(
                    intrinsics_proto,
                    require_concrete_external_distortion_parameters(
                        camera_model_parameters.external_distortion_parameters
                    ),
                )
                yield AvailableCamerasReturn.AvailableCamera(
                    intrinsics=intrinsics_proto,
                    logical_id=logical_cam_id,
                    rig_to_camera=se3_to_grpc_pose(camera.T_camera_rig.numpy()),
                )
            elif isinstance(camera_model_parameters, ncore.data.OpenCVFisheyeCameraModelParameters):
                intrinsics_proto = CameraSpec(
                    logical_id=logical_cam_id,
                    opencv_fisheye_param=self._populate_opencv_fisheye_camera_param(camera_model_parameters),
                    resolution_w=camera_model_parameters.resolution[0],
                    resolution_h=camera_model_parameters.resolution[1],
                    shutter_type=camera_model_parameters.shutter_type.value,  # type: ignore
                )
                CameraBank._populate_external_distortion_parameters(
                    intrinsics_proto,
                    require_concrete_external_distortion_parameters(
                        camera_model_parameters.external_distortion_parameters
                    ),
                )
                yield AvailableCamerasReturn.AvailableCamera(
                    intrinsics=intrinsics_proto,
                    logical_id=logical_cam_id,
                    rig_to_camera=se3_to_grpc_pose(camera.T_camera_rig.numpy()),
                )
            else:
                raise TypeError("Unsupported camera model parameters type.")


# eq=False: the generated __eq__ would compare a torch.Tensor field, whose elementwise
# result has no unambiguous truth value. Identity comparison is what callers want anyway.
@dataclass(eq=False)
class LidarEntry:
    """One lidar of the artifact, as recorded in its ``RigTrajectories``.

    ``logical_id`` is ``LidarCalibration.logical_sensor_name``, not the
    ``lidar_id@sequence_id`` key: it is what clients address and what ``PerSensor`` config
    is indexed by. ``model_parameters`` is optional -- such a lidar is still discoverable,
    it just cannot generate rays. ``bundle()`` is lazy because it allocates on the GPU.
    """

    logical_id: str
    unique_sensor_idx: int
    T_lidar_rig: torch.Tensor
    model_parameters: Optional[ncore.data.ConcreteLidarModelParametersUnion]
    # Observed firing pattern, for units with no nominal model; carried by the artifact.
    firing_pattern: Optional[LidarFiringPattern] = None

    _bundle: Optional[LidarModelBundle] = field(default=None, init=False, repr=False)
    _bundle_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def bundle(self) -> Optional[LidarModelBundle]:
        """The ray-generation bundle for this lidar, built on first use (None without intrinsics)."""
        if self.model_parameters is None:
            return None
        with self._bundle_lock:
            if self._bundle is None:
                self._bundle = LidarModelBundle.load_from_config(self.model_parameters.to_dict())
            return self._bundle


class LidarBank:
    """The artifact's lidars, keyed by logical sensor id.

    All of them are retained so discovery and per-sensor config work on a multi-lidar
    artifact; rendering still targets one lidar per request (see :meth:`get_entry`).
    """

    def __init__(self, artifact: Artifact):
        super().__init__()

        rig_trajectories = RigTrajectories.from_dict(artifact.rig_trajectories)
        stored_patterns = artifact.firing_patterns

        self.lidars: OrderedDict[str, LidarEntry] = OrderedDict()
        for unique_lidar_id, lidar_calibration in rig_trajectories.lidar_calibrations.items():
            logical_id = lidar_calibration.logical_sensor_name
            if logical_id in self.lidars:
                # Unique only within one rig trajectory; a collision makes lookup ambiguous.
                raise ValueError(
                    f"LidarBank: duplicate logical lidar id '{logical_id}' (from unique id "
                    f"'{unique_lidar_id}'); logical lidar ids must be unique within an artifact."
                )
            firing_pattern = None
            if logical_id in stored_patterns:
                firing_pattern = LidarFiringPattern.load(io.BytesIO(stored_patterns[logical_id]))
                log.info(f"LidarBank: lidar '{logical_id}' uses the firing pattern stored in the artifact")
            self.lidars[logical_id] = LidarEntry(
                logical_id=logical_id,
                unique_sensor_idx=lidar_calibration.unique_sensor_idx,
                T_lidar_rig=lidar_calibration.T_sensor_rig,
                model_parameters=lidar_calibration.lidar_model_parameters,
                firing_pattern=firing_pattern,
            )

    def get_entry(self, logical_id: Optional[str] = None) -> Optional[LidarEntry]:
        """The requested lidar, or the only one when ``logical_id`` is omitted.

        Omitting the id is only unambiguous for a single-lidar artifact; on a multi-lidar
        artifact it raises rather than picking one arbitrarily.
        """
        if logical_id is not None:
            entry = self.lidars.get(logical_id)
            if entry is None:
                raise ValueError(
                    f"Unknown lidar '{logical_id}'; this artifact has {sorted(self.lidars)}. "
                    "Call get_available_lidars to discover them."
                )
            return entry
        if not self.lidars:
            return None
        if len(self.lidars) > 1:
            raise ValueError(
                f"This artifact has multiple lidars {sorted(self.lidars)}; the request must name one. "
                "Call get_available_lidars to discover them."
            )
        return next(iter(self.lidars.values()))


class AssetBank:
    """
    Manages external track assets (PLY files) from artifact.
    Follows the same pattern as CameraBank and LidarBank.
    """

    def __init__(self, artifact: Artifact):
        super().__init__()
        self.external_assets = artifact.external_assets  # Dict[track_id, ply_bytes]
        self._dims_offset = OmegaConf.select(
            OmegaConf.create(artifact.parsed_config),
            "dataset.valid_pixels_cuboid_track_params.track_padding_m",
            default=[0.0, 0.0, 0.0],
        )

        if self.external_assets:
            log.info(
                f"AssetBank initialized with {len(self.external_assets)} track assets: "
                f"{list(self.external_assets.keys())}"
            )
        else:
            log.info("AssetBank initialized with no external assets")

    @property
    def dims_offset(self) -> torch.Tensor:
        return torch.tensor(list(self._dims_offset), dtype=torch.float32, device="cuda")

    def has_track(self, track_id: str) -> bool:
        """Check if a track ID exists in external assets."""
        return track_id in self.external_assets

    def get_track_ids(self) -> List[str]:
        """Get all available track IDs."""
        return list(self.external_assets.keys())

    def get_track_asset(self, track_id: str, cuboid_dims: torch.Tensor, device: str = "cuda") -> Asset:
        """
        Get parsed Asset for a track with specified cuboid dimensions.

        Args:
            track_id: Track ID
            cuboid_dims: Cuboid dimensions tensor
            device: Device to load tensors onto

        Raises:
            KeyError: If track_id not found
        """
        if not self.has_track(track_id):
            raise KeyError(f"Track ID '{track_id}' not found. Available: {self.get_track_ids()}")

        asset_data = self.external_assets[track_id]
        ply_bytes = asset_data["ply_bytes"]

        return Asset.from_ply_bytes(ply_bytes, device=device, cuboids_dims=cuboid_dims)

    def get_track_cuboid_dims(self, track_id: str, device: str = "cuda") -> torch.Tensor:
        """
        Get the cuboid dimensions for a track ID.
        """
        if not self.has_track(track_id):
            raise KeyError(f"Track ID '{track_id}' not found. Available: {self.get_track_ids()}")

        asset_data = self.external_assets[track_id]
        return torch.tensor(asset_data["metadata"]["cuboids_dims"], dtype=torch.float32, device=device)


def with_render_lock():
    """
    Decorator to ensure that Backend render function is thread-safe.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(backend: Backend, *args, **kwargs):
            with backend._render_lock:
                return func(backend, *args, **kwargs)

        return wrapper

    return decorator


def _overlay_bounding_box_source(request: RGBRenderRequest | CameraRenderRequest) -> OverlayBoundingBoxSource:
    if isinstance(request, CameraRenderRequest):
        return request.overlay_bounding_boxes
    return OverlayBoundingBoxSource.OVERLAY_BOUNDING_BOX_SOURCE_UNSPECIFIED


_DEFAULT_DISTANCE_COLORMAP_VMAX_M = 80.0

# Cap LUT allocation for sparse or malformed semantic class IDs.
_SEMANTICS_COLOR_LUT_MAX_SIZE = 4096


def _install_edit_overrides(backend: "Backend", artifact: Artifact) -> None:
    """Install the asset-edit runtime when the artifact is an edit-baked USDZ.

    Auto-detected on both counts — the manifest embedded in the USDZ, and a companion
    ``<stem>.schedule.json`` beside it — so plain artifacts are unaffected and no extra
    process is involved. Without a schedule file the registry starts empty and awaits
    live ``AspectState`` messages.

    See ``nurec/utils/edit/README.md`` for what any of this means.
    """
    source = Path(str(getattr(artifact, "source", "") or ""))
    if not source.is_file():
        return
    manifest = load_manifest_from_usdz(source)
    if manifest is None:
        return
    composite = getattr(backend.renderable_model, "_model", None)
    if composite is None or not hasattr(composite, "set_edit_overrides"):
        # Fail loud (escalated to ModelIncompatibilityError by the caller): with
        # replace_original the bake hid the captured actor, so rendering without
        # the edits would silently show a hole where it was.
        raise ValueError(
            f"artifact {source.name} has an edit manifest but the model "
            f"({type(composite).__name__}) does not support edit overrides."
        )
    # The manifest's row indices address the exact checkpoint it was baked against;
    # refuse a stale manifest / re-trained / re-ordered artifact before it corrupts
    # renders. Same gate as the offline render CLI — see EditManifest.validate_against_layers.
    nodes = dict(getattr(composite, "gaussians_nodes", None) or {})
    manifest.validate_against_layers(nodes)

    # Track index for LIVE aspect driving (AspectState on render requests): cleaned
    # track-id string -> (layer_id, nurec_track_id) for every edit the manifest
    # bakes clusters for. Pure addressing — all animation/semantic evaluation happens
    # in the SENDER (WSE/animator); the server only shows the aspects it is told to.
    track_index: dict[str, tuple[str, int]] = {}
    unresolvable: list[tuple[str, int]] = []
    for layer_id, nurec_track_id in manifest.by_layer_track():
        node = nodes.get(layer_id)
        ids = getattr(getattr(node, "cuboid_tracks", None), "tracks_id", None)
        if ids is None or not (0 <= nurec_track_id < len(ids)):
            unresolvable.append((layer_id, nurec_track_id))
            continue
        # Node track ids are cleaned of "@<source>" suffixes at model construction
        # (GaussiansComposite.__init__), so this is a no-op today — kept so the index
        # key is normalized by the SAME function the request lookup uses.
        track_index[DataSourceSummary._clean_track_id_str(str(ids[nurec_track_id]))] = (layer_id, nurec_track_id)
    if unresolvable:
        # Actually drop them. Omitting them from the addressing index alone would leave
        # apply() still iterating their clusters every frame, so the log below would be
        # a lie: their rows would keep being written, just unaddressable.
        manifest.drop_tracks(unresolvable)
        log.warning(
            "edit manifest key(s) %s are not resolvable in this composite; their clusters are "
            "dropped and will not render.",
            unresolvable,
        )
    backend.edit_track_index = track_index
    # The aspects this scene can actually show, per track. Built here so an unknown
    # aspect string is a request error the SENDER sees, not an invisibly-unlit actor and a
    # server-side log line no gRPC client can read.
    backend.edit_track_aspects = {
        track_str: {c.aspect for c in manifest.by_layer_track().get(key, ())} for track_str, key in track_index.items()
    }

    # Only now build the runtime: it snapshots the manifest's cluster grouping in its
    # constructor, so any dropping must already have happened.
    registry = AspectScheduleRegistry()
    # Companion schedule convention: strip ONLY the final extension (Path.stem), so
    # dotted artifact names keep their full stem — "scene.v2.usdz" pairs with
    # "scene.v2.schedule.json" (a first-dot split would silently probe the wrong name).
    schedule_path = source.parent / (source.stem + ".schedule.json")
    if schedule_path.is_file():
        schedule = load_schedule_from_path(schedule_path)
        # A schedule naming a track or aspect this bake does not have renders nothing,
        # while the install log below still reports it as loaded.
        manifest.validate_schedule(schedule)
        registry.set_schedule(schedule)
    composite.set_edit_overrides(AspectOverrideRuntime(manifest=manifest, registry=registry))
    backend.aspect_registry = registry
    # Immutable file baseline: every render re-derives the registry from this plus
    # the request's own AspectState messages (see _update_aspect_states).
    backend.file_schedules = dict(registry.tracks)

    log.info(
        "asset-edit overrides installed for %s: %d cluster(s), live-drivable track(s) %s, schedule %s",
        source.name,
        len(manifest.clusters),
        sorted(track_index),
        f"{schedule_path.name} ({len(registry.tracks)} track schedule(s))"
        if schedule_path.is_file()
        else f"none (no {schedule_path.name}; awaiting set_many)",
    )


@dataclass
class Backend:
    renderable_model: RenderableModel
    camera_bank: CameraBank
    world_to_nurec: FrameConversion
    lidar_bank: LidarBank
    asset_bank: AssetBank
    source_artifact: Artifact = field(repr=False)
    backend_cache_key: BackendCacheKey
    distance_vmax_m: float = _DEFAULT_DISTANCE_COLORMAP_VMAX_M
    # Asset-edit state (None/empty when the artifact carries no edit manifest). The
    # registry is rebuilt on EVERY render from file_schedules (the immutable
    # install-time baseline) plus the request's own AspectState messages — no
    # cross-request state, so pooled/replicated backends of one scene cannot drift.
    aspect_registry: Optional[AspectScheduleRegistry] = None
    # Cleaned track-id string -> (layer_id, nurec_track_id): request addressing only.
    edit_track_index: Optional[dict[str, tuple[str, int]]] = None
    # Cleaned track-id string -> that track's baked aspect strings (rejection only).
    edit_track_aspects: dict[str, set[str]] = field(default_factory=dict)
    file_schedules: dict[tuple[str, int], TrackSchedule] = field(default_factory=dict)
    # Logging only (not behavioral state): emit the "live aspect control" line once.
    _live_aspects_logged: bool = field(default=False, repr=False)
    # for thread-safe access (see with_render_lock)
    _render_lock: threading.Lock = field(default_factory=threading.Lock)
    _artifact_actor_tracks: Optional[ActorTracks] = field(default=None, init=False, repr=False)
    # Separate from _render_lock so cache initialization does not block rendering.
    _artifact_actor_tracks_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # Per-backend CUDA stream for lidar renders, threaded into the composite's
    # forward_gaussians (see _get_lidar_stream). This makes the lidar stream owned by the
    # request context rather than created inside GaussiansComposite.__init__, so
    # concurrent lidar requests on distinct replicas don't contend on one shared stream.
    _lidar_stream: Optional[torch.cuda.Stream] = None

    # Opaque lookup key used internally by BackendCache.
    cache_key: str | None = None

    class RendererType(Enum):
        GSPLAT = auto()
        DEFAULT = auto()

    def __hash__(self):
        # Use object identity for hashing since cache_key is mutable
        # This ensures hash stability even before cache_key is set
        return hash(id(self))

    def __eq__(self, other):
        # Use object identity for equality to match __hash__
        # This maintains the hash/eq contract: if a == b, then hash(a) == hash(b)
        return self is other

    @property
    def scene_id(self) -> str:
        """Return the scene used for backend-cache capacity limits."""
        return self.backend_cache_key.scene_key.scene_id

    @staticmethod
    def from_artifact(
        artifact: Artifact,
        renderer: Backend.RendererType,
        tracks_timestamp_ray_stride: int = 1,
        *,
        backend_cache_key: BackendCacheKey,
    ) -> Backend:
        try:
            config_overrides: tuple[str, ...] = ()

            if renderer == Backend.RendererType.GSPLAT:
                # gRPC serves many artifacts; prefer an explicit flag over arbitrary overrides.
                config_overrides = ("model.renderer.name=3dgut-gsplat",)

            # Distance is a first-class CameraRender product any client may request against this
            # shared, cached backend, and the renderer is fixed at load time. Enable it
            # unconditionally so DISTANCE_COLORIZED is available: gsplat only produces the depth
            # channel when this gate is set (renderers.py:1055-1060). Mirrors the render CLI,
            # which re-enables
            # enable_distance for --output-type distance-colorized. Trade-off: forgoes the CDIM 4->3
            # optimization for color-only gsplat camera renders.
            config_overrides += ("model.renderer.outputs.camera.enable_distance=True",)

            # Semantic logits require extended features.
            if artifact.semantic_class_names is not None:
                config_overrides += (
                    "model.renderer.outputs.camera.enable_extended_features=True",
                    "model.renderer.outputs.camera.enable_sensor_features=True",
                )

            # Inference-only ray-stride; scoped to present track layers inside load_from_artifact.

            renderable_model = RenderableModel.load_from_artifact(
                artifact,
                config_overrides=config_overrides,
                tracks_timestamp_ray_stride=tracks_timestamp_ray_stride,
            )
        except (TypeError, ValueError) as exc:
            raise ModelIncompatibilityError(str(exc)) from exc

        world_to_nurec = RigTrajectories.from_dict(artifact.rig_trajectories).world_to_nurec

        lidar_bank = LidarBank(artifact)

        backend = Backend(
            renderable_model=renderable_model,
            camera_bank=CameraBank(artifact, world_to_nurec),
            world_to_nurec=world_to_nurec,
            lidar_bank=lidar_bank,
            asset_bank=AssetBank(artifact),
            source_artifact=artifact,
            backend_cache_key=backend_cache_key,
        )

        try:
            _install_edit_overrides(backend, artifact)
        except Exception as exc:
            # An edit-baked artifact whose runtime cannot install must refuse to serve:
            # with replace_original the captured actor is hidden in the bake, so
            # serving without the edits would silently render a hole where it was.
            raise ModelIncompatibilityError(f"asset-edit install failed: {exc}") from exc

        torch.cuda.empty_cache()
        gc.collect()

        return backend

    def _validate_dynamic_object_updates_request(self, num_dynamic_objects: int, enable_editing_actors: bool) -> None:
        """Validate whether dynamic object updates are allowed for the current request."""
        if num_dynamic_objects == 0:
            return

        if not enable_editing_actors:
            raise ValueError(
                "Got render request with DynamicObject updates but actor editing is disabled on the server. "
                "Restart serve-grpc with --enable-editing-actors."
            )

        if not self.renderable_model.supports_edit_actors():
            raise ValueError("Got render request with DynamicObject updates but model does not support actor updates")

    @staticmethod
    def _stream_on_device(existing: Optional[torch.cuda.Stream], device: torch.device) -> torch.cuda.Stream:
        """Return a CUDA stream on ``device``, reusing ``existing`` when it already matches.

        Streams are created lazily (and on the model device) rather than at
        construction so a Backend that is built but never used pays nothing, and so the
        stream always lands on the device the model actually lives on.
        """
        if existing is None or existing.device != device:
            return torch.cuda.Stream(device=device)
        return existing

    def _get_lidar_stream(self) -> torch.cuda.Stream:
        """The per-backend lidar stream, created lazily on the model device.

        Passed into the composite's render path (``forward_gaussians``) so the lidar
        collect+render run on a request-owned stream instead of one created inside the
        composite. Safe to create lazily here because lidar renders hold ``_render_lock``.
        """
        self._lidar_stream = self._stream_on_device(self._lidar_stream, self.renderable_model.device)
        return self._lidar_stream

    def _update_aspect_states(self, request: RGBRenderRequest | CameraRenderRequest | LidarRenderRequest) -> None:
        """Apply per-actor AspectState messages: show EXACTLY the named aspects.

        Stateless per the design (nurec/utils/edit/README.md): each request's
        states fully determine that frame; a track without an AspectState renders its
        file-scheduled/baked state. Callers hold the render lock, so the registry
        rebuild is atomic with the render.

        ADDRESSING is validated here: an unknown track id, unknown aspect, or duplicate
        entry raises ``ValueError`` (mapped to ``INVALID_ARGUMENT``) — silently ignoring
        it would leave a sender that mistyped one character with unlit lights and a
        success status, and gRPC clients never see server logs. Addressing only: the
        server still assigns no MEANING to any aspect string.
        """
        driven = [d for d in request.dynamic_objects if d.HasField("aspect_state")]
        if self.aspect_registry is None or not self.edit_track_index:
            if driven:
                detail = (
                    "its asset-edit manifest has no drivable tracks in this composite"
                    if self.aspect_registry is not None
                    else "it has no asset-edit manifest (serve an artifact baked with 'nurec compile-asset-edits')"
                )
                raise ValueError(
                    f"request carries aspect_state for {len(driven)} actor(s), but this scene "
                    f"cannot show any aspect: {detail}. Drop aspect_state from the request."
                )
            return
        live: dict[tuple[str, int], TrackSchedule] = {}
        for dyn in driven:
            track_str = DataSourceSummary._clean_track_id_str(dyn.track_id)
            key = self.edit_track_index.get(track_str)
            if key is None:
                raise ValueError(
                    f"aspect_state names track {track_str!r}, which this scene has no baked asset "
                    f"edit for. Edit-drivable tracks: {sorted(self.edit_track_index)}."
                )
            if key in live:
                # Silently letting the last entry win would make a sender bug undetectable.
                raise ValueError(f"request carries aspect_state for track {track_str!r} more than once.")
            known = self.edit_track_aspects.get(track_str, set())
            requested = set(dyn.aspect_state.aspects) | {g.aspect for g in dyn.aspect_state.gains}
            unknown = sorted(requested - known)
            if unknown:
                raise ValueError(
                    f"aspect_state for track {track_str!r} names aspect(s) {unknown} that no baked "
                    f"cluster carries; they would render as nothing. Baked aspects for this track: "
                    f"{sorted(known)}. (Aspect strings are case-sensitive.)"
                )
            gains: dict[str, tuple[float, Optional[list[float]]]] = {}
            for g in dyn.aspect_state.gains:
                if g.aspect in gains:
                    raise ValueError(
                        f"aspect_state for track {track_str!r} lists gains for aspect {g.aspect!r} more than once."
                    )
                gains[g.aspect] = validate_aspect_gain(
                    g.aspect,
                    g.gain if g.HasField("gain") else 1.0,
                    [float(c) for c in g.cold_rgb] or None,
                    ctx=f"AspectState for track {track_str!r}",
                )
            live[key] = TrackSchedule(
                layer_id=key[0],
                nurec_track_id=key[1],
                edit_id=f"live:{track_str}",
                switches=[AspectSwitch(aspects=list(dyn.aspect_state.aspects), t_us=0)],
                aspect_gains=gains,
            )
        if live and not self._live_aspects_logged:
            self._live_aspects_logged = True
            log.info(
                "live aspect control active (driven this request: %s)%s.",
                sorted(live),
                f"; file schedule applies to undriven {sorted(k for k in self.file_schedules if k not in live)}"
                if any(k not in live for k in self.file_schedules)
                else "",
            )
        self.aspect_registry.set_many({**self.file_schedules, **live}.values())

    @with_render_lock()
    @ScopedTimer()
    def render_rgb_request(
        self, request: RGBRenderRequest, enable_editing_actors: bool = False
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        self._validate_dynamic_object_updates_request(
            num_dynamic_objects=len(request.dynamic_objects),
            enable_editing_actors=enable_editing_actors,
        )
        self._update_aspect_states(request)

        camera = self.camera_bank._get_camera(request.camera_intrinsics)
        ray_bundle = camera.build_ray_bundle(request)

        if enable_editing_actors and self.renderable_model.supports_edit_actors():
            actors_snapshot = actors_snapshot_from_render_request(request)
        else:
            actors_snapshot = None

        rendered = self.renderable_model.render_camera_frame_from_ray_bundle(
            ray_bundle,
            actors_snapshot=actors_snapshot,
            frame_start_us=request.frame_start_us,
            frame_end_us=request.frame_end_us,
        )
        w, h = ray_bundle.rendering_data.w, ray_bundle.rendering_data.h

        assert rendered.color_image is not None, "Renderer did not return a color image"

        return rendered.color_image, (h, w)

    def _get_artifact_actor_tracks(self) -> ActorTracks:
        if self._artifact_actor_tracks is not None:
            return self._artifact_actor_tracks
        with self._artifact_actor_tracks_lock:
            if self._artifact_actor_tracks is not None:
                return self._artifact_actor_tracks
            raw = self.source_artifact.sequence_tracks
            if not raw:
                result = ActorTracks()
            else:
                sequence_tracks = {k: CuboidTracks.from_dict(v) for k, v in raw.items()}
                artifact_cuboid_tracks = next(iter(sequence_tracks.values()))
                # Artifact overlays include annotations that are not marked controllable.
                result = ActorTracks._from_cuboid_tracks(artifact_cuboid_tracks, controllable_only=False)
            self._artifact_actor_tracks = result
            return result

    @with_render_lock()
    @ScopedTimer()
    def render_camera_request(
        self,
        request: CameraRenderRequest,
        enable_editing_actors: bool = False,
    ) -> Tuple[CameraFrame, Tuple[int, int], Optional[ActorsSnapshot]]:
        """Render requested products and return a MODEL overlay snapshot captured under the render lock."""
        self._validate_dynamic_object_updates_request(
            num_dynamic_objects=len(request.dynamic_objects),
            enable_editing_actors=enable_editing_actors,
        )
        self._update_aspect_states(request)

        camera = self.camera_bank._get_camera(request.camera_intrinsics)
        ray_bundle = camera.build_ray_bundle(request)

        if enable_editing_actors and self.renderable_model.supports_edit_actors():
            actors_snapshot = actors_snapshot_from_render_request(request)
        else:
            actors_snapshot = None

        products = list(request.render_products) or [CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR]
        render_options = CameraRenderOptions(
            color_image=CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR in products,
            distance_image=CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED in products,
            semantic_image=CameraRenderProduct.CAMERA_RENDER_PRODUCT_SEMANTICS_COLORIZED in products,
        )

        frame = self.renderable_model.render_camera_frame_from_ray_bundle(
            ray_bundle,
            render_options=render_options,
            actors_snapshot=actors_snapshot,
            frame_start_us=request.frame_start_us,
            frame_end_us=request.frame_end_us,
        )
        h, w = ray_bundle.rendering_data.h, ray_bundle.rendering_data.w

        model_overlay_snapshot: Optional[ActorsSnapshot] = None
        if _overlay_bounding_box_source(request) == OverlayBoundingBoxSource.OVERLAY_BOUNDING_BOX_SOURCE_MODEL:
            if not self.renderable_model.supports_edit_actors():
                raise ValueError("OVERLAY_BOUNDING_BOX_SOURCE_MODEL requested but model does not support actor tracks.")
            model_tracks = self.renderable_model.get_actor_tracks()
            if model_tracks.num_tracks() == 0:
                raise ValueError(
                    "Bounding box overlay requested with source OVERLAY_BOUNDING_BOX_SOURCE_MODEL "
                    "but the resolved actor tracks contain no tracks."
                )
            if request.dynamic_objects:
                # The renderer used request-derived poses; use the same so boxes track the rendered
                # actors. Cuboid dims are not in the request, so look them up from the model tracks.
                req_snapshot = actors_snapshot_from_render_request(request)
                known_ids = set(model_tracks.known_track_ids())
                unknown_ids = [aid for aid in req_snapshot.actor_ids if aid not in known_ids]
                if unknown_ids:
                    # Request track ids are normalized via _clean_track_id_str while model-track keys
                    # are the raw tracks_id[0]; surface the mismatch with the offending and known ids
                    # instead of the cryptic "No cuboid dimensions for actor track id ..." from the
                    # bare get_cuboid_dims lookup.
                    known_preview = sorted(known_ids)
                    if len(known_preview) > 20:
                        known_str = ", ".join(known_preview[:20]) + f", ... ({len(known_preview)} total)"
                    else:
                        known_str = ", ".join(known_preview)
                    raise ValueError(
                        "MODEL bounding-box overlay requested with dynamic_objects whose track_id(s) "
                        f"{unknown_ids} are not among the model's actor tracks. Known model track ids: "
                        f"[{known_str}]."
                    )
                model_overlay_snapshot = ActorsSnapshot(
                    actor_ids=req_snapshot.actor_ids,
                    actor_poses=req_snapshot.actor_poses,
                    cuboid_dims=model_tracks.get_cuboid_dims(req_snapshot.actor_ids),
                )
            else:
                model_overlay_snapshot = model_tracks.get_snapshot_at_frame(
                    request.frame_start_us, request.frame_end_us
                )

        return frame, (h, w), model_overlay_snapshot

    def get_available_cameras(self) -> AvailableCamerasReturn:
        return AvailableCamerasReturn(available_cameras=list(self.camera_bank.get_available_cameras()))

    def get_available_trajectories(self) -> AvailableTrajectoriesReturn:
        return AvailableTrajectoriesReturn(
            available_trajectories=[
                AvailableTrajectoriesReturn.AvailableTrajectory(
                    trajectory_idx=0, trajectory=self.camera_bank.trajectory
                )
            ]
        )

    def get_external_asset_objects(self) -> ExternalAssetObjectsReturn:
        """
        Get list of track IDs for external asset objects.

        Returns:
            ExternalAssetObjectsReturn containing list of available track IDs
        """
        return ExternalAssetObjectsReturn(track_ids=self.asset_bank.get_track_ids())

    def get_dynamic_objects(self) -> List[sensorsim_types.DynamicObjectTrack]:
        """Return DynamicObjectTrack for all controllable actors in gRPC format."""
        if not self.renderable_model.supports_edit_actors():
            return []  # Return empty list instead of raising error

        actor_tracks = self.renderable_model.get_actor_tracks()
        return actor_tracks_to_grpc(actor_tracks)

    def replace_assets(self, replace_list: List[sensorsim_types.ReplaceAssetAction]) -> None:
        def is_empty_aabb(aabb: grpc_types.AABB) -> bool:
            """Check if AABB is all zeros."""
            return aabb.size_x == 0.0 and aabb.size_y == 0.0 and aabb.size_z == 0.0

        for replacement in replace_list:
            original_id = replacement.original_id
            replacement_id = replacement.replacement_id

            if not replacement_id:
                raise ValueError(
                    f"Replacement_id is empty for original_id '{original_id}', please provide a valid replacement_id"
                )

            device = str(self.renderable_model.device)

            # Check if object_size is provided and valid, otherwise use fallback
            if not replacement.HasField("object_size") or is_empty_aabb(replacement.object_size):
                if not self.asset_bank.has_track(replacement_id):
                    raise ValueError(f"object_size required for filesystem path replacement_id '{replacement_id}' ")
                log.info(
                    f"object_size not provided or empty for original_id '{original_id}', "
                    f"using fallback from replacement asset '{replacement_id}'"
                )
                cuboid_dims = self.asset_bank.get_track_cuboid_dims(replacement_id, device=device)
            else:
                cuboid_dims = torch.tensor(
                    [replacement.object_size.size_x, replacement.object_size.size_y, replacement.object_size.size_z],
                    dtype=torch.float32,
                    device=device,
                )

            if self.asset_bank.has_track(replacement_id):
                asset = self.asset_bank.get_track_asset(replacement_id, cuboid_dims=cuboid_dims, device=device)
            else:
                asset = Asset(Path(replacement_id), device=device)
                asset.cuboids_dims = cuboid_dims
                log.info(
                    f"Loaded PLY from filesystem for replacement '{original_id}': {replacement_id} "
                    f"({asset.positions.shape[0]} gaussians)"
                )
            replaced = self.renderable_model.replace_asset(
                original_id, asset=asset, dims_offset=self.asset_bank.dims_offset
            )
            if not replaced:
                raise RuntimeError(
                    f"Failed to replace original_id='{original_id}' with replacement_id='{replacement_id}'"
                )

    def insert_assets(self, dynamic_objects: List[sensorsim_types.DynamicObjectTrack]) -> None:
        """Insert dynamic object assets with proper coordinate frame conversion.

        Supports both AssetBank track IDs and filesystem PLY paths as asset_id.
        AssetBank is checked first; if the ID is not found, it is treated as a filesystem path.
        """
        actor_tracks = actor_tracks_from_grpc(dynamic_objects)

        track_to_asset = {obj.id: obj.asset_id for obj in dynamic_objects}

        for cuboid_track in actor_tracks._cuboid_tracks_list:
            (track_id,) = cuboid_track.tracks_id

            asset_id = track_to_asset[track_id]

            nurec_cuboid_track = CuboidTracks.Ops.transform_with_frame_conversion(
                cuboid_track, self.world_to_nurec, None
            )

            cuboid_dims = torch.tensor(
                nurec_cuboid_track.cuboids_dims, dtype=torch.float32, device=str(self.renderable_model.device)
            )

            device = str(self.renderable_model.device)
            if self.asset_bank.has_track(asset_id):
                asset = self.asset_bank.get_track_asset(asset_id, cuboid_dims=cuboid_dims, device=device)
            else:
                asset = Asset(Path(asset_id), device=device)
                asset.cuboids_dims = cuboid_dims
                log.info(
                    f"Loaded PLY from filesystem for track '{track_id}': {asset_id} "
                    f"({asset.positions.shape[0]} gaussians)"
                )

            # dims_offset (track_padding_m) applies to replacement, not insertion; pass a zero
            # offset so the inserted asset is scaled to its full cuboid dimensions.
            inserted = self.renderable_model.insert_asset(
                asset, nurec_cuboid_track, torch.zeros_like(self.asset_bank.dims_offset)
            )
            if not inserted:
                raise RuntimeError(f"Failed to insert track_id='{track_id}' with asset_id='{asset_id}'")

    @with_render_lock()
    def edit_assets(
        self,
        replace_assets: List[sensorsim_types.ReplaceAssetAction],
        insert_assets: List[sensorsim_types.DynamicObjectTrack],
    ) -> None:
        """
        Edit assets by replacing and inserting assets in the renderable_model.
        The client should send a one-time edit_assets request with the desired asset list to be edited before sending their RGBRenderRequests.
        Each edit operation will change the internal state of the renderable_model

        Args:
            replace_assets: List of ReplaceAssetAction specifying original_id (track from artifact) and replacement_id (asset from AssetBank)
            insert_assets: List of DynamicObjectTrack to insert, the ply data of the inserted assets should be present in the AssetBank
        """
        if not self.renderable_model.supports_edit_actors():
            raise RuntimeError("Trying to edit assets but model does not support actor updates")

        # Asset replacement/insertion reorders and regrows layer rows, which silently
        # invalidates a baked asset-edits manifest (its row indices address the exact
        # bake). Refuse the combination rather than corrupt aspect rendering.
        if (replace_assets or insert_assets) and self.aspect_registry is not None and self.edit_track_index:
            raise ValueError(
                "This scene is an edit-baked artifact (asset-edits manifest installed); "
                "edit_assets would reorder layer rows and silently invalidate the baked "
                "cluster indices. Bake the desired assets with compile-asset-edits instead, "
                "or serve a plain artifact for edit_assets."
            )

        # Save the original training parameters of the model before applying asset edit operations, user can restore the original parameters
        self.renderable_model.save_training_parameters()

        try:
            if replace_assets:
                self.replace_assets(replace_assets)

            if insert_assets:
                self.insert_assets(insert_assets)
        except Exception:
            self.renderable_model.restore_training_parameters()
            raise

    @with_render_lock()
    def restore_model_parameters(self) -> None:
        """
        Restore the original training parameters of the model.
        This undoes any asset editing operations.
        """
        if not self.renderable_model.supports_edit_actors():
            raise RuntimeError("Model does not support actor updates")

        self.renderable_model.restore_training_parameters()

    @with_render_lock()
    @ScopedTimer()
    def render_lidar_request(
        self, request: LidarRenderRequest, enable_editing_actors: bool = False
    ) -> LidarRenderReturn:
        """
        Go-to gRPC lidar rendering API.
        Currently missing
            * proper serialization of Lidar spec
            * proper trajectories for ego and traffic objects: currently only start and end pose are passed
        """

        self._validate_dynamic_object_updates_request(
            num_dynamic_objects=len(request.dynamic_objects),
            enable_editing_actors=enable_editing_actors,
        )
        self._update_aspect_states(request)

        # Update dynamic tracks
        actors_snapshot: Optional[ActorsSnapshot] = None
        if enable_editing_actors and self.renderable_model.supports_edit_actors():
            actors_snapshot = actors_snapshot_from_render_request(request)

        # generate lidar rays from a nominal Lidar model
        with ScopedTimer("LidarRayBundle.build"):
            tquat_sensor_world_start = torch.tensor(grpc_pose_to_tquat(request.sensor_pose.start_pose))
            tquat_sensor_world_end = torch.tensor(grpc_pose_to_tquat(request.sensor_pose.end_pose))

            lidar_to_world = PoseRange(
                tquat_sensor_world_start, tquat_sensor_world_end, request.frame_start_us, request.frame_end_us
            )
            # An empty lidar_id means "the artifact's only lidar"; on a
            # multi-lidar artifact get_lidar rejects the ambiguity.
            lidar_entry = self.lidar_bank.get_entry(getattr(request, "lidar_id", "") or None)
            if lidar_entry is None:
                raise ValueError("Lidar rendering is not supported: artifact has no lidar sensor")
            # A lidar with intrinsics gets its rays synthesised from the sensor model. One
            # without (a model-free unit) needs an observed firing pattern to say where and
            # when it fires; prefer the model when both are present, since it is the
            # artifact's own record of the sensor.
            lidar_model = lidar_entry.bundle()
            if lidar_model is not None:
                lidar_ray_bundle = LidarRayBundle.build(
                    lidar_model=lidar_model,
                    lidar_to_world=lidar_to_world,
                    world_to_nurec=self.world_to_nurec,
                    unique_sensor_idx=lidar_entry.unique_sensor_idx,
                )
            elif lidar_entry.firing_pattern is not None:
                lidar_ray_bundle = LidarRayBundle.build_model_free(
                    firing_pattern=lidar_entry.firing_pattern,
                    lidar_to_world=lidar_to_world,
                    world_to_nurec=self.world_to_nurec,
                    unique_sensor_idx=lidar_entry.unique_sensor_idx,
                )
            else:
                raise ValueError(
                    f"Lidar rendering is not supported for '{lidar_entry.logical_id}': the artifact "
                    "carries neither lidar model parameters nor a firing pattern for it, so there is "
                    "nothing to say where it fires. Re-export the artifact with "
                    "checkpoint.artifact.firing_patterns.enabled=true."
                )

        # Extract filter parameters from request (use None to signal "use default")
        raydrop_threshold = None
        opacity_threshold = None
        enable_distance_filter = None
        distance_filter_threshold = None
        # neighbor_source is enum-like, not Optional: seed it from the server-config default and let
        # the request override it per call, so a concrete value flows through (no None sentinel).
        neighbor_source = self.renderable_model.default_lidar_neighbor_source()
        elevation_resolution_deg = None
        azimuth_max_gap_deg = None

        if request.HasField("render_filter"):
            render_filter: LidarRenderFilter = request.render_filter
            if render_filter.HasField("raydrop_threshold"):
                raydrop_threshold = render_filter.raydrop_threshold
            if render_filter.HasField("opacity_threshold"):
                opacity_threshold = render_filter.opacity_threshold
            if render_filter.HasField("enable_distance_filter"):
                enable_distance_filter = render_filter.enable_distance_filter
            if render_filter.HasField("distance_filter_threshold"):
                distance_filter_threshold = render_filter.distance_filter_threshold
            if render_filter.HasField("neighbor_source"):
                neighbor_source = (
                    "angle"
                    if render_filter.neighbor_source == sensorsim_types.LIDAR_FLOATING_FILTER_SOURCE_ANGLE
                    else "rangeview"
                )
            if render_filter.HasField("elevation_resolution_deg"):
                elevation_resolution_deg = render_filter.elevation_resolution_deg
            if render_filter.HasField("azimuth_max_gap_deg"):
                azimuth_max_gap_deg = render_filter.azimuth_max_gap_deg

        with ScopedTimer("render_lidar_frame_from_ray_bundle"):
            lidar_frame = self.renderable_model.render_lidar_frame_from_ray_bundle(
                lidar_ray_bundle,
                raydrop_threshold=raydrop_threshold,
                opacity_threshold=opacity_threshold,
                enable_distance_filter=enable_distance_filter,
                distance_filter_threshold=distance_filter_threshold,
                neighbor_source=neighbor_source,
                elevation_resolution_deg=elevation_resolution_deg,
                azimuth_max_gap_deg=azimuth_max_gap_deg,
                actors_snapshot=actors_snapshot,
                frame_start_us=request.frame_start_us,
                frame_end_us=request.frame_end_us,
                lidar_stream=self._get_lidar_stream(),
            )

        with ScopedTimer("unpack_results"):
            pc_tensor = lidar_frame.point_positions.cpu().flatten().numpy().astype(np.float32)
            if lidar_frame.point_intensities is not None:
                intensity_tensor = lidar_frame.point_intensities.cpu().flatten().numpy().astype(np.float32)
            else:
                intensity_tensor = None

        with ScopedTimer("protobuf encoding"):
            # Create protobuf message directly from tensors
            final_result = LidarRenderReturn()

            if pc_tensor is not None:
                # Use numpy array view for zero-copy conversion
                final_result.point_xyzs_buffer = pc_tensor.tobytes()
                final_result.num_points = len(pc_tensor) // 3

            if intensity_tensor is not None:
                final_result.point_intensities_buffer = intensity_tensor.tobytes()
            else:
                log.info("The model loaded has no intensity prediction")

        return final_result


class ServerMetricsManager:
    """Manages metrics collection for the gRPC server."""

    def __init__(self, metrics_output_dir: Optional[str], artifacts: Optional[dict[str, Artifact]] = None):
        self._metrics_output_dir = metrics_output_dir
        self._metrics_collector: Optional[MetricsCollector] = None
        self._warmup_rgb_counter = 0
        self._warmup_batch_rgb_counter = 0
        self._warmup_lidar_counter = 0
        self._warmup_skip = 3
        self._save_timer: Optional[threading.Timer] = None
        self._timer_lock = threading.Lock()

        # Initialize metrics collection if output directory is specified
        if self._metrics_output_dir:
            self._initialize_metrics_collection(artifacts)

    def _initialize_metrics_collection(self, artifacts: Optional[dict[str, Artifact]] = None) -> None:
        """Initialize metrics collection with run_id detection."""
        # Create output directory if it doesn't exist
        if self._metrics_output_dir is None:
            raise ValueError("Metrics output directory is None")
        Path(self._metrics_output_dir).mkdir(parents=True, exist_ok=True)

        # Try to get run_id from artifact metadata, fallback to directory-based deduction
        run_id = "grpc_server"
        if artifacts:
            # Use the first artifact's metadata to get run_id
            first_artifact = next(iter(artifacts.values()))
            try:
                metadata = first_artifact.metadata
                if metadata and "logger" in metadata and "run_id" in metadata["logger"]:
                    artifact_run_id = metadata["logger"]["run_id"]
                    if artifact_run_id:  # Check if it's not None or empty
                        run_id = artifact_run_id
                        log.info(f"Using run_id from artifact metadata: {run_id}")
                    else:
                        log.warning("run_id in artifact metadata is empty, falling back to directory-based deduction")
                        run_id = self._deduce_run_id_from_path(first_artifact)
                else:
                    log.warning("No run_id found in artifact metadata, falling back to directory-based deduction")
                    run_id = self._deduce_run_id_from_path(first_artifact)
            except Exception as e:
                log.warning(
                    f"Failed to read run_id from artifact metadata: {e}, falling back to directory-based deduction"
                )
                run_id = self._deduce_run_id_from_path(first_artifact)

        # Create metrics collector with individual parameters
        self._metrics_collector = MetricsCollector(mode="grpc_server", run_id=run_id)
        log.info(f"Metrics collection enabled, output directory: {self._metrics_output_dir}")

    def _deduce_run_id_from_path(self, artifact: Artifact) -> str:
        """
        Fallback method to deduce run_id from artifact file path.
        This is used when run_id is not available in artifact metadata.
        """
        artifact_path = Path(artifact.source)
        # Try to extract a meaningful run identifier from the artifact path
        # Look for common patterns like timestamps, experiment names, etc.
        if artifact_path.parent.name != "." and artifact_path.parent.name:
            if (
                artifact_path.parent.stem == "artifacts"
                and artifact_path.parent.parent.name != "."
                and artifact_path.parent.parent.name
            ):
                return artifact_path.parent.parent.name
            else:
                return artifact_path.parent.name
        else:
            return artifact_path.stem  # filename without extension

    def record_camera_render_time(self, render_time_ms: float, scene_id: str | None = None) -> None:
        if self._metrics_collector is None:
            return

        if self._warmup_rgb_counter >= self._warmup_skip:
            self._metrics_collector.set_metric_metadata("render_time_ms", timing_method="wall_clock")
            self._metrics_collector.save_metric(
                create_metric_sample("render_time_ms", render_time_ms), sequence_id=scene_id
            )
            self._save_metrics_deferred()

        self._warmup_rgb_counter += 1

    def record_ipc_write_time(self, ipc_write_ms: float) -> None:
        """Record RGB CUDA-IPC-write duration. Skips the same warmup frames as record_camera_render_time,
        but does not itself advance _warmup_rgb_counter — that happens once per render, in
        record_camera_render_time, regardless of delivery path."""
        if self._metrics_collector is None:
            return

        if self._warmup_rgb_counter > self._warmup_skip:
            self._metrics_collector.save_metric(create_metric_sample("ipc_write_ms", ipc_write_ms))
            self._save_metrics_deferred()

    def record_lidar_render_time(self, render_time_ms: float, scene_id: str | None = None) -> None:
        """Record LiDAR render time if metrics are enabled and warmup is complete."""
        if self._metrics_collector is None:
            return

        if self._warmup_lidar_counter >= self._warmup_skip:
            metric_sample = create_metric_sample("lidar_render_time_ms", render_time_ms)
            self._metrics_collector.set_metric_metadata("lidar_render_time_ms", timing_method="wall_clock")
            self._metrics_collector.save_metric(metric_sample, sequence_id=scene_id, is_lidar=True)
            self._save_metrics_deferred()

        self._warmup_lidar_counter += 1

    def record_batch_camera_render_time(
        self, render_time_ms: float, num_cameras: int, scene_id: str | None = None
    ) -> None:
        if self._metrics_collector is None:
            return

        if self._warmup_batch_rgb_counter >= self._warmup_skip:
            # A batch spans several cameras, so the aggregate belongs to no single sensor and is
            # left ungrouped. Only the scene is attributable.
            for name in ("batch_render_time_ms", "batch_render_time_per_camera_ms"):
                self._metrics_collector.set_metric_metadata(name, timing_method="wall_clock")
            self._metrics_collector.save_metric(
                create_metric_sample("batch_render_time_ms", render_time_ms), sequence_id=scene_id
            )
            if num_cameras > 0:
                self._metrics_collector.save_metric(
                    create_metric_sample("batch_render_time_per_camera_ms", render_time_ms / num_cameras),
                    sequence_id=scene_id,
                )
            self._save_metrics_deferred()

        self._warmup_batch_rgb_counter += 1

    def shutdown(self) -> None:
        """Cancel any pending timer and save final metrics before shutdown."""
        if self._metrics_collector is None:
            return

        with self._timer_lock:
            if self._save_timer is not None:
                self._save_timer.cancel()
                self._save_timer = None
        self._save_metrics()

    def record_duration(self, name: str, duration_ms: float) -> None:
        """Record a generic duration metric (e.g., gRPC ser/de) if metrics are enabled."""
        if self._metrics_collector is None:
            return

        # Every span this records is measured with perf_counter, so the file says so rather than
        # leaving a consumer to infer it from the name.
        self._metrics_collector.set_metric_metadata(name, timing_method="wall_clock")
        self._metrics_collector.save_metric(create_metric_sample(name, duration_ms))
        self._save_metrics_deferred()

    def _save_metrics(self) -> None:
        """Save collected metrics to the output directory."""
        if self._metrics_collector is None or self._metrics_output_dir is None:
            return

        try:
            # Write metrics to YAML file (the method creates metrics.yaml in the directory)
            self._metrics_collector.write_metrics_to_yaml(self._metrics_output_dir)
        except Exception:
            log.exception("Failed to save metrics")

    def _save_metrics_deferred(self) -> None:
        """Schedule a deferred save of metrics that gets delayed by 1 second each time it's called."""
        if self._metrics_collector is None or self._metrics_output_dir is None:
            return

        with self._timer_lock:
            # Cancel any existing timer
            if self._save_timer is not None:
                self._save_timer.cancel()

            # Start a new timer that will save metrics after 1 second of inactivity
            self._save_timer = threading.Timer(1.0, self._save_metrics)
            self._save_timer.start()


def is_local_scene_uri(uri: str) -> bool:
    """Return True if uri is a file:// URI or absolute local path (for on-the-fly scene registration).

    Public so the multiproc proxy's on-demand registration can share the exact single-process dispatch
    (file:// / absolute path = local; else remote) rather than reaching into a private serve.py name."""
    stripped = uri.strip()
    return stripped.lower().startswith("file://") or Path(stripped).is_absolute()


def resolve_local_scene_uri(local_uri: str) -> Path:
    """Resolve a ``file://`` URI or local path to an existing artifact file path (no load).

    Shared by the single-process `SensorSimService._register_local_scene` and the multiproc proxy's
    on-demand scene registration. Raises `ValueError` if the path does not exist or is not a file.
    """
    u = local_uri.strip()
    if u.lower().startswith("file://"):
        parsed = urlparse(u)
        path = Path(unquote(parsed.path))
    else:
        path = Path(u)
    resolved = path.resolve()
    if not resolved.exists():
        raise ValueError(f"{resolved} does not exist")
    if not resolved.is_file():
        raise ValueError(f"{resolved} is not a file")
    return resolved


def download_scene_to_cache(scene_id: str, uri: str, scene_cache: "SceneCache", downloader: "Downloader") -> Path:
    """Download a remote scene into the `SceneCache` and return its cached path (no load).

    Per-scene file locking (`SceneCache.lock_scene_path`) serializes concurrent requests so the scene is
    downloaded **once**. Used by the multiproc proxy's on-demand download (via `resolve_scene_artifact_path`);
    the single-process `SensorSimService._download_scene` instead downloads through the lease-based
    `SceneCache.add_and_acquire_scene` path.
    """
    # Fast (no file lock) check if scene is already cached. The entry can be evicted between the membership
    # check and the path lookup (TOCTOU), so treat the resulting KeyError as a miss and fall through to the
    # locked (re-)download path instead of surfacing it to the caller.
    if scene_cache.has_scene(scene_id):
        try:
            return scene_cache.get_scene_path(scene_id)
        except KeyError:
            pass

    with scene_cache.lock_scene_path(scene_id):
        # Re-download under the lock as many times as the entry keeps vanishing out from under us (cross-process
        # cache churn). Each pass either returns a resolved path or downloads a fresh copy; a `continue` re-enters
        # this locked path, treating a just-evicted entry as a retryable miss instead of surfacing a KeyError.
        while True:
            # Re-check under the lock in case the scene was added while we waited (same TOCTOU as above).
            if scene_cache.has_scene(scene_id):
                try:
                    return scene_cache.get_scene_path(scene_id)
                except KeyError:
                    pass

            temp_dir = Path(tempfile.gettempdir())
            temp_path = temp_dir / f"nurec_download_{scene_id}_{int(time.time())}_{os.getpid()}.tmp"
            try:
                download_start = time.time()
                total_bytes_downloaded = downloader.download(uri, temp_path)
                download_time = time.time() - download_start
                download_speed = total_bytes_downloaded / (1024 * 1024 * download_time) if download_time > 0 else 0
                log.info(
                    f"Downloaded {total_bytes_downloaded / (1024 * 1024):.2f} MB in {download_time:.2f}s ({download_speed:.2f} MB/s)"
                )
                try:
                    return scene_cache.add_scene(scene_id, temp_path)
                except ValueError:
                    # Cross-process re-register race: another registrant (e.g. a separate process sharing this
                    # cache dir, outside this process's in-memory lock) cached scene_id between our under-lock
                    # miss check and this add. add_scene raises ValueError *before* moving temp_path, so our
                    # download is unused — clean it up and return the path the winner published, rather than
                    # surfacing the benign race as an error.
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    try:
                        return scene_cache.get_scene_path(scene_id)
                    except KeyError:
                        # The winner's entry was itself evicted before we could read it (further cache churn).
                        # Treat it as a retryable miss and re-run the locked download rather than surfacing the
                        # KeyError as a confusing INVALID_ARGUMENT.
                        continue
            except Exception:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise


def resolve_scene_artifact_path(
    scene_url: str,
    scene_id: str,
    *,
    scene_cache: Optional["SceneCache"] = None,
    downloader: Optional["Downloader"] = None,
) -> Path:
    """Resolve a client-supplied ``(scene_url, scene_id)`` to a local artifact path **without loading it**.

    A ``file://`` / local path resolves directly; a remote URI downloads into the `SceneCache`. This is the
    shared seam used by the multiproc proxy's on-demand scene registration: the proxy resolves the
    path here and the owner loads + IPC-exports it lazily on the first render's attach barrier. (The
    single-process service instead registers the resolved artifact in `self._artifact_registrations`.)
    """
    if is_local_scene_uri(scene_url):
        return resolve_local_scene_uri(scene_url)
    if scene_cache is None or downloader is None:
        raise ValueError("a remote scene URL requires a scene_cache + downloader")
    return download_scene_to_cache(scene_id, scene_url, scene_cache, downloader)


class SceneDownloadInterceptor(grpc.ServerInterceptor):
    """Interceptor that checks for x-nre-scene-url header and downloads an external scene (http(s)://) or registers from local filesystem (file:// or local path).

    Applies to any unary RPC whose call metadata carries x-nre-scene-url;
    x-nre-scene-id is required alongside it. Scene ID and canonical URL identify
    the registration; the loaded-backend key additionally includes the content SHA.
    """

    def __init__(self):
        self.service = None

    def set_service(self, sensor_sim_service: SensorSimService):
        self.service = sensor_sim_service

    def _terminate(self, code: grpc.StatusCode, details: str) -> grpc.UnaryUnaryRpcCallback:
        def _terminator(ignored_request, context):
            context.abort(code, details)

        return grpc.unary_unary_rpc_method_handler(_terminator)

    def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], grpc.RpcMethodHandler],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        handler = continuation(handler_call_details)
        metadata = dict(handler_call_details.invocation_metadata)

        if SCENE_URL_HEADER not in metadata:
            return handler

        assert self.service is not None, "Service not set"
        if handler is None or handler.unary_unary is None:
            return handler

        scene_url = metadata[SCENE_URL_HEADER]

        try:
            scene_id = require_scene_id(metadata.get(SCENE_ID_HEADER))
            scene_key = SceneCacheKey.from_url(scene_id, scene_url)
        except ValueError as err:
            return self._terminate(grpc.StatusCode.INVALID_ARGUMENT, str(err))

        if not check_safe_scene_id(scene_id):
            return self._terminate(grpc.StatusCode.INVALID_ARGUMENT, f"Invalid scene ID: {scene_id}")

        is_local = is_local_scene_uri(scene_url)
        unary_unary = handler.unary_unary

        def invoke_with_scene(request, context):
            lease: SceneCacheLease | None = None
            try:
                try:
                    if is_local:
                        self.service._register_local_scene(scene_key, scene_url)
                    else:
                        lease = self.service._download_scene(scene_key, scene_url)
                except Exception as exc:
                    operation = "register local" if is_local else "download"
                    log.exception(
                        "Failed to %s scene scene_id=%s cache_key=%s",
                        operation,
                        scene_id,
                        scene_key.key_hash,
                    )
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        f"Failed to {operation} scene {scene_id!r}: {exc}",
                    )

                return unary_unary(request, context)
            finally:
                if lease is not None:
                    self.service._release_scene_lease(lease)

        return grpc.unary_unary_rpc_method_handler(
            invoke_with_scene,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )


class MetricsInterceptor(grpc.ServerInterceptor):
    """Interceptor to measure decode/encode and end-to-end durations per RPC."""

    def __init__(self) -> None:
        self.metrics_manager: Optional[ServerMetricsManager] = None

    def set_metrics_manager(self, manager: ServerMetricsManager) -> None:
        self.metrics_manager = manager

    def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], grpc.RpcMethodHandler],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        handler = continuation(handler_call_details)
        # Only wrap unary-unary for our service methods
        if handler is None or handler.unary_unary is None:
            return handler

        # Determine method-specific metric names
        method_name = handler_call_details.method.split("/")[-1]
        decode_metric = f"grpc.decode.{method_name}_ms"
        encode_metric = f"grpc.encode.{method_name}_ms"
        e2e_sent_metric = f"grpc.e2e_sent.{method_name}_ms"

        metrics = self.metrics_manager

        # Per-RPC E2E start captured in deserializer and shared with close callback.
        # Using a closure keeps state per request without thread-local/global storage.
        e2e_start = 0.0

        def request_deserializer(raw: bytes):
            nonlocal e2e_start
            if metrics is not None:
                # Capture earliest per-RPC start (before deserialization).
                e2e_start = time.perf_counter()

            obj = handler.request_deserializer(raw)
            if metrics is not None:
                metrics.record_duration(decode_metric, (time.perf_counter() - e2e_start) * 1000.0)
            return obj

        def response_serializer(resp):
            nonlocal e2e_start
            t0 = 0.0
            if metrics is not None:
                t0 = time.perf_counter()
            out = handler.response_serializer(resp)
            if metrics is not None:
                t1 = time.perf_counter()
                if e2e_start:
                    metrics.record_duration(e2e_sent_metric, (t1 - e2e_start) * 1000.0)

                metrics.record_duration(encode_metric, (t1 - t0) * 1000.0)

            return out

        return grpc.unary_unary_rpc_method_handler(
            handler.unary_unary,
            request_deserializer=request_deserializer,
            response_serializer=response_serializer,
        )


@dataclass
class PendingBatchRGBEncode:
    camera_name: str
    image_u8: torch.Tensor
    request: RGBRenderRequest


@dataclass
class PendingCameraRenderOutput:
    image_u8: torch.Tensor
    request: CameraRenderRequest
    product: CameraRenderProduct
    semantics_class_mapping: Optional[List[Tuple[int, str, Tuple[int, ...]]]]
    distance_colormap_vmax_m: Optional[float]


BATCH_RENDER_REQUEST_CHUNK_SIZE = 16


class SensorSimService(SensorsimServiceServicer):
    def __init__(
        self,
        server: Optional[grpc.Server],  # None during tests
        artifacts_glob: Optional[str],
        ray_chunk_size: int,
        egocar_hoods_dir: str | None = None,
        downloader: Optional[Downloader] = None,  # None during tests
        scene_cache: Optional[SceneCache] = None,  # None during tests
        cache_size: int = 10,
        max_backends_per_scene: Optional[int] = None,
        metrics_output_dir: Optional[str] = None,
        enable_editing_actors: Optional[bool] = False,
        server_config: Optional[GrpcServerConfig] = None,
        tracks_timestamp_ray_stride: int = 1,
    ) -> None:
        self.server = server
        self.downloader = downloader
        self.scene_cache = scene_cache
        self.ray_chunk_size = ray_chunk_size
        self.enable_editing_actors = enable_editing_actors
        # Inference-only ray-stride for per-track timestamp estimation; threaded to Backend.from_artifact.
        self._tracks_timestamp_ray_stride = tracks_timestamp_ray_stride
        self.version_id = self._create_version_id()
        self.server_config_map = self.build_server_config_map(server_config)
        # Surface the active ray-stride in the reported config (it is popped before
        # GrpcServerConfig, so it would otherwise be invisible to get_server_config).
        self.server_config_map["tracks_timestamp_ray_stride"] = str(self._tracks_timestamp_ray_stride)
        # Topology so a client never infers the serving mode from absent keys. This is the
        # single-process server's response; the multiprocess proxy overrides it with
        # `serving.mode=multiprocess` from its own effective config map.
        self.server_config_map["serving.mode"] = "single_process"
        self.server_config = server_config
        self._allowed_image_formats = allowed_image_format_values(
            server_config.allowed_image_formats if server_config is not None else None
        )
        log.info(f"Identifying as {self.version_id=}")

        if artifacts_glob is None:
            artifacts_list = []
        else:
            artifacts_list = Artifact.discover_from_glob(artifacts_glob)

        # Check for duplicates -- the desired abstraction is to index by scene_id
        scene_id_to_artifact_paths: dict[str, list[Path]] = {}
        for artifact in artifacts_list:
            scene_id_to_artifact_paths.setdefault(artifact.scene_id, []).append(artifact.source)
        duplicates = {artifact: paths for artifact, paths in scene_id_to_artifact_paths.items() if len(paths) >= 2}
        if duplicates:
            raise AssertionError(f"Duplicate scene IDs found. Duplicates (scene_id: artifact paths): {duplicates}.")

        self._artifact_registry_lock = threading.RLock()
        self._artifact_registrations: dict[str, ArtifactRegistration] = {}
        for artifact in artifacts_list:
            scene_key = SceneCacheKey(scene_id=artifact.scene_id)
            registration = ArtifactRegistration(artifact=artifact, downloaded=False, scene_key=scene_key)
            # Finalize exact-content identity before the service exposes this
            # registration. This keeps full-file I/O outside the registry lock.
            registration.backend_cache_key
            self._artifact_registrations[scene_key.key_hash] = registration

        log.info(
            "Available scenes: %s.",
            [registration.scene_key.scene_id for registration in self._artifact_registrations.values()],
        )

        self.harmonizer_enabled = False
        # Temporal harmonizer: per-session frame history, keyed by server-generated session id.
        self.harmonizer_temporal_enabled = False
        self._harmonizer_sessions: dict[str, HarmonizerState] = {}
        self._harmonizer_sessions_lock = threading.Lock()
        self._max_harmonizer_sessions = 32
        # Loaded off the startup path so the port binds first; see set_harmonizer.
        self._harmonizer_model: Any = None
        self._harmonizer_ready = threading.Event()
        self._harmonizer_error: Optional[BaseException] = None

        self.egocar_rig_bank = (
            EgocarRigBank.load_from_dir(egocar_hoods_dir) if egocar_hoods_dir else EgocarRigBank.empty()
        )
        log.info(f"Available egocar masks: {self.egocar_rig_bank}.")

        self.nvjpeg_batch_encoder: Any = None
        self.nvjpeg_batch_encoder_lock = threading.Lock()

        # This pool limits loaded rendering backends, independently of the on-disk
        # SceneCache. The per-scene limit makes excess requests wait for capacity
        # rather than letting one scene construct every globally available
        # backend. The global size remains the aggregate bound.
        self.backend_cache = BackendCache(maxsize=cache_size, max_per_key=max_backends_per_scene)

        # Initialize metrics manager
        self._metrics_manager = ServerMetricsManager(metrics_output_dir, self._artifact_snapshot())

        # CUDA IPC transport: lazy-init-on-first-use, gate-under-lock.
        self._ipc_mgr: Optional[Any] = None
        self._ipc_mgr_lock = threading.Lock()
        self._ipc_open_sessions = 0
        # NRE_CUDA_IPC_MAX_SESSIONS alone doesn't meaningfully bound GPU memory: each session
        # can independently request up to --cuda-ipc-max-buffer-count buffers of up to
        # kMaxRequestedDimension^2*3 bytes each (native IpcManager.open_session), so the count
        # cap allows a reachable client to still pin most of a device's memory (or several
        # devices', across sessions) well within that session count. _ipc_reserved_bytes tracks
        # the aggregate byte footprint of currently-open sessions against NRE_CUDA_IPC_MAX_TOTAL_BYTES
        # (see open_ipc_session); _ipc_session_bytes records each open session's own footprint so
        # close_ipc_session can give back exactly what that session reserved.
        self._ipc_reserved_bytes = 0
        self._ipc_session_bytes: dict[str, int] = {}
        self._ipc_device_index: Optional[int] = None

    def _artifact_snapshot(self) -> dict[str, Artifact]:
        with self._artifact_registry_lock:
            return {
                registration.scene_key.scene_id: registration.artifact
                for registration in self._artifact_registrations.values()
            }

    def _remove_downloaded_registrations_locked(self, scene_ids: list[str]) -> None:
        for scene_id in scene_ids:
            registration = self._artifact_registrations.get(scene_id)
            if registration is not None and registration.downloaded:
                del self._artifact_registrations[scene_id]

    def _release_scene_lease(self, lease: SceneCacheLease) -> None:
        with self._artifact_registry_lock:
            self._remove_downloaded_registrations_locked(lease.release())

    @staticmethod
    def build_server_config_map(server_config: Optional[GrpcServerConfig]) -> dict[str, str]:
        """Flatten a `GrpcServerConfig` into the string discovery map reported by `get_server_config`.

        Public so the multiproc proxy can build the same map from its own effective config rather than
        reaching into a private serve.py name."""
        if server_config is None:
            return {}

        raw_config_dict = server_config.model_dump()

        def stringify_value(value: Any) -> str:
            if value is None:
                return "null"
            if isinstance(value, Enum):
                return str(value.value)
            if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                return str(value)
            else:
                raise ValueError(f"Unsupported config type: {type(value)}")

        config_map = {key: stringify_value(value) for key, value in raw_config_dict.items()}

        # Capability discovery for DynamicObject.aspect_state: an older server drops the
        # unknown field silently, so clients gate on this key. "stateless-v1" means the
        # server keeps no aspect state between requests.
        #
        # Belongs in this method, not on the instance map, so the multiprocess proxy
        # advertises it too. Gated on enable_editing_actors because aspect_state rides
        # DynamicObject, which is rejected outright without that flag.
        if server_config.enable_editing_actors:
            config_map["asset_edits.aspect_state"] = "stateless-v1"
        return config_map

    @ScopedTimer()
    def _encode_image_bytes(self, image_u8: torch.Tensor, request: RGBRenderRequest | CameraRenderRequest) -> bytes:
        match request.image_format:
            case ImageFormat.UNDEFINED:
                raise TypeError("Request did not specify image format (default value found).")
            case ImageFormat.PNG:
                image = Image.fromarray(image_u8.cpu().numpy())
                img_io = io.BytesIO()
                image.save(img_io, "PNG")
                img_io.seek(0)
                return img_io.getvalue()
            case ImageFormat.JPEG:
                return self._encode_jpeg_images([image_u8], [request])[0].image_bytes
            case ImageFormat.JPEG2000:
                raise TypeError("JPEG2k is not currently supported.")
            case ImageFormat.RGB_UINT8_PLANAR:
                image_u8_np = image_u8.cpu().numpy()
                planar_image_np = image_u8_np.transpose(2, 0, 1)
                return planar_image_np.tobytes()
            case _:
                raise TypeError(f"Unknown image format requested ({request.image_format=}).")

    @ScopedTimer()
    def encode_image(self, image_u8: torch.Tensor, request: RGBRenderRequest) -> RGBRenderReturn:
        return RGBRenderReturn(image_bytes=self._encode_image_bytes(image_u8, request))

    def _encode_jpeg_images(
        self,
        images_u8: list[torch.Tensor],
        requests: list[RGBRenderRequest | CameraRenderRequest],
        batch_encoder: Any = None,
    ) -> list[RGBRenderReturn]:
        if not images_u8:
            return []
        if len(images_u8) != len(requests):
            raise ValueError("Number of images and requests must match.")

        image_quality = self._validated_jpeg_quality(requests[0])
        if any(self._validated_jpeg_quality(request) != image_quality for request in requests[1:]):
            raise ValueError("Batched JPEG encoding requires all images to use the same image quality.")

        contiguous_images = [
            image_u8.contiguous() for image_u8 in images_u8
        ]  # nvjpeg requires contiguous tensors and e.g. harmonizer can return non-contiguous tensors

        if batch_encoder is None:
            batch_encoder = self._get_nvjpeg_batch_encoder()
        encoded_images = batch_encoder.encode(contiguous_images, image_quality)

        if len(encoded_images) != len(contiguous_images):
            raise RuntimeError(
                f"nvJPEG batch encode returned {len(encoded_images)} outputs for {len(contiguous_images)} inputs."
            )

        return [RGBRenderReturn(image_bytes=encoded_image) for encoded_image in encoded_images]

    @staticmethod
    def _validated_jpeg_quality(request: RGBRenderRequest | CameraRenderRequest) -> int:
        image_quality = float(request.image_quality)
        if not math.isfinite(image_quality):
            raise ValueError("JPEG image quality must be finite.")
        if image_quality == 0:
            raise ValueError("Request did not specify image quality (found default of 0).")
        if image_quality < 1 or image_quality > 100:
            raise ValueError("JPEG image quality must be in [1, 100].")
        return int(image_quality)

    def _get_nvjpeg_batch_encoder(self) -> Any:
        if self.nvjpeg_batch_encoder is not None:
            return self.nvjpeg_batch_encoder

        with self.nvjpeg_batch_encoder_lock:
            if self.nvjpeg_batch_encoder is not None:
                return self.nvjpeg_batch_encoder

            from nurec.grpc.nvjpeg_batch_encoder import NvjpegBatchEncoder

            self.nvjpeg_batch_encoder = NvjpegBatchEncoder()
            return self.nvjpeg_batch_encoder

    def _encode_jpeg_images_with_failure_isolation(
        self,
        images_u8: list[torch.Tensor],
        requests: list[RGBRenderRequest | CameraRenderRequest],
    ) -> list[RGBRenderReturn | Exception]:
        if not images_u8:
            return []
        batch_encoder = self._get_nvjpeg_batch_encoder()
        max_batch_size = int(batch_encoder.max_batch_size)
        if max_batch_size <= 0:
            raise RuntimeError(f"nvJPEG encoder reported invalid max batch size {max_batch_size}.")

        outcomes: list[RGBRenderReturn | Exception] = []
        for chunk_start in range(0, len(images_u8), max_batch_size):
            chunk_end = chunk_start + max_batch_size
            outcomes.extend(
                self._encode_jpeg_image_chunk_with_failure_isolation(
                    images_u8[chunk_start:chunk_end],
                    requests[chunk_start:chunk_end],
                    batch_encoder,
                )
            )
        return outcomes

    def _encode_jpeg_image_chunk_with_failure_isolation(
        self,
        images_u8: list[torch.Tensor],
        requests: list[RGBRenderRequest | CameraRenderRequest],
        batch_encoder: Any,
    ) -> list[RGBRenderReturn | Exception]:
        try:
            return list(self._encode_jpeg_images(images_u8, requests, batch_encoder))
        except NvjpegImageEncodingError as exc:
            if len(images_u8) <= 1:
                return [exc]

            midpoint = len(images_u8) // 2
            return self._encode_jpeg_image_chunk_with_failure_isolation(
                images_u8[:midpoint],
                requests[:midpoint],
                batch_encoder,
            ) + self._encode_jpeg_image_chunk_with_failure_isolation(
                images_u8[midpoint:],
                requests[midpoint:],
                batch_encoder,
            )

    def _finish_pending_batch_rgb_results(
        self,
        pending_results: list[BatchRGBRenderReturnItem | PendingBatchRGBEncode],
    ) -> list[BatchRGBRenderReturnItem]:
        results: list[BatchRGBRenderReturnItem] = []
        pending_jpegs: list[PendingBatchRGBEncode] = []
        pending_jpeg_result_indices: list[int] = []

        for pending_result in pending_results:
            if isinstance(pending_result, BatchRGBRenderReturnItem):
                results.append(pending_result)
                continue

            if pending_result.request.image_format == ImageFormat.JPEG:
                pending_jpeg_result_indices.append(len(results))
                pending_jpegs.append(pending_result)
                results.append(BatchRGBRenderReturnItem(camera_name=pending_result.camera_name, success=False))
                continue

            try:
                encoded = self.encode_image(pending_result.image_u8, pending_result.request)
                results.append(
                    BatchRGBRenderReturnItem(
                        camera_name=pending_result.camera_name,
                        result=encoded,
                        success=True,
                    )
                )
            except Exception as exc:
                results.append(self._batch_rgb_error_item(pending_result.camera_name, exc))

        if not pending_jpegs:
            return results

        quality_groups: dict[int, list[tuple[int, PendingBatchRGBEncode]]] = {}
        for result_index, pending_jpeg in zip(pending_jpeg_result_indices, pending_jpegs):
            try:
                quality = self._validated_jpeg_quality(pending_jpeg.request)
            except ValueError as exc:
                results[result_index] = self._batch_rgb_error_item(
                    pending_jpeg.camera_name,
                    exc,
                )
                continue
            quality_groups.setdefault(quality, []).append((result_index, pending_jpeg))

        for quality_group in quality_groups.values():
            encoded_outcomes = self._encode_jpeg_images_with_failure_isolation(
                [pending_jpeg.image_u8 for _, pending_jpeg in quality_group],
                [pending_jpeg.request for _, pending_jpeg in quality_group],
            )
            for (result_index, pending_jpeg), encoded_outcome in zip(
                quality_group,
                encoded_outcomes,
                strict=True,
            ):
                if isinstance(encoded_outcome, Exception):
                    results[result_index] = self._batch_rgb_error_item(pending_jpeg.camera_name, encoded_outcome)
                else:
                    results[result_index] = BatchRGBRenderReturnItem(
                        camera_name=pending_jpeg.camera_name,
                        result=encoded_outcome,
                        success=True,
                    )

        return results

    def _maybe_apply_harmonizer(
        self,
        raster_f32: torch.Tensor,
        img_size: tuple[int, int],
        request: RGBRenderRequest | CameraRenderRequest,
        harmonizer_state: Optional[HarmonizerState] = None,
        context: grpc.ServicerContext | None = None,
    ) -> torch.Tensor:
        """Apply Harmonizer postprocessing when requested and enabled on the server."""
        if not request.apply_harmonizer:
            return raster_f32

        if not (self.harmonizer_enabled or self.harmonizer_temporal_enabled):
            raise HarmonizerNotEnabledError("Harmonizer was requested but is not enabled on the server")

        harmonizer_model = self._await_harmonizer_model(context)
        flat_raster = rearrange(raster_f32, "h w c -> (h w) c")
        if self.harmonizer_enabled:
            # The non-temporal wrapper's third argument controls legacy colour transfer.
            flat_raster = harmonizer_model(flat_raster, img_size, False)
        elif self.harmonizer_temporal_enabled:
            if harmonizer_state is None:
                raise RuntimeError("temporal Harmonizer request was not associated with a resolved session")
            flat_raster = harmonizer_model(flat_raster, img_size, harmonizer_state)
        else:
            raise AssertionError("an enabled Harmonizer mode is required")
        return rearrange(flat_raster, "(h w) c -> h w c", h=img_size[0])

    @staticmethod
    def _camera_request_includes_color_product(request: CameraRenderRequest) -> bool:
        """Return whether a CameraRender request produces colour (the default product)."""
        return not request.render_products or CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR in request.render_products

    def _render_result_to_image_and_mapping(
        self,
        raster_tensor: torch.Tensor,
        product: int,
        distance_colormap_vmax_m: Optional[float],
        request: RGBRenderRequest | CameraRenderRequest,
        semantic_class_colors: Optional[np.ndarray],
        *,
        semantic_class_names: Optional[Dict[int, str]],
        harmonizer_state: Optional[HarmonizerState] = None,
        context: grpc.ServicerContext | None = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[int, str, Tuple[int, ...]]]]]:
        semantics_class_mapping: Optional[List[Tuple[int, str, Tuple[int, ...]]]] = None

        if product == CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED:
            assert distance_colormap_vmax_m is not None, "caller must resolve vmax for distance product"
            distance_np = raster_tensor.cpu().numpy()
            raster_u8 = scalar2img(distance_np, vmin=0.0, vmax=distance_colormap_vmax_m, cmap="turbo")
            image_u8 = torch.from_numpy(raster_u8).to(device=raster_tensor.device)
        elif product == CameraRenderProduct.CAMERA_RENDER_PRODUCT_SEMANTICS_COLORIZED:
            sem_np = raster_tensor.cpu().numpy().astype(np.int64)
            names = semantic_class_names
            if names:
                uncapped = max(names) + 1
                semantics_lut_size = min(uncapped, _SEMANTICS_COLOR_LUT_MAX_SIZE)
                if uncapped > _SEMANTICS_COLOR_LUT_MAX_SIZE:
                    log.warning(
                        "Artifact semantic_class_names contains class id %d (>= cap %d); LUT truncated.",
                        uncapped - 1,
                        _SEMANTICS_COLOR_LUT_MAX_SIZE,
                    )
            else:
                semantics_lut_size = SEMANTICS_COLOR_LUT_DEFAULT_SIZE
            color_remap = build_semantics_class_colors(semantics_lut_size, semantic_class_colors)
            if sem_np.size and int(sem_np.max()) >= 0:
                if int(sem_np.max()) >= semantics_lut_size:
                    log.warning(
                        "Semantic class id(s) >= %d; clipping to LUT for visualization.",
                        semantics_lut_size,
                    )
                # Negative ids (e.g. -1 ray-miss / unmapped sentinels) get colored black to
                # distinguish them from class 0 — np.clip alone would collapse them onto class 0.
                neg_mask = sem_np < 0
                sem_safe = np.clip(sem_np, 0, semantics_lut_size - 1)
                raster_u8 = sem2img(sem_safe, color_remap)
                raster_u8[neg_mask] = 0
            else:
                raster_u8 = np.zeros((*sem_np.shape, 3), dtype=np.uint8)
            image_u8 = torch.from_numpy(raster_u8).to(device=raster_tensor.device)
            semantics_class_mapping = [
                (
                    i,
                    (names.get(i, f"class_{i}") if names else f"class_{i}"),
                    tuple(int(c) for c in color_remap[i]),
                )
                for i in range(semantics_lut_size)
            ]
        else:
            raster_f32 = self._apply_color_postprocessing_f32(raster_tensor, request, harmonizer_state, context)
            image_u8 = (raster_f32 * 255).clamp(0, 255).to(torch.uint8)

        return image_u8, semantics_class_mapping

    def _apply_color_postprocessing_f32(
        self,
        raster_f32: torch.Tensor,
        request: RGBRenderRequest | CameraRenderRequest,
        harmonizer_state: Optional[HarmonizerState],
        context: grpc.ServicerContext | None,
    ) -> torch.Tensor:
        """Harmonizer + egocar-mask postprocessing shared by every CAMERA_RENDER_PRODUCT_COLOR
        delivery path -- encoded image_bytes (via _render_result_to_image_and_mapping, which
        converts to uint8 after this) and CUDA IPC alike (which hands this straight to the pack
        kernel) -- applied while the raster is still float32, before either path's own
        float->uint8 conversion."""
        img_size = (int(raster_f32.shape[0]), int(raster_f32.shape[1]))
        raster_f32 = self._maybe_apply_harmonizer(raster_f32, img_size, request, harmonizer_state, context)
        if (egocar_mask := self.egocar_rig_bank.select_from_request(request)) is not None:
            raster_f32 = egocar_mask.overlay_on_image(raster_f32)
        return raster_f32

    def _apply_bounding_box_overlay_if_requested(
        self,
        image_u8: torch.Tensor,
        request: RGBRenderRequest | CameraRenderRequest,
        backend: Backend,
        *,
        model_actors_snapshot: Optional[ActorsSnapshot] = None,
    ) -> torch.Tensor:
        """Draw cuboid wireframes when :class:`CameraRenderRequest` requests an overlay source.

        ``model_actors_snapshot`` is the MODEL-source snapshot captured by
        ``Backend.render_camera_request`` under ``_render_lock``. Threading it through here lets
        the overlay reflect the same actor state the render saw even if an ``edit_actors`` call
        lands between render and overlay. Required when ``overlay_bounding_boxes`` is MODEL.
        """
        src = _overlay_bounding_box_source(request)
        if src == OverlayBoundingBoxSource.OVERLAY_BOUNDING_BOX_SOURCE_UNSPECIFIED:
            return image_u8
        if src == OverlayBoundingBoxSource.OVERLAY_BOUNDING_BOX_SOURCE_ARTIFACT:
            actor_tracks = backend._get_artifact_actor_tracks()
            if actor_tracks.num_tracks() == 0:
                raise ValueError(
                    "Bounding box overlay requested with source OVERLAY_BOUNDING_BOX_SOURCE_ARTIFACT "
                    "but the resolved actor tracks contain no tracks."
                )
            actors_snapshot = actor_tracks.get_snapshot_at_frame(request.frame_start_us, request.frame_end_us)
        elif src == OverlayBoundingBoxSource.OVERLAY_BOUNDING_BOX_SOURCE_MODEL:
            if model_actors_snapshot is None:
                raise RuntimeError(
                    "OVERLAY_BOUNDING_BOX_SOURCE_MODEL requested but the caller did not pre-resolve "
                    "the model actors snapshot via Backend.render_camera_request."
                )
            actors_snapshot = model_actors_snapshot
        else:
            # Unknown enum value (e.g., a newer proto value this server doesn't recognize); surface
            # as INVALID_ARGUMENT instead of silently no-op'ing the overlay the client asked for.
            try:
                name = OverlayBoundingBoxSource.Name(src)
            except ValueError:
                name = "<unknown>"
            raise ValueError(f"Unknown overlay_bounding_boxes value: {name} ({int(src)}).")
        if actors_snapshot is None or actors_snapshot.num_actors() == 0:
            return image_u8
        if actors_snapshot.cuboid_dims is None:
            return image_u8
        camera = backend.camera_bank._get_camera(request.camera_intrinsics)
        img_size = (int(image_u8.shape[0]), int(image_u8.shape[1]))
        cam_params = transform_intrinsics_to_resolution(
            camera.camera_model_parameters,
            (int(img_size[1]), int(img_size[0])),
        )
        camera_to_world_tquat = torch.stack(
            [
                grpc_pose_to_tquat(request.sensor_pose.start_pose),
                grpc_pose_to_tquat(request.sensor_pose.end_pose),
            ]
        )
        bbox_color_proto = (
            request.bbox_color if isinstance(request, CameraRenderRequest) and request.HasField("bbox_color") else None
        )
        if bbox_color_proto is not None:
            raw_rgb = (bbox_color_proto.r, bbox_color_proto.g, bbox_color_proto.b)
            if any(c > 255 for c in raw_rgb):
                raise ValueError(f"BboxColor channels must be in [0, 255]; got {raw_rgb}")
            box_color_rgb: Tuple[int, int, int] = (int(raw_rgb[0]), int(raw_rgb[1]), int(raw_rgb[2]))
        else:
            box_color_rgb = (0, 255, 0)
        image_np = image_u8.detach().cpu().numpy()
        out_np = overlay_bounding_boxes(
            image_np,
            cam_params,
            camera_to_world_tquat,
            actors_snapshot,
            box_color_rgb=box_color_rgb,
        )
        return torch.from_numpy(np.asarray(out_np, dtype=np.uint8)).to(device=image_u8.device)

    def encode_camera_render_output(
        self,
        image_u8: torch.Tensor,
        request: CameraRenderRequest,
        product: CameraRenderProduct,
        *,
        semantics_class_mapping: Optional[List[Tuple[int, str, Tuple[int, ...]]]],
        distance_colormap_vmax_m: Optional[float],
    ) -> CameraRenderOutput:
        pending_output = PendingCameraRenderOutput(
            image_u8=image_u8,
            request=request,
            product=product,
            semantics_class_mapping=semantics_class_mapping,
            distance_colormap_vmax_m=distance_colormap_vmax_m,
        )
        return self._camera_render_output_from_bytes(
            pending_output,
            self._encode_image_bytes(image_u8, request),
        )

    @staticmethod
    def _camera_render_output_from_bytes(
        pending_output: PendingCameraRenderOutput,
        image_bytes: bytes,
    ) -> CameraRenderOutput:
        height, width = int(pending_output.image_u8.shape[0]), int(pending_output.image_u8.shape[1])
        out_pb = CameraRenderOutput(
            image_bytes=image_bytes,
            image_height=height,
            image_width=width,
            render_product=pending_output.product,
        )
        if pending_output.semantics_class_mapping:
            for class_index, class_name, rgb in pending_output.semantics_class_mapping:
                out_pb.semantics_class_mapping.append(
                    SemanticsClassEntry(
                        class_index=class_index,
                        class_name=class_name,
                        color_r=rgb[0],
                        color_g=rgb[1],
                        color_b=rgb[2],
                    )
                )
        if pending_output.distance_colormap_vmax_m is not None:
            out_pb.distance_colormap_vmax_m = pending_output.distance_colormap_vmax_m
        return out_pb

    def _encode_pending_camera_render_output_bytes(
        self,
        pending_outputs: list[PendingCameraRenderOutput],
    ) -> list[bytes | Exception]:
        encoded_outcomes: list[bytes | Exception | None] = [None] * len(pending_outputs)
        jpeg_quality_groups: dict[int, list[int]] = {}

        for index, pending_output in enumerate(pending_outputs):
            if pending_output.request.image_format == ImageFormat.JPEG:
                try:
                    quality = self._validated_jpeg_quality(pending_output.request)
                except ValueError as exc:
                    encoded_outcomes[index] = exc
                    continue
                jpeg_quality_groups.setdefault(quality, []).append(index)
                continue

            try:
                encoded_outcomes[index] = self._encode_image_bytes(
                    pending_output.image_u8,
                    pending_output.request,
                )
            except (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError):
                raise
            except Exception as exc:
                encoded_outcomes[index] = exc

        for group_indices in jpeg_quality_groups.values():
            encoded_outcomes_for_group = self._encode_jpeg_images_with_failure_isolation(
                [pending_outputs[index].image_u8 for index in group_indices],
                [pending_outputs[index].request for index in group_indices],
            )
            for index, jpeg_outcome in zip(group_indices, encoded_outcomes_for_group, strict=True):
                if isinstance(jpeg_outcome, Exception):
                    encoded_outcomes[index] = jpeg_outcome
                else:
                    encoded_outcomes[index] = jpeg_outcome.image_bytes

        finalized_outcomes: list[bytes | Exception] = []
        for pending_outcome in encoded_outcomes:
            if pending_outcome is None:
                raise AssertionError("Pending camera output was not encoded.")
            finalized_outcomes.append(pending_outcome)
        return finalized_outcomes

    def _finish_pending_camera_render_outputs(
        self,
        pending_outputs: list[PendingCameraRenderOutput],
    ) -> list[CameraRenderOutput]:
        encoded_outcomes = self._encode_pending_camera_render_output_bytes(pending_outputs)
        outputs: list[CameraRenderOutput] = []
        for pending_output, encoded_outcome in zip(pending_outputs, encoded_outcomes):
            if isinstance(encoded_outcome, Exception):
                raise encoded_outcome
            outputs.append(self._camera_render_output_from_bytes(pending_output, encoded_outcome))
        return outputs

    def _prepare_camera_render_outputs_from_request(
        self,
        backend: Backend,
        request: CameraRenderRequest,
        *,
        enable_editing_actors: bool,
        harmonizer_state: Optional[HarmonizerState] = None,
        context: grpc.ServicerContext | None = None,
    ) -> list[PendingCameraRenderOutput]:
        products = list(request.render_products) or [CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR]
        if len(set(products)) != len(products):
            raise ValueError("render_products entries must be unique.")

        is_color_requested = CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR in products
        if request.insert_ego_mask and not is_color_requested:
            raise ValueError("insert_ego_mask requires CAMERA_RENDER_PRODUCT_COLOR among render_products.")

        frame, _, model_overlay_snapshot = backend.render_camera_request(
            request, enable_editing_actors=enable_editing_actors
        )
        semantic_colors = backend.source_artifact.semantic_class_colors
        semantic_names_list = backend.source_artifact.semantic_class_names

        pending_outputs: list[PendingCameraRenderOutput] = []
        for product in products:
            if product == CameraRenderProduct.CAMERA_RENDER_PRODUCT_SEMANTICS_COLORIZED:
                # render_camera_request already raised MissingRenderOutputError (-> FAILED_PRECONDITION)
                # if the model lacks the semantic head, so the field is populated here.
                assert frame.semantic_image is not None
                tensor_result = frame.semantic_image
                distance_colormap_vmax_m: Optional[float] = None
                names_arg = semantic_names_list
            elif product == CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED:
                assert frame.distance_image is not None  # guaranteed by render_camera_request (see above)
                tensor_result = frame.distance_image
                distance_colormap_vmax_m = (
                    float(request.distance_colormap_vmax_m)
                    if request.HasField("distance_colormap_vmax_m")
                    else float(backend.distance_vmax_m)
                )
                names_arg = None
            elif product == CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR:
                if frame.color_image is None:
                    raise ValueError("Renderer did not return a color image.")
                tensor_result = frame.color_image
                distance_colormap_vmax_m = None
                names_arg = None
            else:
                raise ValueError(f"Unsupported render product: {product}.")

            image_u8, semantics_mapping = self._render_result_to_image_and_mapping(
                tensor_result,
                product,
                distance_colormap_vmax_m,
                request,
                semantic_colors,
                semantic_class_names=names_arg,
                harmonizer_state=harmonizer_state,
                context=context,
            )

            image_u8 = self._apply_bounding_box_overlay_if_requested(
                image_u8, request, backend, model_actors_snapshot=model_overlay_snapshot
            )

            pending_outputs.append(
                PendingCameraRenderOutput(
                    image_u8=image_u8,
                    request=request,
                    product=product,
                    semantics_class_mapping=semantics_mapping,
                    distance_colormap_vmax_m=(
                        distance_colormap_vmax_m
                        if product == CameraRenderProduct.CAMERA_RENDER_PRODUCT_DISTANCE_COLORIZED
                        else None
                    ),
                )
            )

        return pending_outputs

    def _camera_render_return_from_request(
        self,
        backend: Backend,
        request: CameraRenderRequest,
        *,
        enable_editing_actors: bool,
        harmonizer_state: Optional[HarmonizerState] = None,
        context: grpc.ServicerContext | None = None,
    ) -> CameraRenderReturn:
        if request.use_cuda_ipc:
            # require_cuda_ipc_camera_request (called by render_camera before this) has already
            # confirmed this request asks for a single CAMERA_RENDER_PRODUCT_COLOR output with no
            # bounding-box overlay -- the only shape the IPC path can deliver, since the fused
            # pack kernel writes straight from the float32 raster with no host-visible image for
            # a second, differently-typed product or an overlay to draw onto.
            frame, _, _ = backend.render_camera_request(request, enable_editing_actors=enable_editing_actors)
            if frame.color_image is None:
                raise ValueError("Renderer did not return a color image.")
            raster_f32 = self._apply_color_postprocessing_f32(frame.color_image, request, harmonizer_state, context)
            ipc_start = time.perf_counter()
            with _nvtx_range("serve_grpc.render_camera.ipc_write"):
                output = self._ipc_write_camera_color(raster_f32, request)
            self._metrics_manager.record_ipc_write_time((time.perf_counter() - ipc_start) * 1000.0)
            return CameraRenderReturn(camera_render_outputs=[output])

        pending_outputs = self._prepare_camera_render_outputs_from_request(
            backend,
            request,
            enable_editing_actors=enable_editing_actors,
            harmonizer_state=harmonizer_state,
            context=context,
        )
        return CameraRenderReturn(camera_render_outputs=self._finish_pending_camera_render_outputs(pending_outputs))

    def set_harmonizer(
        self,
        harmonizer_enabled: bool,
        harmonizer_url: str,
        harmonizer_cache_dir: str,
        model_filename: str,
        harmonizer_resolution: tuple[int, int],
        *,
        harmonizer_temporal_enabled: bool = False,
        harmonizer_temporal_url: str = HARMONIZER_TEMPORAL_MODEL_URL,
        harmonizer_temporal_model_filename: str = HARMONIZER_TEMPORAL_MODEL_FILENAME,
        harmonizer_temporal_resolution: tuple[int, int] = HARMONIZER_TEMPORAL_RESOLUTION,
        harmonizer_temporal_cosmos_base_model_url: str = HARMONIZER_TEMPORAL_COSMOS_BASE_MODEL_URL,
        harmonizer_temporal_cosmos_tokenizer_url: str = HARMONIZER_TEMPORAL_COSMOS_TOKENIZER_URL,
        harmonizer_temporal_cosmos_cache_dir: str | None = None,
        max_harmonizer_sessions: int = 32,
    ) -> None:
        """Optionally enable the (non-temporal or temporal) Harmonizer for render postprocessing.

        Non-temporal and temporal are mutually exclusive (enforced by GrpcServerConfig). The
        temporal model is driven per-session via create_harmonizer_session / end_harmonizer_session
        plus `apply_harmonizer` and `harmonizer_session_id` on RGBRenderRequest / CameraRenderRequest.
        """
        self.harmonizer_enabled = harmonizer_enabled
        self.harmonizer_temporal_enabled = harmonizer_temporal_enabled
        self._max_harmonizer_sessions = max_harmonizer_sessions
        if not (harmonizer_enabled or harmonizer_temporal_enabled):
            return

        # Flags are set synchronously, so a render mid-load still takes the harmonized path
        # and blocks in the property rather than returning an unharmonized frame.
        def load_model() -> None:
            try:
                if harmonizer_enabled:
                    self._harmonizer_model = HarmonizerModelFactory.get(
                        harmonizer_url, harmonizer_cache_dir, model_filename, harmonizer_resolution
                    )
                elif harmonizer_temporal_enabled:
                    self._harmonizer_model = HarmonizerModelFactory.get(
                        harmonizer_temporal_url,
                        harmonizer_cache_dir,
                        harmonizer_temporal_model_filename,
                        harmonizer_temporal_resolution,
                        temporal=True,
                        cosmos_base_model_url=harmonizer_temporal_cosmos_base_model_url,
                        cosmos_tokenizer_url=harmonizer_temporal_cosmos_tokenizer_url,
                        cosmos_cache_dir=harmonizer_temporal_cosmos_cache_dir,
                    )
            except BaseException as exc:  # re-raised to the first render that needs the model
                self._harmonizer_error = exc
                log.exception("Harmonizer model load failed")
            finally:
                self._harmonizer_ready.set()

        threading.Thread(target=load_model, name="harmonizer-load", daemon=True).start()

    def _await_harmonizer_model(self, context: grpc.ServicerContext | None = None) -> Any:
        """The Harmonizer model, waiting for the background load.

        Loading before the bind left clients on a closed socket, unable to tell a slow start
        from a dead server; the wait now happens inside an RPC, where they are connected. It
        polls rather than blocking outright so a cancelled or expired RPC releases the worker --
        max_workers defaults to 1, so one stuck handler is the whole server.
        """
        deadline = time.monotonic() + _HARMONIZER_LOAD_TIMEOUT_S
        while not self._harmonizer_ready.wait(timeout=_HARMONIZER_POLL_S):
            if context is not None:
                if not context.is_active():
                    raise CacheWaitAbortedError(
                        "client went away while the Harmonizer model was loading",
                        CacheWaitAbortedReason.CANCELLED,
                    )
                remaining = context.time_remaining()
                if remaining is not None and remaining <= 0:
                    raise CacheWaitAbortedError(
                        "RPC deadline expired while the Harmonizer model was loading",
                        CacheWaitAbortedReason.DEADLINE_EXCEEDED,
                    )
            if time.monotonic() >= deadline:
                raise CacheWaitAbortedError(
                    f"Harmonizer model still loading after {_HARMONIZER_LOAD_TIMEOUT_S}s",
                    CacheWaitAbortedReason.DEADLINE_EXCEEDED,
                )
        if self._harmonizer_error is not None:
            raise self._harmonizer_error
        return self._harmonizer_model

    @property
    def harmonizer_load_pending(self) -> bool:
        """True while a Harmonizer model is enabled but its background load has not finished."""
        return (self.harmonizer_enabled or self.harmonizer_temporal_enabled) and not self._harmonizer_ready.is_set()

    def wait_for_harmonizer_load(self) -> None:
        """Block until the background Harmonizer load finishes, successfully or not."""
        self._harmonizer_ready.wait()

    @property
    def harmonizer_load_error(self) -> Optional[BaseException]:
        """The exception raised by the background Harmonizer load, or None if it succeeded."""
        return self._harmonizer_error

    def create_harmonizer_session(
        self, request: grpc_types.Empty, context: grpc.ServicerContext
    ) -> sensorsim_types.HarmonizerSessionId:
        log.info("create_harmonizer_session")
        if not self.harmonizer_temporal_enabled:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "temporal harmonizer is not enabled on this server (start with --enable-harmonizer-temporal).",
            )
        with self._harmonizer_sessions_lock:
            if len(self._harmonizer_sessions) >= self._max_harmonizer_sessions:
                context.abort(
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    f"max concurrent harmonizer sessions ({self._max_harmonizer_sessions}) reached.",
                )
            session_id = uuid.uuid4().hex
            self._harmonizer_sessions[session_id] = HarmonizerState()
        return sensorsim_types.HarmonizerSessionId(session_id=session_id)

    def end_harmonizer_session(
        self, request: sensorsim_types.HarmonizerSessionId, context: grpc.ServicerContext
    ) -> grpc_types.Empty:
        log.info("end_harmonizer_session")
        with self._harmonizer_sessions_lock:
            state = self._harmonizer_sessions.pop(request.session_id, None)
        if state is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"unknown harmonizer session: {request.session_id}")
            raise  # unreachable, but needed for mypy
        state.close()
        return grpc_types.Empty()

    def _resolve_harmonizer_session(
        self, request: RGBRenderRequest | CameraRenderRequest, context: grpc.ServicerContext
    ) -> HarmonizerState:
        """Resolve the supplied session ID for a temporal colour render.

        The ID must be present (INVALID_ARGUMENT) and exist (NOT_FOUND).
        """
        if not request.HasField("harmonizer_session_id"):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "harmonizer_session_id is required for a temporal colour render.",
            )
        with self._harmonizer_sessions_lock:
            state = self._harmonizer_sessions.get(request.harmonizer_session_id)
        if state is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"unknown harmonizer session: {request.harmonizer_session_id}")
        assert state is not None  # context.abort raises; this narrows the type for callers
        return state

    @staticmethod
    def _request_scene_cache_key(
        scene_id: str,
        context: grpc.ServicerContext | None = None,
    ) -> SceneCacheKey:
        """Build the scene-cache key selected by a scene-bearing request."""
        metadata = dict(context.invocation_metadata()) if context is not None else {}
        scene_url = metadata.get(SCENE_URL_HEADER)
        if scene_url is None:
            return SceneCacheKey(scene_id=scene_id)

        metadata_scene_id = require_scene_id(metadata.get(SCENE_ID_HEADER))
        if metadata_scene_id != scene_id:
            raise SceneNotFoundError(
                f"Requested scene {scene_id!r} does not match metadata scene {metadata_scene_id!r}."
            )
        return SceneCacheKey.from_url(metadata_scene_id, scene_url)

    def test_all_scenes(self) -> None:
        """
        Attempts to load each available scene to fail fast
        """
        log.info("Testing gathered scenes...")
        with self._artifact_registry_lock:
            scene_keys = [registration.scene_key for registration in self._artifact_registrations.values()]
        for scene_key in scene_keys:
            with self.get_backend(scene_key) as _:
                pass
        log.info("...done testing gathered scenes.")

    def _create_version_id(self) -> grpc_types.VersionId:
        nurec_version = unpack_optional(
            get_version(
                # Use an empty default if version is not available in current env (e.g. during sandboxed unit test execution)
                allow_empty=True
            )
        )
        return grpc_types.VersionId(
            version_id=nurec_version.semantic_string(),
            git_hash=nurec_version.git_commit_sha_short,
            # 0.0.2 -> 0.0.3 covers two additive, wire-compatible changes shipping together:
            #  - the CameraRender / BatchCameraRender handlers, and
            #  - the multi-process discovery keys (`serving.mode` and the proxy's `multiproc.*`
            #    effective-config entries in get_server_config).
            # 0.0.3 -> 0.0.4: the CUDA IPC (same-node) transport -- open_ipc_session/
            # close_ipc_session and RGBRenderRequest's use_cuda_ipc/ipc_session_id/
            # released_buffer_id + RGBRenderReturn's format/buffer_id fields. Additive,
            # wire-compatible; a client that never sets use_cuda_ipc is unaffected.
            # 0.0.4 -> 0.0.5: extends CUDA IPC delivery to CameraRenderRequest/render_camera --
            # the same three request fields and CameraRenderOutput's format/buffer_id, mirroring
            # RGBRenderRequest/RGBRenderReturn. Additive, wire-compatible.
            # Patch bump only: no request/response field removal or behavior change.
            grpc_api_version=grpc_types.VersionId.APIVersion(major=0, minor=0, patch=5),
        )

    def get_version(self, request: grpc_types.Empty, context: grpc.ServicerContext) -> grpc_types.VersionId:
        log.info("get_version")
        return self.version_id

    def get_server_config(
        self, request: grpc_types.Empty, context: grpc.ServicerContext
    ) -> sensorsim_types.ServerConfig:
        log.info("get_server_config")
        return sensorsim_types.ServerConfig(
            server_config=self.server_config_map,
        )

    def get_available_scenes(
        self, request: grpc_types.Empty, context: grpc.ServicerContext
    ) -> grpc_types.AvailableScenesReturn:
        log.info("get_available_scenes")
        return grpc_types.AvailableScenesReturn(scene_ids=self._artifact_snapshot())

    def get_loaded_scenes(
        self, request: grpc_types.Empty, context: grpc.ServicerContext
    ) -> sensorsim_types.LoadedScenesReturn:
        log.info("get_loaded_scenes")
        capacity, entries = self.backend_cache.loaded_scene_counts()
        return sensorsim_types.LoadedScenesReturn(
            loaded_instance_capacity=capacity,
            scenes=[
                sensorsim_types.LoadedSceneEntry(
                    scene_id=scene_id,
                    loaded_instance_count=loaded_count,
                    reusable_instance_count=reusable_count,
                )
                for scene_id, loaded_count, reusable_count in entries
            ],
        )

    def get_available_cameras(
        self, request: AvailableCamerasRequest, context: grpc.ServicerContext
    ) -> AvailableCamerasReturn:
        log.info("get_available_cameras")
        try:
            with self.get_backend(self._request_scene_cache_key(request.scene_id, context), context) as backend:
                return backend.get_available_cameras()
        except (CacheWaitAbortedError, SceneNotFoundError) as exc:
            _abort_with_classified_exception(context, exc, "get_available_cameras")
            raise  # unreachable, but needed for mypy
        except (CacheFullError, NoSpareBackendsError) as exc:
            log.warning(f"Cannot serve scene {request.scene_id}: {exc}")
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise  # unreachable, but needed for mypy
        except Exception:
            log.error(traceback.format_exc())
            raise
        return  # unreachable, but needed for mypy

    def get_available_trajectories(
        self, request: AvailableTrajectoriesRequest, context: grpc.ServicerContext
    ) -> AvailableTrajectoriesReturn:
        log.info("get_available_trajectories")
        try:
            with self.get_backend(self._request_scene_cache_key(request.scene_id, context), context) as backend:
                return backend.get_available_trajectories()
        except (CacheWaitAbortedError, SceneNotFoundError) as exc:
            _abort_with_classified_exception(context, exc, "get_available_trajectories")
            raise  # unreachable, but needed for mypy
        except (CacheFullError, NoSpareBackendsError) as exc:
            log.warning(f"Cannot serve scene {request.scene_id}: {exc}")
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise  # unreachable, but needed for mypy
        except Exception:
            log.error(traceback.format_exc())
            raise

    def get_available_ego_masks(
        self, request: grpc_types.Empty, context: grpc.ServicerContext
    ) -> AvailableEgoMasksReturn:
        log.info("get_available_ego_masks")
        return AvailableEgoMasksReturn(ego_mask_metadata=self.egocar_rig_bank.available_metadata())

    def get_external_asset_objects(
        self, request: ExternalAssetObjectsRequest, context: grpc.ServicerContext
    ) -> ExternalAssetObjectsReturn:
        """
        Get list of track IDs for external asset objects in a scene.

        Args:
            request: Contains scene_id to query
            context: gRPC context

        Returns:
            ExternalAssetObjectsReturn with list of track IDs
        """
        log.info("get_external_asset_objects")
        try:
            with self.get_backend(self._request_scene_cache_key(request.scene_id, context), context) as backend:
                return backend.get_external_asset_objects()
        except (CacheWaitAbortedError, SceneNotFoundError) as exc:
            _abort_with_classified_exception(context, exc, "get_external_asset_objects")
            raise  # unreachable, but needed for mypy
        except Exception:
            log.error(traceback.format_exc())
            raise

    def get_dynamic_objects(
        self, request: AvailableDynamicObjectsRequest, context: grpc.ServicerContext
    ) -> AvailableDynamicObjectsReturn:
        log.info(f"get_dynamic_objects for scene {request.scene_id}")
        try:
            with self.get_backend(self._request_scene_cache_key(request.scene_id, context), context) as backend:
                dynamic_objects = backend.get_dynamic_objects()
                return AvailableDynamicObjectsReturn(dynamic_objects=dynamic_objects)
        except (CacheWaitAbortedError, SceneNotFoundError) as exc:
            _abort_with_classified_exception(context, exc, "get_dynamic_objects")
            raise  # unreachable, but needed for mypy
        except (CacheFullError, NoSpareBackendsError) as exc:
            log.warning(f"Cannot serve scene {request.scene_id}: {exc}")
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise  # unreachable, but needed for mypy
        except Exception:
            log.error(traceback.format_exc())
            raise

    def edit_assets(self, request: EditAssetsRequest, context: grpc.ServicerContext) -> EditAssetsResponse:
        log.info(f"edit_assets for scene {request.scene_id}")
        try:
            with self.get_backend(self._request_scene_cache_key(request.scene_id, context), context) as backend:
                backend.edit_assets(replace_assets=list(request.replace), insert_assets=list(request.insert))
                return EditAssetsResponse(success=True, message="Success")
        except (CacheWaitAbortedError, SceneNotFoundError) as exc:
            _abort_with_classified_exception(context, exc, "edit_assets")
            raise  # unreachable, but needed for mypy
        except Exception as e:
            log.error(traceback.format_exc())
            return EditAssetsResponse(success=False, message=f"Failed to edit assets: {str(e)}")

    def restore_model_parameters(
        self, request: RestoreModelParametersRequest, context: grpc.ServicerContext
    ) -> grpc_types.Empty:
        log.info(f"restore_model_parameters for scene: {request.scene_id}")

        try:
            with self.get_backend(self._request_scene_cache_key(request.scene_id, context), context) as backend:
                backend.restore_model_parameters()
                log.info(f"Successfully restored model parameters for scene: {request.scene_id}")
        except (CacheWaitAbortedError, SceneNotFoundError) as exc:
            _abort_with_classified_exception(context, exc, "restore_model_parameters")
            raise  # unreachable, but needed for mypy
        except Exception:
            log.error(traceback.format_exc())
            raise

        return grpc_types.Empty()

    def _get_ipc_mgr(self) -> Any:
        # Import-time constant — fail fast without taking the init lock when the
        # CUDA IPC transport isn't available.
        if not _CUDA_IPC_AVAILABLE:
            raise RuntimeError("CUDA IPC transport not available: nurec.libs.cuda_ipc_transport import failed")
        if self._ipc_mgr is None:
            with self._ipc_mgr_lock:
                if self._ipc_mgr is None:
                    # NRE_CUDA_IPC_DEVICE pins which CUDA device index the server opens
                    # (and allocates the IPC buffer pool on). Defaults to device 0. Recorded
                    # so _ipc_write_rgb can check the render output is on the same device
                    # (see the check there for why a mismatch can't just work via peer access).
                    device_index = int(os.environ.get("NRE_CUDA_IPC_DEVICE", "0"))
                    # --cuda-ipc-max-buffer-count (GrpcServerConfig.cuda_ipc_max_buffer_count)
                    # bounds how many buffers a single open_ipc_session call may request, so an
                    # operator can raise it for larger batches without a rebuild. Falls back to
                    # the config field's own default if this service was constructed without a
                    # server_config (e.g. some unit tests).
                    max_buffer_count = (
                        self.server_config.cuda_ipc_max_buffer_count
                        if self.server_config is not None
                        else GrpcServerConfig.model_fields["cuda_ipc_max_buffer_count"].default
                    )
                    self._ipc_mgr = IpcManager(device_index, max_buffer_count)
                    self._ipc_device_index = device_index
                    log.info(
                        f"CUDA IPC manager initialized: device_index={device_index} max_buffer_count={max_buffer_count}"
                    )
        return self._ipc_mgr

    def open_ipc_session(
        self,
        request: ipc_types.OpenIpcSessionRequest,
        context: grpc.ServicerContext,
    ) -> ipc_types.OpenIpcSessionResponse:
        log.info(
            f"open_ipc_session {request.requested_width}x{request.requested_height} "
            f"buffers={request.requested_buffer_count}"
        )
        # Bounds the GPU memory a single co-located client (or several) can pin via this
        # transport. NRE_CUDA_IPC_MAX_SESSIONS alone doesn't do this meaningfully: each session
        # can independently request up to --cuda-ipc-max-buffer-count buffers of up to
        # kMaxRequestedDimension^2*3 bytes each, so a reachable client can still pin most of a
        # device's memory well within a generous session count. NRE_CUDA_IPC_MAX_TOTAL_BYTES
        # caps the aggregate byte footprint of every currently-open session instead (or as well).
        # requested_width/height/buffer_count are also bounded per-request, natively, in
        # IpcManager.open_session. Checked before the try-block below, like
        # require_allowed_image_format, so this abort is never reclassified by the catch-all
        # exception handler.
        #
        # The check-and-reserve happens together under one lock acquisition (rather than
        # checking here and incrementing only after mgr.open_session succeeds below) so
        # concurrent open_ipc_session calls -- gRPC serves RPCs from a thread pool, so this
        # needs no --multiproc-workers to happen -- can't all read the same pre-reservation
        # state and all pass the check before any of them finishes allocating. The reservation
        # is rolled back if the native allocation subsequently fails.
        max_sessions = int(os.environ.get("NRE_CUDA_IPC_MAX_SESSIONS", "32"))
        max_total_bytes = int(os.environ.get("NRE_CUDA_IPC_MAX_TOTAL_BYTES", str(16 * 1024**3)))
        requested_bytes = (
            int(request.requested_width) * int(request.requested_height) * 3 * int(request.requested_buffer_count)
        )
        with self._ipc_mgr_lock:
            if self._ipc_open_sessions >= max_sessions:
                context.abort(
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    f"server already has {self._ipc_open_sessions} open CUDA IPC sessions "
                    f"(max {max_sessions}); close an existing session before opening another",
                )
                raise  # unreachable, but needed for mypy
            if self._ipc_reserved_bytes + requested_bytes > max_total_bytes:
                context.abort(
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    f"this request's {requested_bytes} bytes would exceed the CUDA IPC aggregate "
                    f"budget ({self._ipc_reserved_bytes} already reserved, {max_total_bytes} max); "
                    "close an existing session or request fewer/smaller buffers",
                )
                raise  # unreachable, but needed for mypy
            self._ipc_open_sessions += 1
            self._ipc_reserved_bytes += requested_bytes
        try:
            mgr = self._get_ipc_mgr()
            session_id, handles, buffer_size, device_uuid, actual_width, actual_height = mgr.open_session(
                request.requested_width,
                request.requested_height,
                request.requested_buffer_count,
            )
            with self._ipc_mgr_lock:
                self._ipc_session_bytes[session_id] = requested_bytes
            log.info(f"open_ipc_session ok: session_id={session_id} buffers={len(handles)}")
            return ipc_types.OpenIpcSessionResponse(
                session_id=session_id,
                cuda_ipc_mem_handles=handles,
                buffer_size=buffer_size,
                device_uuid=device_uuid,
                actual_width=actual_width,
                actual_height=actual_height,
            )
        except Exception as exc:
            with self._ipc_mgr_lock:
                self._ipc_open_sessions -= 1
                self._ipc_reserved_bytes -= requested_bytes
            log.error(traceback.format_exc())
            status, message = _classify_render_exception(exc)
            if status == grpc.StatusCode.UNKNOWN:
                status = grpc.StatusCode.INTERNAL
            context.abort(status, message)
            raise  # unreachable, but needed for mypy

    def close_ipc_session(
        self,
        request: ipc_types.CloseIpcSessionRequest,
        context: grpc.ServicerContext,
    ) -> ipc_types.CloseIpcSessionResponse:
        log.info(f"close_ipc_session session={request.session_id}")
        try:
            # Deliberately broader than open_ipc_session's classification: teardown is
            # meant to be soft-failing (success=False + error_message, never an abort),
            # so any failure here — not just the expected "transport unavailable" or
            # native errors — is reported back rather than propagated.
            mgr = self._get_ipc_mgr()
            # close_session reports whether it actually removed a session. Repeated
            # or unknown-id closes are a no-op there, so decrementing unconditionally
            # would drop the count below the number of live sessions. Teardown stays
            # idempotent for the caller.
            if mgr.close_session(request.session_id):
                with self._ipc_mgr_lock:
                    self._ipc_open_sessions = max(0, self._ipc_open_sessions - 1)
                    self._ipc_reserved_bytes = max(
                        0, self._ipc_reserved_bytes - self._ipc_session_bytes.pop(request.session_id, 0)
                    )
            return ipc_types.CloseIpcSessionResponse(success=True)
        except Exception as exc:
            log.error(traceback.format_exc())
            return ipc_types.CloseIpcSessionResponse(success=False, error_message=str(exc))

    def _prepare_ipc_write(
        self, raster_f32: torch.Tensor, request: RGBRenderRequest | CameraRenderRequest
    ) -> tuple[Any, torch.Tensor, int, int]:
        """Shared validation + sync for every IPC write path (_ipc_write_rgb's own-buffer
        selection, batch_render_rgb's write into a pre-reserved buffer, and
        _ipc_write_camera_color): shape/device checks, resolution check against the request,
        and the render-stream sync the pack kernel depends on. Returns (mgr, raster_f32,
        height, width)."""
        if raster_f32.dim() != 3 or raster_f32.shape[2] != 3:
            raise RuntimeError(f"render output expected shape (H, W, 3); got {tuple(raster_f32.shape)}")
        if not raster_f32.is_cuda:
            raise RuntimeError("render output must be a CUDA tensor for CUDA IPC path")
        raster_f32 = raster_f32.contiguous()
        height = int(raster_f32.shape[0])
        width = int(raster_f32.shape[1])
        # Buffers are fixed-size for the session (set at open_ipc_session); a request
        # that renders a different resolution can't be delivered into them.
        if (
            request.resolution_h
            and request.resolution_w
            and (height != request.resolution_h or width != request.resolution_w)
        ):
            raise ValueError(
                f"render size {width}x{height} does not match requested resolution "
                f"{request.resolution_w}x{request.resolution_h}"
            )
        # The pack kernel writes the tensor's memory directly, so the render stream
        # must finish before it reads from raster_f32.
        torch.cuda.synchronize(raster_f32.device)
        mgr = self._get_ipc_mgr()
        # The IPC session's buffers live on self._ipc_device_index (NRE_CUDA_IPC_DEVICE);
        # the native write path sets that as the current device before launching the pack
        # kernel (see cuda_ipc_module.cpp), so raster_f32 must already be on that same
        # physical GPU -- a raw device pointer from a different device isn't valid there
        # without peer access, which nothing here requests or enables. A multi-GPU process
        # (e.g. a nonzero LOCAL_RANK) that leaves NRE_CUDA_IPC_DEVICE at its default is the
        # likely way to hit this.
        if raster_f32.device.index != self._ipc_device_index:
            raise RuntimeError(
                f"render output is on CUDA device {raster_f32.device.index}, but the CUDA IPC "
                f"transport is pinned to device {self._ipc_device_index} (NRE_CUDA_IPC_DEVICE); "
                "the renderer and NRE_CUDA_IPC_DEVICE must select the same physical GPU"
            )
        return mgr, raster_f32, height, width

    def _ipc_write_rgb(self, raster_f32: torch.Tensor, request: RGBRenderRequest) -> RGBRenderReturn:
        # Fused conversion straight into one of the session's pooled, IPC-exported
        # buffers: a single CUDA kernel writes directly into a server-owned buffer
        # the client maps. No intermediate tensor, no five-op PyTorch chain, no D2D copy.
        mgr, raster_f32, height, width = self._prepare_ipc_write(raster_f32, request)
        released_buffer_id = request.released_buffer_id if request.HasField("released_buffer_id") else None
        filled_buffer_id = mgr.write_render_hwc_f32(
            request.ipc_session_id,
            released_buffer_id,
            raster_f32.data_ptr(),
            height,
            width,
        )
        return RGBRenderReturn(
            format="rgb_uint8_planar",
            buffer_id=filled_buffer_id,
        )

    def _ipc_write_reserved_rgb(
        self, raster_f32: torch.Tensor, request: RGBRenderRequest, buffer_id: int
    ) -> RGBRenderReturn:
        # batch_render_rgb's counterpart to _ipc_write_rgb: writes into a buffer the batch
        # already claimed via IpcManager.reserve_buffers, rather than selecting one itself.
        mgr, raster_f32, height, width = self._prepare_ipc_write(raster_f32, request)
        mgr.write_reserved_buffer(request.ipc_session_id, buffer_id, raster_f32.data_ptr(), height, width)
        return RGBRenderReturn(format="rgb_uint8_planar", buffer_id=buffer_id)

    def _ipc_write_camera_color(self, raster_f32: torch.Tensor, request: CameraRenderRequest) -> CameraRenderOutput:
        # render_camera's counterpart to _ipc_write_rgb: same own-buffer-selection write, but
        # returns a CameraRenderOutput (the CAMERA_RENDER_PRODUCT_COLOR entry) instead of an
        # RGBRenderReturn. require_cuda_ipc_camera_request has already confirmed this request
        # asks for exactly one COLOR output with no bounding-box overlay.
        mgr, raster_f32, height, width = self._prepare_ipc_write(raster_f32, request)
        released_buffer_id = request.released_buffer_id if request.HasField("released_buffer_id") else None
        filled_buffer_id = mgr.write_render_hwc_f32(
            request.ipc_session_id,
            released_buffer_id,
            raster_f32.data_ptr(),
            height,
            width,
        )
        return CameraRenderOutput(
            format="rgb_uint8_planar",
            buffer_id=filled_buffer_id,
            image_height=height,
            image_width=width,
            render_product=CameraRenderProduct.CAMERA_RENDER_PRODUCT_COLOR,
        )

    @ScopedTimer()
    def render_rgb(self, request: RGBRenderRequest, context: grpc.ServicerContext) -> RGBRenderReturn:
        log.info("render_rgb")

        require_allowed_image_format(request, self._allowed_image_formats, context)
        require_cuda_ipc_output_format(request, context)

        # Resolve the temporal session before the render try-block so its INVALID_ARGUMENT/NOT_FOUND
        # is not reclassified by the catch-all handler below.
        harmonizer_state = (
            self._resolve_harmonizer_session(request, context)
            if (
                self.harmonizer_temporal_enabled
                and (request.apply_harmonizer or request.HasField("harmonizer_session_id"))
            )
            else None
        )
        if harmonizer_state is not None and not request.apply_harmonizer:
            # Reset before rendering so a later opt-in cannot use stale history.
            harmonizer_state.reset()
        # Start timing for metrics collection
        render_start_time = time.perf_counter()

        try:
            with self.get_backend(self._request_scene_cache_key(request.scene_id, context), context) as backend:
                raster_f32, img_size = backend.render_rgb_request(request, self.enable_editing_actors)

            raster_f32 = self._maybe_apply_harmonizer(raster_f32, img_size, request, harmonizer_state, context)

            # inpaint the egocar mask
            if (egocar_mask := self.egocar_rig_bank.select_from_request(request)) is not None:
                raster_f32 = egocar_mask.overlay_on_image(raster_f32)

            # Calculate render time and collect metrics
            render_time_ms = (time.perf_counter() - render_start_time) * 1000.0
            self._metrics_manager.record_camera_render_time(render_time_ms, request.scene_id)

            # Mark frame boundary and send plots
            profile.mark_frame_boundary("rgb_render")

            if request.use_cuda_ipc:
                ipc_start = time.perf_counter()
                with _nvtx_range("serve_grpc.rgb_render.ipc_write"):
                    result = self._ipc_write_rgb(raster_f32, request)
                self._metrics_manager.record_ipc_write_time((time.perf_counter() - ipc_start) * 1000.0)
                return result

            return self.encode_image((raster_f32 * 255).clamp(0, 255).to(torch.uint8), request)
        except Exception as exc:
            _abort_with_classified_exception(context, exc, "render_rgb")
            raise  # unreachable, but needed for mypy

    @ScopedTimer()
    def render_camera(self, request: CameraRenderRequest, context: grpc.ServicerContext) -> CameraRenderReturn:
        log.info("render_camera")

        require_allowed_image_format(request, self._allowed_image_formats, context)
        require_cuda_ipc_camera_request(request, context)

        # Only colour output can use the temporal Harmonizer. Resolve its session before the render
        # try-block so INVALID_ARGUMENT/NOT_FOUND is not reclassified by the catch-all handler below.
        is_color_requested = self._camera_request_includes_color_product(request)
        harmonizer_state = (
            self._resolve_harmonizer_session(request, context)
            if (
                self.harmonizer_temporal_enabled
                and is_color_requested
                and (request.apply_harmonizer or request.HasField("harmonizer_session_id"))
            )
            else None
        )
        if harmonizer_state is not None and not request.apply_harmonizer:
            # Reset before rendering so a later opt-in cannot use stale history.
            harmonizer_state.reset()
        render_start_time = time.perf_counter()

        try:
            with self.get_backend(self._request_scene_cache_key(request.scene_id, context), context) as backend:
                result = self._camera_render_return_from_request(
                    backend,
                    request,
                    enable_editing_actors=bool(self.enable_editing_actors),
                    harmonizer_state=harmonizer_state,
                    context=context,
                )
                render_time_ms = (time.perf_counter() - render_start_time) * 1000.0
                self._metrics_manager.record_camera_render_time(render_time_ms, request.scene_id)
                profile.mark_frame_boundary("camera_render")
                return result
        except Exception as exc:
            status, message = _classify_render_exception(exc)
            if status == grpc.StatusCode.UNKNOWN:
                log.error(traceback.format_exc())
                raise
            log.warning(f"render_camera failed with {status.name}: {message}")
            context.abort(status, message)
            raise  # unreachable, but needed for mypy

    @ScopedTimer()
    def batch_render_camera(
        self, request: BatchCameraRenderRequest, context: grpc.ServicerContext
    ) -> BatchCameraRenderReturn:
        log.info(f"batch_render_camera: {len(request.items)} cameras")

        for index, item in enumerate(request.items):
            require_allowed_image_format(
                item.request,
                self._allowed_image_formats,
                context,
                request_description=f"Batch item[{index}] {item.camera_name!r}",
            )
            # CUDA IPC delivery is not implemented for batch_render_camera: each item's buffer
            # would have to be reserved from its session's pool atomically up front (as
            # batch_render_rgb does), which this endpoint doesn't do. Reject explicitly rather
            # than silently falling back to image_bytes for an item that asked for IPC delivery.
            if item.request.use_cuda_ipc:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"batch_render_camera does not support use_cuda_ipc (item[{index}] "
                    f"{item.camera_name!r}); use render_camera for CUDA IPC delivery of a single "
                    "camera at a time.",
                )
                raise  # unreachable, but needed for mypy

        # The temporal harmonizer is autoregressive (each frame conditions on the previous frame's
        # output), so batch requests cannot safely maintain or reset per-session history.
        if self.harmonizer_temporal_enabled:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "batch_render_camera does not support the temporal harmonizer; "
                "use render_camera with a harmonizer session instead.",
            )

        if not request.items:
            return BatchCameraRenderReturn(items=[])

        scene_id = request.items[0].request.scene_id

        mismatched_scenes = [
            (i, item.camera_name, item.request.scene_id)
            for i, item in enumerate(request.items)
            if item.request.scene_id != scene_id
        ]
        if mismatched_scenes:
            error_msg = (
                f"All cameras in batch must use the same scene_id. "
                f"Expected '{scene_id}', but found mismatches: "
                + ", ".join(f"item[{i}] '{name}' has '{sid}'" for i, name, sid in mismatched_scenes)
            )
            log.error(f"batch_render_camera: {error_msg}")
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, error_msg)
            raise ValueError(error_msg)  # unreachable, but needed for mypy

        render_start_time = time.perf_counter()

        results = [BatchCameraRenderReturnItem(camera_name=item.camera_name, success=False) for item in request.items]
        try:
            for chunk_start in range(0, len(request.items), BATCH_RENDER_REQUEST_CHUNK_SIZE):
                chunk_items = request.items[chunk_start : chunk_start + BATCH_RENDER_REQUEST_CHUNK_SIZE]
                pending_jpeg_items: list[tuple[int, BatchCameraRenderRequestItem, list[PendingCameraRenderOutput]]] = []
                with self.get_backend(self._request_scene_cache_key(scene_id, context), context) as backend:
                    for chunk_index, item in enumerate(chunk_items):
                        result_index = chunk_start + chunk_index
                        if item.request.image_format != ImageFormat.JPEG:
                            results[result_index] = self._render_single_camera_for_batch_camera(
                                backend,
                                item,
                                enable_editing_actors=bool(self.enable_editing_actors),
                                context=context,
                            )
                            continue

                        try:
                            pending_outputs = self._prepare_camera_render_outputs_from_request(
                                backend,
                                item.request,
                                enable_editing_actors=bool(self.enable_editing_actors),
                                context=context,
                            )
                            pending_jpeg_items.append((result_index, item, pending_outputs))
                        except (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError):
                            raise
                        except Exception as exc:
                            results[result_index] = self._batch_camera_error_item(item.camera_name, exc)

                pending_outputs = [
                    pending_output
                    for _, _, item_pending_outputs in pending_jpeg_items
                    for pending_output in item_pending_outputs
                ]
                try:
                    encoded_outcomes = self._encode_pending_camera_render_output_bytes(pending_outputs)
                except (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError):
                    self._recover_from_cuda_oom("batch_render_camera JPEG encoding")
                    raise
                outcome_index = 0
                for result_index, item, item_pending_outputs in pending_jpeg_items:
                    item_outcomes = encoded_outcomes[outcome_index : outcome_index + len(item_pending_outputs)]
                    outcome_index += len(item_pending_outputs)
                    item_error = next(
                        (outcome for outcome in item_outcomes if isinstance(outcome, Exception)),
                        None,
                    )
                    if item_error is not None:
                        results[result_index] = self._batch_camera_error_item(item.camera_name, item_error)
                        continue

                    try:
                        camera_outputs: list[CameraRenderOutput] = []
                        for pending_output, encoded_outcome in zip(item_pending_outputs, item_outcomes):
                            if not isinstance(encoded_outcome, bytes):
                                raise AssertionError("Successful camera output did not contain encoded bytes.")
                            camera_outputs.append(
                                self._camera_render_output_from_bytes(pending_output, encoded_outcome)
                            )
                        results[result_index] = BatchCameraRenderReturnItem(
                            camera_name=item.camera_name,
                            result=CameraRenderReturn(camera_render_outputs=camera_outputs),
                            success=True,
                        )
                    except (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError):
                        raise
                    except Exception as exc:
                        results[result_index] = self._batch_camera_error_item(item.camera_name, exc)

            render_time_ms = (time.perf_counter() - render_start_time) * 1000.0
            self._metrics_manager.record_batch_camera_render_time(render_time_ms, len(results), scene_id)
            profile.mark_frame_boundary("batch_camera_render")

            return BatchCameraRenderReturn(items=results)

        except Exception as exc:
            status, message = _classify_render_exception(exc)
            if status == grpc.StatusCode.UNKNOWN:
                log.error(traceback.format_exc())
                raise
            log.warning(f"batch_render_camera failed with {status.name}: {message}")
            context.abort(status, message)
            raise  # unreachable, but needed for mypy

    def _render_single_camera_for_batch_camera(
        self,
        backend: Backend,
        item: BatchCameraRenderRequestItem,
        enable_editing_actors: bool = False,
        context: grpc.ServicerContext | None = None,
    ) -> BatchCameraRenderReturnItem:
        try:
            ret = self._camera_render_return_from_request(
                backend,
                item.request,
                enable_editing_actors=enable_editing_actors,
                context=context,
            )
            return BatchCameraRenderReturnItem(
                camera_name=item.camera_name,
                result=ret,
                success=True,
            )
        except (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError):
            raise
        except Exception as exc:
            return self._batch_camera_error_item(item.camera_name, exc)

    @staticmethod
    def _batch_camera_error_item(camera_name: str, exc: Exception) -> BatchCameraRenderReturnItem:
        status, message = _classify_render_exception(exc)
        error_message = f"{status.name}: {message}" if message else status.name
        if status == grpc.StatusCode.UNKNOWN:
            log.error(f"Failed to render camera {camera_name}: {error_message}")
        return BatchCameraRenderReturnItem(
            camera_name=camera_name,
            success=False,
            error_message=error_message,
        )

    @ScopedTimer()
    def batch_render_rgb(self, request: BatchRGBRenderRequest, context: grpc.ServicerContext) -> BatchRGBRenderReturn:
        """Render multiple cameras in a single call.

        This endpoint batches multiple render requests to reduce network overhead and optimize
        common setup operations. Each item in the batch can represent a different camera, a
        different frame/timestamp for the same camera, or any combination - as long as all
        items share the same scene_id.

        Items may freely mix use_cuda_ipc=true and JPEG-encoded delivery. All use_cuda_ipc
        items in a batch must share one ipc_session_id: every buffer they need is reserved
        from that session's pool atomically, up front, before any item in the batch renders,
        so either the whole batch's IPC delivery fits in the session's currently-free buffers
        (after applying any released_buffer_id releases from this same batch) or none of it
        does -- there is no partial-fit fallback.

        Architecture:
        - Per-camera rendering - sequential loop (designed for future parallelization).
          NOTE: aspect states (Backend._update_aspect_states) are stateless per item —
          each item's AspectState messages fully determine that frame; if this loop is
          ever parallelized, each item still applies its own states before its render.

        Args:
            request: Batch request containing a list of BatchRGBRenderRequestItem, each with:
                - camera_name: Identifier to match request with response
                - request: Full RGBRenderRequest (camera pose, intrinsics, frame timing, etc.)
                All items must use the same scene_id.
            context: gRPC service context for error handling and metadata.

        Returns:
            BatchRGBRenderReturn containing a list of BatchRGBRenderReturnItem, each with:
                - camera_name: Matches the request's camera_name
                - result: RGBRenderReturn with the rendered image (if success=True)
                - success: Whether this specific camera render succeeded
                - error_message: Error details if success=False

        Raises:
            grpc.StatusCode.INVALID_ARGUMENT: If items have different scene_ids, different
                ipc_session_ids among use_cuda_ipc items, or a use_cuda_ipc item declares an
                image_format other than RGB_UINT8_PLANAR.
            grpc.StatusCode.RESOURCE_EXHAUSTED: If the batch's use_cuda_ipc items need more
                buffers than the session currently has free.
            grpc.StatusCode.FAILED_PRECONDITION: If scene cannot be served (cache full, no backends),
                or if the temporal harmonizer is enabled (unsupported in batch mode).
        """
        log.info(f"batch_render_rgb: {len(request.items)} cameras")

        for index, item in enumerate(request.items):
            require_allowed_image_format(
                item.request,
                self._allowed_image_formats,
                context,
                request_description=f"Batch item[{index}] {item.camera_name!r}",
            )
            require_cuda_ipc_output_format(item.request, context)

        # The temporal harmonizer is autoregressive (each frame conditions on the previous frame's
        # output), so batch requests cannot safely maintain or reset per-session history.
        if self.harmonizer_temporal_enabled:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "batch_render_rgb does not support the temporal harmonizer; "
                "use render_rgb with a harmonizer session instead.",
            )

        if not request.items:
            return BatchRGBRenderReturn(items=[])

        # All items must use the same scene
        scene_id = request.items[0].request.scene_id

        # Validate all items use the same scene_id
        mismatched_scenes = [
            (i, item.camera_name, item.request.scene_id)
            for i, item in enumerate(request.items)
            if item.request.scene_id != scene_id
        ]
        if mismatched_scenes:
            error_msg = (
                f"All cameras in batch must use the same scene_id. "
                f"Expected '{scene_id}', but found mismatches: "
                + ", ".join(f"item[{i}] '{name}' has '{sid}'" for i, name, sid in mismatched_scenes)
            )
            log.error(f"batch_render_rgb: {error_msg}")
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, error_msg)
            raise ValueError(error_msg)  # unreachable, but needed for mypy

        # CUDA IPC items all draw from one session's fixed-size buffer pool, so they must
        # all target the same session (like scene_id above) -- otherwise "does this batch
        # fit" has no single pool to check it against. Reserve every IPC item's buffer
        # up front, atomically, rather than letting each item race the others (and any
        # concurrent render_rgb/batch_render_rgb call against the same session) for a
        # free buffer one at a time: reject the whole batch now if it doesn't fit, so a
        # partially-filled batch never leaves some items holding buffers indefinitely.
        ipc_indices = [i for i, item in enumerate(request.items) if item.request.use_cuda_ipc]
        reserved_buffer_ids: dict[int, int] = {}
        ipc_session_id: str | None = None
        # Every buffer reserved above that this batch doesn't hand back to the client --
        # either because its own item failed (handled below, which discards it here as it
        # releases it) or because the *batch* aborts before a response is ever sent, stranding
        # even already-succeeded items' buffers -- must be released before this method exits,
        # or it stays in_use until the session is closed. Discarded as each buffer is
        # accounted for; whatever remains when the outer except below fires gets released then.
        outstanding_ipc_buffer_ids: set[int] = set()
        if ipc_indices:
            ipc_session_id = request.items[ipc_indices[0]].request.ipc_session_id
            mismatched_sessions = [
                (i, request.items[i].camera_name, request.items[i].request.ipc_session_id) for i in ipc_indices[1:]
            ]
            mismatched_sessions = [(i, name, sid) for i, name, sid in mismatched_sessions if sid != ipc_session_id]
            if mismatched_sessions:
                error_msg = (
                    "All use_cuda_ipc items in a batch must use the same ipc_session_id. "
                    f"Expected {ipc_session_id!r}, but found mismatches: "
                    + ", ".join(f"item[{i}] '{name}' has {sid!r}" for i, name, sid in mismatched_sessions)
                )
                log.error(f"batch_render_rgb: {error_msg}")
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, error_msg)
                raise ValueError(error_msg)  # unreachable, but needed for mypy

            released_buffer_ids = [
                request.items[i].request.released_buffer_id
                for i in ipc_indices
                if request.items[i].request.HasField("released_buffer_id")
            ]
            try:
                mgr = self._get_ipc_mgr()
                reserved = mgr.reserve_buffers(ipc_session_id, released_buffer_ids, len(ipc_indices))
            except Exception as exc:
                log.error(traceback.format_exc())
                status, message = _classify_render_exception(exc)
                if status == grpc.StatusCode.UNKNOWN:
                    status = grpc.StatusCode.INTERNAL
                context.abort(status, message)
                raise  # unreachable, but needed for mypy
            reserved_buffer_ids = dict(zip(ipc_indices, reserved))
            outstanding_ipc_buffer_ids = set(reserved_buffer_ids.values())

        # Start timing for metrics collection
        render_start_time = time.perf_counter()

        results: list[BatchRGBRenderReturnItem] = []
        try:
            for chunk_start in range(0, len(request.items), BATCH_RENDER_REQUEST_CHUNK_SIZE):
                chunk_items = request.items[chunk_start : chunk_start + BATCH_RENDER_REQUEST_CHUNK_SIZE]
                pending_results: list[BatchRGBRenderReturnItem | PendingBatchRGBEncode] = []
                with self.get_backend(self._request_scene_cache_key(scene_id, context), context) as backend:
                    # Render only a bounded chunk before releasing the backend and encoding it.
                    for offset, item in enumerate(chunk_items):
                        item_index = chunk_start + offset
                        buffer_id = reserved_buffer_ids.get(item_index)
                        try:
                            if buffer_id is not None:
                                raster_f32 = self._render_single_camera_raster_for_batch(
                                    backend,
                                    item,
                                    self.enable_editing_actors,
                                    context=context,
                                )
                                ipc_start = time.perf_counter()
                                with _nvtx_range("serve_grpc.batch_render_rgb.ipc_write"):
                                    ipc_result = self._ipc_write_reserved_rgb(raster_f32, item.request, buffer_id)
                                self._metrics_manager.record_ipc_write_time((time.perf_counter() - ipc_start) * 1000.0)
                                pending_results.append(
                                    BatchRGBRenderReturnItem(
                                        camera_name=item.camera_name, result=ipc_result, success=True
                                    )
                                )
                            else:
                                image_u8 = self._render_single_camera_image_for_batch(
                                    backend,
                                    item,
                                    self.enable_editing_actors,
                                    context=context,
                                )
                                pending_results.append(
                                    PendingBatchRGBEncode(
                                        camera_name=item.camera_name,
                                        image_u8=image_u8,
                                        request=item.request,
                                    )
                                )
                        except (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError):
                            raise
                        except Exception as exc:
                            if buffer_id is not None:
                                # This item's reserved buffer was never written and will never be
                                # returned to the client (there's no successful result to carry its
                                # id), so give it back now rather than stranding it in_use until the
                                # session closes. Best-effort: a failure freeing it is logged, not
                                # allowed to mask the item's real error below. Discarded from
                                # outstanding_ipc_buffer_ids either way -- if the release itself
                                # failed, a second attempt from the outer except below is no more
                                # likely to succeed and would just duplicate the log line.
                                try:
                                    self._get_ipc_mgr().release_buffer(ipc_session_id, buffer_id)
                                except Exception:
                                    log.error(
                                        f"batch_render_rgb: failed to release reserved IPC buffer "
                                        f"{buffer_id} for session {ipc_session_id} after item failure:\n"
                                        f"{traceback.format_exc()}"
                                    )
                                finally:
                                    outstanding_ipc_buffer_ids.discard(buffer_id)
                            pending_results.append(self._batch_rgb_error_item(item.camera_name, exc))

                try:
                    results.extend(self._finish_pending_batch_rgb_results(pending_results))
                except (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError):
                    self._recover_from_cuda_oom("batch_render_rgb JPEG encoding")
                    raise

            # Calculate render time and collect metrics
            render_time_ms = (time.perf_counter() - render_start_time) * 1000.0
            self._metrics_manager.record_batch_camera_render_time(render_time_ms, len(results), scene_id)

            # Mark frame boundary
            profile.mark_frame_boundary("batch_rgb_render")

            return BatchRGBRenderReturn(items=results)

        except Exception as exc:
            # Whatever wasn't already accounted for by the per-item handler above -- including
            # buffers for items that already succeeded in an earlier chunk -- is about to be
            # stranded: this method is aborting, so no BatchRGBRenderReturn (and no buffer_id)
            # ever reaches the client. Release them now rather than leaving them in_use until
            # the session closes; this covers both a mid-item OOM (re-raised past the per-item
            # handler on purpose, to trigger _recover_from_cuda_oom) and an exception outside
            # the per-item loop entirely (e.g. get_backend failing on the very first chunk).
            if outstanding_ipc_buffer_ids:
                assert ipc_session_id is not None
                mgr = self._get_ipc_mgr()
                for buffer_id in outstanding_ipc_buffer_ids:
                    try:
                        mgr.release_buffer(ipc_session_id, buffer_id)
                    except Exception:
                        log.error(
                            f"batch_render_rgb: failed to release reserved IPC buffer {buffer_id} "
                            f"for session {ipc_session_id} while aborting the batch:\n{traceback.format_exc()}"
                        )
            _abort_with_classified_exception(context, exc, "batch_render_rgb")
            raise  # unreachable, but needed for mypy

    def _render_single_camera_raster_for_batch(
        self,
        backend: Backend,
        item: BatchRGBRenderRequestItem,
        enable_editing_actors: Optional[bool] = None,
        context: grpc.ServicerContext | None = None,
    ) -> torch.Tensor:
        """Renders one batch item through to a raw (H, W, 3) float32 raster -- shared by both
        delivery paths. _render_single_camera_image_for_batch takes it the rest of the way
        to uint8 for JPEG encoding; use_cuda_ipc items instead hand this straight to
        _ipc_write_reserved_rgb, which does its own float32->uint8 conversion in the pack
        kernel."""
        raster_f32, img_size = backend.render_rgb_request(item.request, enable_editing_actors=enable_editing_actors)

        raster_f32 = self._maybe_apply_harmonizer(raster_f32, img_size, item.request, context=context)

        if (egocar_mask := self.egocar_rig_bank.select_from_request(item.request)) is not None:
            raster_f32 = egocar_mask.overlay_on_image(raster_f32)

        return raster_f32

    def _render_single_camera_image_for_batch(
        self,
        backend: Backend,
        item: BatchRGBRenderRequestItem,
        enable_editing_actors: Optional[bool] = None,
        context: grpc.ServicerContext | None = None,
    ) -> torch.Tensor:
        raster_f32 = self._render_single_camera_raster_for_batch(backend, item, enable_editing_actors, context)
        return (raster_f32 * 255).clamp(0, 255).to(torch.uint8)

    def _batch_rgb_error_item(self, camera_name: str, exc: Exception) -> BatchRGBRenderReturnItem:
        status, message = _classify_render_exception(exc)
        error_message = f"{status.name}: {message}" if message else status.name
        log.error(f"Failed to render camera {camera_name}: {error_message}")
        return BatchRGBRenderReturnItem(
            camera_name=camera_name,
            success=False,
            error_message=error_message,
        )

    def render_lidar(self, request: LidarRenderRequest, context: grpc.ServicerContext) -> LidarRenderReturn:
        log.info("render_lidar")

        # Start timing for metrics collection
        render_start_time = time.perf_counter()

        try:
            with self.get_backend(self._request_scene_cache_key(request.scene_id, context), context) as backend:
                result = backend.render_lidar_request(request, self.enable_editing_actors)

            # Calculate render time and collect metrics
            render_time_ms = (time.perf_counter() - render_start_time) * 1000.0
            self._metrics_manager.record_lidar_render_time(render_time_ms, request.scene_id)

            return result
        except (CacheWaitAbortedError, SceneNotFoundError) as exc:
            _abort_with_classified_exception(context, exc, "render_lidar")
            raise  # unreachable, but needed for mypy
        except (CacheFullError, NoSpareBackendsError) as exc:
            log.warning(f"Cannot serve scene {request.scene_id}: {exc}")
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise  # unreachable, but needed for mypy
        except Exception as exc:
            _abort_with_classified_exception(context, exc, "render_lidar")
            raise  # unreachable, but needed for mypy

    def shut_down(self, request: grpc_types.Empty, context: grpc.ServicerContext) -> grpc_types.Empty:
        log.info("shut_down")

        # Handle metrics shutdown
        self._metrics_manager.shutdown()

        context.add_callback(self._shut_down)
        return grpc_types.Empty()

    def _shut_down(self) -> None:
        if self.server is not None:
            self.server.stop(0)
        # self.server is None during tests as we can't shut down a pytest fixture

    def register_resident_backend(self, backend: Backend) -> None:
        """Register an already-loaded, externally-owned backend so scene-bearing RPCs can resolve it.

        Used by the multi-process workers: their backends alias the owner's IPC weight storage and are
        installed straight into the `BackendCache`, with no artifact registration of their own. Since
        `get_backend` resolves the registry BEFORE the cache, such a scene would otherwise be reported
        NOT_FOUND even while resident. The registration is `resident_only`, so a cache miss fails loud
        instead of rebuilding a non-aliased copy (see `ArtifactRegistration.resident_only`).

        Register under the same key `get_backend` derives from the request, and install the backend in the
        cache under `backend.backend_cache_key.key_hash`, so lookup and cache agree.
        """
        backend_cache_key = backend.backend_cache_key
        registration = ArtifactRegistration(
            artifact=backend.source_artifact,
            downloaded=False,
            scene_key=backend_cache_key.scene_key,
            resident_only=True,
        )
        with self._artifact_registry_lock:
            self._artifact_registrations[backend_cache_key.scene_key.key_hash] = registration

    def unregister_resident_backend(self, backend_cache_key: BackendCacheKey) -> None:
        """Drop a `register_resident_backend` registration (the backend was detached/evicted)."""
        with self._artifact_registry_lock:
            self._artifact_registrations.pop(backend_cache_key.scene_key.key_hash, None)

    def _register_local_scene(self, scene_key: SceneCacheKey, local_uri: str) -> None:
        """Register a local scene under its scene-ID and canonical-URI identity."""
        resolved = resolve_local_scene_uri(local_uri)

        registration_key = scene_key.key_hash
        scene_path_lock = (
            self.scene_cache.lock_scene_path(registration_key)
            if self.scene_cache is not None
            else contextlib.nullcontext()
        )
        with scene_path_lock:
            with self._artifact_registry_lock:
                existing = self._artifact_registrations.get(registration_key)
                if existing is not None:
                    log.info(
                        "Scene local cache hit: scene_id=%s cache_key=%s sha256=%s",
                        scene_key.scene_id,
                        registration_key,
                        existing.backend_cache_key.content_sha256,
                    )
                    return

            artifact = Artifact(resolved)
            registration = ArtifactRegistration(artifact=artifact, downloaded=False, scene_key=scene_key)
            backend_cache_key = registration.backend_cache_key
            with self._artifact_registry_lock:
                self._artifact_registrations[registration_key] = registration

            log.info(
                "Scene local cache miss: scene_id=%s cache_key=%s sha256=%s",
                scene_key.scene_id,
                registration_key,
                backend_cache_key.content_sha256,
            )

    def _download_scene(self, scene_key: SceneCacheKey, scene_url: str) -> SceneCacheLease | None:
        """Download or lease a remote scene keyed by scene ID and canonical URL."""
        assert self.scene_cache is not None
        assert self.downloader is not None
        registration_key = scene_key.key_hash
        temp_path: Path | None = None
        with self.scene_cache.lock_scene_path(registration_key):
            with self._artifact_registry_lock:
                existing = self._artifact_registrations.get(registration_key)
                if existing is not None:
                    if not existing.downloaded:
                        log.info(
                            "Scene local cache hit: scene_id=%s cache_key=%s sha256=%s",
                            scene_key.scene_id,
                            registration_key,
                            existing.backend_cache_key.content_sha256,
                        )
                        return None
                    # Reuse the cached bytes only if the on-disk file is actually
                    # present. A file removed out-of-band (manual cleanup, external
                    # eviction) leaves a stale cache entry, so guard on real existence
                    # and fall through to a fresh download rather than lease a gone file.
                    if (
                        self.scene_cache.has_scene(registration_key)
                        and self.scene_cache.get_scene_path(registration_key, update_access_time=False).exists()
                    ):
                        log.info(
                            "Scene disk cache hit: scene_id=%s cache_key=%s sha256=%s",
                            scene_key.scene_id,
                            registration_key,
                            existing.backend_cache_key.content_sha256,
                        )
                        return self.scene_cache.acquire_scene(registration_key)
                    # Registration claims downloaded but the file is gone. Remove the
                    # cache entry before its registration so a still-pinned generation
                    # remains retryable after its last owner releases it.
                    if self.scene_cache.has_scene(registration_key):
                        removed = self.scene_cache.acquire_scene(registration_key).discard()
                        if not removed:
                            raise RuntimeError(
                                f"Cached scene {scene_key.scene_id!r} disappeared while it is still in use; "
                                "retry the request"
                            )
                    del self._artifact_registrations[registration_key]

            log.info(
                "Scene disk cache miss; downloading: scene_id=%s cache_key=%s",
                scene_key.scene_id,
                registration_key,
            )
            temp_dir = Path(tempfile.gettempdir())
            temp_path = temp_dir / f"nurec_download_{registration_key}_{int(time.time())}_{os.getpid()}.tmp.usdz"
            try:
                download_start = time.time()
                total_bytes_downloaded = self.downloader.download(scene_url, temp_path)
                download_time = time.time() - download_start
                download_speed = total_bytes_downloaded / (1024 * 1024 * download_time) if download_time > 0 else 0
                log.info(
                    f"Downloaded {total_bytes_downloaded / (1024 * 1024):.2f} MB in {download_time:.2f}s ({download_speed:.2f} MB/s)"
                )

                artifact = Artifact(temp_path)
                registration = ArtifactRegistration(artifact=artifact, downloaded=True, scene_key=scene_key)
                content_sha256 = registration.backend_cache_key.content_sha256

                lease: SceneCacheLease | None = None
                with self._artifact_registry_lock:
                    try:
                        lease, evicted_scene_ids = self.scene_cache.add_and_acquire_scene(registration_key, temp_path)
                        temp_path = None
                        self._remove_downloaded_registrations_locked(evicted_scene_ids)
                        artifact.source = lease.path
                        self._artifact_registrations[registration_key] = registration
                    except Exception:
                        if lease is not None:
                            lease.discard()
                        raise

                log.info(
                    "Scene disk cache miss downloaded: scene_id=%s cache_key=%s sha256=%s",
                    scene_key.scene_id,
                    registration_key,
                    content_sha256,
                )
                return lease
            except Exception:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink()
                raise

    def _get_renderer(self) -> Backend.RendererType:
        if self.server_config is None:
            return Backend.RendererType.DEFAULT
        match self.server_config.renderer:
            case RendererBackend.GSPLAT:
                return Backend.RendererType.GSPLAT
            case RendererBackend.DEFAULT:
                return Backend.RendererType.DEFAULT

    @contextlib.contextmanager
    def get_backend(
        self,
        requested_scene_key: SceneCacheKey,
        context: grpc.ServicerContext | None = None,
    ) -> Generator[Backend, None, None]:
        """
        A context manager that gets an available backend for the requested artifact.

        It creates a new backend if one doesn't exist. If creation fails due to OOM,
        automatically evicts spare backends and retries until successful or no spares remain.
        """

        artifact_id = requested_scene_key.scene_id
        has_explicit_url = requested_scene_key.canonical_url is not None
        selection_lease: SceneCacheLease | None = None
        with self._artifact_registry_lock:
            if has_explicit_url:
                registration_key = requested_scene_key.key_hash
                registration = self._artifact_registrations.get(registration_key)
            else:
                candidates = [
                    (key, candidate)
                    for key, candidate in self._artifact_registrations.items()
                    if candidate.scene_key.scene_id == artifact_id
                ]
                if len(candidates) == 1:
                    registration_key, registration = candidates[0]
                elif len(candidates) > 1:
                    raise SceneNotFoundError(
                        f"Scene {artifact_id!r} has multiple registrations; x-nre-scene-url is required."
                    )
                else:
                    registration_key = requested_scene_key.key_hash
                    registration = None

            if registration is None:
                raise SceneNotFoundError(f"Scene {artifact_id=} not available.")

            if registration.downloaded and not has_explicit_url:
                assert self.scene_cache is not None
                selection_lease = self.scene_cache.acquire_scene(registration_key)

            backend_cache_key = registration.backend_cache_key
            backend_key = backend_cache_key.key_hash

        def create_backend() -> Backend:
            lease: SceneCacheLease | None = None
            with self._artifact_registry_lock:
                current_registration = self._artifact_registrations.get(registration_key)
                if current_registration is not registration:
                    raise SceneNotFoundError(f"Scene {artifact_id=} not available.")
                if registration.resident_only:
                    # The resident, externally-owned backend is gone (evicted or detached). Rebuilding here
                    # would produce a private, non-aliased copy diverging from the owner's IPC-shared
                    # weights, so fail loud rather than silently serve a wrong-renderer scene.
                    raise SceneNotFoundError(
                        f"Scene {artifact_id=} is no longer resident; it cannot be rebuilt in this process."
                    )
                source = registration.artifact.source
                if registration.downloaded and selection_lease is None:
                    assert self.scene_cache is not None
                    lease = self.scene_cache.acquire_scene(registration_key)

            try:
                with Artifact(source).temporary_cache() as artifact:
                    if not registration.downloaded:
                        current_sha256 = artifact.sha256_footprint
                        if current_sha256 != backend_cache_key.content_sha256:
                            raise ValueError(
                                f"Scene {artifact_id!r} changed after registration; use a new artifact source"
                            )
                    return Backend.from_artifact(
                        artifact,
                        self._get_renderer(),
                        self._tracks_timestamp_ray_stride,
                        backend_cache_key=backend_cache_key,
                    )
            finally:
                if lease is not None:
                    self._release_scene_lease(lease)

        def validate_cache_miss() -> None:
            with self._artifact_registry_lock:
                if self._artifact_registrations.get(registration_key) is not registration:
                    raise SceneNotFoundError(f"Scene {artifact_id=} not available.")

        try:
            backend = self.backend_cache.put_with_retries(
                key=backend_key,
                backend_factory=create_backend,
                max_retries=5,
                enable_eviction=True,
                scene_id=artifact_id,
                wait_policy=_grpc_backend_cache_wait_policy(context),
                cache_miss_validator=validate_cache_miss,
            )
        finally:
            if selection_lease is not None:
                self._release_scene_lease(selection_lease)

        try:
            # Yield the backend to the caller (as a context manager)
            yield backend
        except (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError):
            self._recover_from_cuda_oom(f"operation on {artifact_id}")
            raise
        finally:
            # Always return the backend to the cache, even if an exception occurred
            # This prevents backends from getting stuck in the in_use set forever
            self.backend_cache.checkin(backend)

    def _recover_from_cuda_oom(self, operation: str) -> None:
        log.warning(f"OOM during {operation}, attempting memory cleanup")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        evicted_count = 0
        while self.backend_cache.evict_one_spare():
            evicted_count += 1
        if evicted_count > 0:
            log.info(f"Evicted {evicted_count} spare backend(s) to free GPU memory after OOM")


def _serve_once_harmonizer_loaded(service: SensorSimService, health_servicer: health.HealthServicer) -> None:
    """Flip health to SERVING once the background Harmonizer load has finished, successfully or not."""
    service.wait_for_harmonizer_load()
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    load_error = service.harmonizer_load_error
    if load_error is None:
        log.info("Harmonizer model loaded; reporting SERVING.")
    else:
        # Reported SERVING regardless: every RPC that does not need the model is serviceable, and
        # the load error is re-raised to the first render that does.
        log.error(f"Harmonizer model failed to load ({load_error}); reporting SERVING.")


@click.command("serve-grpc")
@grpc_cli_options()
@click.option(
    "--multiproc-workers",
    type=int,
    default=0,
    help="If > 0, serve via the multi-process pool (owner + N workers + proxy, CUDA-IPC shared "
    "weights) instead of the single-process server. Requires NVIDIA MPS for throughput scaling. A glob "
    "matching multiple artifacts serves every scene lazily by scene_id. `0` (default) = single-process; "
    "a negative value is a usage error. `1` is the real owner + one worker + proxy topology (an extra gRPC "
    "hop, for topology/contract testing — no throughput gain over single-process); `>= 2` is a real fleet. "
    "The explicit multiprocess-only options below are usage errors when --multiproc-workers=0.",
)
@click.option(
    "--require-mps/--no-require-mps",
    default=False,
    help="With --multiproc-workers > 1, fail fast if NVIDIA MPS is not active (instead of warning). MPS "
    "must be started externally (operator / container entrypoint / initContainer); this server never "
    "starts it.",
)
@click.option(
    "--multiproc-bringup-concurrency",
    type=int,
    default=1,
    help="With --multiproc-workers, how many workers to bring up at once at startup (1 = serialized, "
    "the default; >1 overlaps startup; <=0 = all in parallel). Each worker transiently holds a full model "
    "copy on the GPU while loading (before aliasing onto the owner's IPC weights), so this caps the "
    "startup VRAM peak — raise it on GPUs with headroom, keep it low on tight ones.",
)
@click.option(
    "--multiproc-max-in-flight",
    type=int,
    default=0,
    help="Admission control: cap concurrent in-flight (active downstream) renders fleet-wide (0 = "
    "unlimited). Over-limit renders are shed RESOURCE_EXHAUSTED (or wait up to "
    "--multiproc-admission-timeout-s). Bounds active worker renders only — not the gRPC executor queue, "
    "scene download, owner load, or attach barrier; it can lower active concurrency below the worker count "
    "but is not a general OOM guarantee (default: 0).",
)
@click.option(
    "--multiproc-admission-timeout-s",
    type=float,
    default=0.0,
    help="How long an over-limit render waits for an admission slot before shedding RESOURCE_EXHAUSTED "
    "(0 = shed immediately). A nonzero timeout requires --multiproc-max-in-flight > 0 (default: 0.0).",
)
@click.option(
    "--multiproc-require-scene-cache/--no-multiproc-require-scene-cache",
    default=False,
    help="Fail fast if --download-cache-size < --cache-size (the on-disk download cache must hold "
    "at least the resident-scene budget, or an evicted remote scene's file can be dropped before re-attach). "
    "Default auto-bumps and reports the adjustment at INFO; local file:// scenes are never affected.",
)
@cli.scopedtimer_cli_options(print_func=log.info)
@cli.tracks_timestamp_ray_stride_cli_option()
@click.version_option(version=str(unpack_optional(get_version(), default="version-not-available")))
@click.pass_context
def serve_grpc(ctx: click.Context, **grpc_config_options) -> None:
    """Neural Rendering gRPC server"""

    # Pop our extra option before building GrpcServerConfig (which doesn't know this key).
    # No fallback: the click option always supplies a value, so a KeyError here is the right
    # failure if it is ever unwired -- and it keeps the default in one place, `cli.py`.
    tracks_timestamp_ray_stride = grpc_config_options.pop("tracks_timestamp_ray_stride")

    # Drop-in switch: these are launch-mode flags, not GrpcServerConfig fields — pop before building.
    multiproc_workers = grpc_config_options.pop("multiproc_workers", 0)
    require_mps = grpc_config_options.pop("require_mps", False)
    multiproc_require_scene_cache = grpc_config_options.pop("multiproc_require_scene_cache", False)
    multiproc_bringup_concurrency = grpc_config_options.pop("multiproc_bringup_concurrency", 1)
    multiproc_max_in_flight = grpc_config_options.pop("multiproc_max_in_flight", 0)
    multiproc_admission_timeout_s = grpc_config_options.pop("multiproc_admission_timeout_s", 0.0)

    # Launch-mode contract: a negative worker count is a usage error (never
    # silently routed to single-process), and the multiprocess-only options are usage errors in
    # single-process mode. Use Click's parameter source so a *default* value (the decorator default) does not
    # make a normal --multiproc-workers=0 invocation invalid — only an *explicitly* supplied incompatible
    # option does. --multiproc-workers=1 stays on the multiprocess path (routing it to single-process would
    # silently change admission/discovery/config-reporting semantics).
    if multiproc_workers < 0:
        raise click.UsageError(f"--multiproc-workers must be >= 0 (0 = single-process); got {multiproc_workers}")

    def _param_is_explicit(param_name: str) -> bool:
        get_parameter_source = getattr(ctx, "get_parameter_source", None)
        if get_parameter_source is None:
            return False
        source = get_parameter_source(param_name)
        return source is not None and source != click.core.ParameterSource.DEFAULT

    # (flag, current value, off/default value). A multiproc-only option is a usage error in single-process
    # mode only when it is BOTH explicitly supplied AND set to something other than its off/default value:
    # a semantically-inert OFF form (`--no-require-mps`, `--multiproc-max-in-flight 0`,
    # `--no-multiproc-require-scene-cache`, ...) still registers as COMMANDLINE via Click's parameter source
    # but changes nothing, so it must be accepted for drop-in wrappers that always emit every flag.
    multiproc_only_options = {
        "require_mps": ("--require-mps", require_mps, False),
        "multiproc_bringup_concurrency": ("--multiproc-bringup-concurrency", multiproc_bringup_concurrency, 1),
        "multiproc_require_scene_cache": ("--multiproc-require-scene-cache", multiproc_require_scene_cache, False),
        "multiproc_max_in_flight": ("--multiproc-max-in-flight", multiproc_max_in_flight, 0),
        "multiproc_admission_timeout_s": ("--multiproc-admission-timeout-s", multiproc_admission_timeout_s, 0.0),
    }
    if multiproc_workers == 0:
        explicit_multiproc_only = [
            flag
            for param_name, (flag, value, off_value) in multiproc_only_options.items()
            if _param_is_explicit(param_name) and value != off_value
        ]
        if explicit_multiproc_only:
            raise click.UsageError(
                f"{', '.join(explicit_multiproc_only)} require --multiproc-workers > 0 (they are ignored in "
                "single-process mode); set --multiproc-workers to a positive value to use them."
            )

    # --multiproc-workers=1 keeps the real owner+worker+proxy topology (an extra gRPC hop, for
    # contract/topology parity), but NVIDIA MPS only helps multiple workers time-share one GPU — so an
    # enabled MPS flag has no effect at one worker. Warn (don't fail, don't change the topology) so an
    # operator who set one at a single worker isn't misled into expecting MPS scaling.
    if multiproc_workers == 1:
        enabled_mps_flags = [flag for enabled, flag in ((require_mps, "--require-mps"),) if enabled]
        if enabled_mps_flags:
            log.warning(
                f"{', '.join(enabled_mps_flags)} set with --multiproc-workers=1 has no effect: NVIDIA MPS is "
                "only used with more than one worker sharing a GPU. Continuing with the single-worker "
                "owner+worker+proxy topology and no MPS."
            )

    # Instantiate config (this will trigger model_post_init and update CLI context if needed)
    grpc_config = GrpcServerConfig(**grpc_config_options)

    # Capture and log CLI settings
    collector = SettingsCollector.from_click_context(ctx, "serve-grpc")
    collector.log_settings(log)

    log.info(f"Renderer backend: {grpc_config.renderer.value}")

    if multiproc_workers > 0:
        # Drop-in multi-process path: same config surface, owner+workers+proxy topology.
        log.info(f"Starting MULTI-PROCESS gRPC server with --multiproc-workers={multiproc_workers}")
        from nurec.grpc.multiproc_serve import (  # type: ignore[import-untyped]
            run_drop_in,
            validate_multiproc_admission,
            validate_multiproc_download_cache,
            validate_multiproc_harmonizer,
            validate_multiproc_mps,
        )

        # Admission cross-option validation surfaces as a usage error.
        admission_error = validate_multiproc_admission(
            max_in_flight=multiproc_max_in_flight,
            admission_timeout_s=multiproc_admission_timeout_s,
        )
        if admission_error is not None:
            raise click.UsageError(admission_error)
        # Temporal-harmonizer sessions are per-process state the proxy cannot route; refuse at launch.
        harmonizer_error = validate_multiproc_harmonizer(grpc_config)
        if harmonizer_error is not None:
            raise click.UsageError(harmonizer_error)
        # Pre-validate the two operator-facing fail-fasts that otherwise raise raw tracebacks from inside
        # run_drop_in (mps_preflight's RuntimeError, _resolve_download_cache_size's ValueError), so the CLI
        # surfaces one clean `Error:` line — matching the admission/harmonizer pattern above.
        mps_error = validate_multiproc_mps(num_workers=multiproc_workers, require_mps=require_mps)
        if mps_error is not None:
            raise click.UsageError(mps_error)
        download_cache_error = validate_multiproc_download_cache(
            grpc_config, require_scene_cache=multiproc_require_scene_cache
        )
        if download_cache_error is not None:
            raise click.UsageError(download_cache_error)

        exit_code = run_drop_in(
            grpc_config,
            num_workers=multiproc_workers,
            require_mps=require_mps,
            require_scene_cache=multiproc_require_scene_cache,
            max_concurrent_bringup=multiproc_bringup_concurrency,
            max_in_flight=multiproc_max_in_flight,
            admission_timeout_s=multiproc_admission_timeout_s,
            # Popped from the CLI options above (not a GrpcServerConfig field); threaded to every
            # owner/worker Backend.from_artifact for render parity with single-process serve.
            tracks_timestamp_ray_stride=tracks_timestamp_ray_stride,
        )
        ScopedTimer.print_summary()
        # Propagate the fleet's terminal exit code so worker death surfaces as a nonzero process exit
        # (supervisor restart); a requested shutdown stays 0.
        ctx.exit(exit_code)

    log.info(
        f"Starting grpc server with {grpc_config.max_workers=} and cache_size={grpc_config.cache_size}, "
        f"max_backends_per_scene={grpc_config.max_backends_per_scene} (LRU with OOM retry)"
    )

    cache_capacity_warning = grpc_config.cache_capacity_warning()
    if cache_capacity_warning is not None:
        log.warning(cache_capacity_warning)

    with ScopedTimer("serve_grpc"):
        scene_download_interceptor = SceneDownloadInterceptor()
        metrics_interceptor = MetricsInterceptor()

        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=grpc_config.max_workers),
            interceptors=[metrics_interceptor, scene_download_interceptor],
            # for lidar output
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50MB
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50MB
            ],
        )

        scene_cache = SceneCache(
            cache_dir=Path(grpc_config.download_cache_dir).expanduser(), max_size=grpc_config.download_cache_size
        )

        downloader = Downloader()

        service = SensorSimService(
            server,
            grpc_config.artifact_glob,
            ray_chunk_size=grpc_config.ray_chunk_size,
            egocar_hoods_dir=grpc_config.egocar_hood_dir,
            downloader=downloader,
            scene_cache=scene_cache,
            cache_size=grpc_config.cache_size,
            max_backends_per_scene=grpc_config.max_backends_per_scene,
            metrics_output_dir=grpc_config.metrics_output_dir,
            enable_editing_actors=grpc_config.enable_editing_actors,
            server_config=grpc_config,
            tracks_timestamp_ray_stride=tracks_timestamp_ray_stride,
        )

        scene_download_interceptor.set_service(service)
        metrics_interceptor.set_metrics_manager(service._metrics_manager)

        if grpc_config.test_scenes_are_valid:
            service.test_all_scenes()

        service.set_harmonizer(
            grpc_config.enable_harmonizer,
            grpc_config.harmonizer_url,
            grpc_config.harmonizer_cache,
            grpc_config.harmonizer_model_filename,
            grpc_config.harmonizer_resolution,
            harmonizer_temporal_enabled=grpc_config.enable_harmonizer_temporal,
            harmonizer_temporal_url=grpc_config.harmonizer_temporal_url,
            harmonizer_temporal_model_filename=grpc_config.harmonizer_temporal_model_filename,
            harmonizer_temporal_resolution=grpc_config.harmonizer_temporal_resolution,
            harmonizer_temporal_cosmos_base_model_url=grpc_config.harmonizer_temporal_cosmos_base_model_url,
            harmonizer_temporal_cosmos_tokenizer_url=grpc_config.harmonizer_temporal_cosmos_tokenizer_url,
            harmonizer_temporal_cosmos_cache_dir=grpc_config.harmonizer_temporal_cosmos_cache_dir,
            max_harmonizer_sessions=grpc_config.max_harmonizer_sessions,
        )

        address = f"{grpc_config.host}:{grpc_config.port}"
        server.add_insecure_port(address)

        add_SensorsimServiceServicer_to_server(service, server)

        health_servicer = health.HealthServicer()
        # HealthServicer starts "" as SERVING, and a dedicated --health-port server starts below,
        # ahead of the main one. Gate readiness here so neither can answer SERVING while the
        # Harmonizer is still loading. A failed load also releases the gate -- the error surfaces on
        # the first render that needs the model, and the rest of the service serves either way.
        if service.harmonizer_load_pending:
            health_servicer.set("", health_pb2.HealthCheckResponse.NOT_SERVING)
            threading.Thread(
                target=_serve_once_harmonizer_loaded,
                args=(service, health_servicer),
                name="harmonizer-health",
                daemon=True,
            ).start()

        health_server: Optional[grpc.Server] = None

        # Health check serving:
        # - Default/legacy: register health service on the main server.
        # - If --health-port is set and differs from --port: start a dedicated health-only gRPC server.
        if grpc_config.health_port is None or grpc_config.health_port == grpc_config.port:
            health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
            log.info(f"Serving on {address} (health on same port)")
        else:
            health_address = f"{grpc_config.host}:{grpc_config.health_port}"
            health_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
            health_pb2_grpc.add_HealthServicer_to_server(health_servicer, health_server)
            health_server.add_insecure_port(health_address)
            log.info(f"Serving main gRPC on {address} and health on {health_address}")
            health_server.start()

        # The port binds before the Harmonizer model is loaded, so readiness is reported by the
        # health service, gated above, rather than inferred from the transport.
        server.start()

        try:
            server.wait_for_termination()
        finally:
            # Cleanup: mark health as down and stop health server if separate
            if health_server is not None:
                try:
                    health_servicer.set("", health_pb2.HealthCheckResponse.NOT_SERVING)
                except Exception:
                    log.exception("Failed to update health status to NOT_SERVING during shutdown")
                health_server.stop(0)

    ScopedTimer.print_summary()
