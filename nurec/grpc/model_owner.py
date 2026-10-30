# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
import threading
import uuid

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import torch


TensorKind = Literal["parameter", "buffer"]


class ModelOwnerError(RuntimeError):
    """Base class for model-owner lifecycle errors."""


class SceneNotLoadedError(ModelOwnerError):
    """Raised when a worker asks for a scene the owner has not loaded."""


class SceneStillAttachedError(ModelOwnerError):
    """Raised when replacing or releasing a scene that still has live attachments."""


class AttachmentNotFoundError(ModelOwnerError):
    """Raised when a worker detaches an unknown or already-detached attachment."""


class WorkerLifecycleError(ModelOwnerError):
    """Raised when a worker's owner->worker lifecycle channel is closed or carries a bad payload."""


class WorkerLifecycleCommand(enum.Enum):
    """Typed owner->worker lifecycle control message sent over the one-way lifecycle channel.

    Worker liveness derives from the OS (the spawned worker's multiprocessing parent-process sentinel), so
    this channel carries only explicit lifecycle control rather than any liveness signal. ``STOP`` is a
    neutral directive to stop serving and exit gracefully; the owner sends it from every teardown path
    (clean shutdown, startup rollback, control-plane degradation, sibling-worker failure), so it deliberately
    does not encode *why* the fleet is stopping — the owner's own terminal reason carries that."""

    STOP = "stop"


@dataclass(frozen=True)
class SharedTensorInfo:
    name: str
    kind: TensorKind
    shape: tuple[int, ...]
    dtype: str
    device: str
    numel: int
    nbytes: int
    data_ptr: int


@dataclass(frozen=True)
class SharedTensorManifest:
    tensors: tuple[SharedTensorInfo, ...]
    total_nbytes: int

    @classmethod
    def from_tensors(cls, tensors: Mapping[str, tuple[TensorKind, torch.Tensor]]) -> SharedTensorManifest:
        infos = tuple(
            _shared_tensor_info(name, kind, tensor)
            for name, (kind, tensor) in sorted(tensors.items(), key=lambda item: item[0])
        )
        return cls(tensors=infos, total_nbytes=sum(info.nbytes for info in infos))

    def by_name(self) -> dict[str, SharedTensorInfo]:
        return {info.name: info for info in self.tensors}


@dataclass(frozen=True)
class SharedTensorBundle:
    tensors: Mapping[str, torch.Tensor]
    manifest: SharedTensorManifest


@dataclass(frozen=True)
class ModelOwnerAttachment:
    scene_key: str
    attachment_id: str
    tensors: Mapping[str, torch.Tensor]
    manifest: SharedTensorManifest


@dataclass(frozen=True)
class AttachSceneRequest:
    scene_key: str


@dataclass(frozen=True)
class DetachSceneRequest:
    attachment_id: str


@dataclass(frozen=True)
class OwnerResponse:
    ok: bool
    attachment: ModelOwnerAttachment | None = None
    error: str | None = None


@dataclass(frozen=True)
class _NamedTensor:
    name: str
    kind: TensorKind
    tensor: torch.Tensor


@dataclass
class _SceneEntry:
    scene_key: str
    owner: Any
    tensors: dict[str, torch.Tensor]
    manifest: SharedTensorManifest
    attachments: set[str] = field(default_factory=set)


def _shared_tensor_info(name: str, kind: TensorKind, tensor: torch.Tensor) -> SharedTensorInfo:
    return SharedTensorInfo(
        name=name,
        kind=kind,
        shape=tuple(int(dim) for dim in tensor.shape),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        numel=int(tensor.numel()),
        nbytes=int(tensor.numel() * tensor.element_size()),
        data_ptr=int(tensor.data_ptr()),
    )


def canonical_model(renderable_or_module: Any) -> torch.nn.Module:
    """Return the canonical torch module whose weights should be shared."""

    model = getattr(renderable_or_module, "_model", renderable_or_module)
    if not hasattr(model, "named_parameters") or not hasattr(model, "named_buffers"):
        raise TypeError(f"Expected a RenderableModel or torch module, got {type(renderable_or_module)!r}")
    return cast(torch.nn.Module, model)


def iter_shared_tensor_entries(renderable_or_module: Any) -> Iterator[_NamedTensor]:
    """Enumerate canonical parameter and buffer tensors for owner-to-worker sharing."""

    model = canonical_model(renderable_or_module)

    for name, parameter in model.named_parameters(recurse=True):
        if parameter is None:
            continue
        yield _NamedTensor(name=name, kind="parameter", tensor=parameter.detach())

    for name, buffer in model.named_buffers(recurse=True):
        if buffer is None:
            continue
        yield _NamedTensor(name=name, kind="buffer", tensor=buffer.detach())


def collect_shared_tensor_bundle(renderable_or_module: Any) -> SharedTensorBundle:
    """Collect detached tensor views plus metadata for a loaded model.

    Detaching the tensors keeps the export autograd-free while preserving the
    same backing storage. The owner registry also stores the original model
    object, so both the module and exported tensor views stay alive.
    """

    tensors_with_kind: dict[str, tuple[TensorKind, torch.Tensor]] = {}
    for entry in iter_shared_tensor_entries(renderable_or_module):
        if entry.name in tensors_with_kind:
            raise ValueError(f"Duplicate shared tensor name: {entry.name}")
        tensors_with_kind[entry.name] = (entry.kind, entry.tensor)

    tensors = {name: tensor for name, (_kind, tensor) in tensors_with_kind.items()}
    return SharedTensorBundle(tensors=tensors, manifest=SharedTensorManifest.from_tensors(tensors_with_kind))


class ModelOwnerRegistry:
    """Owns loaded models and issues ref-counted tensor attachments to workers."""

    def __init__(self, *, attachment_id_factory: Callable[[], str] | None = None) -> None:
        self._attachment_id_factory = attachment_id_factory or (lambda: uuid.uuid4().hex)
        self._lock = threading.RLock()
        self._scenes: dict[str, _SceneEntry] = {}
        self._attachment_to_scene: dict[str, str] = {}

    def load_scene_from_artifact(
        self,
        scene_key: str,
        artifact: Any,
        *,
        config_overrides: tuple[str, ...] = (),
    ) -> SharedTensorManifest:
        from nurec.render.render import RenderableModel

        renderable_model = RenderableModel.load_from_artifact(artifact, config_overrides=config_overrides)
        return self.put_loaded(scene_key, renderable_model)

    def put_loaded(self, scene_key: str, renderable_or_module: Any) -> SharedTensorManifest:
        bundle = collect_shared_tensor_bundle(renderable_or_module)
        with self._lock:
            existing = self._scenes.get(scene_key)
            if existing is not None and existing.attachments:
                raise SceneStillAttachedError(
                    f"Cannot replace scene {scene_key!r}; {len(existing.attachments)} worker(s) are attached"
                )

            self._scenes[scene_key] = _SceneEntry(
                scene_key=scene_key,
                owner=renderable_or_module,
                tensors=dict(bundle.tensors),
                manifest=bundle.manifest,
            )
            return bundle.manifest

    def attach(self, scene_key: str) -> ModelOwnerAttachment:
        with self._lock:
            entry = self._get_entry_locked(scene_key)
            attachment_id = self._new_attachment_id_locked()
            entry.attachments.add(attachment_id)
            self._attachment_to_scene[attachment_id] = scene_key
            return ModelOwnerAttachment(
                scene_key=scene_key,
                attachment_id=attachment_id,
                tensors=dict(entry.tensors),
                manifest=entry.manifest,
            )

    def detach(self, attachment_id: str) -> None:
        with self._lock:
            scene_key = self._attachment_to_scene.pop(attachment_id, None)
            if scene_key is None:
                raise AttachmentNotFoundError(f"Unknown model-owner attachment: {attachment_id!r}")

            entry = self._scenes.get(scene_key)
            if entry is not None:
                entry.attachments.discard(attachment_id)

    def attached_count(self, scene_key: str) -> int:
        with self._lock:
            return len(self._get_entry_locked(scene_key).attachments)

    def manifest_for(self, scene_key: str) -> SharedTensorManifest:
        with self._lock:
            return self._get_entry_locked(scene_key).manifest

    def release_if_idle(self, scene_key: str) -> bool:
        with self._lock:
            entry = self._get_entry_locked(scene_key)
            if entry.attachments:
                return False
            del self._scenes[scene_key]
            return True

    def scene_keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._scenes))

    def _get_entry_locked(self, scene_key: str) -> _SceneEntry:
        entry = self._scenes.get(scene_key)
        if entry is None:
            raise SceneNotLoadedError(f"Model owner has not loaded scene {scene_key!r}")
        return entry

    def _new_attachment_id_locked(self) -> str:
        for _ in range(100):
            attachment_id = self._attachment_id_factory()
            if attachment_id not in self._attachment_to_scene:
                return attachment_id
        raise RuntimeError("Failed to allocate a unique model-owner attachment id")


def handle_owner_request(
    registry: ModelOwnerRegistry, request: AttachSceneRequest | DetachSceneRequest
) -> OwnerResponse:
    try:
        if isinstance(request, AttachSceneRequest):
            return OwnerResponse(ok=True, attachment=registry.attach(request.scene_key))
        if isinstance(request, DetachSceneRequest):
            registry.detach(request.attachment_id)
            return OwnerResponse(ok=True)
    except ModelOwnerError as exc:
        return OwnerResponse(ok=False, error=str(exc))

    raise TypeError(f"Unsupported model-owner request type: {type(request)!r}")


def send_worker_stop(lifecycle_tx: Any) -> None:
    """Send the neutral stop lifecycle command to one worker (owner side).

    A worker that receives this stops serving and exits gracefully (not treated as owner death). The command
    is intentionally neutral: the owner uses it for every teardown path, so it does not imply a client
    ``shut_down`` request. One-way channel: the owner holds the write end; the worker waits on the read end."""
    lifecycle_tx.send(WorkerLifecycleCommand.STOP)


def recv_worker_lifecycle(lifecycle_rx: Any) -> WorkerLifecycleCommand:
    """Receive one already-ready lifecycle command; raise `WorkerLifecycleError` on EOF or a bad payload.

    The caller must have already observed `lifecycle_rx` as readable (e.g. via
    `multiprocessing.connection.wait`), so the `recv` never blocks. It exists so a worker's multi-object
    wait loop (the lifecycle channel **and** the owner's parent-process death sentinel) can own the
    readiness wait itself and use this only to drain and validate the control message. An EOF means the
    owner closed the write end (e.g. the owner exited); a non-command payload is a control-plane contract
    violation. Both surface as `WorkerLifecycleError` so the worker can classify the stop reason."""
    try:
        command = lifecycle_rx.recv()
    except (EOFError, OSError) as exc:
        raise WorkerLifecycleError("Worker lifecycle channel closed") from exc

    if not isinstance(command, WorkerLifecycleCommand):
        raise WorkerLifecycleError(f"Unexpected worker lifecycle payload: {command!r}")
    return command
