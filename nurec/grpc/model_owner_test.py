# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import queue
import subprocess
import sys

from collections.abc import Callable, Mapping

import pytest
import torch

from nurec.grpc.model_owner import (
    AttachSceneRequest,
    DetachSceneRequest,
    ModelOwnerAttachment,
    ModelOwnerRegistry,
    SceneNotLoadedError,
    SceneStillAttachedError,
    WorkerLifecycleCommand,
    WorkerLifecycleError,
    collect_shared_tensor_bundle,
    handle_owner_request,
    recv_worker_lifecycle,
    send_worker_stop,
)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(3, 2))
        self.register_buffer("scale", torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))


class RenderableWrapper:
    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model


def _attachment_id_factory(*attachment_ids: str) -> Callable[[], str]:
    iterator = iter(attachment_ids)
    return lambda: next(iterator)


def _checksums(tensors: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(tensor.detach().to(dtype=torch.float32).sum().cpu().item()) for name, tensor in tensors.items()}


def _attachment_checksum_worker(request_queue, response_queue) -> None:
    try:
        attachment: ModelOwnerAttachment = request_queue.get(timeout=10.0)
        response_queue.put(("ok", attachment.attachment_id, _checksums(attachment.tensors)))
    except BaseException as exc:
        response_queue.put(("error", repr(exc), {}))


def test_collect_shared_tensor_bundle_enumerates_parameters_and_buffers() -> None:
    model = TinyModel()

    bundle = collect_shared_tensor_bundle(model)

    assert set(bundle.tensors) == {"linear.weight", "scale"}
    by_name = bundle.manifest.by_name()
    assert by_name["linear.weight"].kind == "parameter"
    assert by_name["linear.weight"].shape == (3, 2)
    assert by_name["linear.weight"].dtype == "torch.float32"
    assert by_name["linear.weight"].data_ptr == model.linear.weight.data_ptr()
    assert by_name["scale"].kind == "buffer"
    assert by_name["scale"].shape == (3,)
    assert bundle.manifest.total_nbytes == sum(
        tensor.numel() * tensor.element_size() for tensor in bundle.tensors.values()
    )


def test_collect_shared_tensor_bundle_uses_renderable_private_model() -> None:
    model = TinyModel()

    bundle = collect_shared_tensor_bundle(RenderableWrapper(model))

    assert bundle.tensors["linear.weight"].data_ptr() == model.linear.weight.data_ptr()
    assert bundle.tensors["scale"].data_ptr() == model.get_buffer("scale").data_ptr()


def test_registry_attach_detach_keeps_scene_until_idle_release() -> None:
    model = TinyModel()
    registry = ModelOwnerRegistry(attachment_id_factory=_attachment_id_factory("attach-1"))
    registry.put_loaded("scene-a", model)

    attachment = registry.attach("scene-a")

    assert attachment.attachment_id == "attach-1"
    assert attachment.scene_key == "scene-a"
    assert registry.attached_count("scene-a") == 1
    assert registry.release_if_idle("scene-a") is False
    assert attachment.tensors["linear.weight"].data_ptr() == model.linear.weight.data_ptr()

    registry.detach(attachment.attachment_id)

    assert registry.attached_count("scene-a") == 0
    assert registry.release_if_idle("scene-a") is True
    with pytest.raises(SceneNotLoadedError):
        registry.attach("scene-a")


def test_registry_rejects_replacing_scene_with_live_attachment() -> None:
    registry = ModelOwnerRegistry(attachment_id_factory=_attachment_id_factory("attach-1"))
    registry.put_loaded("scene-a", TinyModel())
    attachment = registry.attach("scene-a")

    with pytest.raises(SceneStillAttachedError):
        registry.put_loaded("scene-a", TinyModel())

    registry.detach(attachment.attachment_id)
    registry.put_loaded("scene-a", TinyModel())
    assert registry.attached_count("scene-a") == 0


def test_handle_owner_request_attaches_detaches_and_reports_errors() -> None:
    registry = ModelOwnerRegistry(attachment_id_factory=_attachment_id_factory("attach-1"))

    missing = handle_owner_request(registry, AttachSceneRequest("missing-scene"))

    assert not missing.ok
    assert missing.error is not None and "missing-scene" in missing.error

    registry.put_loaded("scene-a", TinyModel())
    attached = handle_owner_request(registry, AttachSceneRequest("scene-a"))

    assert attached.ok
    assert attached.attachment is not None
    assert registry.attached_count("scene-a") == 1

    detached = handle_owner_request(registry, DetachSceneRequest(attached.attachment.attachment_id))

    assert detached.ok
    assert registry.attached_count("scene-a") == 0


def test_worker_lifecycle_stop_round_trips() -> None:
    # Fast in-process helper coverage; the standalone lifecycle spawn test covers production serialization.
    ctx = torch.multiprocessing.get_context("fork")
    lifecycle_rx, lifecycle_tx = ctx.Pipe(duplex=False)
    try:
        send_worker_stop(lifecycle_tx)
        command = recv_worker_lifecycle(lifecycle_rx)
    finally:
        lifecycle_rx.close()
        lifecycle_tx.close()

    assert command is WorkerLifecycleCommand.STOP


def test_recv_worker_lifecycle_reports_channel_eof_as_error() -> None:
    # A closed write end (owner exited) makes the read end readable-with-EOF; recv must surface loss so the
    # worker can classify it (owner-death vs. control-plane failure) against the parent-process sentinel.
    ctx = torch.multiprocessing.get_context("fork")
    lifecycle_rx, lifecycle_tx = ctx.Pipe(duplex=False)
    lifecycle_tx.close()
    try:
        with pytest.raises(WorkerLifecycleError, match="closed"):
            recv_worker_lifecycle(lifecycle_rx)
    finally:
        lifecycle_rx.close()


def test_recv_worker_lifecycle_rejects_unexpected_payload() -> None:
    # A non-command payload is a control-plane contract violation, surfaced as a lifecycle error.
    ctx = torch.multiprocessing.get_context("fork")
    lifecycle_rx, lifecycle_tx = ctx.Pipe(duplex=False)
    try:
        lifecycle_tx.send("not-a-lifecycle-command")
        with pytest.raises(WorkerLifecycleError, match="Unexpected"):
            recv_worker_lifecycle(lifecycle_rx)
    finally:
        lifecycle_rx.close()
        lifecycle_tx.close()


def test_attachment_round_trips_through_torch_multiprocessing_queue() -> None:
    registry = ModelOwnerRegistry(attachment_id_factory=_attachment_id_factory("attach-1"))
    registry.put_loaded("scene-a", TinyModel())
    attachment = registry.attach("scene-a")
    expected_checksums = _checksums(attachment.tensors)

    # Spawn-process coverage lives in //nurec/grpc:worker_lifecycle_spawn_test. Under Bazel's pytest
    # wrapper, a spawned child re-enters pytest during bootstrap, so this unit test uses fork to
    # keep the queue serialization check hermetic and fast.
    ctx = torch.multiprocessing.get_context("fork")
    request_queue = ctx.Queue()
    response_queue = ctx.Queue()
    process = ctx.Process(target=_attachment_checksum_worker, args=(request_queue, response_queue))

    result = None
    process.start()
    try:
        request_queue.put(attachment)
        result = response_queue.get(timeout=20.0)
    except queue.Empty:
        pytest.fail("timed out waiting for model-owner attachment worker")
    finally:
        process.join(timeout=20.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=20.0)
        request_queue.close()
        response_queue.close()
        request_queue.join_thread()
        response_queue.join_thread()

    assert process.exitcode == 0
    assert result == ("ok", attachment.attachment_id, expected_checksums)


# A standalone (fresh-interpreter) worker that opens the owner's CUDA IPC handle. It must run in a real
# separate process (CUDA IPC is cross-process by definition) and cannot use torch.multiprocessing `spawn`
# here — the repo's pytest launcher is not import-safe as a multiprocessing main (it re-runs pytest). So the
# worker is launched via `subprocess` with a clean __main__, receiving the CUDA tensor as bytes produced by
# torch's ForkingPickler (which emits the CUDA IPC handle, not the data), and rebuilds it with plain pickle.
_CUDA_IPC_WORKER_SRC = """
import os, pickle, sys, time, torch

# On DGX Spark the GPU allocates from ordinary system memory and a CUDA primary context needs
# 2 MiB-contiguous pages, which a parallel Bazel build starves for seconds at a time. The
# rules_python stage2 bootstrap retries for a normal test process, but this worker is started as a
# plain interpreter and so never runs it. Establish the context here instead, retrying while the
# driver reports it cannot back one. Gated to aarch64, as the bootstrap is.
if os.uname().machine == "aarch64":
    # "busy or unavailable" is the other symptom the bootstrap's flock guards against. This
    # worker does not take that lock, so it retries out of both conditions instead.
    _retryable = ("out of memory", "busy or unavailable")
    _deadline = time.monotonic() + 60.0
    while True:
        try:
            torch.zeros(1, device="cuda")
            break
        except RuntimeError as _e:
            if not any(m in str(_e) for m in _retryable) or time.monotonic() >= _deadline:
                raise
            time.sleep(0.5)

with open(sys.argv[1], "rb") as handle:
    tensor = pickle.load(handle)  # opens the owner's CUDA IPC handle -> a real cross-process GPU map
if not tensor.is_cuda:
    raise SystemExit("worker did not receive a CUDA tensor")
checksum = float(tensor.detach().to(torch.float32).sum().cpu().item())
with open(sys.argv[2], "w") as out:
    out.write(repr(checksum))
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU for real CUDA-IPC")
def test_cuda_ipc_attachment_lifetime_across_real_worker_exit(tmp_path) -> None:
    # Anchor the CUDA-IPC lifetime contract with REAL CUDA tensors and a REAL separate worker process (the
    # existing coverage uses CPU tensors / monkeypatched torch.cuda). The owner must keep a scene's CUDA
    # tensors alive for the entire window a worker maps them via IPC, and free them only after the worker's
    # proven exit -> otherwise release_if_idle would cudaFree tensors a live worker still maps (a UAF).
    from multiprocessing.reduction import ForkingPickler

    import torch.multiprocessing  # noqa: F401 - importing it registers torch's CUDA reducers on ForkingPickler

    model = TinyModel().cuda()
    registry = ModelOwnerRegistry(attachment_id_factory=_attachment_id_factory("attach-cuda"))
    registry.put_loaded("scene-cuda", model)

    attachment = registry.attach("scene-cuda")
    weight = attachment.tensors["linear.weight"]
    assert weight.is_cuda  # a real CUDA tensor is exported for IPC
    assert weight.data_ptr() == model.linear.weight.data_ptr()  # zero-copy alias of the owner's storage
    expected_checksum = float(weight.detach().to(torch.float32).sum().cpu().item())

    # While the worker is (about to be) attached, the owner MUST NOT free the scene.
    assert registry.release_if_idle("scene-cuda") is False

    # Serialize the CUDA tensor the way a torch.multiprocessing queue would (ForkingPickler emits the CUDA
    # IPC handle, not the bytes), hand it to a separate process, and prove that process opened the handle.
    handle_path = tmp_path / "cuda_attachment.pkl"
    handle_path.write_bytes(bytes(ForkingPickler.dumps(weight)))
    result_path = tmp_path / "checksum.txt"

    # Propagate the parent's import paths so the fresh interpreter can `import torch` (bazel runfiles).
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    worker = subprocess.run(
        [sys.executable, "-c", _CUDA_IPC_WORKER_SRC, str(handle_path), str(result_path)],
        capture_output=True,
        text=True,
        timeout=180,
        env=child_env,
    )
    assert worker.returncode == 0, f"CUDA IPC worker failed (rc={worker.returncode}): {worker.stderr}"
    assert float(result_path.read_text()) == pytest.approx(expected_checksum)  # worker read the owner tensor via IPC

    # The worker has now provably exited (subprocess.run joined). Only after that proven exit does the owner
    # drop its refcount and free the scene: release stays False while attached, True once detached.
    assert registry.release_if_idle("scene-cuda") is False  # still charged until the owner detaches
    registry.detach(attachment.attachment_id)
    assert registry.release_if_idle("scene-cuda") is True  # idle after the proven worker exit -> freeable
