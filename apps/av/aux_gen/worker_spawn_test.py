# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise aux loader payloads through Python's multiprocessing spawn path."""

import multiprocessing

from pathlib import Path
from typing import Any
from unittest.mock import patch

from python.runfiles import runfiles
from upath import UPath

from apps.av.aux_gen.device_frame_service import DeviceFrameStopSignal
from apps.av.aux_gen.ncore_aux_data import (
    LoaderWorkerPayload,
    ManifestLoaderWorkerPayload,
    _segmentation_host_worker_when_ready,
    _SegmentationHostWorkerConfig,
    _SegmentationHostWorkerReady,
)
from apps.av.aux_gen.worker_spawn_test_support import SPAWN_TEST_TIMEOUT_SECONDS, open_payload_in_spawned_process
from nurec.ncore_backends import DatasetRequest, resolve_dataset


RUNFILES = runfiles.Create()
SEQUENCE = "clipgt-9048443e-c482-4228-8326-5b3dff3be711"


def _runfile(repository: str, filename: str) -> Path:
    path = Path(RUNFILES.Rlocation(f"{repository}/{filename}"))
    if not path.is_file():
        raise AssertionError(f"Test data not found at {path}")
    return path


def _manifest_payload() -> LoaderWorkerPayload:
    manifest_path = str(_runfile("test_data_ncore_v4", f"{SEQUENCE}.json"))
    resolved = resolve_dataset(DatasetRequest(metadata_path=UPath(manifest_path)))
    return ManifestLoaderWorkerPayload(resolved_dataset=resolved)


def _receive_ready_in_spawned_process(config: _SegmentationHostWorkerConfig, updates: Any, results: Any) -> None:
    """Observe the actual ready payload without running camera inference."""

    def report(startup: _SegmentationHostWorkerConfig, ready: _SegmentationHostWorkerReady) -> None:
        assert isinstance(ready, _SegmentationHostWorkerReady)
        assert ready.auxiliary_camera_stats is not None
        ready.auxiliary_camera_stats.append({"camera_id": startup.camera_id})
        results.put((ready.semantic_metadata, ready.sam2_metadata))

    with patch("apps.av.aux_gen.ncore_aux_data._segmentation_host_worker", side_effect=report):
        _segmentation_host_worker_when_ready(config, updates)


def main() -> None:
    payload = _manifest_payload()
    backend_name, sequence_id = open_payload_in_spawned_process(payload)
    if backend_name != "ncore-v4" or sequence_id != SEQUENCE:
        raise AssertionError(f"Expected {('ncore-v4', SEQUENCE)}, got {(backend_name, sequence_id)}")

    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        statistics = manager.list()
        ready_receiver, ready_sender = context.Pipe(duplex=False)
        results = context.Queue(maxsize=1)
        config = _SegmentationHostWorkerConfig(
            worker_id=0,
            loader_payload=payload,
            zarr_store_type="itar",
            output_path=Path("/unused"),
            camera_id="camera",
            camera_frame_range=range(0),
            resolution=(1, 1),
            visualize=False,
            slot_queue=context.Queue(),
            sam2_sample_indices=(),
            ego_mask_aggregation_method="majority",
            request_queue=context.Queue(),
            result_queue=context.Queue(),
            error_queue=context.Queue(),
            stop_event=DeviceFrameStopSignal.create(context),
        )
        ready = _SegmentationHostWorkerReady(
            semantic_metadata={"stuff_classes": ["road"]},
            sam2_metadata={"method": "sam2"},
            auxiliary_camera_stats=statistics,
        )
        owned_queues = [
            results,
            config.slot_queue,
            config.request_queue,
            config.result_queue,
            config.error_queue,
        ]
        owned_connections = [ready_receiver, ready_sender]
        children: list[Any] = []
        try:
            child = context.Process(target=_receive_ready_in_spawned_process, args=(config, ready_receiver, results))
            child.start()
            children.append(child)
            ready_receiver.close()
            ready_sender.send(ready)
            assert results.get(timeout=SPAWN_TEST_TIMEOUT_SECONDS) == (ready.semantic_metadata, ready.sam2_metadata)
            child.join(timeout=SPAWN_TEST_TIMEOUT_SECONDS)
            assert not child.is_alive(), "Ready worker did not exit"
            assert child.exitcode == 0

            config.stop_event.set()
            cancelled_receiver, cancelled_sender = context.Pipe(duplex=False)
            owned_connections.extend((cancelled_receiver, cancelled_sender))
            cancelled = context.Process(
                target=_receive_ready_in_spawned_process, args=(config, cancelled_receiver, results)
            )
            cancelled.start()
            children.append(cancelled)
            cancelled_receiver.close()
            cancelled.join(timeout=SPAWN_TEST_TIMEOUT_SECONDS)
            assert not cancelled.is_alive(), "Cancelled worker did not observe stop_event"
            assert cancelled.exitcode == 0
            assert list(statistics) == [{"camera_id": "camera"}]

            invalid_receiver, invalid_sender = context.Pipe(duplex=False)
            owned_connections.extend((invalid_receiver, invalid_sender))
            invalid_queue = context.Queue()
            owned_queues.append(invalid_queue)
            invalid_ready = _SegmentationHostWorkerReady(
                semantic_metadata=None, sam2_metadata=None, auxiliary_camera_stats=invalid_queue
            )
            try:
                invalid_sender.send(invalid_ready)
            except RuntimeError as error:
                assert "Queue objects should only be shared" in str(error)
            else:
                raise AssertionError("Pipe.send accepted a readiness payload that cannot be pickled")
        finally:
            for process in children:
                if process.is_alive():
                    process.terminate()
                    process.join()
            config.stop_event.close()
            for connection in owned_connections:
                connection.close()
            for owned_queue in owned_queues:
                owned_queue.cancel_join_thread()
                owned_queue.close()


if __name__ == "__main__":
    main()
