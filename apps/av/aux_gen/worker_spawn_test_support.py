# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared process-spawn harness for aux-gen loader payload tests."""

import multiprocessing

from multiprocessing.connection import Connection

from apps.av.aux_gen.ncore_aux_data import LoaderWorkerPayload, _open_worker_sequence


def _open_payload_worker(payload: LoaderWorkerPayload, sender: Connection) -> None:
    backend, loader = _open_worker_sequence(payload)
    sender.send((backend.name, str(loader.sequence_id)))
    sender.close()


def open_payload_in_spawned_process(payload: LoaderWorkerPayload) -> tuple[str, str]:
    """Reopen a payload in a fresh interpreter and return its identity."""

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_open_payload_worker, args=(payload, sender))
    process.start()
    sender.close()
    process.join(timeout=60)
    if process.is_alive():
        process.terminate()
        process.join()
        raise AssertionError("Timed out while reopening an NCore dataset in a spawned worker")
    if process.exitcode != 0:
        raise AssertionError(f"Spawned worker exited with code {process.exitcode}")
    if not receiver.poll():
        raise AssertionError("Spawned worker exited without returning a loader result")
    return receiver.recv()
