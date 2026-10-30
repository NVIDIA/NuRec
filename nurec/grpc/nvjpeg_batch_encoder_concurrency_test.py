# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone deadlock regression test for concurrent native nvJPEG encoding.

The repository's pytest launcher is not safe to re-import as a multiprocessing
``__main__``. This guarded executable owns a spawn child so the parent can kill
the child and fail cleanly if the native encoder deadlocks while reacquiring the
Python GIL.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import sys
import threading

from multiprocessing.connection import Connection

import torch

from nurec.grpc.nvjpeg_batch_encoder import NvjpegBatchEncoder


def _run_concurrent_encodes(result_tx: Connection) -> None:
    """Share one encoder across two threads and report completion to the parent."""
    try:
        if not torch.cuda.is_available():
            result_tx.send(("SKIP", "native nvJPEG encoding requires a CUDA GPU"))
            return

        encoder = NvjpegBatchEncoder()
        rows = torch.arange(512, dtype=torch.int16, device="cuda")[:, None, None]
        columns = torch.arange(512, dtype=torch.int16, device="cuda")[None, :, None]
        images = [
            torch.cat(
                (
                    (rows + columns + offset) % 256,
                    (rows * 3 + columns + offset) % 256,
                    (rows + columns * 5 + offset) % 256,
                ),
                dim=2,
            )
            .to(torch.uint8)
            .contiguous()
            for offset in (17, 93)
        ]
        qualities = (35, 95)
        batch_sizes = (3, 5)
        expected = [
            NvjpegBatchEncoder().encode([image] * batch_size, quality)
            for image, batch_size, quality in zip(images, batch_sizes, qualities, strict=True)
        ]

        # Validation must happen before any work is queued, and a rejected batch must not poison
        # the encoder for the next call.
        try:
            encoder.encode([images[0], torch.empty((0, 8, 3), dtype=torch.uint8, device="cuda")], quality=90)
            raise AssertionError("zero-height image was accepted")
        except (RuntimeError, ValueError):
            pass
        recovered = encoder.encode([images[0]], quality=90)
        if len(recovered) != 1 or not recovered[0].startswith(b"\xff\xd8"):
            raise AssertionError("encoder did not recover after rejecting an invalid batch")

        try:
            encoder.encode([images[0]] * (encoder.max_batch_size + 1), quality=90)
            raise AssertionError("oversized native batch was accepted")
        except (RuntimeError, ValueError):
            pass

        if torch.cuda.device_count() > 1:
            second_device_image = images[0].to("cuda:1")
            cross_device = encoder.encode([second_device_image], quality=90)
            if len(cross_device) != 1 or not cross_device[0].startswith(b"\xff\xd8"):
                raise AssertionError("per-device encoder returned an invalid JPEG")

        start_barrier = threading.Barrier(2)
        errors: list[str] = []
        result_tx.send(("READY", ""))

        def worker(index: int) -> None:
            try:
                for _ in range(3):
                    start_barrier.wait(timeout=10.0)
                    encoded = encoder.encode(
                        [images[index]] * batch_sizes[index],
                        quality=qualities[index],
                    )
                    if len(encoded) != batch_sizes[index]:
                        raise AssertionError(f"thread {index} received {len(encoded)} outputs")
                    if any(not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9") for jpeg in encoded):
                        raise AssertionError(f"thread {index} received an invalid JPEG payload")
                    if encoded != expected[index]:
                        raise AssertionError(
                            f"thread {index} received bytes from another image, batch size, or quality"
                        )
            except BaseException as exc:
                errors.append(f"thread {index}: {type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=worker, args=(index,), daemon=True) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30.0)

        alive_threads = [thread.name for thread in threads if thread.is_alive()]
        if alive_threads:
            errors.append(f"encoder threads did not finish: {alive_threads}")
        result_tx.send(("ERROR", "; ".join(errors)) if errors else ("PASS", ""))
    except BaseException as exc:
        result_tx.send(("ERROR", f"{type(exc).__name__}: {exc}"))
    finally:
        result_tx.close()


def main() -> int:
    ctx = multiprocessing.get_context("spawn")
    result_rx, result_tx = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_run_concurrent_encodes, args=(result_tx,))
    process.start()
    result_tx.close()
    try:
        # The child cold-imports torch, initializes CUDA, and builds two sizeable inputs before
        # READY. This budget covers startup on a loaded CI runner; only a failing run uses it all.
        if not result_rx.poll(120.0):
            raise AssertionError("nvJPEG concurrency child did not finish CUDA startup")
        status, detail = result_rx.recv()
        if status == "SKIP":
            print(f"SKIP: {detail}")
            process.join(timeout=10.0)
            return 0
        if status != "READY":
            raise AssertionError(f"nvJPEG concurrency child failed during startup: {detail}")

        # A timeout here is the regression signal: one native call is waiting on the encoder
        # mutex while holding the GIL, and the mutex owner cannot reacquire the GIL to return.
        if not result_rx.poll(30.0):
            raise AssertionError(
                "concurrent calls on one NvjpegBatchEncoder deadlocked; the child made no progress for 30 seconds"
            )
        status, detail = result_rx.recv()
        if status != "PASS":
            raise AssertionError(f"concurrent nvJPEG encode failed: {detail}")

        process.join(timeout=10.0)
        if process.is_alive():
            raise AssertionError("nvJPEG concurrency child reported success but did not exit")
        if process.exitcode != 0:
            raise AssertionError(f"nvJPEG concurrency child exited with status {process.exitcode}")
        return 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5.0)
        with contextlib.suppress(Exception):
            result_rx.close()


if __name__ == "__main__":
    sys.exit(main())
