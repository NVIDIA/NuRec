# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone spawn-boundary acceptance test for worker lifecycle supervision.

The repository's pytest launcher is not safely re-importable as multiprocessing ``__main__``. This guarded
test executable therefore owns the production ``spawn`` process tree directly and verifies typed lifecycle
serialization, fail-closed EOF/bad-payload exits, and real parent-sentinel behavior after owner SIGKILL.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import signal
import sys
import time

from typing import Any, Literal

from nurec.grpc.model_owner import send_worker_stop
from nurec.grpc.multiproc_serve import (
    WorkerStopReason,
    _owner_parent_signals,
    _wait_for_worker_stop_reason,
)


def _lifecycle_target(lifecycle_rx: Any, result_tx: Any) -> None:
    """Resolve one lifecycle event in a spawn child and exit with the production reason code."""
    try:
        parent_sentinel, is_parent_alive = _owner_parent_signals()
        if parent_sentinel is None:
            result_tx.send("NO_PARENT_SENTINEL")
            raise SystemExit(2)
        reason = _wait_for_worker_stop_reason(lifecycle_rx, parent_sentinel, is_parent_alive)
        result_tx.send(reason.name)
        raise SystemExit(reason.exit_code)
    finally:
        lifecycle_rx.close()
        result_tx.close()


def _run_lifecycle_case(
    case: Literal["stop", "bad", "eof"], expected_reason: WorkerStopReason, expected_exit: int
) -> None:
    ctx = multiprocessing.get_context("spawn")
    lifecycle_rx, lifecycle_tx = ctx.Pipe(duplex=False)
    result_rx, result_tx = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_lifecycle_target, args=(lifecycle_rx, result_tx))
    proc.start()
    lifecycle_rx.close()
    result_tx.close()
    try:
        if case == "stop":
            send_worker_stop(lifecycle_tx)
        elif case == "bad":
            lifecycle_tx.send("not-a-lifecycle-command")
        else:
            lifecycle_tx.close()

        # The spawn child cold-imports the full venv (torch et al.) before _lifecycle_target runs;
        # on a loaded CI runner that alone can exceed 10s, so the budget must cover worst-case
        # import time, not just the lifecycle exchange. Only a failing run waits this long.
        if not result_rx.poll(120.0):
            raise AssertionError(f"spawn lifecycle {case} did not produce a result")
        result = result_rx.recv()
        if result != expected_reason.name:
            raise AssertionError(f"spawn lifecycle {case}: got {result!r}, expected {expected_reason.name!r}")
        proc.join(timeout=30.0)
        if proc.is_alive():
            raise AssertionError(f"spawn lifecycle {case} did not exit")
        if proc.exitcode != expected_exit:
            raise AssertionError(f"spawn lifecycle {case}: exit {proc.exitcode}, expected {expected_exit}")
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5.0)
        with contextlib.suppress(Exception):
            lifecycle_tx.close()
        result_rx.close()


def _sigkill_worker_target(lifecycle_rx: Any, result_tx: Any) -> None:
    """Wait for the direct spawn parent's sentinel and report why the worker stopped.

    Sends "READY" before entering the sentinel wait so the test only SIGKILLs the owner once this
    worker is provably up and watching — killing the owner while the worker is still mid-spawn
    (cold venv import, payload unpickle from the dying parent) is a different scenario than the
    one under test and is racy by nature.
    """
    try:
        parent_sentinel, is_parent_alive = _owner_parent_signals()
        if parent_sentinel is None:
            result_tx.send("NO_PARENT_SENTINEL")
            return
        result_tx.send("READY")
        reason = _wait_for_worker_stop_reason(lifecycle_rx, parent_sentinel, is_parent_alive)
        result_tx.send(reason.name)
    finally:
        lifecycle_rx.close()
        result_tx.close()


def _sigkill_owner_target(lifecycle_rx: Any, result_tx: Any, pid_tx: Any) -> None:
    """Spawn a worker, publish its PID, and stay alive until the test SIGKILLs this owner."""
    ctx = multiprocessing.get_context("spawn")
    worker = ctx.Process(target=_sigkill_worker_target, args=(lifecycle_rx, result_tx))
    worker.start()
    lifecycle_rx.close()
    result_tx.close()
    pid_tx.send(worker.pid)
    pid_tx.close()
    worker.join()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _run_owner_sigkill_case() -> None:
    ctx = multiprocessing.get_context("spawn")
    lifecycle_rx, lifecycle_tx = ctx.Pipe(duplex=False)
    result_rx, result_tx = ctx.Pipe(duplex=False)
    pid_rx, pid_tx = ctx.Pipe(duplex=False)
    owner = ctx.Process(target=_sigkill_owner_target, args=(lifecycle_rx, result_tx, pid_tx))
    owner.start()
    lifecycle_rx.close()
    result_tx.close()
    pid_tx.close()
    owner_pid = owner.pid
    if owner_pid is None:
        raise AssertionError("spawn owner has no PID")
    worker_pid: int | None = None
    try:
        # The owner is itself a spawn child: this budget must cover its cold venv import (torch et
        # al., >10s on a loaded CI runner), not just the trivial spawn+publish.
        if not pid_rx.poll(120.0):
            raise AssertionError("spawn owner did not publish its worker PID")
        published_worker_pid = pid_rx.recv()
        if not isinstance(published_worker_pid, int):
            raise AssertionError(f"spawn owner published invalid worker PID {published_worker_pid!r}")
        worker_pid = published_worker_pid

        # Wait for the worker's READY handshake (its own cold import) before killing the owner:
        # the scenario under test is "owner dies while the worker is watching the sentinel", not
        # "owner dies while the worker is still bootstrapping" (which can crash the spawn payload
        # read and is racy by nature).
        if not result_rx.poll(120.0):
            raise AssertionError("spawn worker never became READY")
        ready = result_rx.recv()
        if ready != "READY":
            raise AssertionError(f"spawn worker reported {ready!r} instead of READY")
        if not _pid_alive(published_worker_pid):
            raise AssertionError("spawn worker exited before owner SIGKILL")

        os.kill(owner_pid, signal.SIGKILL)
        # The worker is provably watching the sentinel now, so resolution is fast; 30s is margin,
        # not an import budget.
        if not result_rx.poll(30.0):
            raise AssertionError("spawn worker did not resolve its parent sentinel after owner SIGKILL")
        result = result_rx.recv()
        if result != WorkerStopReason.OWNER_DEATH.name:
            raise AssertionError(f"spawn owner SIGKILL resolved as {result!r}")

        deadline = time.monotonic() + 30.0
        while _pid_alive(published_worker_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        if _pid_alive(published_worker_pid):
            raise AssertionError("spawn worker remained alive after owner SIGKILL")
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(owner_pid, signal.SIGKILL)
        owner.join(timeout=5.0)
        if worker_pid is not None and _pid_alive(worker_pid):
            with contextlib.suppress(ProcessLookupError):
                os.kill(worker_pid, signal.SIGKILL)
        for connection in (lifecycle_tx, result_rx, pid_rx):
            with contextlib.suppress(Exception):
                connection.close()


def main() -> int:
    _run_lifecycle_case("stop", WorkerStopReason.STOP, 0)
    _run_lifecycle_case("bad", WorkerStopReason.CONTROL_PLANE_FAILURE, 1)
    _run_lifecycle_case("eof", WorkerStopReason.CONTROL_PLANE_FAILURE, 1)
    _run_owner_sigkill_case()
    return 0


if __name__ == "__main__":
    sys.exit(main())
