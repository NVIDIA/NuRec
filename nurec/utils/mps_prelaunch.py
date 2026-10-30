# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stdlib-only NVIDIA MPS *detection* helper.

MPS is started **externally** — by the operator, the container entrypoint, a Kubernetes initContainer, or
systemd — never by this process. This module only *detects* whether a live MPS control daemon is already
running, so the multi-process gRPC preflight (``nurec.grpc.multiproc_serve.mps_active`` / ``mps_preflight``,
the sole consumer) can warn — or, with ``--require-mps``, fail fast — when a multi-worker pool is launched
without MPS.

Detection is kept to the standard library on purpose: it probes MPS by shelling out to
``nvidia-cuda-mps-control`` with a read-only control query rather than by touching torch/CUDA, so the
preflight can run in the launcher before the worker pool is spawned without initializing a CUDA context in
that process. Nothing here imports torch.
"""

from __future__ import annotations

import enum
import os
import shutil
import stat
import subprocess


_LIVE_PROBE_TIMEOUT_S = 3.0  # bound on the read-only control query used to detect a live daemon
# A read-only control command: a live daemon accepts it (rc 0); a stale/absent socket fails to connect.
_LIVE_PROBE_COMMAND = "get_default_active_thread_percentage\n"
_CONTROL_SOCKET = "control"  # the control daemon's socket inside the pipe dir


def _pipe_dir_candidates() -> list[str]:
    """Pipe dirs to probe when *detecting* an already-running daemon.

    If ``CUDA_MPS_PIPE_DIRECTORY`` is set, only that dir counts. Otherwise probe both the CUDA default
    ``/tmp/nvidia-mps`` (where an operator-managed daemon lands with no env override) and the per-uid dir
    ``/tmp/nvidia-mps-{uid}`` (where a per-user daemon lands). Keeping this list in one helper is what lets
    ``multiproc_serve.mps_active`` (the in-process preflight) and this detection helper agree on what
    "MPS is up" means.
    """
    env_dir = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    if env_dir:
        return [env_dir]
    return [f"/tmp/nvidia-mps-{os.getuid()}", "/tmp/nvidia-mps"]


def _trusted_pipe_dir(path: str) -> bool:
    """True iff ``path`` is a directory whose control socket is worth probing.

    The candidate dirs live under ``/tmp``, which is world-writable, so any local user can create
    ``/tmp/nvidia-mps`` before a real daemon does and leave something at ``control``. A daemon's own pipe
    dir is created by whoever started it — root for a system-wide or initContainer-managed daemon, this
    user for a per-uid one — and is not writable by others, so requiring both ownership and no world-write
    skips a planted directory instead of probing it.

    Fails closed like the rest of detection: a dir that cannot be stat'ed is not trusted, and an untrusted
    candidate reads as "not up" rather than as an error.
    """
    try:
        info = os.stat(path)
    except OSError:
        return False
    if not stat.S_ISDIR(info.st_mode):
        return False
    if info.st_uid not in (0, os.getuid()):
        return False
    return not info.st_mode & stat.S_IWOTH


class _ProbeResult(enum.Enum):
    """Tri-state outcome of a read-only control-socket probe (distinguishes "gone" from "cannot tell").

    A boolean would collapse ``ABSENT`` and ``UNKNOWN`` into ``False`` and lose the distinction between a
    daemon whose absence is *verified* and an *indeterminate* probe (a hung binary, a probe timeout, or a
    missing binary). Detection (``_control_daemon_responds`` / ``mps_active``) treats anything short of a
    confirmed ``LIVE`` as "not up".
    """

    LIVE = "live"  # the daemon answered the read-only query (rc 0)
    ABSENT = "absent"  # the control binary ran but could not connect (rc != 0): the daemon is gone
    UNKNOWN = "unknown"  # indeterminate: missing binary, probe timeout, or other error -> cannot verify absence


def _probe_control_daemon(pipe_dir: str, timeout_s: float = _LIVE_PROBE_TIMEOUT_S) -> _ProbeResult:
    """Bounded read-only control probe of the daemon at ``pipe_dir``, as a tri-state ``_ProbeResult``.

    Runs ``nvidia-cuda-mps-control`` with a read-only command: a live daemon accepts it (rc 0 -> ``LIVE``),
    while a stale or absent control socket makes the control binary fail to connect (rc != 0 -> ``ABSENT``, a
    *verified* absence — a crashed daemon's leftover socket does not pass). A missing binary, a probe timeout,
    or any other error is ``UNKNOWN``: liveness could not be determined, so detection treats it as "not up"
    rather than assuming the daemon is dead.
    """
    control_bin = shutil.which("nvidia-cuda-mps-control")
    if control_bin is None:
        return _ProbeResult.UNKNOWN  # cannot probe -> cannot verify absence
    env = dict(os.environ)
    env["CUDA_MPS_PIPE_DIRECTORY"] = os.path.expanduser(pipe_dir)
    try:
        result = subprocess.run(
            [control_bin],
            input=_LIVE_PROBE_COMMAND,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return _ProbeResult.UNKNOWN  # missing/hung binary or timeout -> indeterminate (never "proven absent")
    return _ProbeResult.LIVE if result.returncode == 0 else _ProbeResult.ABSENT


def _control_daemon_responds(pipe_dir: str) -> bool:
    """True iff a bounded read-only control query confirms a *live* daemon at ``pipe_dir`` (detection helper).

    A thin boolean view of ``_probe_control_daemon`` for *detection* (``mps_active``): only a confirmed
    ``LIVE`` counts as up, so a stale socket (``ABSENT``) and an indeterminate probe (``UNKNOWN`` — missing
    binary / timeout / error) both read as "not up" and the caller then treats MPS as down.
    """
    return _probe_control_daemon(pipe_dir) is _ProbeResult.LIVE


def mps_active(pipe_dir: str | None = None) -> bool:
    """True iff a *live* MPS control daemon is validated in a candidate pipe dir.

    A daemon counts as up only when its control socket (``<dir>/control``) exists **and** a bounded
    read-only control query to it succeeds, in a dir that passes ``_trusted_pipe_dir``. A stale control
    socket left by a crashed daemon (nothing listening) therefore returns ``False`` and never satisfies
    ``--require-mps``. With ``pipe_dir`` given,
    only that dir is checked; otherwise every candidate from ``_pipe_dir_candidates()`` is probed.
    Stdlib-only (no torch import) so it is safe to call before the CUDA-initializing imports;
    ``multiproc_serve.mps_active`` delegates here so all callers agree on what "MPS is up" means.
    """
    candidates = [pipe_dir] if pipe_dir else _pipe_dir_candidates()
    for candidate in candidates:
        expanded = os.path.expanduser(candidate)
        if not _trusted_pipe_dir(expanded):
            continue
        if os.path.exists(os.path.join(expanded, _CONTROL_SOCKET)) and _control_daemon_responds(expanded):
            return True
    return False
