# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the stdlib-only NVIDIA MPS *detection* helper.

MPS is started externally; this module only detects a live control daemon, so these tests cover the
candidate pipe dirs, the tri-state control probe, and the ``mps_active`` / ``_control_daemon_responds``
detection wrappers the multi-process gRPC preflight relies on.
"""

import os
import stat

from pathlib import Path
from types import SimpleNamespace

from nurec.utils import mps_prelaunch


def _seq_raise(exc):
    """A stubbed callable that always raises ``exc`` (arg-tolerant)."""

    def _raise(*_a, **_k):
        raise exc

    return _raise


# ---------------------------------------------------------------------------
# candidate pipe dirs probed during detection
# ---------------------------------------------------------------------------


def test_pipe_dir_candidates_with_env(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/custom/pipe")
    assert mps_prelaunch._pipe_dir_candidates() == ["/custom/pipe"]


def test_pipe_dir_candidates_without_env(monkeypatch) -> None:
    # With no env override, BOTH the CUDA default and the per-uid dir must be probed so an operator daemon
    # at either path is detected (matches multiproc_serve.mps_active).
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    candidates = mps_prelaunch._pipe_dir_candidates()
    assert "/tmp/nvidia-mps" in candidates
    assert f"/tmp/nvidia-mps-{os.getuid()}" in candidates


# ---------------------------------------------------------------------------
# pipe-dir trust: /tmp is world-writable, so a planted dir must not be probed
# ---------------------------------------------------------------------------


def test_world_writable_pipe_dir_is_not_trusted(tmp_path: Path) -> None:
    tmp_path.chmod(0o777)
    assert mps_prelaunch._trusted_pipe_dir(str(tmp_path)) is False


def test_pipe_dir_owned_by_another_user_is_not_trusted(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(mps_prelaunch.os, "getuid", lambda: os.stat(tmp_path).st_uid + 1)
    assert mps_prelaunch._trusted_pipe_dir(str(tmp_path)) is False


def test_own_private_pipe_dir_is_trusted(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    assert mps_prelaunch._trusted_pipe_dir(str(tmp_path)) is True


def test_missing_or_non_directory_is_not_trusted(tmp_path: Path) -> None:
    assert mps_prelaunch._trusted_pipe_dir(str(tmp_path / "nope")) is False
    plain = tmp_path / "file"
    plain.write_text("")
    assert mps_prelaunch._trusted_pipe_dir(str(plain)) is False


def test_mps_active_skips_planted_dir_without_probing(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "control").write_text("")
    tmp_path.chmod(0o777)

    def boom(*a, **k):
        raise AssertionError("an untrusted pipe dir must not be probed")

    monkeypatch.setattr(mps_prelaunch.subprocess, "run", boom)
    assert mps_prelaunch.mps_active(str(tmp_path)) is False
    tmp_path.chmod(stat.S_IRWXU)  # leave it removable by the tmp_path fixture


# ---------------------------------------------------------------------------
# Live-control detection: socket + bounded control query (stale/absent => False)
# ---------------------------------------------------------------------------


def _arrange_control_query(monkeypatch, *, returncode: int = 0) -> None:
    """Arrange the control binary + a bounded read-only query that returns ``returncode``."""
    monkeypatch.setattr(mps_prelaunch.shutil, "which", lambda _n: "/usr/bin/nvidia-cuda-mps-control")
    monkeypatch.setattr(mps_prelaunch.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=returncode))


def test_mps_active_true_when_socket_and_control_query_succeeds(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "control").write_text("")
    _arrange_control_query(monkeypatch, returncode=0)  # a live daemon answers the read-only query
    assert mps_prelaunch.mps_active(str(tmp_path)) is True


def test_mps_active_false_when_query_fails_on_stale_socket(monkeypatch, tmp_path: Path) -> None:
    # A stale control socket (nothing listening) makes the control query fail; it must not pass.
    (tmp_path / "control").write_text("")
    _arrange_control_query(monkeypatch, returncode=1)
    assert mps_prelaunch.mps_active(str(tmp_path)) is False


def test_mps_active_false_when_query_raises(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "control").write_text("")
    monkeypatch.setattr(mps_prelaunch.shutil, "which", lambda _n: "/usr/bin/nvidia-cuda-mps-control")
    monkeypatch.setattr(mps_prelaunch.subprocess, "run", _seq_raise(OSError("cannot connect")))
    assert mps_prelaunch.mps_active(str(tmp_path)) is False


def test_mps_active_false_when_control_binary_missing(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "control").write_text("")
    monkeypatch.setattr(mps_prelaunch.shutil, "which", lambda _n: None)
    monkeypatch.setattr(mps_prelaunch.subprocess, "run", _seq_raise(AssertionError("no query without a binary")))
    assert mps_prelaunch.mps_active(str(tmp_path)) is False


def test_mps_active_no_socket_does_not_probe(monkeypatch, tmp_path: Path) -> None:
    def boom(*a, **k):
        raise AssertionError("no control socket -> must not probe the daemon")

    monkeypatch.setattr(mps_prelaunch.subprocess, "run", boom)
    assert mps_prelaunch.mps_active(str(tmp_path / "nope")) is False


def test_mps_active_detects_any_live_candidate(monkeypatch, tmp_path: Path) -> None:
    # A live daemon at only ONE of the (no-env) candidate dirs must be found.
    active_dir = tmp_path / "nvidia-mps"
    active_dir.mkdir()
    (active_dir / "control").write_text("")
    monkeypatch.setattr(
        mps_prelaunch, "_pipe_dir_candidates", lambda: [str(tmp_path / "nvidia-mps-uid"), str(active_dir)]
    )
    _arrange_control_query(monkeypatch, returncode=0)
    assert mps_prelaunch.mps_active() is True


# ---------------------------------------------------------------------------
# tri-state control probe: LIVE / ABSENT / UNKNOWN (UNKNOWN never == "gone")
# ---------------------------------------------------------------------------


def test_probe_control_daemon_live_on_rc_zero(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(mps_prelaunch.shutil, "which", lambda _n: "/usr/bin/nvidia-cuda-mps-control")
    monkeypatch.setattr(mps_prelaunch.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    assert mps_prelaunch._probe_control_daemon(str(tmp_path)) is mps_prelaunch._ProbeResult.LIVE


def test_probe_control_daemon_absent_on_nonzero(monkeypatch, tmp_path: Path) -> None:
    # The control binary ran but could not connect (stale/absent socket): a VERIFIED absence.
    monkeypatch.setattr(mps_prelaunch.shutil, "which", lambda _n: "/usr/bin/nvidia-cuda-mps-control")
    monkeypatch.setattr(mps_prelaunch.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    assert mps_prelaunch._probe_control_daemon(str(tmp_path)) is mps_prelaunch._ProbeResult.ABSENT


def test_probe_control_daemon_unknown_on_timeout(monkeypatch, tmp_path: Path) -> None:
    # A probe timeout is indeterminate -> UNKNOWN (must never be read as "daemon dead").
    monkeypatch.setattr(mps_prelaunch.shutil, "which", lambda _n: "/usr/bin/nvidia-cuda-mps-control")
    monkeypatch.setattr(
        mps_prelaunch.subprocess, "run", _seq_raise(mps_prelaunch.subprocess.TimeoutExpired(["mps"], 3.0))
    )
    assert mps_prelaunch._probe_control_daemon(str(tmp_path)) is mps_prelaunch._ProbeResult.UNKNOWN


def test_probe_control_daemon_unknown_on_oserror(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(mps_prelaunch.shutil, "which", lambda _n: "/usr/bin/nvidia-cuda-mps-control")
    monkeypatch.setattr(mps_prelaunch.subprocess, "run", _seq_raise(OSError("cannot connect")))
    assert mps_prelaunch._probe_control_daemon(str(tmp_path)) is mps_prelaunch._ProbeResult.UNKNOWN


def test_probe_control_daemon_unknown_on_missing_binary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(mps_prelaunch.shutil, "which", lambda _n: None)
    monkeypatch.setattr(mps_prelaunch.subprocess, "run", _seq_raise(AssertionError("no probe without a binary")))
    assert mps_prelaunch._probe_control_daemon(str(tmp_path)) is mps_prelaunch._ProbeResult.UNKNOWN


def test_control_daemon_responds_only_true_for_live(monkeypatch, tmp_path: Path) -> None:
    # The detection wrapper collapses ABSENT/UNKNOWN to False; only a confirmed LIVE reads as "up".
    for result, expected in (
        (mps_prelaunch._ProbeResult.LIVE, True),
        (mps_prelaunch._ProbeResult.ABSENT, False),
        (mps_prelaunch._ProbeResult.UNKNOWN, False),
    ):
        monkeypatch.setattr(mps_prelaunch, "_probe_control_daemon", lambda *a, _r=result, **k: _r)
        assert mps_prelaunch._control_daemon_responds(str(tmp_path)) is expected
