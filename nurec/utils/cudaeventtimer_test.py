# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check CUDA event ordering with fakes and process-isolated GPU execution."""

import os
import subprocess
import sys

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import nurec.utils.cudaeventtimer as timer_module


def fake_cuda(monkeypatch):
    """Record CUDA API calls so tests can assert ordering around the render body."""
    calls = []
    events = []

    class Stream:
        device = "cuda:0"

        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            calls.append(("wait", self.name, event.index))

    current = Stream("current")

    class Event:
        def __init__(self, *, enable_timing):
            self.index = len(events)
            events.append(self)

        def record(self, stream):
            calls.append(("record", self.index, stream.name))

        def synchronize(self):
            calls.append(("event-sync", self.index))

        def elapsed_time(self, stop):
            calls.append(("elapsed", self.index, stop.index))
            return 2.5

    cuda = SimpleNamespace(
        nvtx=SimpleNamespace(range_push=lambda label: None, range_pop=lambda: None),
        device=lambda device: nullcontext(),
        current_device=lambda: 0,
        current_stream=lambda device: current,
        Event=Event,
        synchronize=lambda device: calls.append(("device-sync",)),
    )
    monkeypatch.setattr(timer_module, "torch", SimpleNamespace(cuda=cuda))
    return calls, Stream


def test_cuda_time_excludes_prior_gpu_work_and_is_read_after_completion(monkeypatch):
    calls, _ = fake_cuda(monkeypatch)
    timer = timer_module.CudaEventTimer()
    with timer("cuda:0"):
        calls.append(("render",))
    assert calls[-6:] == [
        ("device-sync",),
        ("record", 0, "current"),
        ("render",),
        ("record", 1, "current"),
        ("event-sync", 1),
        ("elapsed", 0, 1),
    ]
    assert timer.milliseconds == [2.5]
    assert timer.valid == [True]
    with timer("cuda:0"):
        calls.append(("render",))
    assert timer.milliseconds == [2.5, 2.5]
    assert timer.valid == [True, True]


def test_warmup_uses_same_events_and_boundaries_without_publishing_samples(monkeypatch):
    calls, _ = fake_cuda(monkeypatch)
    labels = []
    monkeypatch.setattr(timer_module.torch.cuda.nvtx, "range_push", labels.append)
    monkeypatch.setattr(timer_module.torch.cuda.nvtx, "range_pop", lambda: labels.append(None))
    timer = timer_module.CudaEventTimer()
    warmup_label = "nurec.render.warmup(frame=0, iteration=1)"
    with timer("cuda:0", label=warmup_label, record_sample=False):
        calls.append(("render",))
    expected = [
        ("device-sync",),
        ("record", 0, "current"),
        ("render",),
        ("record", 1, "current"),
        ("event-sync", 1),
        ("elapsed", 0, 1),
    ]
    assert calls[-6:] == expected
    assert timer.milliseconds == []
    assert timer.valid == []
    before = len(calls)
    with timer("cuda:0"):
        calls.append(("render",))
    # Measured rendering reuses the materialized events and restores its own label.
    assert calls[before:] == expected
    assert timer.milliseconds == [2.5]
    assert timer.valid == [True]
    assert labels == [warmup_label, None, "nurec.render.forward", None]


def test_every_producer_waits_for_start_and_is_joined_before_stop(monkeypatch):
    calls, Stream = fake_cuda(monkeypatch)
    timer = timer_module.CudaEventTimer(streams=[Stream("producer")])
    with timer("cuda:0"):
        calls.append(("render",))
    assert calls[-9:] == [
        ("device-sync",),
        ("record", 0, "current"),
        ("wait", "producer", 0),
        ("render",),
        ("record", 2, "producer"),
        ("wait", "current", 2),
        ("record", 1, "current"),
        ("event-sync", 1),
        ("elapsed", 0, 1),
    ]
    assert timer.milliseconds == [2.5]
    assert timer.valid == [True]


def test_producer_on_another_device_is_rejected_before_rendering(monkeypatch):
    calls, Stream = fake_cuda(monkeypatch)
    producer = Stream("producer")
    producer.device = "cuda:1"
    timer = timer_module.CudaEventTimer(streams=[producer])

    with pytest.raises(ValueError, match="measured CUDA device"), timer("cuda:0"):
        pytest.fail("Rendering must not begin with a producer on another device")

    assert calls == []
    assert timer.milliseconds == []
    assert timer.valid == []


@pytest.mark.parametrize("record_sample", [False, True])
def test_failed_forward_does_not_publish_a_timing(monkeypatch, record_sample):
    calls, _ = fake_cuda(monkeypatch)
    timer = timer_module.CudaEventTimer()
    with pytest.raises(RuntimeError, match="render failed"), timer("cuda:0", record_sample=record_sample):
        raise RuntimeError("render failed")
    assert timer.milliseconds == []
    assert timer.valid == []
    assert not any(call[0] == "elapsed" for call in calls)


def test_timer_can_be_reused_when_a_producer_becomes_current(monkeypatch):
    _, Stream = fake_cuda(monkeypatch)
    producer = Stream("producer")
    timer = timer_module.CudaEventTimer(streams=[producer])
    with timer("cuda:0"):
        pass
    monkeypatch.setattr(timer_module.torch.cuda, "current_stream", lambda device: producer)
    with timer("cuda:0"):
        pass
    assert timer.milliseconds == [2.5, 2.5]
    assert timer.valid == [True, True]


@pytest.mark.parametrize("invalid_ms", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_invalid_duration_records_zero_and_validity_without_stopping_profiling(monkeypatch, caplog, invalid_ms):
    fake_cuda(monkeypatch)
    durations = iter([invalid_ms, 2.5, invalid_ms, 3.5])
    monkeypatch.setattr(timer_module.torch.cuda.Event, "elapsed_time", lambda start, stop: next(durations))
    timer = timer_module.CudaEventTimer()
    expected_ms = [0.0, 2.5, 0.0, 3.5]
    expected_valid = [False, True, False, True]
    for index in range(1, 5):
        with timer("cuda:0"):
            pass
        assert timer.milliseconds == expected_ms[:index]
        assert timer.valid == expected_valid[:index]
    assert len(caplog.records) == 2
    assert all(record.levelname == "WARNING" for record in caplog.records)
    assert "Invalid CUDA rendering duration for nurec.render.forward" in caplog.text
    assert "recording 0 ms with valid=False" in caplog.text


def _real_cuda_producer_worker() -> None:
    device = torch.device("cuda:0")
    producer = torch.cuda.Stream(device=device)
    sentinel = torch.cuda.Event()
    # Initialize the sleep kernel and sentinel before checking asynchronous work.
    with torch.cuda.stream(producer):
        torch.cuda._sleep(1)
        sentinel.record()
    producer.synchronize()

    timer = timer_module.CudaEventTimer(streams=[producer])
    with timer(device):
        with torch.cuda.stream(producer):
            # Keep work pending long enough to expose a missing producer join.
            torch.cuda._sleep(500_000_000)
            sentinel.record()
        assert not sentinel.query(), "Producer work finished too early to exercise the timer's wait"

    # Query before any extra synchronization that could hide an early return.
    assert sentinel.query(), "Timer returned before its producer stream completed"
    assert timer.valid == [True]
    assert len(timer.milliseconds) == 1
    assert timer.milliseconds[0] > 0


def test_real_cuda_timer_waits_for_producer_in_isolated_process() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA event integration test requires a GPU")
    if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") not in ("", "0"):
        pytest.skip("CUDA_LAUNCH_BLOCKING prevents testing asynchronous producer completion")
    # Preserve Bazel's import paths in the fresh Python process.
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    completed = subprocess.run(
        [sys.executable, __file__, "--real-cuda-producer-worker"],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout


if __name__ == "__main__" and sys.argv[1:] == ["--real-cuda-producer-worker"]:
    _real_cuda_producer_worker()
