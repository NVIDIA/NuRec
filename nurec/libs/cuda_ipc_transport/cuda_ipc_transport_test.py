# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the CUDA IPC transport's IpcManager.

Covers the server-side buffer pool lifecycle (allocate/select/release) and, since
cudaIpcOpenMemHandle can only be called from a *different* process than the one
that called cudaIpcGetMemHandle, a cross-process roundtrip that opens an exported
handle in a child process and verifies the pack-kernel output byte-for-byte —
the actual correctness risk this transport design calls out. Needs a GPU.

The child process is launched with subprocess.run(sys.executable, ...) rather than
the multiprocessing module: multiprocessing's "spawn" start method (required here —
forking a process that has already initialized a CUDA context is undefined behavior)
re-imports the running interpreter's __main__ module to bootstrap the child. Under
this repo's pytest runner (bazel/pytest/pytest_main.py), that module calls
pytest.main() unconditionally at import time with no `if __name__ == "__main__":`
guard, so the re-import recursively re-invokes pytest inside the spawn bootstrap and
trips Python's own reentrant-spawn guard (RuntimeError: "An attempt has been made to
start a new process before the current process has finished its bootstrapping
phase"). A plain subprocess sidesteps that machinery entirely: it's a fresh
interpreter with no relationship to the parent's __main__.
"""

import ctypes
import subprocess
import sys

import pytest
import torch

from nurec.libs.cuda_ipc_transport import IpcManager  # type: ignore[attr-defined]


_CHILD_OPEN_AND_READ_SCRIPT = """
import ctypes
import os
import sys
import time

_CUDA_SUCCESS = 0
_CUDA_MEMCPY_DEVICE_TO_HOST = 2
_CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1


def _error_string(err):
    lib.cudaGetErrorString.restype = ctypes.c_char_p
    lib.cudaGetErrorString.argtypes = [ctypes.c_int]
    try:
        return lib.cudaGetErrorString(err).decode(errors="replace")
    except Exception:
        return "<cudaGetErrorString failed>"


def _check(err, what):
    if err != _CUDA_SUCCESS:
        # cudaGetErrorString turns e.g. cudaError_t=2 into "out of memory" -- worth the extra
        # call whenever this actually fails, since the bare numeric code alone isn't enough to
        # tell a real OOM apart from, say, a device/driver mismatch that happens to also map to
        # cudaErrorMemoryAllocation.
        raise RuntimeError(f"{what} failed: cudaError_t={err} ({_error_string(err)})")


# cudaIpcOpenMemHandle's real signature is
#   cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags)
# -- handle is a 64-byte struct passed BY VALUE, not a pointer. Without an explicit
# ctypes.Structure and argtypes, ctypes has no way to know that and instead passes
# whatever raw bytes/pointer it's given per its own (wrong) guess, corrupting the
# call. Declare the real struct and argtypes so the ABI matches.
class _CudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * 64)]


handle_bytes = bytes.fromhex(sys.argv[1])
size = int(sys.argv[2])
cudart_path = sys.argv[3]

lib = ctypes.CDLL(cudart_path)
lib.cudaSetDevice.argtypes = [ctypes.c_int]
lib.cudaSetDevice.restype = ctypes.c_int
lib.cudaIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), _CudaIpcMemHandle, ctypes.c_uint]
lib.cudaIpcOpenMemHandle.restype = ctypes.c_int
lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
lib.cudaMemcpy.restype = ctypes.c_int
lib.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
lib.cudaIpcCloseMemHandle.restype = ctypes.c_int

# On DGX Spark (unified host/device memory) a CUDA primary context needs 2 MiB-contiguous
# pages, which a parallel Bazel build starves for seconds at a time -- the same transient race
# nurec/grpc/model_owner_test.py's _CUDA_IPC_WORKER_SRC retries around for its own bare-interpreter
# worker (see that file's comment). This script is the same kind of bare interpreter (no
# rules_python stage2 bootstrap, which retries this for a normal test process), so it needs the
# same retry: cudaSetDevice here is this process's first CUDA call, and is exactly where that
# race is observed (cudaErrorMemoryAllocation, decoded as "out of memory" below).
if os.uname().machine == "aarch64":
    _retryable = ("out of memory", "busy or unavailable")
    _deadline = time.monotonic() + 60.0
    while True:
        err = lib.cudaSetDevice(0)
        if err == _CUDA_SUCCESS:
            break
        if not any(m in _error_string(err).lower() for m in _retryable) or time.monotonic() >= _deadline:
            _check(err, "cudaSetDevice")
        time.sleep(0.5)
else:
    _check(lib.cudaSetDevice(0), "cudaSetDevice")

handle_struct = _CudaIpcMemHandle.from_buffer_copy(handle_bytes)
dev_ptr = ctypes.c_void_p()
_check(
    lib.cudaIpcOpenMemHandle(ctypes.byref(dev_ptr), handle_struct, _CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS),
    "cudaIpcOpenMemHandle",
)

host_buf = (ctypes.c_ubyte * size)()
_check(
    lib.cudaMemcpy(host_buf, dev_ptr, size, _CUDA_MEMCPY_DEVICE_TO_HOST),
    "cudaMemcpy",
)
lib.cudaIpcCloseMemHandle(dev_ptr)
sys.stdout.write(bytes(host_buf).hex())
"""


def _loaded_libcudart_path() -> str:
    """Absolute path of the libcudart.so this process already has mapped (via torch /
    the native IpcManager extension). Bare `ctypes.CDLL("libcudart.so")` relies on the
    default dynamic-linker search path (LD_LIBRARY_PATH, ld.so.cache, RPATH of the
    Python interpreter itself), which the compiled pybind extension gets for free from
    its own baked-in RPATH but a plain `python -c` child does not -- in a hermetic
    sandbox (e.g. CI) with no system-wide CUDA install, that bare name isn't found at
    all. Reading /proc/self/maps for what's actually mapped works regardless of how
    this environment's CUDA runtime happens to be packaged (system install, a pip
    wheel's bundled .so, etc.), so the child can be handed a real path instead of
    guessing a name.
    """
    with open("/proc/self/maps") as f:
        for line in f:
            path = line.rstrip("\n").split(maxsplit=5)[-1]
            if "libcudart.so" in path:
                return path
    raise RuntimeError("libcudart.so not found in /proc/self/maps -- is CUDA initialized in this process?")


def _child_open_and_read(handle_bytes: bytes, size: int) -> bytes:
    """Runs the IPC-handle-open-and-copy in a genuinely separate OS process (not a
    multiprocessing child — see the module docstring) and returns the bytes it read."""
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_OPEN_AND_READ_SCRIPT, handle_bytes.hex(), str(size), _loaded_libcudart_path()],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"child process failed (exit {result.returncode}): {result.stderr}")
    return bytes.fromhex(result.stdout.strip())


def _reference_chw_u8_bytes(src_hwc_f32: torch.Tensor) -> bytes:
    # .numpy() would be the obvious way to get raw bytes out, but numpy isn't a
    # dependency of this test's hermetic venv; read the CPU tensor's own contiguous
    # storage directly instead.
    chw_u8 = (src_hwc_f32 * 255.0).clamp(0, 255).to(torch.uint8).permute(2, 0, 1).reshape(-1).contiguous().cpu()
    return ctypes.string_at(chw_u8.data_ptr(), chw_u8.numel())


def test_open_session_allocates_requested_buffers():
    if not torch.cuda.is_available():
        pytest.skip("CUDA IPC needs a CUDA device")
    mgr = IpcManager(0)
    width, height, buffer_count = 8, 6, 2
    session_id, handles, buffer_size, device_uuid, actual_w, actual_h = mgr.open_session(width, height, buffer_count)
    try:
        assert len(handles) == buffer_count
        assert all(len(h) == 64 for h in handles)  # sizeof(cudaIpcMemHandle_t)
        assert buffer_size == width * height * 3
        assert (actual_w, actual_h) == (width, height)
        assert len(device_uuid) == 16
    finally:
        assert mgr.close_session(session_id) is True


def test_open_session_respects_configurable_max_buffer_count():
    if not torch.cuda.is_available():
        pytest.skip("CUDA IPC needs a CUDA device")
    # The compiled-in default (64) is generous enough that this needs its own lower cap to
    # actually exercise the check -- serve.py's --cuda-ipc-max-buffer-count wires this same
    # constructor argument through per-deployment.
    mgr = IpcManager(0, max_buffer_count=3)
    with pytest.raises(RuntimeError, match="requested_buffer_count exceeds the maximum of 3"):
        mgr.open_session(8, 6, 4)
    # At the cap, not over it, still succeeds.
    session_id, handles, _buffer_size, _uuid, _w, _h = mgr.open_session(8, 6, 3)
    try:
        assert len(handles) == 3
    finally:
        mgr.close_session(session_id)


def test_write_render_hwc_f32_errors_without_a_free_buffer():
    if not torch.cuda.is_available():
        pytest.skip("CUDA IPC needs a CUDA device")
    mgr = IpcManager(0)
    width, height = 4, 3
    session_id, _handles, _buffer_size, _uuid, _w, _h = mgr.open_session(width, height, 2)
    try:
        src = torch.zeros(height, width, 3, device="cuda", dtype=torch.float32)
        first = mgr.write_render_hwc_f32(session_id, None, src.data_ptr(), height, width)
        second = mgr.write_render_hwc_f32(session_id, None, src.data_ptr(), height, width)
        assert {first, second} == {0, 1}
        with pytest.raises(RuntimeError, match="no free CUDA IPC buffer"):
            mgr.write_render_hwc_f32(session_id, None, src.data_ptr(), height, width)
        # Releasing one frees it back up for the next request.
        third = mgr.write_render_hwc_f32(session_id, first, src.data_ptr(), height, width)
        assert third == first
    finally:
        mgr.close_session(session_id)


def test_ipc_handle_roundtrip_cross_process():
    if not torch.cuda.is_available():
        pytest.skip("CUDA IPC roundtrip needs a CUDA device")

    width, height = 4, 3
    mgr = IpcManager(0)
    session_id, handles, buffer_size, _uuid, _w, _h = mgr.open_session(width, height, 2)
    try:
        torch.manual_seed(0)
        src = (torch.rand(height, width, 3, device="cuda", dtype=torch.float32) * 1.4 - 0.2).contiguous()
        filled_id = mgr.write_render_hwc_f32(session_id, None, src.data_ptr(), height, width)

        result = _child_open_and_read(handles[filled_id], buffer_size)

        assert result == _reference_chw_u8_bytes(src)
    finally:
        mgr.close_session(session_id)
