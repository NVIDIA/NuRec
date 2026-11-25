<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CUDA IPC frame delivery

CUDA IPC delivers GPU-resident images to a client on the same host and physical GPU.
The server packs float RGB output directly into shared planar uint8 buffers, without
copying image data through CPU memory or protobuf.

The server owns the buffers. The client maps them once per session and treats them
as read-only.

## Camera batches and completion

`batch_render_camera` accepts 1–8 IPC requests for one scene and session. Each item
must set `use_cuda_ipc=true` and request only color output, using `RGB_UINT8_PLANAR`
at the session dimensions, without bounding-box overlays.

On temporal-Harmonizer servers, IPC camera batches must leave `apply_harmonizer`
false or unset and omit `harmonizer_session_id`. Multiprocess serving is not supported.

Choose the completion mode when opening the session:

- `COMPLETION_ON_RETURN` is the default. Returned images are ready to read.
- `EVENT_ORDERED` is supported only by `batch_render_camera`. The RPC may return
  before GPU work finishes. Before reading a returned buffer, enqueue a wait on its
  ready event in the consuming CUDA stream.

Check the accepted `completion_mode` in the session response. Older servers may
ignore the requested mode.

Camera batching works with either completion mode.

## Buffer ownership

Wait for all client reads or copies to finish before sending `released_buffer_id`
in a subsequent request. Send each release once. After release, the server may
reuse the buffer.

A completed ready event means the server has finished writing that image. It does
not mean the client has finished reading it.

Before closing a session, finish outstanding requests and GPU work, destroy imported
event handles, and unmap all buffers with `cudaIpcCloseMemHandle`. Then call
`close_ipc_session`.

## Components

| Component     | Role                                                                                                    |
| ------------- | ------------------------------------------------------------------------------------------------------- |
| `IpcManager`  | Own sessions, buffers, and ready events; reserve buffers, pack images, and reclaim unpublished buffers. |
| `render_pack` | Convert float HWC images directly into planar uint8 IPC buffers on the supplied CUDA stream.            |
| `serve.py`    | Validate requests, render images, and return buffer IDs through the existing RPCs.                      |

## Build

The transport is always built in — no opt-in flag:

```bash
bazel build //:run
```

If the native IPC module cannot be imported, the server still starts but IPC RPCs
are unavailable.

### Optional CUDA 12.9 runtime

The default build uses CUDA 12.8. For CUDA 12.9 IPC clients, build with:

```text
--//nurec/libs/cuda_ipc_transport:cuda_12_9=true
```

This links a private CUDA 12.9 runtime into the IPC extension without changing the
runtime used by the rest of the server.

## Runtime requirements

- Client and server processes on the same node, sharing the same physical GPU
  (verified via `device_uuid`).
- On DGX Spark (unified host/device memory, `aarch64`), establishing a _new_ process's first
  CUDA context can transiently fail with `cudaErrorMemoryAllocation` ("out of memory") or
  "busy or unavailable" if a parallel Bazel build is starving the 2 MiB-contiguous pages a
  primary context needs — the same race `nurec/grpc/model_owner_test.py`'s
  `_CUDA_IPC_WORKER_SRC` retries around for its own bare-interpreter worker.
  `cuda_ipc_transport_test.py`'s cross-process test retries its child's `cudaSetDevice` for up
  to 60s on `aarch64` for the same reason; a real client hitting this should retry similarly
  rather than treat it as a hard failure.
- `NRE_CUDA_IPC_DEVICE` may pin which CUDA device index the server opens; defaults
  to device 0.
- Buffers are fixed-size for the life of a session (sized from
  `OpenIpcSessionRequest.requested_width/height` at open time) — a client that wants
  to change render resolution must close and reopen the session.
- **No authentication beyond "same node, reaches the gRPC endpoint":** any co-located
  process that can call `open_ipc_session` gets `cudaIpcMemHandle_t`s mapping GPU
  memory it can then read/write directly — a broader capability than the encoded-bytes
  `render_rgb` response already exposed. There is no per-client identity or authorization
  check, so this transport assumes every co-located process is trusted. Session ids are
  128 bits of random entropy (not a sequential counter) specifically so that one
  co-located process cannot guess another's session id and close it or release/overwrite
  buffers it is still reading — but a session id is still the _only_ capability check any
  session-scoped operation makes, so possession of it (e.g. read off another process via
  some other channel) is equivalent to ownership.
- **GPU memory caps:** `requested_width`/`requested_height`/`requested_buffer_count` are
  bounded per-request server-side (native `IpcManager::open_session`); concurrent sessions
  are capped (`NRE_CUDA_IPC_MAX_SESSIONS`, default 32); and the aggregate byte footprint of
  every currently-open session is capped (`NRE_CUDA_IPC_MAX_TOTAL_BYTES`, default 16 GiB) —
  a session count cap alone doesn't bound memory, since a single session can already
  request up to `--cuda-ipc-max-buffer-count` buffers at up to the maximum resolution.

## Testing

```bash
# proto wire tags (no hardware)
bazel test //nurec/grpc/protos:ipc_proto_test

# buffer-pool lifecycle + cross-process IPC handle roundtrip (needs a GPU)
bazel test //nurec/libs/cuda_ipc_transport:cuda_ipc_transport_test
```

Full end-to-end verification against the real `nurec-directdisplay` client is a
manual smoke test, not run in CI.
