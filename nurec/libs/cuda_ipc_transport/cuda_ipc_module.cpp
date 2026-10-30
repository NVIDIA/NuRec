// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python binding for the NRE CUDA IPC transport. Exposes a single IpcManager
// class that owns a pool of server-allocated, IPC-exported GPU frame buffers
// per session, for a client co-located on the same node:
//   open_session             -> cudaMalloc N buffers, export each as a
//                                cudaIpcMemHandle_t
//   write_render_hwc_f32     -> mark a released buffer free (if given), pick a
//                                free buffer, fused pack kernel into it,
//                                return its id
//   close_session            -> cudaFree each buffer
//
// Buffer selection is entirely server-side; the client only ever tells the
// server, via a released_buffer_id on its next request, that it is done
// reading a previously-returned buffer (having already fenced that with a
// CPU-side cudaEventSynchronize on its own consumption event before sending
// the request — no CUDA event plumbing needed here).
//
// CUDA IPC needs no network handshake — this file is the entire native
// surface for the transport.

#include "nurec/libs/render_pack/render_pack.h"

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

// Sane upper bound on requested_width/requested_height, so a single co-located
// process cannot exhaust the server's GPU memory with an unbounded buffer
// dimension. Deliberately generous (well above any real render target) rather
// than tuned to a specific deployment. The buffer *count* cap is configurable
// per IpcManager instance instead (see max_buffer_count_ below) -- deployments
// batch different numbers of cameras per IPC session, so a single compiled-in
// constant does not fit every one.
constexpr uint32_t kMaxRequestedDimension = 16384; // width or height, pixels

// Session ids are the only capability check session-scoped operations (write, release, close)
// make, so they must be unguessable: 128 bits straight from the OS CSPRNG.
std::string generate_session_id() {
    std::random_device rd;
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 0; i < 4; ++i) {
        oss << std::setw(8) << rd();
    }
    return oss.str();
}

struct Buffer {
    void* ptr   = nullptr;
    bool in_use = false;
};

struct Session {
    std::vector<Buffer> buffers;
    uint32_t width     = 0;
    uint32_t height    = 0;
    size_t buffer_size = 0; // width * height * 3 (uint8 planar)
    int device         = 0;
    std::mutex mu; // serializes write_render_hwc_f32 on this session

    ~Session() {
        int prev = -1;
        cudaGetDevice(&prev);
        cudaSetDevice(device);
        for (auto& b : buffers) {
            if (b.ptr)
                cudaFree(b.ptr);
        }
        if (prev >= 0)
            cudaSetDevice(prev);
    }
};

class IpcManager {
public:
    explicit IpcManager(int device_index, uint32_t max_buffer_count)
        : device_(device_index), max_buffer_count_(max_buffer_count) {
        if (max_buffer_count_ == 0)
            throw std::runtime_error("max_buffer_count must be > 0");
        cudaError_t cerr = cudaSetDevice(device_);
        if (cerr != cudaSuccess)
            throw std::runtime_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(cerr));
        cudaDeviceProp prop{};
        cerr = cudaGetDeviceProperties(&prop, device_);
        if (cerr != cudaSuccess)
            throw std::runtime_error(std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(cerr));
        device_uuid_.assign(reinterpret_cast<const uint8_t*>(prop.uuid.bytes),
                            reinterpret_cast<const uint8_t*>(prop.uuid.bytes) + sizeof(prop.uuid.bytes));
    }

    py::bytes device_uuid() const {
        return py::bytes(reinterpret_cast<const char*>(device_uuid_.data()), device_uuid_.size());
    }

    // Returns (session_id, cuda_ipc_mem_handles, buffer_size, device_uuid, actual_width, actual_height).
    std::tuple<std::string, std::vector<py::bytes>, uint64_t, py::bytes, uint32_t, uint32_t>
    open_session(uint32_t width, uint32_t height, uint32_t buffer_count) {
        if (width == 0 || height == 0)
            throw std::runtime_error("requested_width/requested_height must be > 0");
        if (width > kMaxRequestedDimension || height > kMaxRequestedDimension)
            throw std::runtime_error("requested_width/requested_height exceeds the maximum of " +
                                     std::to_string(kMaxRequestedDimension) + " pixels");
        if (buffer_count == 0)
            throw std::runtime_error("requested_buffer_count must be > 0");
        if (buffer_count > max_buffer_count_)
            throw std::runtime_error("requested_buffer_count exceeds the maximum of " +
                                     std::to_string(max_buffer_count_));

        // Build the session in a local shared_ptr and publish it into sessions_
        // only after every buffer is allocated and exported, so a concurrent
        // close_session cannot observe a partially-built session.
        const std::string sid = generate_session_id();
        auto sess             = std::make_shared<Session>();
        sess->device          = device_;
        sess->width           = width;
        sess->height          = height;
        sess->buffer_size     = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;

        cudaError_t cerr = cudaSetDevice(device_);
        if (cerr != cudaSuccess)
            throw std::runtime_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(cerr));

        std::vector<py::bytes> handles;
        sess->buffers.resize(buffer_count);
        for (uint32_t i = 0; i < buffer_count; ++i) {
            void* buf = nullptr;
            cerr      = cudaMalloc(&buf, sess->buffer_size);
            if (cerr != cudaSuccess || buf == nullptr)
                throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(cerr));
            sess->buffers[i].ptr = buf;

            // cudaMalloc does not zero device memory. The client receives every
            // buffer's IPC handle up front (before any of them are ever written by
            // write_render_hwc_f32), so an unzeroed buffer would let the client read
            // whatever physical GPU memory was previously there via cudaIpcOpenMemHandle
            // before the server ever writes to it. Zero-fill so the worst a client can
            // observe before its first render is a black frame.
            cerr = cudaMemset(buf, 0, sess->buffer_size);
            if (cerr != cudaSuccess)
                throw std::runtime_error(std::string("cudaMemset failed: ") + cudaGetErrorString(cerr));

            cudaIpcMemHandle_t handle{};
            cerr = cudaIpcGetMemHandle(&handle, buf);
            if (cerr != cudaSuccess)
                throw std::runtime_error(std::string("cudaIpcGetMemHandle failed: ") + cudaGetErrorString(cerr));
            handles.emplace_back(reinterpret_cast<const char*>(&handle), sizeof(handle));
        }

        // cudaMemset is asynchronous with respect to the host; block until every
        // buffer's zero-fill has actually landed before handing the handles back,
        // so a client that maps and reads immediately never observes stale memory.
        cerr = cudaStreamSynchronize(nullptr);
        if (cerr != cudaSuccess)
            throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(cerr));

        {
            std::lock_guard<std::mutex> lk(sessions_mu_);
            sessions_.emplace(sid, sess);
        }

        return std::make_tuple(sid, handles, static_cast<uint64_t>(sess->buffer_size), device_uuid(), width, height);
    }

    // Fused: mark released_buffer_id free (if given), pick a free buffer, run
    // the shared render-pack kernel to scale+saturate+permute the renderer's
    // (H, W, 3) float32 GPU tensor directly into it, return its id.
    uint32_t write_render_hwc_f32(const std::string& session_id,
                                  std::optional<uint32_t> released_buffer_id,
                                  uint64_t hwc_src_ptr,
                                  uint32_t height,
                                  uint32_t width) {
        auto s = get_session(session_id);
        std::lock_guard<std::mutex> lk(s->mu);

        if (width != s->width || height != s->height)
            throw std::runtime_error("render size (" + std::to_string(width) + "x" + std::to_string(height) +
                                     ") does not match session buffer size (" + std::to_string(s->width) + "x" +
                                     std::to_string(s->height) + ")");

        if (released_buffer_id.has_value()) {
            const uint32_t rid = *released_buffer_id;
            if (rid >= s->buffers.size())
                throw std::runtime_error("released_buffer_id out of range: " + std::to_string(rid));
            // Trusts the client's release protocol: it must have already fenced with
            // its own cudaEventSynchronize before sending this id (see the top-of-file
            // comment), and we do not verify the buffer was actually marked in_use.
            // A buggy or malicious client that double-releases, or releases a buffer
            // it is still reading, can cause that buffer to be selected as the next
            // write target and overwritten mid-read on the client side. This is an
            // accepted trust boundary for a same-node, co-located client, not a
            // server-side bug to harden against here.
            s->buffers[rid].in_use = false;
        }

        int free_id = -1;
        for (size_t i = 0; i < s->buffers.size(); ++i) {
            if (!s->buffers[i].in_use) {
                free_id = static_cast<int>(i);
                break;
            }
        }
        if (free_id < 0)
            throw std::runtime_error("no free CUDA IPC buffer available for session " + session_id +
                                     " — client must release a buffer before requesting another frame");

        void* dst = s->buffers[free_id].ptr;
        {
            py::gil_scoped_release release;
            // The calling thread's current CUDA device is whatever the gRPC worker
            // thread last touched (e.g. the render backend's device), not necessarily
            // this session's device — set it explicitly before launching, mirroring
            // the constructor and open_session above. On a multi-GPU host with
            // NRE_CUDA_IPC_DEVICE pinned to a non-default index, skipping this would
            // launch the kernel against the wrong device context.
            cudaError_t cerr = cudaSetDevice(s->device);
            if (cerr != cudaSuccess)
                throw std::runtime_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(cerr));
            // Launched on the legacy default stream (nullptr), which is safe here only
            // because the Python caller (serve.py's _ipc_write_rgb) already runs
            // torch.cuda.synchronize(raster_f32.device) — a full-device barrier — right
            // before calling this, so hwc_src_ptr's write is guaranteed complete by
            // the time this kernel reads it regardless of which stream produced it.
            cerr = nurec_render_pack::launch_render_pack_hwc_f32_to_chw_u8(
                reinterpret_cast<const float*>(hwc_src_ptr),
                static_cast<uint8_t*>(dst),
                width,
                height,
                /*stream=*/nullptr);
            if (cerr != cudaSuccess)
                throw std::runtime_error(std::string("render_pack kernel launch failed: ") + cudaGetErrorString(cerr));
            // The client maps this buffer directly (no completion signal beyond
            // the blocking render_rgb RPC return), so the kernel must finish
            // writing it before we hand the buffer id back to the caller.
            cerr = cudaStreamSynchronize(nullptr);
            if (cerr != cudaSuccess)
                throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(cerr));
        }

        s->buffers[free_id].in_use = true;
        return static_cast<uint32_t>(free_id);
    }

    // Atomically applies releases and reserves `count` buffers for a batch write, so a
    // batch_render_rgb caller (serve.py) can check "does this batch's IPC delivery fit"
    // and claim the buffers in one lock acquisition -- without this, checking free count
    // and then writing per-item later would race against any other request against the
    // same session (another render_rgb call, or a concurrent batch) in between.
    // released_buffer_ids are marked free even if the reservation itself then fails, so a
    // rejected batch doesn't strand buffers the client already said it was done with.
    // Returns the reserved buffer ids, in ascending order; the caller writes into them via
    // write_reserved_buffer.
    std::vector<uint32_t> reserve_buffers(const std::string& session_id,
                                          const std::vector<uint32_t>& released_buffer_ids,
                                          uint32_t count) {
        auto s = get_session(session_id);
        std::lock_guard<std::mutex> lk(s->mu);

        for (uint32_t rid : released_buffer_ids) {
            if (rid >= s->buffers.size())
                throw std::runtime_error("released_buffer_id out of range: " + std::to_string(rid));
            // Same trust boundary as write_render_hwc_f32's released_buffer_id handling above:
            // no check that rid was actually in_use, so a buggy or malicious client can mark a
            // buffer free while it's still reading it. Accepted for a same-node, co-located
            // client -- see the comment there for the full rationale.
            s->buffers[rid].in_use = false;
        }

        std::vector<uint32_t> reserved;
        for (uint32_t i = 0; i < s->buffers.size() && reserved.size() < count; ++i) {
            if (!s->buffers[i].in_use)
                reserved.push_back(i);
        }
        if (reserved.size() < count) {
            uint32_t free_count = 0;
            for (const auto& b : s->buffers)
                if (!b.in_use)
                    ++free_count;
            throw std::runtime_error("insufficient free CUDA IPC buffers: requested " + std::to_string(count) +
                                     ", " + std::to_string(free_count) + " free of " +
                                     std::to_string(s->buffers.size()) + " total in session " + session_id);
        }

        for (uint32_t id : reserved)
            s->buffers[id].in_use = true;
        return reserved;
    }

    // Runs the pack kernel into a buffer previously claimed by reserve_buffers. Unlike
    // write_render_hwc_f32, this never selects a buffer or changes in_use state itself --
    // the reservation already did that -- it only validates the id is actually reserved.
    void write_reserved_buffer(const std::string& session_id,
                               uint32_t buffer_id,
                               uint64_t hwc_src_ptr,
                               uint32_t height,
                               uint32_t width) {
        auto s = get_session(session_id);
        std::lock_guard<std::mutex> lk(s->mu);

        if (width != s->width || height != s->height)
            throw std::runtime_error("render size (" + std::to_string(width) + "x" + std::to_string(height) +
                                     ") does not match session buffer size (" + std::to_string(s->width) + "x" +
                                     std::to_string(s->height) + ")");
        if (buffer_id >= s->buffers.size())
            throw std::runtime_error("buffer_id out of range: " + std::to_string(buffer_id));
        if (!s->buffers[buffer_id].in_use)
            throw std::runtime_error("buffer_id " + std::to_string(buffer_id) +
                                     " is not reserved -- call reserve_buffers first");

        void* dst = s->buffers[buffer_id].ptr;
        {
            py::gil_scoped_release release;
            cudaError_t cerr = cudaSetDevice(s->device);
            if (cerr != cudaSuccess)
                throw std::runtime_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(cerr));
            cerr = nurec_render_pack::launch_render_pack_hwc_f32_to_chw_u8(
                reinterpret_cast<const float*>(hwc_src_ptr),
                static_cast<uint8_t*>(dst),
                width,
                height,
                /*stream=*/nullptr);
            if (cerr != cudaSuccess)
                throw std::runtime_error(std::string("render_pack kernel launch failed: ") + cudaGetErrorString(cerr));
            cerr = cudaStreamSynchronize(nullptr);
            if (cerr != cudaSuccess)
                throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(cerr));
        }
    }

    // Marks a reserved-but-never-written buffer free again, without running any kernel.
    // Used by a batch caller to give back a buffer reserve_buffers claimed for an item
    // whose render then failed before write_reserved_buffer ran -- otherwise that buffer
    // id was never returned to the client (the failed item has no result to carry it),
    // so it would sit in_use forever until the session is closed.
    void release_buffer(const std::string& session_id, uint32_t buffer_id) {
        auto s = get_session(session_id);
        std::lock_guard<std::mutex> lk(s->mu);
        if (buffer_id >= s->buffers.size())
            throw std::runtime_error("buffer_id out of range: " + std::to_string(buffer_id));
        s->buffers[buffer_id].in_use = false;
    }

    // Returns whether a session was actually removed, so the caller can keep an
    // accurate live-session count across repeated or unknown-id closes.
    bool close_session(const std::string& session_id) {
        // Extract the shared_ptr under the lock, then drop the lock before it goes
        // out of scope. If this was the last reference, ~Session() runs here and
        // does a device-synchronizing cudaFree per buffer; running that while
        // holding sessions_mu_ would stall every other session's open_session/
        // close_session/get_session (the latter called from write_render_hwc_f32 on
        // every frame) for the duration of this session's teardown.
        std::shared_ptr<Session> victim;
        {
            std::lock_guard<std::mutex> lk(sessions_mu_);
            auto it = sessions_.find(session_id);
            if (it == sessions_.end())
                return false;
            victim = std::move(it->second);
            sessions_.erase(it);
        }
        return true; // victim's destructor runs here, unlocked
    }

private:
    // Returns a shared_ptr so callers keep the Session alive for the duration of
    // their operation even if a concurrent close_session removes it from the map
    // (the object is destroyed only when the last reference drops).
    std::shared_ptr<Session> get_session(const std::string& session_id) {
        std::lock_guard<std::mutex> lk(sessions_mu_);
        auto it = sessions_.find(session_id);
        if (it == sessions_.end())
            throw std::runtime_error("session not found: " + session_id);
        return it->second;
    }

    int device_                = 0;
    uint32_t max_buffer_count_ = 0;
    std::vector<uint8_t> device_uuid_;
    std::unordered_map<std::string, std::shared_ptr<Session>> sessions_;
    std::mutex sessions_mu_;
};

} // namespace

PYBIND11_MODULE(_nre_cuda_ipc, m) {
    m.doc() = "NRE CUDA IPC transport — pybind11 wrapper around cudaIpcGetMemHandle";

    py::class_<IpcManager>(m, "IpcManager")
        .def(py::init<int, uint32_t>(), py::arg("device_index") = 0, py::arg("max_buffer_count") = 64)
        .def_property_readonly("device_uuid", &IpcManager::device_uuid)
        .def("open_session", &IpcManager::open_session,
             py::arg("width"),
             py::arg("height"),
             py::arg("buffer_count"))
        .def("write_render_hwc_f32", &IpcManager::write_render_hwc_f32,
             py::arg("session_id"),
             py::arg("released_buffer_id"),
             py::arg("hwc_src_ptr"),
             py::arg("height"),
             py::arg("width"))
        .def("reserve_buffers", &IpcManager::reserve_buffers,
             py::arg("session_id"),
             py::arg("released_buffer_ids"),
             py::arg("count"))
        .def("write_reserved_buffer", &IpcManager::write_reserved_buffer,
             py::arg("session_id"),
             py::arg("buffer_id"),
             py::arg("hwc_src_ptr"),
             py::arg("height"),
             py::arg("width"))
        .def("release_buffer", &IpcManager::release_buffer,
             py::arg("session_id"),
             py::arg("buffer_id"))
        .def("close_session", &IpcManager::close_session,
             py::arg("session_id"));
}
