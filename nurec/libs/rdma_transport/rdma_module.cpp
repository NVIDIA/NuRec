// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python binding for the NRE RDMA transport. Exposes a single RdmaManager
// class that owns the ibv_context + PD and per-session RC QPs:
//   open_session              -> connect a QP, allocate a cudaMalloc'd staging MR
//   register_buffer           -> record a client (remote_addr, rkey, size) under buffer_id
//   rdma_write_render_hwc_f32 -> fused pack kernel into staging, post RDMA WRITE, poll CQ
//   close_session             -> destroy the QP and staging MR
//
// GPUDirect server-side path: the staging buffer is cudaMalloc'd GPU memory
// registered via ibv_reg_mr through nvidia_peermem, so the NIC reads the
// payload directly from GPU memory (no D2H copy). Requires nvidia_peermem
// loaded and PCIe ACS cleared between the chosen GPU and the NIC.
//
// Writes are synchronous: rdma_write_render_hwc_f32 spin-polls the CQ inline
// (with the GIL released), which fits the one-frame-in-flight HIL use case.
//
// Errors map to Python types serve.py classifies without parsing messages:
// ValueError for invalid arguments, KeyError for an unknown session,
// PreconditionError (a RuntimeError) for a write the session cannot take, and
// a plain RuntimeError for device, CUDA and ibverbs failures.

#include "nurec/libs/render_pack/render_pack.h"
#include "rdma_utils.h"

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

// CUDA device the staging buffers live on: device 0 from the renderer's view, which with
// CUDA_VISIBLE_DEVICES set maps to the physical GPU chosen to share a PCIe switch with the
// NIC. Exposed as RdmaManager.cuda_device so serve.py can check the render output is there.
constexpr int kStagingDevice = 0;

// Upper bound on max_buf_size: one 16384x16384 rgb_uint8_planar frame, the same per-dimension
// limit the CUDA IPC transport applies. Keeps a single open_session from making the server
// cudaMalloc an arbitrary amount of GPU memory.
constexpr uint64_t kMaxBufSize = 16384ull * 16384ull * 3ull;
// A write is posted as one scatter-gather entry, whose length is 32-bit.
static_assert(kMaxBufSize <= std::numeric_limits<uint32_t>::max(), "a staging buffer must fit one ibv_sge");

// A write the session cannot take as asked; surfaced to Python as PreconditionError.
struct PreconditionError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Session ids are the only capability check session-scoped operations (register, write,
// close) make, so they must be unguessable: a client that guessed another's id could
// register its own (remote_addr, rkey) there and receive that client's frames. 128 bits
// straight from the OS CSPRNG.
std::string generate_session_id() {
    std::random_device rd;
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 0; i < 4; ++i) {
        oss << std::setw(8) << rd();
    }
    return oss.str();
}

struct RemoteBuffer {
    uint64_t remote_addr = 0;
    uint32_t rkey        = 0;
    uint64_t size        = 0;
};

struct Session {
    QueuePair qp{};
    void* staging_buf   = nullptr; // cudaMalloc'd on kStagingDevice
    ibv_mr* staging_mr  = nullptr;
    size_t staging_size = 0;
    std::unordered_map<uint32_t, RemoteBuffer> remote_buffers;
    std::mutex mu; // serializes rdma_write / register_buffer on this session
    // Set once a write fails to collect its completion. The work request stays
    // outstanding, so its completion can still land in the CQ later and be
    // mistaken for a subsequent write's. The session is refused from then on;
    // the client must reopen to get a fresh QP and CQ.
    bool broken = false;

    ~Session() {
        if (staging_mr)
            ibv_dereg_mr(staging_mr);
        if (staging_buf) {
            int prev = -1;
            cudaGetDevice(&prev);
            cudaSetDevice(kStagingDevice);
            cudaFree(staging_buf);
            if (prev >= 0)
                cudaSetDevice(prev);
        }
        if (qp.qp)
            ibv_destroy_qp(qp.qp);
        if (qp.cq)
            ibv_destroy_cq(qp.cq);
    }
};

class RdmaManager {
public:
    explicit RdmaManager(const std::string& device_name)
        : rctx_(rdma_open(device_name)) {}

    ~RdmaManager() {
        {
            std::lock_guard<std::mutex> lk(sessions_mu_);
            sessions_.clear();
        }
        if (rctx_.pd)
            ibv_dealloc_pd(rctx_.pd);
        if (rctx_.ctx)
            ibv_close_device(rctx_.ctx);
    }

    uint32_t local_gid_index() const { return rctx_.gid_index; }

    // Returns (session_id, server_gid, server_gid_index, server_qp_num, server_psn).
    // Called with the GIL held, since it reads and returns Python bytes; it releases the GIL
    // around the native setup itself.
    std::tuple<std::string, py::bytes, uint32_t, uint32_t, uint32_t>
    open_session(const py::bytes& client_gid, uint32_t client_qp_num, uint32_t client_psn, uint64_t max_buf_size) {
        if (max_buf_size == 0)
            throw std::invalid_argument("max_buf_size must be > 0");
        if (max_buf_size > kMaxBufSize)
            throw std::invalid_argument("max_buf_size " + std::to_string(max_buf_size) + " exceeds the " +
                                        std::to_string(kMaxBufSize) + "-byte limit (one 16384x16384 RGB frame)");

        const std::string gid = client_gid;
        if (gid.size() != 16)
            throw std::invalid_argument("client_gid must be 16 bytes");

        RemotePeer remote{};
        std::memcpy(remote.gid.raw, gid.data(), 16);
        remote.qp_num = client_qp_num;
        remote.psn    = client_psn;

        const std::string sid  = generate_session_id();
        uint32_t server_qp_num = 0;
        uint32_t server_psn    = 0;
        {
            py::gil_scoped_release release;

            // Build the session in a local shared_ptr and publish it into sessions_
            // only after initialization fully succeeds. Until then it is unreachable
            // by close_session, so a concurrent close cannot destroy it mid-init; on
            // any failure the local shared_ptr drops and ~Session cleans up (no map
            // entry to erase).
            auto sess = std::make_shared<Session>();
            sess->qp  = create_qp(rctx_);

            // GPUDirect staging: cudaMalloc'd GPU buffer, registered as the RDMA MR
            // via nvidia_peermem so the NIC reads the payload from GPU memory directly.
            cudaError_t cerr = cudaSetDevice(kStagingDevice);
            if (cerr != cudaSuccess)
                throw std::runtime_error(std::string("cudaSetDevice(staging) failed: ") + cudaGetErrorString(cerr));
            void* buf = nullptr;
            cerr      = cudaMalloc(&buf, max_buf_size);
            if (cerr != cudaSuccess)
                throw std::runtime_error(std::string("cudaMalloc(staging) failed: ") + cudaGetErrorString(cerr));
            sess->staging_buf  = buf;
            sess->staging_size = max_buf_size;

            sess->staging_mr = ibv_reg_mr(rctx_.pd, buf, max_buf_size, IBV_ACCESS_LOCAL_WRITE);
            if (!sess->staging_mr)
                throw std::runtime_error("ibv_reg_mr(GPU staging) failed — "
                                         "check nvidia_peermem and PCIe ACS between GPU and NIC");

            connect_qp(rctx_, sess->qp, remote);

            server_qp_num = sess->qp.qp->qp_num;
            server_psn    = sess->qp.psn;

            std::lock_guard<std::mutex> lk(sessions_mu_);
            sessions_.emplace(sid, std::move(sess));
        }

        py::bytes server_gid(reinterpret_cast<const char*>(rctx_.gid.raw), 16);
        return std::make_tuple(sid, server_gid, rctx_.gid_index, server_qp_num, server_psn);
    }

    void register_buffer(const std::string& session_id,
                         uint64_t remote_addr,
                         uint32_t rkey,
                         uint64_t size,
                         uint32_t buffer_id) {
        auto s = get_session(session_id);
        std::lock_guard<std::mutex> lk(s->mu);
        s->remote_buffers[buffer_id] = RemoteBuffer{remote_addr, rkey, size};
    }

    // Fused: read the renderer's (H, W, 3) float32 GPU tensor, run a single
    // CUDA kernel that scales+saturates+permutes into planar uint8 written
    // directly into the pre-registered staging buffer, then post the RDMA
    // WRITE. No intermediate tensor and no D2D copy into staging. The caller
    // guarantees the tensor is on kStagingDevice and fully written.
    uint64_t rdma_write_render_hwc_f32(const std::string& session_id,
                                       uint32_t buffer_id,
                                       uint64_t hwc_src_ptr,
                                       uint32_t height,
                                       uint32_t width) {
        auto s = get_session(session_id);
        std::lock_guard<std::mutex> lk(s->mu);

        if (s->broken)
            throw PreconditionError("RDMA session unusable after an earlier completion failure; reopen it: " +
                                    session_id);

        const uint64_t len = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * 3u;
        if (len == 0)
            return 0;
        if (len > s->staging_size)
            throw PreconditionError("payload exceeds staging buffer size");

        auto it = s->remote_buffers.find(buffer_id);
        if (it == s->remote_buffers.end())
            throw PreconditionError("buffer_id not registered: " + std::to_string(buffer_id));
        if (len > it->second.size)
            throw PreconditionError("payload exceeds remote buffer size");

        // gRPC may serve this write on a different thread than open_session, and the
        // current device is per-thread.
        cudaError_t cerr = cudaSetDevice(kStagingDevice);
        if (cerr != cudaSuccess)
            throw std::runtime_error(std::string("cudaSetDevice(staging) failed: ") + cudaGetErrorString(cerr));
        cerr = nurec_render_pack::launch_render_pack_hwc_f32_to_chw_u8(reinterpret_cast<const float*>(hwc_src_ptr),
                                                                       static_cast<uint8_t*>(s->staging_buf),
                                                                       width,
                                                                       height,
                                                                       /*stream=*/nullptr);
        if (cerr != cudaSuccess)
            throw std::runtime_error(std::string("render_pack kernel launch failed: ") + cudaGetErrorString(cerr));
        // Wait for the kernel to finish writing the staging buffer
        // before initiating the RDMA WRITE that reads it.
        cerr = cudaStreamSynchronize(nullptr);
        if (cerr != cudaSuccess)
            throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(cerr));

        ibv_sge sge{};
        sge.addr   = reinterpret_cast<uint64_t>(s->staging_buf);
        sge.length = static_cast<uint32_t>(len);
        sge.lkey   = s->staging_mr->lkey;

        ibv_send_wr wr{};
        wr.opcode              = IBV_WR_RDMA_WRITE;
        wr.sg_list             = &sge;
        wr.num_sge             = 1;
        wr.send_flags          = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = it->second.remote_addr;
        wr.wr.rdma.rkey        = it->second.rkey;

        ibv_send_wr* bad = nullptr;
        int ret          = ibv_post_send(s->qp.qp, &wr, &bad);
        if (ret)
            throw std::runtime_error("ibv_post_send failed: errno=" + std::to_string(ret));

        // s->mu keeps one write in flight per session and a failed poll poisons the session,
        // so the completion polled here is always this write's.
        try {
            poll_cq_one(s->qp.cq);
        } catch (...) {
            s->broken = true;
            throw;
        }
        return len;
    }

    // Returns whether a session was actually removed, so the caller can keep an
    // accurate live-session count across repeated or unknown-id closes. The
    // session is torn down after sessions_mu_ is released (or later, by a write
    // still holding it), so other sessions' lookups never wait on its teardown.
    bool close_session(const std::string& session_id) {
        std::shared_ptr<Session> closed;
        {
            std::lock_guard<std::mutex> lk(sessions_mu_);
            auto it = sessions_.find(session_id);
            if (it == sessions_.end())
                return false;
            closed = std::move(it->second);
            sessions_.erase(it);
        }
        return true;
    }

private:
    // Returns a shared_ptr so callers keep the Session alive for the duration of
    // their operation even if a concurrent close_session removes it from the map
    // (the object is destroyed only when the last reference drops).
    std::shared_ptr<Session> get_session(const std::string& session_id) {
        std::lock_guard<std::mutex> lk(sessions_mu_);
        auto it = sessions_.find(session_id);
        if (it == sessions_.end())
            throw py::key_error("session not found: " + session_id);
        return it->second;
    }

    RdmaContext rctx_;
    std::unordered_map<std::string, std::shared_ptr<Session>> sessions_;
    std::mutex sessions_mu_;
};

} // namespace

PYBIND11_MODULE(_nre_rdma, m) {
    m.doc() = "NRE RDMA transport — pybind11 wrapper around libibverbs RC QPs";

    py::register_exception<PreconditionError>(m, "PreconditionError", PyExc_RuntimeError);

    // Everything that blocks on the device or the NIC runs with the GIL released, so a write
    // spinning on its completion never stalls the server's other gRPC threads. No method takes
    // a session lock while holding the GIL, which would deadlock against a write waiting to
    // re-acquire it.
    py::class_<RdmaManager>(m, "RdmaManager")
        .def(py::init<const std::string&>(), py::arg("device_name") = "", py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("local_gid_index", &RdmaManager::local_gid_index)
        .def_property_readonly("cuda_device", [](const RdmaManager&) { return kStagingDevice; })
        .def("open_session",
             &RdmaManager::open_session,
             py::arg("client_gid"),
             py::arg("client_qp_num"),
             py::arg("client_psn"),
             py::arg("max_buf_size"))
        .def("register_buffer",
             &RdmaManager::register_buffer,
             py::arg("session_id"),
             py::arg("remote_addr"),
             py::arg("rkey"),
             py::arg("size"),
             py::arg("buffer_id"),
             py::call_guard<py::gil_scoped_release>())
        .def("rdma_write_render_hwc_f32",
             &RdmaManager::rdma_write_render_hwc_f32,
             py::arg("session_id"),
             py::arg("buffer_id"),
             py::arg("hwc_src_ptr"),
             py::arg("height"),
             py::arg("width"),
             py::call_guard<py::gil_scoped_release>())
        .def("close_session",
             &RdmaManager::close_session,
             py::arg("session_id"),
             py::call_guard<py::gil_scoped_release>());
}
