// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rdma_utils.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>

RdmaContext rdma_open(const std::string& dev_name) {
    int num_devs      = 0;
    ibv_device** devs = ibv_get_device_list(&num_devs);
    if (!devs)
        throw std::runtime_error("No IB/RoCE devices found");

    // Open the named device, or when auto-selecting the first one whose port 1 is PORT_ACTIVE.
    RdmaContext rc;
    for (int i = 0; i < num_devs && !rc.ctx; ++i) {
        const char* name = ibv_get_device_name(devs[i]);
        if (!dev_name.empty() && dev_name != name)
            continue;

        ibv_context* ctx = ibv_open_device(devs[i]);
        if (!ctx)
            continue;

        ibv_port_attr pa{};
        if (dev_name.empty() && !(ibv_query_port(ctx, 1, &pa) == 0 && pa.state == IBV_PORT_ACTIVE)) {
            std::cout << "[rdma] Skipping " << name << " (port not active)\n";
            ibv_close_device(ctx);
            continue;
        }
        rc.ctx = ctx;
    }
    ibv_free_device_list(devs);
    if (!rc.ctx)
        throw std::runtime_error(dev_name.empty() ? (num_devs == 0 ? "No IB/RoCE devices found"
                                                                   : "No active IB/RoCE device found")
                                                  : "Device not found or could not be opened: " + dev_name);

    // Release what this function acquired if a later step throws; disarmed on success.
    struct Cleanup {
        RdmaContext* rc;
        explicit Cleanup(RdmaContext* r)
            : rc(r) {}
        Cleanup(const Cleanup&)            = delete;
        Cleanup& operator=(const Cleanup&) = delete;
        ~Cleanup() {
            if (!rc)
                return;
            if (rc->pd)
                ibv_dealloc_pd(rc->pd);
            ibv_close_device(rc->ctx);
        }
    } cleanup{&rc};

    rc.pd = ibv_alloc_pd(rc.ctx);
    if (!rc.pd)
        throw std::runtime_error("ibv_alloc_pd failed");

    int gid_idx = find_rocev2_gid(rc.ctx, rc.port);
    if (gid_idx < 0)
        throw std::runtime_error("No RoCEv2 GID found on port 1 — is this a RoCE device?");
    rc.gid_index = static_cast<uint32_t>(gid_idx);

    if (ibv_query_gid(rc.ctx, rc.port, rc.gid_index, &rc.gid))
        throw std::runtime_error("ibv_query_gid failed");

    // Print GID as hex for diagnostics.
    std::cout << "[rdma] Device: " << ibv_get_device_name(rc.ctx->device)
              << "  port: " << (int)rc.port
              << "  GID index: " << rc.gid_index
              << "  GID: ";
    for (int i = 0; i < 16; ++i)
        std::cout << std::hex << (int)rc.gid.raw[i] << (i < 15 ? ":" : "");
    std::cout << std::dec << "\n";
    cleanup.rc = nullptr; // the caller owns the context and PD now
    return rc;
}

static bool is_ipv4_mapped(const ibv_gid& gid) {
    // ::ffff:a.b.c.d — bytes 0-9 zero, bytes 10-11 = 0xff
    for (int i = 0; i < 10; ++i)
        if (gid.raw[i] != 0)
            return false;
    return gid.raw[10] == 0xff && gid.raw[11] == 0xff;
}

int find_rocev2_gid(ibv_context* ctx, uint8_t port) {
    std::string dev  = ibv_get_device_name(ctx->device);
    std::string base = "/sys/class/infiniband/" + dev +
                       "/ports/" + std::to_string(port) + "/gid_attrs/types/";

    int first = -1; // first non-zero RoCEv2 GID, used when none is IPv4-mapped

    for (int i = 0; i < 256; ++i) {
        std::ifstream f(base + std::to_string(i));
        if (!f)
            break;
        std::string type;
        std::getline(f, type);
        if (type.find("RoCE v2") == std::string::npos)
            continue;

        ibv_gid gid{};
        if (ibv_query_gid(ctx, port, i, &gid) != 0)
            continue;
        bool nonzero = false;
        for (int b = 0; b < 16; ++b)
            nonzero |= (gid.raw[b] != 0);
        if (!nonzero)
            continue;

        // Prefer IPv4-mapped (routable) GIDs — these work for same-host loopback.
        if (is_ipv4_mapped(gid))
            return i;
        if (first < 0)
            first = i;
    }
    return first;
}

QueuePair create_qp(const RdmaContext& rctx, int cq_depth) {
    QueuePair qp;
    // The starting PSN is exchanged with the peer in the clear, so it need not be secret,
    // only different across QPs and server restarts: a repeated starting PSN would let a
    // stale packet from a previous connection fall inside the new QP's acceptance window.
    qp.psn = std::random_device{}() & 0x00FFFFFFu;

    qp.cq = ibv_create_cq(rctx.ctx, cq_depth, nullptr, nullptr, 0);
    if (!qp.cq)
        throw std::runtime_error("ibv_create_cq failed");

    ibv_qp_init_attr init{};
    init.send_cq          = qp.cq;
    init.recv_cq          = qp.cq;
    init.cap.max_send_wr  = cq_depth;
    init.cap.max_recv_wr  = 1;
    init.cap.max_send_sge = 1;
    init.cap.max_recv_sge = 1;
    init.qp_type          = IBV_QPT_RC;

    qp.qp = ibv_create_qp(rctx.pd, &init);
    if (!qp.qp) {
        ibv_destroy_cq(qp.cq);
        throw std::runtime_error("ibv_create_qp failed");
    }
    return qp;
}

void connect_qp(const RdmaContext& rctx, QueuePair& local, const RemotePeer& remote) {
    // RESET → INIT
    {
        ibv_qp_attr attr{};
        attr.qp_state        = IBV_QPS_INIT;
        attr.pkey_index      = 0;
        attr.port_num        = rctx.port;
        attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
        if (ibv_modify_qp(local.qp, &attr,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
            throw std::runtime_error("QP RESET->INIT failed");
    }

    // INIT → RTR  (RoCE: is_global=1, always fill GRH)
    {
        ibv_qp_attr attr{};
        attr.qp_state           = IBV_QPS_RTR;
        attr.path_mtu           = IBV_MTU_1024;
        attr.dest_qp_num        = remote.qp_num;
        attr.rq_psn             = remote.psn;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer      = 12;

        attr.ah_attr.is_global         = 1;
        attr.ah_attr.grh.dgid          = remote.gid;
        attr.ah_attr.grh.sgid_index    = static_cast<uint8_t>(rctx.gid_index);
        attr.ah_attr.grh.hop_limit     = 64;
        attr.ah_attr.grh.traffic_class = 0;
        attr.ah_attr.port_num          = rctx.port;
        attr.ah_attr.sl                = 0;

        int ret = ibv_modify_qp(local.qp, &attr,
                                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                                    IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                    IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
        if (ret)
            throw std::runtime_error("QP INIT->RTR failed: errno=" + std::to_string(ret) + " (" + strerror(ret) + ")" + " dest_qp=" + std::to_string(remote.qp_num) + " rq_psn=" + std::to_string(remote.psn) + " sgid_idx=" + std::to_string(rctx.gid_index));
    }

    // RTR → RTS
    {
        ibv_qp_attr attr{};
        attr.qp_state      = IBV_QPS_RTS;
        attr.timeout       = 14;
        attr.retry_cnt     = 7;
        attr.rnr_retry     = 7;
        attr.sq_psn        = local.psn;
        attr.max_rd_atomic = 1;

        if (ibv_modify_qp(local.qp, &attr,
                          IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                              IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC))
            throw std::runtime_error("QP RTR->RTS failed");
    }
}

ibv_wc poll_cq_one(ibv_cq* cq) {
    ibv_wc wc{};
    int n;
    // Bound the busy-poll so a lost/failed completion (dead peer, cable pull) surfaces
    // as an error instead of hanging the render thread forever. 5 s is orders of
    // magnitude above a healthy single-frame RDMA WRITE completion.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    do {
        n = ibv_poll_cq(cq, 1, &wc);
        if (n == 0 && std::chrono::steady_clock::now() > deadline)
            throw std::runtime_error("ibv_poll_cq timed out after 5s waiting for completion");
    } while (n == 0);
    if (n < 0)
        throw std::runtime_error("ibv_poll_cq returned error");
    if (wc.status != IBV_WC_SUCCESS)
        throw std::runtime_error(std::string("RDMA WC error: ") + ibv_wc_status_str(wc.status));
    return wc;
}
