// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <infiniband/verbs.h>
#include <stdexcept>
#include <string>

struct RdmaContext {
    ibv_context* ctx   = nullptr;
    ibv_pd* pd         = nullptr;
    uint8_t port       = 1;
    uint32_t gid_index = 0;
    ibv_gid gid        = {};
};

struct QueuePair {
    ibv_qp* qp   = nullptr;
    ibv_cq* cq   = nullptr;
    uint32_t psn = 0;
};

struct RemotePeer {
    ibv_gid gid;
    uint32_t qp_num;
    uint32_t psn;
};

// Open the first available RoCE device, allocate a PD, discover the RoCEv2 GID.
// Pass a specific device name (e.g. "mlx5_0") to override auto-selection.
RdmaContext rdma_open(const std::string& dev_name = "");

// Scan sysfs gid_attrs/types to find a non-zero RoCEv2 GID index.
// Returns -1 if none found.
int find_rocev2_gid(ibv_context* ctx, uint8_t port);

// Create an RC QP with its own CQ (cq_depth entries each for send and recv).
QueuePair create_qp(const RdmaContext& rctx, int cq_depth = 128);

// Transition a QP: RESET -> INIT -> RTR -> RTS using the remote peer's info.
void connect_qp(const RdmaContext& rctx, QueuePair& local, const RemotePeer& remote);

// Spin-poll CQ until one completion arrives. Throws on error WC.
ibv_wc poll_cq_one(ibv_cq* cq);
