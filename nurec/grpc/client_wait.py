# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging

import grpc

from grpc_health.v1 import health_pb2, health_pb2_grpc  # type: ignore


log = logging.getLogger(__name__)

_HEALTH_POLL_S = 1.0


async def _wait_until_serving(channel: grpc.aio.Channel) -> None:
    """Wait for the transport, then for the health service to report SERVING."""
    await channel.channel_ready()
    stub = health_pb2_grpc.HealthStub(channel)
    while True:
        try:
            response = await stub.Check(health_pb2.HealthCheckRequest(service=""))
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                # No health service on this channel; the transport check is all there is.
                return
            # A server still loading can drop the call (keepalive ping timeout); retry until
            # the caller's deadline rather than reporting the server down.
            await asyncio.sleep(_HEALTH_POLL_S)
            continue
        if response.status == health_pb2.HealthCheckResponse.SERVING:
            return
        await asyncio.sleep(_HEALTH_POLL_S)


async def wait_for_grpc_server(channel: grpc.aio.Channel, timeout_s: int) -> None:
    """Block until the server is reachable and reports itself SERVING.

    Transport readiness alone is not a readiness signal: the server binds its port before the
    Harmonizer model has finished loading, so a client that proceeds on ``channel_ready()`` can
    send its first RPC into a server that cannot yet answer it. The health service reports
    NOT_SERVING for that window, so it is what decides readiness here; the transport check only
    gets the channel far enough to ask.

    A server that does not answer health on this channel -- ``--health-port`` pointing elsewhere,
    or an image predating the health service -- replies UNIMPLEMENTED, and transport readiness is
    then all there is to wait for.

    Args:
        channel: An open async gRPC channel to the server.
        timeout_s: Maximum seconds to wait. ``0`` waits indefinitely.

    Raises:
        TimeoutError: If the server is not reachable and serving within ``timeout_s`` seconds.
    """
    label = "infinite" if timeout_s == 0 else f"{timeout_s}s"
    log.info(f"Waiting for gRPC server to come up (timeout={label})...")
    try:
        await asyncio.wait_for(_wait_until_serving(channel), None if timeout_s == 0 else timeout_s)
    except asyncio.TimeoutError as e:
        raise TimeoutError(f"gRPC server not reachable within {timeout_s}s") from e
    log.info("gRPC server is ready.")
