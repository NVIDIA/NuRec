# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module defining convenience wrappers around debugpy for usage withing bazel targets."""

import logging
import os

from pathlib import Path
from typing import Final, Optional

import debugpy

from nurec.utils.misc import is_env_true, map_optional


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)  # we explicitly want the logging messages here for the user when debugging.

os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

DEBUGPY_EXCEPTION_STR: Final = "Address already in use"


def breakpoint(
    port: int = 5678,
    log_dir: Optional[Path] = None,
    allow_port_increment: bool = False,
    break_on_connection: bool = False,
) -> None:
    """Open a debugpy port, wait for a remote connection e.g. from VSCode, and break remote debugger

    For help with setup, see the README.md file next to this file.

    Args:
        port: the port on which debugpy will wait for a client to connect.
        log_dir: a directory to which detailed debugpy logs are optionally written.
        allow_port_increment: if True, increment the port when failing on an occupied port.
            this also enables connecting multiple debugger processes to the same target running process.
        break_on_connection: break the remote debugger on this breakpoint after connection
    """

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        debugpy.log_to(log_dir.as_posix())

    if not debugpy.is_client_connected():
        try:
            debugpy.listen(port)
            log.warning("Waiting for a client to connect to the debugpy port: %d", port)
            debugpy.wait_for_client()
        except RuntimeError as e:
            if any(DEBUGPY_EXCEPTION_STR in arg for arg in e.args) and allow_port_increment:
                log.info("Debugpy port already in use - incrementing an retrying...")
                port += 1
                breakpoint(port, log_dir)
            else:
                msg = "To find the next free port, use: remote_debug.breakpoint(..., allow_increment=True)"
                log.info(msg)
                raise e

    if break_on_connection:
        debugpy.breakpoint()


def breakpoint_env() -> None:
    """Conditionally activate debugpy remote debugging based on environment variables.

    Environment Variables:
        NUREC_DEBUGPY_ENABLED: if set to "1", "true", "True", or "TRUE", enables remote debugging.
        NUREC_DEBUGPY_PORT: the port on which debugpy will wait for a client to connect. Default is 5678.
        NUREC_DEBUGPY_LOG_DIR: optional directory to which detailed debugpy logs are written.
        NUREC_DEBUGPY_ALLOW_PORT_INCREMENT: if set to "1", "true", "True", or "TRUE", enables port incrementing.
        NUREC_DEBUGPY_BREAK_ON_CONNECTION: if set to "1", "true", "True", or "TRUE", breaks at the breakpoint after connection.
    """
    if is_env_true("NUREC_DEBUGPY_ENABLED", False):
        breakpoint(
            port=int(os.environ.get("NUREC_DEBUGPY_PORT", "5678")),
            log_dir=map_optional(os.environ.get("NUREC_DEBUGPY_LOG_DIR", None), lambda log_dir_str: Path(log_dir_str)),
            allow_port_increment=is_env_true("NUREC_DEBUGPY_ALLOW_PORT_INCREMENT", False),
            break_on_connection=is_env_true("NUREC_DEBUGPY_BREAK_ON_CONNECTION", False),
        )
