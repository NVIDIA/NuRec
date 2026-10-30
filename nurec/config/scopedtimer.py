# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from enum import Enum
from typing import Literal

from nurec.config.base_schema import BaseConfigSchema, Field


log = logging.getLogger(__name__)


class ProfilerBackend(Enum):
    """Available profiler backends for ScopedTimer.

    Attributes:
        NONE: No profiling backend (default, fastest)
        TRACY: Tracy real-time profiler with visual timeline
        NVTX: NVIDIA Tools Extension for GPU profiling with NSys
    """

    NONE = "NONE"
    TRACY = "TRACY"
    NVTX = "NVTX"


# Define a type alias for the verbosity literal type
VerbosityLiteral = Literal["NONE", "SUMMARY", "BASIC", "DETAILS"]


class VerbosityLevel:
    """
    Verbosity level for scoped timer.

    Members:
        NONE: No output.
        SUMMARY: Output a summary.
        BASIC: Basic information.
        DETAILS: Detailed output.
    """

    def __init__(self, value=None):
        # Accept but ignore the value argument for pickle compatibility with artifacts
        # where VerbosityLevel was serialized into checkpoint
        # This allows the class to be unpickled when it was previously an Enum
        pass

    NONE: VerbosityLiteral = "NONE"
    SUMMARY: VerbosityLiteral = "SUMMARY"
    BASIC: VerbosityLiteral = "BASIC"
    DETAILS: VerbosityLiteral = "DETAILS"


class ScopedTimerConfig(BaseConfigSchema):
    """
    Configuration for ScopedTimer.
        Parameters:
            enabled (bool): Enables scoped timer timing measurements

            verbosity (VerbosityLevel): At context manager exit, print elapsed time to func_print_host
            synchronize (bool): Synchronize the CPU thread with any outstanding CUDA work to return accurate GPU timings
            logfile (str): File to write timing results to
            profiling_backend (ProfilerBackend): Profiler backend to use

            emit_initialization (bool): Emit initialization as a capture window before step windows.
            emit_post_train (bool): Emit post-training work as a capture window after step windows.
            emit_start_step (int | None): First step to emit backend ranges.
            emit_num_steps (int | None): Number of steps to emit after start.
            emit_repeat_interval (int | None): Repeat period in steps for emission windows (0/None => single window).
            backward_nvtx_detail (Literal["off", "loss"]): off, or loss to label each autograd node by owning losses.
    """

    enabled: bool = Field(default=False, description="Global variable that determines whether to enable timing.")
    # TODO: If we define verbosity as being VerbosityLevel(Enum), generated checkpoints will be dependent on this nurec type.
    # To make the checkpoint independent of nurec, we need to define verbosity as Literal.
    verbosity: VerbosityLiteral = Field(default=VerbosityLevel.NONE, description="Verbosity level for scoped timer.")
    synchronize: bool = Field(default=False, description="Synchronize GPU work for accurate timing")
    logfile: str | None = Field(default=None, description="File to write timing results to")
    profiling_backend: ProfilerBackend = Field(default=ProfilerBackend.NONE, description="Profiler backend to use")
    emit_initialization: bool = Field(
        default=False, description="Emit initialization as a capture window before configured step windows."
    )
    emit_post_train: bool = Field(
        default=False, description="Emit post-training work as a capture window after configured step windows."
    )
    emit_start_step: int | None = Field(default=None, description="First global step to emit backend ranges.")
    emit_num_steps: int | None = Field(default=None, description="Number of steps to emit after emit_start_step.")
    emit_repeat_interval: int | None = Field(
        default=None, description="Repeat interval (steps) for emission windows (0/None => single window)."
    )
    backward_nvtx_detail: Literal["off", "loss"] = Field(
        default="off",
        description="Loss-owned backward NVTX detail. off, or loss to label each autograd node by owning losses.",
    )
