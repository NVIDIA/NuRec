# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import importlib
import threading

from typing import Any

import torch


class NvjpegImageEncodingError(RuntimeError):
    """A failure that may be isolated to one image in a native batch."""


class NvjpegEncoderPoisonedError(RuntimeError):
    """The native encoder became unsafe to reuse after an asynchronous CUDA failure."""


class NvjpegBatchEncoder:
    def __init__(self) -> None:
        if getattr(builtins, "__PYCENA__", False):
            raise ImportError("C++ nvJPEG batch encoder is not packaged in pycena runtimes.")

        module_name = ".".join(("nurec", "grpc", "libnurec_nvjpeg_batch_encoder_cc"))
        self._module = importlib.import_module(module_name)
        self.max_batch_size = int(self._module.MAX_BATCH_SIZE)
        self._encoders: dict[int, Any] = {}
        self._encoders_lock = threading.Lock()

    def encode(self, images_u8: list[torch.Tensor], quality: int) -> list[bytes]:
        if not images_u8:
            return []
        if not images_u8[0].is_cuda:
            raise ValueError("nvJPEG encoder expects CUDA tensors.")

        device = images_u8[0].get_device()
        if any(not image.is_cuda or image.get_device() != device for image in images_u8):
            raise ValueError("All images in a batch must be on the same CUDA device.")

        with self._encoders_lock:
            encoder = self._encoders.get(device)
            if encoder is None:
                encoder = self._module.NvjpegBatchEncoder(device)
                self._encoders[device] = encoder

        try:
            return encoder.encode_batch(images_u8, int(quality))
        except self._module.NvjpegOutOfMemoryError as exc:
            raise torch.cuda.OutOfMemoryError(str(exc)) from exc
        except self._module.NvjpegEncoderPoisonedError as exc:
            with self._encoders_lock:
                if self._encoders.get(device) is encoder:
                    del self._encoders[device]
            raise NvjpegEncoderPoisonedError(str(exc)) from exc
        except self._module.NvjpegImageEncodingError as exc:
            raise NvjpegImageEncodingError(str(exc)) from exc
