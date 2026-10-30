# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import logging
import os
import platform
import shutil
import tarfile
import threading

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union, cast

import kornia
import torch
import torchvision.transforms as T

from einops import rearrange
from torch import nn

# Import create_model_registry as a module-level name so patch("nurec.harmonizer.model.create_model_registry") works.
from nurec.utils.model_registry import create_model_registry


logger = logging.getLogger(__name__)


TORCHSCRIPT_MODEL_SUFFIXES = {".pt"}
CHECKPOINT_MODEL_SUFFIXES = {".pkl"}
COSMOS_PREDICT2_TEXT2IMAGE_DIR = "Cosmos-Predict2-0.6B-Text2Image"
COSMOS_MODEL_FILENAME = "model.pt"
COSMOS_TOKENIZER_FILENAME = "tokenizer.pth"


def color_transfer(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Transfers the color distribution from the target image to the source image
    using the method of Reinhard et al., 2001.

    Parameters:
    - source: tensor of the source image.
    - target: tensor of the target image.

    Returns:
    - Color transferred image as a tensor.
    """
    # Convert images from BGR to L*a*b* color space
    source_lab = kornia.color.rgb_to_lab(source.permute(2, 0, 1)).permute(1, 2, 0)
    target_lab = kornia.color.rgb_to_lab(target.permute(2, 0, 1)).permute(1, 2, 0)

    src_mean = source_lab.view(-1, 1, 3).mean(dim=0, keepdim=True)
    src_std = source_lab.view(-1, 1, 3).std(dim=0, keepdim=True)

    target_mean = target_lab.view(-1, 1, 3).mean(dim=0, keepdim=True)
    target_std = target_lab.view(-1, 1, 3).std(dim=0, keepdim=True)

    # Subtract the mean from the source image
    lab = source_lab - src_mean

    # Scale by the standard deviation ratio
    lab = lab * (target_std / src_std)

    # Add the target mean
    lab += target_mean

    # Clip the values to valid range
    lab = lab.clamp(-128, 127)

    # Convert back to BGR color space
    transferred = kornia.color.lab_to_rgb(lab.permute(2, 0, 1)).permute(1, 2, 0)

    return transferred


class HarmonizerSessionClosedError(RuntimeError):
    """Raised when a render targets a HarmonizerState whose session has already been ended."""


@dataclass
class HarmonizerState:
    """Per-session frame history for temporal Harmonizer wrappers.

    Holds up to `max_history` previously enhanced frames normalized to [-1, 1]
    at model_resolution. Frames are stored on whatever device they arrive on
    (in practice the model's output device); no device transfer is performed.

    Represents one coherent render sequence at a time. `reset()` clears the history so the same
    state can start a new sequence; reusing it for unrelated or interleaved sequences without a
    reset mixes their histories and corrupts temporal conditioning.
    """

    history: deque[torch.Tensor] = field(default_factory=deque)
    max_history: int = 4
    # Serializes access to this session's history; held by temporal model forward calls.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    # True once the session has been ended.
    closed: bool = False

    def close(self) -> None:
        """End the session: mark it closed and drop history (frees the GPU frames)."""
        with self.lock:
            self.closed = True
            self.history.clear()

    def reset(self) -> None:
        """Drop history while keeping the session open for a new coherent sequence."""
        with self.lock:
            self.history.clear()

    def push(self, frame: torch.Tensor) -> None:
        """Add an enhanced frame (C, H_model, W_model) normalized to [-1, 1] to history.

        The frame is detached, cloned, and kept on its current device (in practice the model's device).
        Cloning prevents a frame view from retaining the full temporal model output allocation.
        """
        if len(self.history) >= self.max_history:
            self.history.popleft()
        self.history.append(frame.detach().clone())

    def build_fixed_length_temporal_input(self, current: torch.Tensor) -> torch.Tensor:
        """Stack current frame with history into a fixed-length temporal input.

        Missing history frames are padded by repeating the current frame, so the
        returned tensor always has T = 1 + max_history. Frame order is
        ``[current, t-1, t-2, ..., t-K]`` per the DiffusionHarmonizer paper.
        """
        # history is oldest-first; reversed gives [t-1, t-2, ..., t-K].
        history_frames = [f.unsqueeze(0) for f in reversed(self.history)]
        n_pad = self.max_history - len(history_frames)
        if n_pad > 0:
            history_frames.extend([current] * n_pad)
        all_frames = [current] + history_frames
        return torch.cat([f.unsqueeze(2) for f in all_frames], dim=2)

    def build_variable_length_temporal_input(self, current: torch.Tensor) -> torch.Tensor:
        """Stack current frame with history into the variable-length checkpoint input.

        The upstream checkpoint path uses only T=1 until ``max_history`` outputs
        exist, then uses ``[current, t-1, ..., t-K]``.
        """
        if len(self.history) < self.max_history:
            return current.unsqueeze(2)
        history_frames = [f.unsqueeze(0) for f in reversed(self.history)]
        all_frames = [current] + history_frames[: self.max_history]
        return torch.cat([f.unsqueeze(2) for f in all_frames], dim=2)


class BaseHarmonizerModel(nn.Module):
    """Shared infrastructure for Harmonizer model wrappers."""

    def __init__(
        self,
        model_url: str,
        cache_dir: str,
        model_filename: str,
        model_resolution: tuple[int, int],
    ):
        """ """
        super().__init__()
        self.model_url = model_url
        self.model: Any = None
        self.cache_dir = cache_dir
        self.model_filename = model_filename
        self._lock = threading.Lock()

        if (
            not isinstance(model_resolution, tuple)
            or len(model_resolution) != 2
            or not all(isinstance(x, int) for x in model_resolution)
        ):
            raise ValueError(f"{self.__class__.__name__}: model_resolution must be a tuple of two integers")
        self.model_resolution = torch.Size(model_resolution)

    @staticmethod
    def _download_checkpoint(model_url: str, cache_dir: str, model_filename: str) -> str:
        """Download the model using the model_registry module."""
        cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        cache_dir_path = Path(cache_dir)
        cached_checkpoint_file = os.path.join(cache_dir, model_filename)

        if os.path.exists(cached_checkpoint_file):
            logger.info("Using cached Harmonizer model (%s).", cached_checkpoint_file)
            return cached_checkpoint_file

        # If the checkpoint file does not exist, create the model registry and download the model
        registry = create_model_registry(model_url, cache_dir_path)
        downloaded_file = registry.get_model()

        assert os.path.exists(downloaded_file), "Harmonizer: Model file not found"

        # If the downloaded file is a tarball, extract it
        if downloaded_file.endswith(".tar.gz"):
            # Extract file
            with tarfile.open(downloaded_file, "r:gz") as tar:
                # Get model files from the tarball matching the configured artifact kind.
                model_suffix = Path(model_filename).suffix.lower()
                supported_suffixes = TORCHSCRIPT_MODEL_SUFFIXES | CHECKPOINT_MODEL_SUFFIXES
                model_files = [
                    member
                    for member in tar.getmembers()
                    if Path(member.name).suffix.lower() == model_suffix
                    or (not model_suffix and Path(member.name).suffix.lower() in supported_suffixes)
                ]
                # Assert that there is only one model file in the tarball
                if not len(model_files) == 1:
                    raise ValueError(
                        f"Harmonizer: Expected exactly one model file in the tarball, got {len(model_files)}",
                    )
                # Extract the model file
                logger.debug("Extracting model file from %s to %s", downloaded_file, cache_dir)
                tar.extract(model_files[0], path=cache_dir)

            # Move the model file to the cached_checkpoint_file
            logger.debug("Moving model file to %s", cached_checkpoint_file)
            shutil.move(os.path.join(cache_dir, model_files[0].name), cached_checkpoint_file)

            # Remove the tarball
            os.remove(downloaded_file)
        else:
            # If not a tarball, assume it's already the checkpoint
            # Some registries download direct checkpoint files to the final cache path.
            if Path(downloaded_file).resolve() != Path(cached_checkpoint_file).resolve():
                shutil.move(downloaded_file, cached_checkpoint_file)

        return cached_checkpoint_file

    def _model_device(self) -> torch.device:
        """Return the device of the loaded model."""
        assert self.model is not None
        first = next(self.model.parameters(), None)
        if first is None:
            first = next(self.model.buffers(), None)
        if first is None:
            return torch.device("cuda", torch.cuda.current_device())
        return first.device

    @staticmethod
    def _preprocess_input(
        img: torch.Tensor, img_size: torch.Size, model_resolution: torch.Size
    ) -> tuple[torch.Tensor, torch.dtype]:
        """preprocess input tensors to correspond to pre-defined resolution, rescale to [-1,1]"""

        # Check if input is fp16 and convert to bf16 if needed (SANA model does not support fp16)
        img_dtype = img.dtype
        if img_dtype == torch.float16:
            img = img.to(torch.bfloat16)
        img = rearrange(img, "(h w) c -> 1 c h w", h=img_size[0])

        transforms = T.Compose(
            [
                #  T.GaussianBlur(kernel_size=(11, 11), sigma=3.0), # TODO - add blurring to avoid aliasing!!!
                T.Resize(
                    model_resolution,
                    interpolation=T.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                T.Lambda(lambda x: x * 2 - 1),
            ]
        )
        return transforms(img), img_dtype

    @staticmethod
    def _postprocess_output(img: torch.Tensor, img_size: torch.Size, img_dtype: torch.dtype) -> torch.Tensor:
        if img.dtype == torch.bfloat16 and img_dtype != torch.bfloat16:
            img = img.to(img_dtype)
        if img_size != torch.Size(img.shape[-2:]):
            logger.debug(f"Harmonizer processed image will be resampled to its original resolution: {img_size}.")
        transforms = T.Compose(
            [
                T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                T.Lambda(lambda x: (x + 1) / 2),
            ]
        )
        img = rearrange(transforms(img), "1 c h w -> (h w) c")

        return img


class BaseHarmonizerModelTraced(BaseHarmonizerModel):
    """Base class for TorchScript-exported Harmonizer artifacts."""

    def _load_pretrained_model(self):
        cached_checkpoint = self._download_checkpoint(self.model_url, self.cache_dir, self.model_filename)
        # transformer_engine dlopens cuDNN when it is imported. On x86_64 this transparently
        # resolves to the nvidia-cudnn-cu12 wheel pulled in by torch. On aarch64, cuDNN is instead
        # bundled under torch's torch/lib, we point CUDNN_PATH at it to resolve it, if not already
        # set by the user.
        if platform.machine() == "aarch64":
            os.environ.setdefault("CUDNN_PATH", os.path.join(os.path.dirname(torch.__file__), "lib"))
        # Load-bearing import: transformer_engine registers the tex_ts::* TorchScript custom ops the
        # JIT checkpoint calls, so it must run before torch.jit.load. It looks unused — do NOT remove.
        # TODO: Investigate why importing transformer_engine causes CUDA_ERROR_UNSUPPORTED_PTX_VERSION.
        import transformer_engine  # noqa: F401

        logger.info(f"Loading model with TE saved at {cached_checkpoint}, {os.path.exists(cached_checkpoint)}")
        if hasattr(torch.ops, "tex_ts"):
            logger.info(f"has rmsnorm_fwd_inf_ts: {hasattr(torch.ops.tex_ts, 'rmsnorm_fwd_inf_ts')}")

        model = torch.jit.load(cached_checkpoint)
        model = model.cuda()
        # logger.info('------ model', logger.info(model.inlined_graph))
        logger.info(f"model {cached_checkpoint} loaded successfully")
        return model


class NonTemporalHarmonizerModelTraced(BaseHarmonizerModelTraced):
    """Image-based (non-temporal) Harmonizer. One frame in, one frame out.

    Supports both the 5-D Harmonizer checkpoint and the legacy 4-D (difix)
    checkpoint; `_detect_needs_temporal_dim` probes which one was loaded so
    `forward` can add/strip the temporal axis accordingly.
    """

    def __init__(
        self,
        model_url: str,
        cache_dir: str,
        model_filename: str,
        model_resolution: tuple[int, int],
    ):
        super().__init__(model_url, cache_dir, model_filename, model_resolution)
        self.needs_temporal_dim = False

    def _detect_needs_temporal_dim(self) -> None:
        """Probe the loaded model with a dummy input to determine whether it expects a temporal dimension."""
        assert self.model is not None, "Model must be loaded before calling _detect_needs_temporal_dim"
        dummy = torch.zeros(1, 3, *self.model_resolution, device=self._model_device())
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
            try:
                self.model(dummy)
                self.needs_temporal_dim = False
                logger.info("Harmonizer model accepts 4-D input (no temporal dim).")
            except RuntimeError as e:
                message = str(e).lower()
                # Only catch dim-mismatch errors here; matching every RuntimeError would
                # mask unrelated failures (checkpoint load, CUDA OOM, etc.). Tokens are
                # wordings observed empirically — extend as new variants surface.
                if not any(token in message for token in ("expected 5-d", "expected 5d", "dimension")):
                    raise

                self.model(dummy.unsqueeze(2))
                self.needs_temporal_dim = True
                logger.info("Harmonizer model requires 5-D input (temporal dim detected).")

    def forward(self, input: torch.Tensor, img_size: torch.Size, use_color_transfer: bool) -> torch.Tensor:
        if self.model is not None:
            # A JIT model may load onto the default CUDA device. Align it to the input's
            # device for single-process callers that select another GPU.
            if self._model_device() != input.device:
                self.model = self.model.to(input.device)
            preprocessed_input, img_dtype = self._preprocess_input(input, img_size, self.model_resolution)
            if self.needs_temporal_dim:
                preprocessed_input = preprocessed_input.unsqueeze(2)
            # Disable autocast to avoid mixed input types into the model (SANA model does not support fp16)
            with torch.amp.autocast(device_type="cuda", enabled=False):
                result = self.model(preprocessed_input)
            if self.needs_temporal_dim:
                result = result.squeeze(2)
            result = self._postprocess_output(result, img_size, img_dtype)
            if use_color_transfer:
                result = color_transfer(result.view(*img_size, -1), input.view(*img_size, -1)).view(input.shape)
            return result
        else:
            raise ValueError("Harmonizer: failed to find model instance.")

    # TODO __call__ = module_call_type(forward) - currently module_call_type lives in nurec.model.base which causes circular dependencies


class TemporalHarmonizerModelTraced(BaseHarmonizerModelTraced):
    """Temporal Harmonizer. Conditions each frame on up to K=4 previously enhanced frames.

    This class *wraps* the JIT checkpoint of the temporal harmonizer with a frozen history length.
    Its `forward` takes the currently rendered frame, plus a per-session state holding the past frames:

        forward(input: (H*W, C) image, img_size, state, use_color_transfer=False)
            -> (H*W, C) refined current frame

    Internally it preprocesses `input`, asks `state` to build the temporal stack,
    and calls the JIT model:

        JIT input:  (1, C, T, H_model, W_model),  T = 1 + max_history
        JIT output: (1, C, T, H_model, W_model),  same temporal dim

    HarmonizerState pads missing history slots with the current frame so T is
    always fixed at what the published checkpoint was traced for. The refined
    "current" frame is at output T-slot 0 (per the paper's Eq. 2); the remaining
    T-slots are conditioning outputs and are discarded. History is kept normalized
    to [-1, 1] at model_resolution on the model device.

    Requires a HarmonizerState per render session; the state is updated in-place on each call
    of the TemporalHarmonizerModelTraced.
    """

    def forward(
        self,
        input: torch.Tensor,
        img_size: torch.Size,
        state: HarmonizerState,
        use_color_transfer: bool = False,
    ) -> torch.Tensor:
        if self.model is None:
            raise ValueError("TemporalHarmonizerModelTraced: model not loaded")
        if use_color_transfer:
            raise ValueError(
                "TemporalHarmonizerModelTraced does not support external color transfer "
                "(that is legacy difix / non-temporal only); the temporal harmonizer handles color itself."
            )
        preprocessed, img_dtype = self._preprocess_input(input, img_size, self.model_resolution)
        # preprocessed: (1, C, H_model, W_model) in [-1, 1]
        # Hold state.lock across the whole critical section (build temporal input -> harmonizer
        # inference -> update harmonizer state): same-session renders must serialize.
        with state.lock:
            if state.closed:
                raise HarmonizerSessionClosedError("harmonizer session has been ended")
            temporal_input = state.build_fixed_length_temporal_input(preprocessed)  # (1, C, T, H_model, W_model)
            # self._lock serializes the shared JIT model across sessions.
            with self._lock, torch.amp.autocast(device_type="cuda", enabled=False):
                # TODO: remove this try/except once multi-GPU issue is fixed — the JIT trace
                # hard-codes a cuda device that .to() cannot relocate.
                try:
                    out = self.model(temporal_input)  # (1, C, T_in, H_model, W_model)
                except RuntimeError as e:
                    # A cross-device error when input and model already share a device points at
                    # a cuda device hard-coded in the trace, not a caller mistake; hint and re-raise.
                    message = str(e).lower()
                    if input.device == self._model_device() and any(
                        token in message for token in ("same device", "two devices", "device mismatch")
                    ):
                        logger.error(
                            "Temporal harmonizer device mismatch though input and model are both on %s. "
                            "The JIT trace hard-codes a cuda device that .to() cannot relocate; run with "
                            "a single GPU visible (e.g. CUDA_VISIBLE_DEVICES).",
                            input.device,
                        )
                    raise
            # The published JIT checkpoint returns the full temporal stack (1, C, T_in, H, W);
            # take T-slot 0 (the current frame, per paper's Eq. 2).
            out_frame = out[:, :, 0, :, :]  # (1, C, H_model, W_model) in [-1, 1]
            # Store output frame in [-1, 1] for use as history input next call.
            state.push(out_frame.squeeze(0))  # → (C, H_model, W_model) on the model device
        return self._postprocess_output(out_frame, img_size, img_dtype)


class BaseHarmonizerModelCheckpointed(BaseHarmonizerModel):
    """Base class for upstream PyTorch checkpoint-backed Harmonizer artifacts.

    The checkpoint path reconstructs the upstream Harmonizer/Cosmos Predict2
    PyTorch module and then loads `state_dict_unet` / `state_dict_vae`.
    """

    def __init__(
        self,
        model_url: str,
        cache_dir: str,
        model_filename: str,
        model_resolution: tuple[int, int],
        *,
        cosmos_base_model_url: str | None = None,
        cosmos_tokenizer_url: str | None = None,
        cosmos_cache_dir: str | None = None,
    ):
        super().__init__(model_url, cache_dir, model_filename, model_resolution)
        self.cosmos_base_model_url = cosmos_base_model_url
        self.cosmos_tokenizer_url = cosmos_tokenizer_url
        self.cosmos_cache_dir = cosmos_cache_dir

    def _cosmos_dir(self) -> Path:
        default_cosmos_dir = (
            Path(os.path.abspath(os.path.expanduser(self.cache_dir))) / "nvidia" / COSMOS_PREDICT2_TEXT2IMAGE_DIR
        )
        configured_cosmos_dir = (
            os.environ.get("NUREC_HARMONIZER_COSMOS_DIR") or self.cosmos_cache_dir or str(default_cosmos_dir)
        )
        return Path(configured_cosmos_dir).expanduser().resolve()

    def _download_cosmos_weights(self) -> Path:
        """Download missing Cosmos Predict2 base weights into the expected cache layout."""
        cosmos_dir = self._cosmos_dir()
        model_path = cosmos_dir / COSMOS_MODEL_FILENAME
        tokenizer_path = cosmos_dir / "tokenizer" / COSMOS_TOKENIZER_FILENAME

        missing_config = []
        if not model_path.exists() and not self.cosmos_base_model_url:
            missing_config.append("cosmos_base_model_url")
        if not tokenizer_path.exists() and not self.cosmos_tokenizer_url:
            missing_config.append("cosmos_tokenizer_url")
        if missing_config:
            raise FileNotFoundError(
                "Checkpointed temporal Harmonizer requires Cosmos Predict2 base weights. "
                f"Missing files: {[str(p) for p in (model_path, tokenizer_path) if not p.exists()]}. "
                f"Configure {missing_config} or place the files under {cosmos_dir}."
            )

        if not model_path.exists():
            assert self.cosmos_base_model_url is not None
            self._download_checkpoint(self.cosmos_base_model_url, str(cosmos_dir), COSMOS_MODEL_FILENAME)
        if not tokenizer_path.exists():
            assert self.cosmos_tokenizer_url is not None
            self._download_checkpoint(
                self.cosmos_tokenizer_url, str(cosmos_dir / "tokenizer"), COSMOS_TOKENIZER_FILENAME
            )

        return cosmos_dir

    def _ensure_cosmos_weights(self) -> Path:
        """Download and validate the Cosmos Predict2 weights required by the upstream model."""
        cosmos_dir = self._download_cosmos_weights()
        required_files = [
            cosmos_dir / COSMOS_MODEL_FILENAME,
            cosmos_dir / "tokenizer" / COSMOS_TOKENIZER_FILENAME,
        ]
        missing = [path for path in required_files if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Checkpointed temporal Harmonizer requires Cosmos Predict2 weights in "
                f"{cosmos_dir}. Missing: {[str(path) for path in missing]}"
            )

        return cosmos_dir

    def _load_pretrained_model(self):
        cached_checkpoint = self._download_checkpoint(self.model_url, self.cache_dir, self.model_filename)
        logger.info("Loading checkpointed temporal Harmonizer from %s", cached_checkpoint)

        try:
            Pix2Pix_Turbo = getattr(importlib.import_module("pix2pix_turbo_harmonizer"), "Pix2Pix_Turbo")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Checkpointed temporal Harmonizer loading requires the upstream Harmonizer Bazel library "
                "plus its Cosmos Predict2 runtime dependencies (`cosmos_predict2` and `imaginaire`). "
                "The TorchScript `.pt` path does not need these dependencies."
            ) from e

        device = torch.device("cuda", torch.cuda.current_device())
        cosmos_dir = self._ensure_cosmos_weights()
        model = Pix2Pix_Turbo(
            pretrained_path=cached_checkpoint,
            timestep=250,
            train_full_unet=True,
            freeze_vae=False,
            vae_skip_connection=False,
            use_sched=True,
            device=device,
            dtype=torch.bfloat16,
            cosmos_dir=str(cosmos_dir),
        )
        model.set_eval()
        return model.to(device=device, dtype=torch.bfloat16)


class TemporalHarmonizerModelCheckpointed(BaseHarmonizerModelCheckpointed):
    """Temporal Harmonizer loaded from the research `.pkl` checkpoint.

    The checkpoint path reconstructs the upstream Harmonizer/Cosmos Predict2
    PyTorch module and then loads `state_dict_unet` / `state_dict_vae`.
    """

    def forward(
        self,
        input: torch.Tensor,
        img_size: torch.Size,
        state: HarmonizerState,
        use_color_transfer: bool = False,
    ) -> torch.Tensor:
        if self.model is None:
            raise ValueError("TemporalHarmonizerModelCheckpointed: model not loaded")
        if use_color_transfer:
            raise ValueError(
                "TemporalHarmonizerModelCheckpointed does not support external color transfer; "
                "the checkpointed harmonizer handles color itself."
            )

        preprocessed, img_dtype = self._preprocess_input(input, img_size, self.model_resolution)
        preprocessed = preprocessed.to(device=self._model_device(), dtype=torch.bfloat16)
        with state.lock:
            if state.closed:
                raise HarmonizerSessionClosedError("harmonizer session has been ended")
            temporal_input = state.build_variable_length_temporal_input(preprocessed)
            with self._lock:
                with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
                    # Try to use the patched upstream fast path, which only decodes the current
                    # frame in temporal slot 0; fall back to forward() if unavailable.
                    current_frame_forward = getattr(self.model, "forward_current_only", None)
                    out = (
                        current_frame_forward(temporal_input)
                        if current_frame_forward is not None
                        else self.model(temporal_input)
                    )
            out_frame = out[:, :, 0, :, :]
            state.push(out_frame.squeeze(0))
        return self._postprocess_output(out_frame, img_size, img_dtype)


HarmonizerModelType = Union[
    NonTemporalHarmonizerModelTraced, TemporalHarmonizerModelTraced, TemporalHarmonizerModelCheckpointed
]


class HarmonizerModelFactory:
    """Process-wide singleton factory for Harmonizer models.

    Dispatches to the concrete traced/checkpointed wrapper from the requested
    artifact and the `temporal` flag. At most one model instance is held at a
    time; switching modes within a process requires an explicit clear_cache() first.
    """

    _initialized_model: HarmonizerModelType | None = None
    _lock = threading.Lock()

    @staticmethod
    def _infer_artifact_kind(model_filename: str) -> str:
        suffix = Path(model_filename).suffix.lower()
        if suffix in TORCHSCRIPT_MODEL_SUFFIXES:
            return "torchscript"
        if suffix in CHECKPOINT_MODEL_SUFFIXES:
            return "checkpoint"
        raise ValueError(
            f"Unsupported Harmonizer model file suffix {suffix!r} for {model_filename!r}. "
            f"TorchScript suffixes: {sorted(TORCHSCRIPT_MODEL_SUFFIXES)}; "
            f"checkpoint suffixes: {sorted(CHECKPOINT_MODEL_SUFFIXES)}."
        )

    @staticmethod
    def _expected_model_class(model_filename: str, temporal: bool) -> type[BaseHarmonizerModel]:
        artifact_kind = HarmonizerModelFactory._infer_artifact_kind(model_filename)
        if artifact_kind == "torchscript":
            return TemporalHarmonizerModelTraced if temporal else NonTemporalHarmonizerModelTraced
        if temporal:
            return TemporalHarmonizerModelCheckpointed
        raise ValueError(
            "Checkpointed Harmonizer artifacts are only wired for temporal inference right now. "
            "Use the exported TorchScript `.pt` model for non-temporal inference."
        )

    @classmethod
    def get(
        cls,
        model_url: str,
        cache_dir: str,
        model_filename: str,
        model_resolution: tuple[int, int],
        temporal: bool = False,
        *,
        cosmos_base_model_url: str | None = None,
        cosmos_tokenizer_url: str | None = None,
        cosmos_cache_dir: str | None = None,
    ) -> HarmonizerModelType:
        # Keep this check outside the lock so every DDP rank fails before model initialization or download.
        # The Harmonizer model isn't multi-GPU-safe (shared-cache download race + device-baked constants).
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if world_size > 1:
            raise RuntimeError(
                f"Harmonizer does not support multi-GPU execution (world_size={world_size}). Its "
                "TorchScript model is not multi-GPU-safe (shared model-cache download race and "
                "device-baked constants). Re-run with world_size=1."
            )
        with cls._lock:
            expected_cls = cls._expected_model_class(model_filename, temporal)
            if cls._initialized_model is not None:
                if not isinstance(cls._initialized_model, expected_cls):
                    raise RuntimeError(
                        f"HarmonizerModelFactory already holds a {type(cls._initialized_model).__name__} "
                        f"but get() was called with temporal={temporal}. Call clear_cache() before "
                        "switching modes within the same process."
                    )
                return cls._initialized_model
            model_kwargs = {}
            if issubclass(expected_cls, BaseHarmonizerModelCheckpointed):
                model_kwargs = {
                    "cosmos_base_model_url": cosmos_base_model_url,
                    "cosmos_tokenizer_url": cosmos_tokenizer_url,
                    "cosmos_cache_dir": cosmos_cache_dir,
                }
            model_instance = cast(
                HarmonizerModelType,
                expected_cls(model_url, cache_dir, model_filename, model_resolution, **model_kwargs),
            )
            # Eagerly download checkpoint
            cached = BaseHarmonizerModel._download_checkpoint(model_url, cache_dir, model_filename)
            logger.info(f"HarmonizerModelFactory: checkpoint ready at {cached}")
            # Eagerly load pretrained model
            model_instance.model = model_instance._load_pretrained_model()
            logger.info("HarmonizerModelFactory: model loaded successfully")
            # The non-temporal path still supports the legacy 4-D (difix) checkpoint,
            # so probe which input dimensionality the loaded model expects.
            if isinstance(model_instance, NonTemporalHarmonizerModelTraced):
                model_instance._detect_needs_temporal_dim()
            cls._initialized_model = model_instance
            return cls._initialized_model

    @classmethod
    def clear_cache(cls):
        with cls._lock:
            cls._initialized_model = None
