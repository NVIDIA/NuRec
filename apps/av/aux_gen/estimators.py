# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import math
import platform

from contextlib import contextmanager
from pathlib import Path
from typing import List, Literal, Mapping, NamedTuple, Optional

# We are type-ignoring most 3rdparty as these
# wheels are not part of the main toolchains requirement file
import mmseg.apis  # type: ignore
import numpy as np
import point_cloud_utils as pcu
import torch
import torch.nn.functional as F

from PIL import Image as PILImage
from python.runfiles import runfiles
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision import transforms

import ncore.data

from apps.av.aux_gen.device_frame import (
    DeviceFrameOutputSpec,
    DeviceFrameProcessor,
    DeviceFrameProcessorMetadata,
    process_device_frame_from_host,
)
from apps.av.aux_gen.segmentation_device_pipeline import (
    CAMERA_MASK_INPUT,
    MASK2FORMER_PROCESSOR,
    SAM2_MASK_OUTPUT,
    SAM2_PROCESSOR,
    SEMANTIC_LABELS_OUTPUT,
    SEMANTIC_LOGITS_OUTPUT,
    SEMANTIC_UNCERTAINTY_OUTPUT,
    _mask2former_output_specs,
    _sam2_output_specs,
)
from apps.av.aux_gen.segmentation_presentation import (
    is_dark_image,
)
from apps.av.aux_gen.segmentation_presentation import (
    labels_to_paletted_image as _labels_to_paletted_image,
)
from apps.av.aux_gen.segmentation_presentation import (
    save_mask2former_visualization as _save_mask2former_visualization,
)
from apps.av.aux_gen.trt_engine_selection import select_trt_checkpoint
from apps.av.aux_gen.utils import LidarSpinMesh, _quantile, plot_points_on_image_with_color
from ncore.impl.common.transformations import transform_point_cloud
from ncore.sensors import camera_model_from_parameters
from nurec.libs.vren.ensemble import ensemble_cuda, ensemble_numba  # type: ignore
from nurec.utils.geometry import se3_matrix_inverse
from nurec.utils.ncore_utils import AuxDataCameraSemSegProvider


RUNFILES = runfiles.Create()


def find_network_file_path(filename: str, search_roots: Optional[List[Path]] = None) -> Path:
    # external/ is for standard runfiles

    default_search_roots: list[Path] = []

    for search_path in [
        "mmseg_repo/configs/dinov2_sam",
        "mask2former_weights/file",
        "trt_engine_a40_x86_64/file",
        "trt_engine_l40_x86_64/file",
        "trt_engine_l40_aarch64/file",
        "trt_engine_a100_x86_64/file",
        "trt_engine_a6000_x86_64/file",
        "trt_engine_ampereplus_x86_64/file",
        "trt_engine_ampereplus_aarch64/file",
        "trt_engine_5880_x86_64/file",
        "trt_engine_l4_x86_64/file",
        "sam2_egomask_weights/file",
    ]:
        # Add runfiles-based search-roots
        if RUNFILES is not None:
            rpath = RUNFILES.Rlocation(f"{search_path}/{filename}")
            if rpath is not None:
                default_search_roots.append(Path(rpath).parent)

        # Add runfiles-based search-roots
        default_search_roots.append(Path("../" + search_path))

    if search_roots is None:
        search_roots = default_search_roots

    for path in search_roots:
        if path.exists():
            file_matches = list(path.rglob(f"**/{filename}"))

            # Take only first
            if len(file_matches) > 0:
                return file_matches[0]

    raise FileNotFoundError(f"Could not find file: {filename} in {default_search_roots}")


def get_gpu_info() -> tuple[str | None, str | None]:
    """Return the current CUDA device name and compute capability.

    Returns:
        The active device name and ``major.minor`` compute capability, or
        ``(None, None)`` when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return None, None
    device_id = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_id)
    name = torch.cuda.get_device_name(device_id)
    return name, f"{capability[0]}.{capability[1]}"


class SegmentationReturn(NamedTuple):
    semantic_seg: Optional[PILImage.Image]
    semantic_seg_logits: Optional[np.ndarray]
    semantic_dinov2_feats: Optional[np.ndarray]
    instance_seg: Optional[dict[str, np.ndarray]]


@contextmanager
def temporary_torch_load_override():
    """Helper function for applying local override to the torch.load() API"""
    original_load = torch.load
    torch.load = functools.partial(torch.load, weights_only=False)
    try:
        yield
    finally:
        torch.load = original_load


class Mask2FormerSegmentationEstimator(DeviceFrameProcessor):
    """
    Save semantic segmentation logits. Use a constant resolution set by segmentor's config.
    """

    name = MASK2FORMER_PROCESSOR

    def __init__(
        self,
        resolution: list,
        dataset_name: str = "PrivateDataset_20c",
        config: str = "mask2former_dino_nv_redim256_private12k_aug_backboneTTA.py",
        pretrained_checkpoint: str = "mask2former_dinov2_nv_private12k_backboneTTA_76000.pth",
        estimate_logits: bool = False,
        estimate_dinov2_feats: bool = False,
        backbone_feats_dim: int = -1,
        enable_trt: bool = True,
        output_uncertainty_mask: bool = False,
    ) -> None:
        """Initialize a CUDA Mask2Former estimator.

        Args:
            resolution: Raw image resolution as ``[width, height]``.
            dataset_name: Semantic dataset identifier stored in output
                metadata.
            config: MMSeg configuration filename resolved from runfiles.
            pretrained_checkpoint: PyTorch checkpoint filename resolved from
                runfiles.
            estimate_logits: Whether to return per-class semantic logits.
            estimate_dinov2_feats: Whether host predictions include DINO
                backbone features.
            backbone_feats_dim: Requested backbone feature dimension, or
                ``-1`` to preserve the model configuration.
            enable_trt: Whether a compatible precompiled TensorRT engine may
                replace the PyTorch model.
            output_uncertainty_mask: Whether device-processor outputs include
                the semantic uncertainty mask used by visualization.

        Raises:
            FileNotFoundError: If the config, checkpoint, or selected engine
                cannot be resolved.
            ValueError: If model configuration or resolution is invalid.
            RuntimeError: If CUDA model or TensorRT engine initialization
                fails.
        """
        self.resolution = resolution
        self.dataset_name = dataset_name
        self.config = config
        self.method = "mask2former"
        self.estimate_logits = estimate_logits
        self.estimate_dinov2_feats = estimate_dinov2_feats
        self.backbone_feats_dim = backbone_feats_dim
        self.output_uncertainty_mask = output_uncertainty_mask

        config_path = find_network_file_path(self.config)
        checkpoint_path = find_network_file_path(pretrained_checkpoint)

        machine = platform.machine()
        gpu_name, sm_version = get_gpu_info()
        trt_checkpoint = select_trt_checkpoint(gpu_name, sm_version, machine) if enable_trt else None

        if trt_checkpoint is None:
            if enable_trt:
                logging.warning(
                    "TensorRT requested but no precompiled engine available for %s "
                    "(sm_%s, %s); falling back to a slower code path.",
                    gpu_name,
                    sm_version,
                    machine,
                )
            trt_checkpoint_path = None
        else:
            trt_checkpoint_path = find_network_file_path(trt_checkpoint)

        args = {
            # config path
            "config": str(config_path),
            # ckpt path
            "checkpoint": str(checkpoint_path),
            "device": "cuda",
            "cfg_options": {"model.inference_backbone_feats_dim": backbone_feats_dim},
            "trt_checkpoint": trt_checkpoint_path,
            "resolution": resolution if trt_checkpoint_path else None,
        }

        # Set torch.load(weights_only=False) to allow loading the full
        # checkpoint including HistoryBuffer. Minimise scope of this overload
        # with save/restore of original torch.load() function.
        with temporary_torch_load_override():
            self.predictor = mmseg.apis.init_model(
                args["config"],
                args["checkpoint"],
                device=args["device"],
                cfg_options=args["cfg_options"],
                resolution=args["resolution"],
                trt_checkpoint=args["trt_checkpoint"],
            )

        self.pretrained_checkpoint = args["checkpoint"]
        self.trt_checkpoint: Path | None = trt_checkpoint_path

        # Get the semantic metadata
        self.stuff_classes = self.predictor.dataset_meta["classes"]
        self.stuff_colors = self.predictor.dataset_meta["palette"]

    def get_device_input_spec(self) -> tuple[tuple[int, ...], torch.dtype]:
        """Return the raw CUDA image input contract.

        Returns:
            A pair containing the ``[B, 3, H, W]`` input shape and its
            :class:`torch.dtype`. Input uses BGR channel order and must be
            contiguous on the predictor's CUDA device. TensorRT uses its
            binding dtype; PyTorch accepts raw ``uint8`` pixels.

        """
        input_shape, input_dtype, _ = mmseg.apis.get_model_device_input_spec(
            self.predictor,
            tuple(int(value) for value in self.resolution),
        )
        return input_shape, input_dtype

    def _post_process_device_tensors(
        self,
        logits_tensor: torch.Tensor,
        pred_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply uncertainty filtering while keeping all tensors on CUDA.

        Args:
            logits_tensor: Class logits with shape ``[C, H, W]`` on CUDA.
                The tensor is converted to ``float32`` for entropy processing.
            pred_labels: Predicted labels with shape ``[1, H, W]`` on CUDA.

        Returns:
            The filtered ``uint8[H, W]`` labels, ``float32[C, H, W]`` logits,
            and ``bool[H, W]`` uncertainty mask on the prediction tensors'
            device. Logits and labels may reuse or mutate backend storage;
            callers must consume them before the next inference.

        Raises:
            ValueError: If the configured class metadata has no ``building``
                fallback class.
        """
        if logits_tensor.device != pred_labels.device:
            raise ValueError(
                f"Mask2Former logits are on {logits_tensor.device}, but labels are on {pred_labels.device}"
            )
        logits_tensor = logits_tensor.to(dtype=torch.float32)
        probs = F.softmax(logits_tensor, dim=0)
        entropy = torch.special.entr(probs).sum(dim=0)
        quantile = 0.90
        if entropy.numel() <= 4096**2:
            entropy_threshold = torch.quantile(entropy, quantile)
        else:
            entropy_threshold = _quantile(entropy, quantile)

        uncertainty_mask = entropy > entropy_threshold
        labels_tensor = pred_labels.to(dtype=torch.uint8).squeeze(0)
        building_id = self.stuff_classes.index("building")
        labels_tensor[uncertainty_mask] = building_id

        if self.estimate_logits:
            logits_tensor[:, uncertainty_mask] = 0
            logits_tensor[building_id, uncertainty_mask] = 1

        return labels_tensor, logits_tensor, uncertainty_mask

    def infer_device(
        self,
        image_bgr_nchw: torch.Tensor,
        ego_mask: torch.Tensor | None,
        stream: torch.cuda.Stream,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Run the common CUDA inference and semantic post-processing path.

        Args:
            image_bgr_nchw: Contiguous raw BGR ``[1, 3, H, W]`` tensor on the
                predictor's CUDA device. TensorRT requires its binding dtype;
                PyTorch requires ``uint8`` values in ``[0, 255]``.
            ego_mask: Optional read-only CUDA ``bool[H, W]`` mask on the same
                device.
            stream: CUDA stream owning preprocessing, inference, and
                post-processing.

        Returns:
            Filtered ``uint8[H, W]`` labels, ``float32[C, H, W]`` logits,
            optional CUDA DINO backbone features with the configured model
            shape and dtype, and ``bool[H, W]`` uncertainty. Backend outputs
            may alias TensorRT bindings and must be consumed or copied before
            the next inference on this estimator.

        Raises:
            ValueError: If image, stream, mask, or model outputs violate the
                configured tensor contract.
            RuntimeError: If the backend fails to enqueue or omits a required
                output.
        """
        with torch.no_grad(), torch.cuda.stream(stream):
            prediction = mmseg.apis.inference_model_device(self.predictor, image_bgr_nchw, stream)
            logits_tensor = prediction.seg_logits.data
            pred_labels = prediction.pred_sem_seg.data
            backbone_prediction = getattr(prediction, "backbone_feats", None)
            backbone_feats = backbone_prediction.data if backbone_prediction is not None else None
            labels_tensor, logits_tensor, uncertainty_mask = self._post_process_device_tensors(
                logits_tensor, pred_labels
            )
            if ego_mask is not None and "egocar" in self.stuff_classes:
                ego_class_id = self.stuff_classes.index("egocar")
                labels_tensor[ego_mask] = ego_class_id
                if self.estimate_logits:
                    logits_tensor[:, ego_mask] = 0
                    logits_tensor[ego_class_id, ego_mask] = 1
            return labels_tensor, logits_tensor, backbone_feats, uncertainty_mask

    def metadata(self) -> DeviceFrameProcessorMetadata:
        """Return semantic model metadata split by consumer.

        Returns:
            ``storage`` holds the ``store_semantic_meta`` attributes; ``backend``
            identifies the selected inference implementation.
        """
        _, input_dtype = self.get_device_input_spec()
        return DeviceFrameProcessorMetadata(
            storage={
                "resolution": self.resolution,
                "dataset_name": self.dataset_name,
                "method": self.method,
                "pretrained_checkpoint": self.pretrained_checkpoint,
                "stuff_classes": self.stuff_classes,
                "stuff_colors": self.stuff_colors,
            },
            backend={
                "inference_backend": "tensorrt" if self.trt_checkpoint is not None else "pytorch",
                "inference_engine": self.trt_checkpoint.name if self.trt_checkpoint is not None else None,
                "device_input_dtype": str(input_dtype),
            },
        )

    def output_specs(self, resolution: tuple[int, int]) -> tuple[DeviceFrameOutputSpec, ...]:
        """Describe Mask2Former outputs at one frame resolution.

        Args:
            resolution: Frame resolution as ``(width, height)``.

        Returns:
            ``uint8[H, W]`` semantic labels, optional
            ``float16[H, W, C]`` class logits, and optional
            ``uint8[H, W]`` uncertainty.
        """
        return _mask2former_output_specs(
            resolution,
            estimate_logits=self.estimate_logits,
            include_uncertainty=self.output_uncertainty_mask,
            num_classes=len(self.stuff_classes),
        )

    @torch.no_grad()
    def process_device(
        self,
        image_bgr_nchw: torch.Tensor,
        *,
        outputs: Mapping[str, torch.Tensor],
        processor_inputs: Mapping[str, torch.Tensor],
        stream: torch.cuda.Stream,
    ) -> None:
        """Run Mask2Former and copy results into device-frame outputs.

        Args:
            image_bgr_nchw: Contiguous raw BGR ``[1, 3, H, W]`` CUDA tensor
                with the backend-required dtype.
            outputs: Caller-owned CUDA labels ``uint8[H, W]``, optional logits
                ``float16[H, W, C]``, and optional uncertainty
                ``uint8[H, W]`` keyed by semantic name. Buffers remain valid
                for the pipeline-slot lifetime.
            processor_inputs: Optional read-only CUDA inputs, including
                ``bool[H, W]`` camera mask.
            stream: CUDA stream that owns inference, post-processing, and
                output copies. The caller owns synchronization.

        Returns:
            None. Results are enqueued into the supplied output tensors.

        Raises:
            RuntimeError: If an inference backend returns incomplete outputs
                or required logits output storage is absent.
            ValueError: If backend outputs cannot be reshaped to the
                configured resolution and class count.
        """
        labels_tensor, logits_tensor, _, uncertainty_mask = self.infer_device(
            image_bgr_nchw,
            processor_inputs.get(CAMERA_MASK_INPUT),
            stream,
        )
        # TensorRT results may alias reusable output bindings. Keep these
        # ownership copies on the inference stream so they complete before
        # the next inference on that stream overwrites the bindings.
        with torch.cuda.stream(stream):
            outputs[SEMANTIC_LABELS_OUTPUT].copy_(labels_tensor, non_blocking=True)
            if self.estimate_logits:
                semantic_seg_logits_out = outputs.get(SEMANTIC_LOGITS_OUTPUT)
                if semantic_seg_logits_out is None:
                    raise RuntimeError("Mask2Former logits output buffer was not provided")
                semantic_seg_logits_out.copy_(logits_tensor.permute(1, 2, 0), non_blocking=True)
            uncertainty_out = outputs.get(SEMANTIC_UNCERTAINTY_OUTPUT)
            if self.output_uncertainty_mask and uncertainty_out is None:
                raise RuntimeError("Mask2Former uncertainty output buffer was not provided")
            if uncertainty_out is not None:
                uncertainty_out.copy_(uncertainty_mask, non_blocking=True)

    def predict(self, image: PILImage.Image, ego_mask: Optional[np.ndarray] = None, **kwargs) -> SegmentationReturn:
        """Run Mask2Former inference and return host-resident outputs.

        Args:
            image: RGB PIL image matching the estimator resolution.
            ego_mask: Optional host boolean mask with shape ``[H, W]``.
            **kwargs: Compatibility options. ``vis_path`` requests a semantic
                overlay after device inference.

        Returns:
            Host-owned semantic labels as a paletted ``uint8[H, W]`` PIL
            image, optional ``float16[H, W, C]`` logits, optional
            ``[Hf, Wf, C]`` DINO features in their backend dtype, and
            ``None`` instance segmentation.

        Raises:
            ValueError: If image, mask, or model-output dimensions are invalid.
            RuntimeError: If the selected inference backend cannot produce the
                required outputs.
        """
        input_shape, input_dtype, input_device = mmseg.apis.get_model_device_input_spec(
            self.predictor,
            tuple(int(value) for value in self.resolution),
        )
        _, channels, height, width = input_shape
        image_rgb_hwc = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if channels != 3 or tuple(image_rgb_hwc.shape) != (height, width, channels):
            raise ValueError(
                f"Mask2Former image shape {tuple(image_rgb_hwc.shape)} does not match "
                f"device input {(height, width, channels)}"
            )

        vis_path = Path(kwargs["vis_path"]) if kwargs.get("vis_path") else None
        if not self.estimate_dinov2_feats and (vis_path is None or self.output_uncertainty_mask):
            host_outputs = process_device_frame_from_host(
                self,
                image_rgb_hwc,
                input_spec=(input_shape, input_dtype),
                device=input_device,
                processor_inputs=(
                    {CAMERA_MASK_INPUT: np.asarray(ego_mask, dtype=np.bool_)} if ego_mask is not None else None
                ),
            )
            result = SegmentationReturn(
                semantic_seg=_labels_to_paletted_image(host_outputs[SEMANTIC_LABELS_OUTPUT]),
                semantic_seg_logits=host_outputs.get(SEMANTIC_LOGITS_OUTPUT),
                semantic_dinov2_feats=None,
                instance_seg=None,
            )
            if vis_path is not None:
                uncertainty = host_outputs.get(SEMANTIC_UNCERTAINTY_OUTPUT)
                if uncertainty is None:
                    raise RuntimeError("Mask2Former host visualization requires an uncertainty output")
                _save_mask2former_visualization(
                    image_rgb_hwc,
                    host_outputs[SEMANTIC_LABELS_OUTPUT],
                    np.asarray(self.stuff_colors, dtype=np.uint8),
                    vis_path,
                    uncertainty,
                )
            return result

        image_rgb_tensor = torch.from_numpy(np.array(image_rgb_hwc, copy=True))
        image_bgr_nchw = (
            image_rgb_tensor.to(device=input_device, dtype=input_dtype)
            .permute(2, 0, 1)
            .flip(0)
            .unsqueeze(0)
            .contiguous()
        )
        ego_mask_device = (
            torch.as_tensor(ego_mask, dtype=torch.bool, device=input_device) if ego_mask is not None else None
        )
        stream = torch.cuda.current_stream(device=input_device)
        labels_device, logits_device, backbone_feats_device, uncertainty_device = self.infer_device(
            image_bgr_nchw,
            ego_mask_device,
            stream,
        )
        stream.synchronize()

        semantic_seg = _labels_to_paletted_image(labels_device.cpu().numpy())
        semantic_seg_logits = (
            logits_device.permute(1, 2, 0).to(dtype=torch.float16).cpu().numpy() if self.estimate_logits else None
        )
        if self.estimate_dinov2_feats and backbone_feats_device is None:
            raise RuntimeError("Mask2Former backend did not return requested DINO backbone features")
        semantic_dinov2_feats = (
            backbone_feats_device.permute(1, 2, 0).cpu().numpy()
            if self.estimate_dinov2_feats and backbone_feats_device is not None
            else None
        )
        result = SegmentationReturn(
            semantic_seg=semantic_seg,
            semantic_seg_logits=semantic_seg_logits,
            semantic_dinov2_feats=semantic_dinov2_feats,
            instance_seg=None,
        )
        if vis_path is not None:
            if result.semantic_seg is None:
                raise RuntimeError("Mask2Former device inference returned no semantic labels")
            _save_mask2former_visualization(
                np.asarray(image.convert("RGB"), dtype=np.uint8),
                np.asarray(result.semantic_seg, dtype=np.uint8),
                np.asarray(self.stuff_colors, dtype=np.uint8),
                vis_path,
                uncertainty_device.cpu().numpy(),
            )
        return result

    def set_resolution(self, resolution: list) -> None:
        self.resolution = resolution


class DINOv2Estimator:
    """
    DINOv2 feature estimator for feature extraction.
    Images are padded to the nearest patch size (one could also test direct rescaling outside).
    """

    PATCH_SIZE = (14, 14)

    def __init__(self, resolution: list, backend: str, facet: Literal["token", "key"] = "token") -> None:
        self.backend = backend
        self.facet = facet

        if backend.startswith("dinov2"):
            self.model = torch.hub.load("facebookresearch/dinov2", backend)
            self.model.eval().cuda()

        elif backend == "nv_dinov2":
            # TODO [JH]: Figure out a more elegant way to load this model.
            #   Currently we re-use the mask2former backbone since it is frozen during training.
            segm_model = Mask2FormerSegmentationEstimator(resolution, enable_trt=False)
            assert segm_model.predictor.backbone_dino is not None
            self.model = segm_model.predictor.backbone_dino.timm_model  # type: ignore

        else:
            raise ValueError(f"Backend {backend} not supported")

        self.last_extracted_facet = None
        if self.facet == "token":
            self.model.blocks[-1].register_forward_hook(self.extract_token_hook)

        elif self.facet == "key":
            self.model.blocks[-1].attn.register_forward_hook(self.extract_key_hook)

        # Patch size sanity check
        self.patch_size = self.model.patch_embed.patch_size[::-1]
        assert self.patch_size == self.PATCH_SIZE, f"Patch size mismatch: {self.patch_size} vs {self.PATCH_SIZE}"

        self.set_resolution(resolution)

    @classmethod
    def resolution_from_dino_width(cls, original_resolution: list, dino_width: int) -> list:
        dino_height = math.ceil(dino_width * original_resolution[1] / original_resolution[0])
        return [dino_width * cls.PATCH_SIZE[0], dino_height * cls.PATCH_SIZE[1]]

    def set_resolution(self, resolution: list) -> None:
        self.resolution = resolution
        pad_w = self.patch_size[0] - self.resolution[0] % self.patch_size[0]
        pad_h = self.patch_size[1] - self.resolution[1] % self.patch_size[1]
        if pad_w == self.patch_size[0]:
            pad_w = 0
        if pad_h == self.patch_size[1]:
            pad_h = 0

        self.dino_resolution = (
            (self.resolution[0] + pad_w) // self.patch_size[0],
            (self.resolution[1] + pad_h) // self.patch_size[1],
        )
        self.padding = (
            pad_w // 2,
            pad_h // 2,
            pad_w - pad_w // 2,
            pad_h - pad_h // 2,
        )

        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.resolution[::-1]),  # W, H -> H, W
                transforms.Pad(self.padding, padding_mode="reflect"),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.mask_preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.resolution[::-1]),
                transforms.Pad(self.padding, padding_mode="constant", fill=0),
            ]
        )

    def extract_key_hook(self, module, input, output):
        input = input[0]
        B, N, C = input.shape
        qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        # 0 is query, 1 is key, 2 is value
        feature = qkv[1]  # B, H, T, C
        # remove cls token, and permute to B, HC, T
        feature = feature[:, :, 1:].moveaxis(2, -1).flatten(1, 2)
        feature = feature.reshape(B, -1, *self.dino_resolution[::-1])
        self.last_extracted_facet = feature

    def extract_token_hook(self, module, input, output):
        B, _, C = output.shape
        feature = output[:, 1:].reshape(B, *self.dino_resolution[::-1], C)
        self.last_extracted_facet = feature.moveaxis(3, 1)

    @torch.no_grad()
    def predict(
        self, image: PILImage.Image, ego_mask: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        image = self.preprocess(image).unsqueeze(0).cuda()

        self.model(image)
        assert self.last_extracted_facet is not None
        facet = self.last_extracted_facet[0].permute(1, 2, 0).cpu().numpy()

        # Ego mask is used to determine which tokens are usable.
        if ego_mask is not None:
            ego_mask = self.mask_preprocess(ego_mask.astype(np.float32))[0]
            assert ego_mask is not None

            # Patchify and compute mean mask, > 1 pixels to be considered containing ego information.
            ego_mask = (
                ego_mask.reshape(
                    self.dino_resolution[1], self.patch_size[1], self.dino_resolution[0], self.patch_size[0]
                ).sum([1, 3])
                > 1.0
            )

            ego_mask = ego_mask.numpy()

        return facet, ego_mask

    def get_metadata(self) -> dict:
        return {
            "resolution": self.resolution,
            "facet": self.facet,
            "backend": self.backend,
            "window_left": self.padding[0] / self.patch_size[0],
            "window_top": self.padding[1] / self.patch_size[1],
            "window_width": self.dino_resolution[0] - (self.padding[0] + self.padding[2]) / self.patch_size[0],
            "window_height": self.dino_resolution[1] - (self.padding[1] + self.padding[3]) / self.patch_size[1],
        }


class LidarSegmentationByProjectEstimator:
    """Lidar semantic segmentation by projecting image segmentation results."""

    def __init__(
        self,
        loader: ncore.data.SequenceLoaderProtocol,
        aux_camera_semseg_provider: AuxDataCameraSemSegProvider,
        camera_ids: list[str],
        ensemble_cuda: bool = True,
        method: str = "camera-sseg-projection-ensemble",
        ignore_label: int = 255,  # -1 for uint8
        min_ray_length: float = 0.1,
    ) -> None:
        self.loader = loader
        self.aux_camera_semseg_provider = aux_camera_semseg_provider
        self.ignore_label = ignore_label
        self.ensemble_cuda = ensemble_cuda
        self.method = method
        assert len(camera_ids), "Require at least a single active camera"
        self.camera_ids = camera_ids
        self.min_ray_length = min_ray_length

    def set_meta(self) -> None:
        self.stuff_colors = self.aux_camera_semseg_provider.get_semantic_segmentation_meta(self.camera_ids[0])[
            "stuff_colors"
        ]
        for camera_id in self.camera_ids[1:]:
            stuff_color = self.aux_camera_semseg_provider.get_semantic_segmentation_meta(camera_id)["stuff_colors"]
            assert np.array(stuff_color).shape == np.array(self.stuff_colors).shape
            assert np.abs(np.array(stuff_color) - np.array(self.stuff_colors)).sum() == 0, (
                "Stuff colors of all cameras should be consistent."
            )
        self.stuff_colors.append([128, 128, 128])  # for ignore label
        self.stuff_classes = self.aux_camera_semseg_provider.get_semantic_segmentation_meta(self.camera_ids[0])[
            "stuff_classes"
        ]
        for camera_id in self.camera_ids[1:]:
            stuff_class = self.aux_camera_semseg_provider.get_semantic_segmentation_meta(camera_id)["stuff_classes"]
            assert stuff_class == self.stuff_classes, "Stuff calsses of all cameras should be consistent."
        self.stuff_classes.append("ignore")

        # check resolution
        for camera_id in self.camera_ids:
            resolution_semseg_provider = self.aux_camera_semseg_provider.get_semantic_segmentation_meta(camera_id)[
                "resolution"
            ]
            camera_sensor = self.loader.get_camera_sensor(camera_id)
            camera_model = camera_model_from_parameters(
                camera_sensor.model_parameters, device="cpu", dtype=torch.float32
            )
            assert torch.all(camera_model.resolution == torch.tensor(resolution_semseg_provider)), (
                "The resolution from CameraModel and AuxDataCameraSemSegProvider do not match."
            )

        # lidar-camera-visibility order
        self.lidar_camera_visibility_order = {camera_id: idx for idx, camera_id in enumerate(self.camera_ids)}

    def predict(
        self,
        lidar_sensor: ncore.data.LidarSensorProtocol,
        frame_timestamp_us: int,
        vis_path: Path | None = None,
    ) -> tuple[PILImage.Image, np.ndarray]:
        lidar_frame_idx = lidar_sensor.get_closest_frame_index(frame_timestamp_us)

        # load motion-compensated point clouds (represented in end-of-frame sensor-frame)
        pc = lidar_sensor.get_frame_point_cloud(lidar_frame_idx, motion_compensation=True, with_start_points=True)
        xyz_s = pc.xyz_m_start  # start
        assert isinstance(xyz_s, np.ndarray)
        xyz_e = pc.xyz_m_end  # end

        ray_length = np.linalg.norm(xyz_e - xyz_s, axis=-1)
        valid_mask = ray_length >= self.min_ray_length
        xyz_s_valid = xyz_s[valid_mask]
        xyz_e_valid = xyz_e[valid_mask]

        # lidar to world
        T_lidar_to_world = lidar_sensor.get_frames_T_sensor_target("world", lidar_frame_idx)
        xyz_e_valid_world_np = transform_point_cloud(xyz_e_valid, T_lidar_to_world)
        xyz_e_valid_world = torch.from_numpy(xyz_e_valid_world_np).to("cuda", dtype=torch.float32)
        num_points = len(xyz_e_valid)
        lidar_seg_ensemble = np.full((num_points, len(self.camera_ids)), self.ignore_label)
        visibility_mask = np.full((num_points, len(self.camera_ids)), False)

        # create camera mesh from lidar spin to calculate occluded points
        spin_mesh = LidarSpinMesh(xyz_s_valid, xyz_e_valid)
        spin_mesh_vertices = np.ascontiguousarray(spin_mesh.vertices)
        intersector = pcu.RayMeshIntersector(spin_mesh_vertices, spin_mesh.faces)

        visual_dict = {}
        colormap = np.array(self.stuff_colors)

        ego_class_id = self.stuff_classes.index("egocar") if "egocar" in self.stuff_classes else self.ignore_label
        for camera_id, ci in self.lidar_camera_visibility_order.items():
            camera_sensor = self.loader.get_camera_sensor(camera_id)
            camera_frame_idx = camera_sensor.get_closest_frame_index(frame_timestamp_us)
            frame_timestamp_us_camera = camera_sensor.get_frame_timestamp_us(camera_frame_idx)
            image_seg = np.array(
                self.aux_camera_semseg_provider.get_semantic_segmentation(camera_id, frame_timestamp_us_camera)
            )
            image_seg[image_seg == ego_class_id] = self.ignore_label

            camera_model = camera_model_from_parameters(
                camera_sensor.model_parameters, device="cuda", dtype=torch.float32
            )
            lidar_seg = np.full(num_points, self.ignore_label)

            T_world_sensor_startend = torch.from_numpy(
                camera_sensor.get_frames_T_source_sensor(
                    source_node="world",
                    frame_indices=camera_frame_idx,
                    frame_timepoint=None,  # both start and end
                )  # 2x4x4
            ).to("cuda", dtype=torch.float32)

            projection = camera_model.world_points_to_image_points_shutter_pose(
                xyz_e_valid_world,
                T_world_sensor_startend[0],
                T_world_sensor_startend[1],
                return_valid_indices=True,
                return_all_projections=False,
                return_T_world_sensors=True,
            )
            assert projection.valid_indices is not None
            assert (
                projection.T_world_sensors is not None
            )  # world-coordinates of the camera-frame at the time of projection

            # find non occluded points
            # cast rays from camera to lidar points in camera frustrum
            T_world_to_cameras = projection.T_world_sensors.cpu().numpy().astype(np.float32)
            T_lidar_to_cameras = T_world_to_cameras @ T_lidar_to_world
            # camera positions in lidar frame
            camera_positions_lidar = np.ascontiguousarray(
                se3_matrix_inverse(T_lidar_to_cameras, unbatch=False).numpy()[:, :3, -1]
            )  # interpolated positions of the camera at the instant when the 2D projection of each Lidar point is recorded by the rolling shutter.
            camera_pc_rays = xyz_e_valid[projection.valid_indices.cpu()] - camera_positions_lidar
            face_indices, bary_coords, ray_intersection_distances = intersector.intersect_rays(
                camera_positions_lidar, camera_pc_rays
            )
            if len(face_indices.shape) == 0:
                face_indices = np.expand_dims(face_indices, 0)
                bary_coords = np.expand_dims(bary_coords, 0)
                ray_intersection_distances = np.expand_dims(ray_intersection_distances, 0)

            # initialize non-occlusion mask
            non_occluded_points: np.ndarray = face_indices < 0
            # depth check
            valid_rays = face_indices >= 0  # True only at the rays which intersected the shape
            ray_mesh_intersections = pcu.interpolate_barycentric_coords(
                spin_mesh.faces, face_indices[valid_rays], bary_coords[valid_rays], spin_mesh_vertices
            )
            intersection_tol = 0.001
            pc_camera_dist = np.linalg.norm(
                camera_pc_rays[valid_rays],
                axis=-1,
            )
            intersection_camera_dist = np.linalg.norm(
                camera_positions_lidar[valid_rays] - ray_mesh_intersections, axis=-1
            )
            non_occluded_points[valid_rays] = pc_camera_dist <= intersection_camera_dist + intersection_tol

            ij = np.floor(projection.image_points.cpu().numpy()).astype(np.int32)

            assert projection.valid_indices is not None
            valid_indices = projection.valid_indices.cpu()
            lidar_seg[valid_indices[non_occluded_points]] = image_seg[
                ij[non_occluded_points, 1], ij[non_occluded_points, 0]
            ]
            lidar_seg_ensemble[:, ci] = lidar_seg
            visibility_mask[:, ci][valid_indices] = True
            if vis_path:
                visual_dict[camera_id] = [ij, valid_indices]

        # ensamble labels from all cameras
        if self.ensemble_cuda:
            lidar_seg_ensemble = (
                ensemble_cuda(lidar_seg_ensemble, device=torch.device("cuda"), ignore_label=self.ignore_label)
                .cpu()
                .numpy()
            )
        else:
            lidar_seg_ensemble = ensemble_numba(lidar_seg_ensemble.astype(np.uint8), self.ignore_label)

        lidar_seg_result = np.full((len(xyz_e),), self.ignore_label, dtype=np.uint8)
        lidar_seg_result[valid_mask] = lidar_seg_ensemble

        if vis_path:
            for camera_id in self.camera_ids:
                camera_sensor = self.loader.get_camera_sensor(camera_id)
                camera_frame_idx = camera_sensor.get_closest_frame_index(frame_timestamp_us)
                img = camera_sensor.get_frame_image_array(camera_frame_idx)
                ij, indices = visual_dict[camera_id]
                seg = lidar_seg_result[valid_mask][indices]
                seg[seg == self.ignore_label] = self.stuff_classes.index("ignore")
                color = colormap[seg]
                img_vis = plot_points_on_image_with_color(img.copy(), ij, color)
                save_path = vis_path / camera_id / f"{camera_frame_idx}.png"
                PILImage.fromarray(img_vis).save(save_path)

        # Decode semantic segmentation as Image
        decode_seg = PILImage.fromarray(lidar_seg_result.reshape(-1, 1), mode="P")

        # loading a P image removes the color palette, so we need to put it back to prevent losing data on save
        palettedata = np.linspace(0, 255, 256, dtype=np.uint8).tolist()
        decode_seg.putpalette(palettedata)

        return decode_seg, visibility_mask

    def get_semantic_metadata(self) -> dict:
        return {
            "ignore_label": self.ignore_label,
            "method": self.method,
            "stuff_classes": self.stuff_classes,
            "stuff_colors": self.stuff_colors,
        }

    def get_visibility_metadata(self) -> dict:
        return {"camera_visibility_order": self.lidar_camera_visibility_order}


class PromptlessSAM2EgoMaskEstimator(DeviceFrameProcessor):
    """
    Promptless SAM2-based ego mask estimator for auxiliary data generation.
    """

    name = SAM2_PROCESSOR

    def __init__(
        self,
        model_cfg: str = "sam2_hiera_l",
        pretrained_checkpoint: str = "sam2_hiera_l_avfinetuned_v1.pt",
        device: str | torch.device | None = "cuda",
    ) -> None:
        """Initialize a CUDA-only promptless SAM2 estimator.

        Args:
            model_cfg: SAM2 model configuration name.
            pretrained_checkpoint: Checkpoint filename resolved through the
                runfiles search path.
            device: CUDA device used for model state and inference. ``None``
                selects CUDA when available.

        Raises:
            ValueError: If ``device`` does not identify a CUDA device.
            FileNotFoundError: If the checkpoint cannot be resolved.
        """
        resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if resolved_device.type != "cuda":
            raise ValueError(f"PromptlessSAM2EgoMaskEstimator requires CUDA, got {resolved_device}")

        self.dataset_name = model_cfg
        self.method = "promptless_avsam2"
        self.pretrained_checkpoint = str(find_network_file_path(pretrained_checkpoint))

        # Image predictor
        self.sam2_model = build_sam2(model_cfg, None, device=resolved_device)
        self.sam2_model.load_state_dict(torch.load(self.pretrained_checkpoint, weights_only=True))
        self.predictor = SAM2ImagePredictor(self.sam2_model)

    def metadata(self) -> DeviceFrameProcessorMetadata:
        """Return serializable model metadata for auxiliary-data writers.

        Returns:
            ``storage`` holds dataset, method, and checkpoint identifiers.
            SAM2 has no backend fields.
        """
        return DeviceFrameProcessorMetadata(
            storage={
                "dataset_name": self.dataset_name,
                "method": self.method,
                "pretrained_checkpoint": self.pretrained_checkpoint,
            },
            backend={},
        )

    def output_specs(self, resolution: tuple[int, int]) -> tuple[DeviceFrameOutputSpec, ...]:
        """Describe the SAM2 mask at one frame resolution.

        Args:
            resolution: Frame resolution as ``(width, height)``.

        Returns:
            One ``uint8[H, W]`` promptless SAM2 binary-mask output.
        """
        return _sam2_output_specs(resolution)

    @torch.no_grad()
    def process_device(
        self,
        image_bgr_nchw: torch.Tensor,
        *,
        outputs: Mapping[str, torch.Tensor],
        processor_inputs: Mapping[str, torch.Tensor],
        stream: torch.cuda.Stream,
    ) -> None:
        """Run SAM2 and copy its binary mask into device-frame output.

        Args:
            image_bgr_nchw: Contiguous BGR ``[1, 3, H, W]`` CUDA tensor with
                values in ``[0, 255]`` and dtype ``uint8``, ``float16``, or
                ``float32``.
            outputs: Caller-owned CUDA outputs containing a
                ``uint8[H, W]`` SAM2 mask buffer.
            processor_inputs: Unused read-only processor inputs.
            stream: CUDA stream owning inference and the output copy.

        Returns:
            None. The binary mask write is enqueued on ``stream``.

        Raises:
            ValueError: If image, stream, or output contracts are invalid.
            RuntimeError: If SAM2 inference fails.
        """
        del processor_inputs
        ego_mask_out = outputs[SAM2_MASK_OUTPUT]
        if image_bgr_nchw.ndim != 4 or image_bgr_nchw.shape[0] != 1 or image_bgr_nchw.shape[1] != 3:
            raise ValueError(f"SAM2 device input must have shape 1x3xHxW, got {tuple(image_bgr_nchw.shape)}")
        if image_bgr_nchw.device.type != "cuda":
            raise ValueError("SAM2 device input must be a CUDA tensor")
        if image_bgr_nchw.dtype not in (torch.uint8, torch.float16, torch.float32):
            raise ValueError(f"SAM2 device input has unsupported dtype {image_bgr_nchw.dtype}")

        height, width = (int(value) for value in image_bgr_nchw.shape[-2:])
        if ego_mask_out.shape != (height, width) or ego_mask_out.dtype != torch.uint8:
            raise ValueError(
                f"SAM2 output must be uint8 with shape {(height, width)}, got {ego_mask_out.dtype} "
                f"{tuple(ego_mask_out.shape)}"
            )
        if ego_mask_out.device != image_bgr_nchw.device:
            raise ValueError("SAM2 input and output must be on the same CUDA device")
        if stream.device != image_bgr_nchw.device:
            raise ValueError(f"SAM2 stream is on {stream.device}, input is on {image_bgr_nchw.device}")

        with torch.cuda.stream(stream):
            self.predictor.reset_predictor()
            self.predictor._orig_hw = [(height, width)]

            # The Mask2Former staging tensor is BGR uint8 NCHW. Convert it to
            # RGB float in [0, 1], then reuse SAM2's scripted resize and
            # normalization transforms directly on CUDA.
            image_rgb = image_bgr_nchw.flip(1).to(dtype=torch.float32).div_(255.0)
            input_image = self.predictor._transforms.transforms(image_rgb[0]).unsqueeze(0)
            backbone_out = self.sam2_model.forward_image(input_image)
            _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
            if self.sam2_model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed

            features = [
                feat.permute(1, 2, 0).view(1, -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], self.predictor._bb_feat_sizes[::-1])
            ][::-1]
            self.predictor._features = {"image_embed": features[-1], "high_res_feats": features[:-1]}
            self.predictor._is_image_set = True
            self.predictor._is_batch = False

            prompt_coords = torch.zeros((1, 1, 2), dtype=torch.float32, device=image_bgr_nchw.device)
            prompt_labels = -torch.ones((1, 1), dtype=torch.int32, device=image_bgr_nchw.device)
            # SAM2ImagePredictor._predict is its PyTorch tensor interface. The
            # public predict() wrapper converts inputs and outputs through NumPy.
            masks, scores, _ = self.predictor._predict(
                prompt_coords,
                prompt_labels,
                multimask_output=True,
                return_logits=False,
            )
            if masks.numel() == 0:
                raise ValueError("No masks predicted by Estimator")
            best_mask_index = scores.argmax(dim=1)
            best_mask = masks[0].index_select(0, best_mask_index)[0]
            ego_mask_out.copy_(self._post_process_device_mask(best_mask), non_blocking=True)

    @staticmethod
    def _post_process_device_mask(mask: torch.Tensor) -> torch.Tensor:
        """Match the host-side SciPy median filter and dilation on CUDA.

        Args:
            mask: Two-dimensional CUDA-compatible binary mask.

        Returns:
            A ``uint8`` mask with the same shape and device as ``mask``.

        Raises:
            ValueError: If ``mask`` is not two-dimensional or is too small for
                the 15-by-15 median window.
        """
        if mask.ndim != 2:
            raise ValueError(f"SAM2 mask must be two-dimensional, got {tuple(mask.shape)}")

        mask_float = mask.to(dtype=torch.float32)[None, None]
        median_radius = 7
        if mask.shape[0] <= median_radius or mask.shape[1] <= median_radius:
            raise ValueError(f"SAM2 mask {tuple(mask.shape)} is too small for the 15x15 median filter")

        # torchvision's symmetric mode repeats edge pixels, matching
        # scipy.ndimage's half-sample-symmetric ``reflect`` mode.
        mask_float = transforms.functional.pad(
            mask_float,
            [median_radius] * 4,
            padding_mode="symmetric",
        )
        median_mask = F.avg_pool2d(mask_float, kernel_size=15, stride=1) >= 0.5
        dilated_mask = F.max_pool2d(median_mask.to(dtype=torch.float32), kernel_size=11, stride=1, padding=5)
        return dilated_mask[0, 0].to(dtype=torch.uint8)

    def predict(
        self,
        image: PILImage.Image,
        **kwargs,
    ) -> np.ndarray:
        """Run the CUDA tensor path and return its compact host mask.

        Args:
            image: RGB PIL image with values in ``[0, 255]``.
            **kwargs: Ignored compatibility arguments.

        Returns:
            A ``uint8[H, W]`` binary ego mask in host memory.

        Raises:
            ValueError: If the image or model output violates the device-path
                contract.
        """
        image_rgb_hwc = np.asarray(image.convert("RGB"), dtype=np.uint8)
        height, width = image_rgb_hwc.shape[:2]
        if is_dark_image(image_rgb_hwc):
            return np.zeros((height, width), dtype=np.uint8)
        outputs = process_device_frame_from_host(
            self,
            image_rgb_hwc,
            input_spec=((1, 3, height, width), torch.uint8),
            device=torch.device(self.predictor.device),
        )
        return outputs[SAM2_MASK_OUTPUT]
