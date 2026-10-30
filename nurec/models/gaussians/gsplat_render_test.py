# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test suite for GSplatRenderer using msgpack-based test cases.

Test flow:
1. Load test inputs from compressed msgpack files (legacy msgpack format)
2. Extract Gaussian parameters from model dict
3. Run rendering with GSplatRenderer
4. Validate output properties (shape, dtype, finite values, value ranges)

NOTE: This test previously used a live rendering oracle for golden-value comparison.
The oracle was removed as part of the renderer migration. Future work:
replace with baked golden tensors in @gsplat_render_test_assets.

Debug Visualization:
Set GSPLAT_TEST_DEBUG=1 to save rendered outputs as PNG images for visual inspection.
Images are saved to gsplat_test_output/{self.test_name}/

Ex:
  bazel test --test_env GSPLAT_TEST_DEBUG=1 --test_arg=-s --test_output=streamed --test_env GSPLAT_TEST_OUTPUT_DIR=$PWD/out //nurec/models/gaussians:gsplat_render_test
"""

import gzip
import os
import pickle

from typing import Any, Callable, Optional, Union

import msgpack
import numpy as np
import pytest
import torch

from omegaconf import OmegaConf
from PIL import Image

from ncore.data import ConcreteCameraModelParametersUnion
from nurec.config.model import GSplatRendererConfig
from nurec.models.base import BaseModel
from nurec.models.gaussians.renderers import GSplatRenderer
from nurec.utils.batch import FrameMeta, RenderingData
from nurec.utils.renamed_keys import adopt_renamed_key


# Debug flag - set GSPLAT_TEST_DEBUG=1 to enable visual output
DEBUG_OUTPUT_ENABLED = os.environ.get("GSPLAT_TEST_DEBUG", "0") == "1"
# Use Bazel's test outputs directory if available, otherwise fall back to current directory
DEBUG_OUTPUT_DIR = os.environ.get(
    "GSPLAT_TEST_OUTPUT_DIR", os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "gsplat_test_output")
)


def _apply_turbo_colormap(img_normalized, *, invert=False):
    """Apply Google's Turbo colormap to normalized image data."""
    if invert:
        img_normalized = 1.0 - img_normalized

    turbo_colormap_data = np.array(
        [
            [0.000, 0.00000, 0.00000, 0.00000],  # Black
            [0.111, 0.18995, 0.07176, 0.23217],  # Dark blue
            [0.222, 0.25107, 0.38574, 0.63426],  # Blue
            [0.333, 0.19874, 0.59516, 0.74892],  # Cyan
            [0.444, 0.11253, 0.73386, 0.61556],  # Teal
            [0.556, 0.25748, 0.83105, 0.42809],  # Green
            [0.667, 0.57797, 0.87937, 0.24123],  # Yellow-green
            [0.778, 0.87635, 0.82538, 0.17747],  # Yellow
            [0.889, 0.98323, 0.65569, 0.13103],  # Orange
            [1.000, 0.93752, 0.25005, 0.08334],  # Red
        ]
    )

    img_shape = img_normalized.shape
    t = np.clip(img_normalized.flatten(), 0, 1)
    r = np.interp(t, turbo_colormap_data[:, 0], turbo_colormap_data[:, 1]).reshape(img_shape)
    g = np.interp(t, turbo_colormap_data[:, 0], turbo_colormap_data[:, 2]).reshape(img_shape)
    b = np.interp(t, turbo_colormap_data[:, 0], turbo_colormap_data[:, 3]).reshape(img_shape)

    colored = np.zeros((*img_shape, 3), dtype=np.uint8)
    colored[:, :, 0] = (r * 255).astype(np.uint8)
    colored[:, :, 1] = (g * 255).astype(np.uint8)
    colored[:, :, 2] = (b * 255).astype(np.uint8)
    return colored


def _save_rgb(img, path):
    """Save RGB image (H, W, 3) with values in [0, 1]."""
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_uint8, mode="RGB").save(path)


def _save_grayscale(img, path):
    """Save grayscale image (H, W) with values in [0, 1]."""
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_uint8, mode="L").save(path)


def _save_colormap(img, path, *, invert=False, vmin=None, vmax=None):
    """Save single-channel image with Turbo colormap."""
    if vmin is None:
        vmin = img.min()
    if vmax is None:
        vmax = img.max()

    if vmax > vmin:
        img_normalized = (img - vmin) / (vmax - vmin)
    else:
        img_normalized = np.zeros_like(img)

    colored = _apply_turbo_colormap(img_normalized, invert=invert)
    Image.fromarray(colored, mode="RGB").save(path)


class GSplatRenderTestCase:
    """Test case for GSplatRenderer using legacy msgpack test assets.

    Loads test data from msgpack files and validates GSplatRenderer:
    1. Extracts Gaussian parameters from model dict (legacy or modern format)
    2. Maps renderer settings to GSplat config
    3. Runs GSplatRenderer and validates output properties (shape, dtype, finite,
       value ranges)

    Attributes:
        model: dict containing serialized model data
        renderer: dict containing renderer settings
        frame_id: unique frame identifier
        frame_width, frame_height: frame dimensions
        frame_start_timestamp, frame_end_timestamp: frame capture timestamps
        frames_sensor_model: camera model (Pinhole, Fisheye, F-Theta)
        frames_sensor_ids: sensor IDs
        frames_sensor_start_pose, frames_sensor_end_pose: sensor poses [7] (xyz + quat)
        rays: combined ray data [1, H, W, 6] from rays_origin + rays_direction
        rays_radiance_density: reference RGB+density output [HxWx4]
        rays_hit_distance: reference ray distance output [HxW]
        rays_hit_normal: reference ray normal output [HxWx3] (optional)
        gaussian_parameters: dict with positions, rotations, scales, densities, features
        renderer_config: DictConfig for GSplatRenderer settings
        n_active_features: number of active SH features
        device: torch device for tensors
    """

    def __init__(self, test_case_dict: dict, device: torch.device):
        """Initialize from msgpack test case dictionary.

        Args:
            test_case_dict: unpacked msgpack test case
            device: torch device for tensors
        """
        # Declare optional attributes for type checking
        self.test_name: str = ""
        self.frames_sensor_model: Optional[ConcreteCameraModelParametersUnion] = None
        self.frames_sensor_ids: Optional[torch.Tensor] = None
        self.frames_sensor_start_pose: Optional[torch.Tensor] = None
        self.frames_sensor_end_pose: Optional[torch.Tensor] = None
        self.rays_timestamps_us: Optional[torch.Tensor] = None

        def load_tensor(bytes_data, dtype=np.float32) -> torch.Tensor:
            return torch.from_numpy(np.frombuffer(bytes_data, dtype).copy()).to(device=device)

        # Extract model and check if it's compatible.
        # Checked-in msgpack fixtures still use the pre-rename nre_data envelope.
        model_dict = test_case_dict["model"]
        adopt_renamed_key(model_dict, "nurec_data", "nre_data")
        model_config = model_dict["nurec_data"]["config"]

        # Force model name to sh-gaussians
        model_config["name"] = "sh-gaussians"

        # Disable background and PPISP so they aren't applied during the test
        # Background: Set to "skip-background" to disable background rendering
        if "background" in model_config:
            model_config["background"]["name"] = "skip-background"
        # PPISP: Clear post_processings list to disable post-processing
        if "post_processing" in model_config:
            model_config["post_processing"] = []

        gaussian_params = GSplatRenderTestCase._extract_gaussians_from_msgpack_fixture(model_dict, device)

        # Get renderer settings from test asset config
        renderer_config = test_case_dict["renderer"]
        renderer_config["prepare_before_render"] = False
        renderer_config["rasterize_mode"] = "classic"

        # Extract frame metadata
        self.frame_id = test_case_dict["frame_id"]
        self.frame_width = test_case_dict["frame_width"]
        self.frame_height = test_case_dict["frame_height"]
        self.frame_start_timestamp = test_case_dict["frame_start_timestamp"]
        self.frame_end_timestamp = test_case_dict["frame_end_timestamp"]

        # Get n_active_features (sh_degree) from config
        # (sh_degree+1)² = number of SH bands; e.g., sh_degree=0 -> 1 band (DC only), sh_degree=3 -> 16 bands
        config_sh_degree = model_config.get("particle", {}).get("radiance_sph_degree", 3)
        self.n_active_features = config_sh_degree

        # Extract extra_ray_signal_infos from model config (for camera rendering)
        # Structure: tuple[list[str], list[int], list[Callable]]
        # - List 0: signal names
        # - List 1: signal dimensions
        # - List 2: activation functions
        self.camera_extra_ray_signal_infos = GSplatRenderTestCase._extract_extra_ray_signal_infos(
            model_config, sensor_type="camera"
        )
        print(f"Extra ray signal infos: {len(self.camera_extra_ray_signal_infos[0])} signals")
        if len(self.camera_extra_ray_signal_infos[0]) > 0:
            print(f"  Signal names: {self.camera_extra_ray_signal_infos[0]}")
            print(f"  Signal dims: {self.camera_extra_ray_signal_infos[1]}")

        # Load sensor model and poses
        if "frames_sensor_model" in test_case_dict:
            self.frames_sensor_model = pickle.loads(test_case_dict["frames_sensor_model"])
            self.frames_sensor_ids = load_tensor(test_case_dict["frames_sensor_ids"], dtype=np.int32)
            self.frames_sensor_start_pose = load_tensor(test_case_dict["frames_sensor_start_pose"])
            self.frames_sensor_end_pose = load_tensor(test_case_dict["frames_sensor_end_pose"])
            if self.frames_sensor_end_pose is None:
                self.frames_sensor_end_pose = self.frames_sensor_start_pose
        else:
            self.frames_sensor_model = None
            self.frames_sensor_ids = None
            self.frames_sensor_start_pose = None
            self.frames_sensor_end_pose = None

        # Load rays from msgpack (separate origin and direction arrays)
        h, w = self.frame_height, self.frame_width
        rays_origin = load_tensor(test_case_dict["rays_origin"]).reshape(h, w, 3)
        rays_direction = load_tensor(test_case_dict["rays_direction"]).reshape(h, w, 3)
        # Combine into RenderingData format: [1, H, W, 6] (origin + direction)
        self.rays = torch.cat([rays_origin, rays_direction], dim=-1).unsqueeze(0)

        # Load per-ray timestamps for rolling shutter support
        # Note: rays_timestamp is stored as int64, not float32
        if "rays_timestamp" in test_case_dict:
            rays_timestamp = load_tensor(test_case_dict["rays_timestamp"], dtype=np.int64).reshape(h, w, 1)
            self.rays_timestamps_us = rays_timestamp.unsqueeze(0)  # (1, H, W, 1)
        else:
            self.rays_timestamps_us = None

        # Load reference outputs (convert to numpy for comparison)
        # These need to be [H*W, C] format for comparison
        self.rays_radiance_density = load_tensor(test_case_dict["rays_radiance_density"]).cpu().numpy().reshape(-1, 4)
        self.rays_hit_distance = load_tensor(test_case_dict["rays_hit_distance"]).cpu().numpy().reshape(-1)
        self.rays_hit_normal = (
            load_tensor(test_case_dict["rays_hit_normal"]).cpu().numpy().reshape(-1, 3)
            if "rays_hit_normal" in test_case_dict
            else None
        )

        # Store attributes
        self.model = model_dict
        self.gaussian_parameters = gaussian_params
        self.renderer_config = renderer_config
        self.device = device

    @classmethod
    def from_file(cls, file_path: str, device: torch.device):
        """Load test case from a msgpack test asset file.

        Args:
            file_path: path to .msgpack.gz test file
            device: torch device for tensors

        Returns:
            GSplatRenderTestCase instance
        """
        print(f"GSplatRenderer::TestCase loading test case {file_path}")

        with gzip.open(file_path, "rb") as f:
            test_case_dict = msgpack.unpackb(f.read())

        instance = cls(test_case_dict, device)
        instance.test_name = file_path.split("/")[-1].split(".")[0]
        return instance

    @staticmethod
    def _extract_gaussians_from_msgpack_fixture(model_dict: dict, device: torch.device) -> dict[str, torch.Tensor]:
        """Extract Gaussian parameters from a msgpack model dictionary.

        Pure numpy/dict parsing. Expects modern format:
        .gaussians_nodes.{node_name}.* (e.g., gaussians, background).
        Concatenates all gaussian nodes together, matching collect_gaussian_parameters()
        behavior.

        Args:
            model_dict: model dictionary with particle data
            device: torch device

        Returns:
            dict with concatenated positions, rotations, scales, densities, features, extra_signals

        Raises:
            ValueError: if model is not a Gaussian splatting model or keys not found
        """
        adopt_renamed_key(model_dict, "nurec_data", "nre_data")
        state_dict = model_dict["nurec_data"]["state_dict"]

        # Find all gaussian node prefixes (e.g., .gaussians_nodes.background, .gaussians_nodes.gaussians)
        # Pattern: extract everything matching ".gaussians_nodes.<node_name>"
        node_prefixes = []
        for key in state_dict.keys():
            if ".gaussians_nodes." in key:
                # Extract the prefix up to and including the node name
                # e.g., ".gaussians_nodes.background.positions" -> ".gaussians_nodes.background"
                parts = key.split(".")
                # Find the index of "gaussians_nodes"
                idx = parts.index("gaussians_nodes")
                # Node prefix is up to and including the next part (the node name)
                if idx + 1 < len(parts):
                    prefix = ".".join(parts[: idx + 2])  # Include "gaussians_nodes" and node name
                    if prefix not in node_prefixes:
                        node_prefixes.append(prefix)

        if not node_prefixes:
            raise ValueError("Could not find any Gaussian node parameters in model")

        print(f"Found {len(node_prefixes)} gaussian nodes: {[p.split('.')[-1] for p in node_prefixes]}")

        # Collect parameters from all nodes
        from collections import defaultdict

        all_node_parameters = defaultdict(list)

        for prefix in node_prefixes:
            node_name = prefix.split(".")[-1]

            # Load tensors helper for this node
            def load_param(suffix, node_prefix=prefix):
                key = f"{node_prefix}.{suffix}"
                assert key in state_dict, f"Missing required parameter: {key}"
                data = np.frombuffer(state_dict[key], dtype=np.float16).copy()
                if f"{key}.shape" in state_dict:
                    shape = state_dict[f"{key}.shape"]
                    data = data.reshape(shape)
                return torch.from_numpy(data).to(dtype=torch.float32, device=device).requires_grad_(True)

            # Load extra signals helper (optional parameters)
            def load_param_optional(suffix, default_shape):
                key = f"{prefix}.{suffix}"
                if key in state_dict:
                    data = np.frombuffer(state_dict[key], dtype=np.float16).copy()
                    if f"{key}.shape" in state_dict:
                        shape = state_dict[f"{key}.shape"]
                        data = data.reshape(shape)
                    return torch.from_numpy(data).to(dtype=torch.float32, device=device).requires_grad_(True)
                else:
                    # Return empty tensor with correct shape
                    return torch.zeros(default_shape, dtype=torch.float32, device=device, requires_grad=True)

            # Load core parameters (raw, pre-activation from msgpack)
            positions = load_param("positions")
            rotations = load_param("rotations")
            scales = load_param("scales")
            densities = load_param("densities")

            # Get features stored in split format:
            # - "features_albedo": DC band (base color) [N, 3] or [N, 1, 3]
            # - "features_specular": Higher-order SH bands (view-dependent) [N, K-1, 3]
            features_albedo = load_param("features_albedo")
            features_specular = load_param("features_specular")

            # Flatten albedo to [N, 3] if stored as [N, 1, 3]
            if features_albedo.dim() == 3:
                features_albedo = features_albedo.reshape(features_albedo.shape[0], -1)
            # Flatten specular to [N, (K-1)*3] if stored as [N, K-1, 3]
            if features_specular.dim() == 3:
                features_specular = features_specular.reshape(features_specular.shape[0], -1)

            # Concatenate to [N, K*3] format directly
            features = torch.cat([features_albedo, features_specular], dim=1)

            # Apply activations to match collect_gaussian_parameters() behavior:
            # The get_*() methods apply these activations, so we need to do the same
            # - densities: sigmoid → opacities in [0, 1]
            # - scales: exp → positive scales
            # - rotations: normalize quaternion
            densities = torch.sigmoid(densities)
            scales = torch.exp(scales)

            # Normalize quaternions (msgpack stores WXYZ format, as expected by gsplat)
            if rotations.numel() > 0:
                eps = 1e-8
                norm = torch.clamp(torch.linalg.norm(rotations, dim=-1, keepdim=True), min=eps)
                rotations = rotations / norm

            # Load extra signals
            num_gaussians = positions.shape[0]
            extra_signal = load_param_optional("extra_signal", (num_gaussians, 0))
            camera_extra_signal = load_param_optional("camera_extra_signal", (num_gaussians, 0))
            lidar_extra_signal = load_param_optional("lidar_extra_signal", (num_gaussians, 0))

            # Ensure correct shapes for densities
            if densities.dim() == 1:
                densities = densities.unsqueeze(1)

            # Append this node's parameters to the collection
            all_node_parameters["positions"].append(positions)
            all_node_parameters["rotations"].append(rotations)
            all_node_parameters["scales"].append(scales)
            all_node_parameters["densities"].append(densities)
            all_node_parameters["features"].append(features)
            all_node_parameters["extra_signal"].append(extra_signal)
            all_node_parameters["camera_extra_signal"].append(camera_extra_signal)
            all_node_parameters["lidar_extra_signal"].append(lidar_extra_signal)

            print(f"  Node '{node_name}': {num_gaussians} gaussians")

        # Concatenate all nodes together (matching collect_gaussian_parameters behavior)
        concatenated = {}
        for k, tensors_list in all_node_parameters.items():
            result = torch.cat(tensors_list, dim=0)
            # Ensure concatenated tensors are differentiable
            if result.dtype.is_floating_point and not result.requires_grad:
                result.requires_grad_(True)
            concatenated[k] = result

        # Debug: print concatenated results
        print("Debug - _extract_gaussians_from_msgpack_fixture output:")
        for k, v in concatenated.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, requires_grad={v.requires_grad}")

        return concatenated

    @staticmethod
    def _extract_extra_ray_signal_infos(
        model_config: dict, sensor_type: str = "camera"
    ) -> tuple[list[str], list[int], list]:
        """Extract extra_ray_signal_infos from model config.

        Mirrors the logic in BaseGaussianModel.__init__() (gaussians_model.py lines 296-328)
        where camera_extra_ray_signal_infos and lidar_extra_ray_signal_infos are built.

        Args:
            model_config: Model configuration dict from msgpack
            sensor_type: "camera" or "lidar"

        Returns:
            tuple[list[str], list[int], list[Callable]]:
                - List 0: signal names
                - List 1: signal dimensions
                - List 2: activation functions
        """
        from nurec.models.utils import get_activation

        signal_names: list[str] = []
        signal_dims: list[int] = []
        signal_activations: list = []

        extra_signal_configs = model_config.get("extra_signal", None)
        if extra_signal_configs is not None:
            # First pass: Add common signals to both camera and lidar ray outputs
            for signal_name, params in extra_signal_configs.items():
                if params["sensor_type"] == "common":
                    signal_dim = params["n_signal_dim"]
                    signal_activation = get_activation(params["activation"])
                    signal_names.append(str(signal_name))
                    signal_dims.append(signal_dim)
                    signal_activations.append(signal_activation)

            # Second pass: Add sensor-specific signals
            for signal_name, params in extra_signal_configs.items():
                if params["sensor_type"] == sensor_type:
                    signal_dim = params["n_signal_dim"]
                    signal_activation = get_activation(params["activation"])
                    signal_names.append(str(signal_name))
                    signal_dims.append(signal_dim)
                    signal_activations.append(signal_activation)

        return (signal_names, signal_dims, signal_activations)

    def run(self, use_rays: bool):
        """Run rendering test verifying GSplatRenderer output properties.

        NOTE: This test previously used a live rendering oracle for golden-value comparison.
        The oracle was removed as part of the renderer migration. Future work:
        replace with baked golden tensors in @gsplat_render_test_assets.

        Args:
            use_rays: If True, pass rays to GSplat renderer instead of using
                sensor model intrinsics directly.

        Raises:
            AssertionError: if outputs have wrong shape, dtype, or fall outside
                valid ranges (finite, opacity in [0, 1], depth >= 0, etc.)
        """
        torch.cuda.synchronize()

        # Create mock model that provides msgpack model_dict to GSplatRenderer
        class MockModel(BaseModel):
            def __init__(self, config, model_dict):
                super().__init__(config)
                self._model_dict = model_dict

            def serialize_to_json_dict(self, with_state_dict: bool = True) -> dict:
                import copy

                result = copy.deepcopy(self._model_dict)
                # If state_dict not requested, replace it with empty dict (GSplatRenderer
                # needs the key but not the data during initialization)
                adopt_renamed_key(result, "nurec_data", "nre_data")
                if not with_state_dict and "nurec_data" in result and "state_dict" in result["nurec_data"]:
                    result["nurec_data"]["state_dict"] = {}
                return result

        # Create GSplat config
        gsplat_config = OmegaConf.create(dict(self.renderer_config))
        gsplat_config.name = "3dgut-gsplat"
        gsplat_config.use_rays = use_rays
        gsplat_config_validated = GSplatRendererConfig.model_validate(gsplat_config)

        # Create GSplat renderer
        model_gsplat = MockModel(config=gsplat_config, model_dict=self.model)
        gsplat_renderer = GSplatRenderer(gsplat_config_validated, model_gsplat)

        # Get rays and frame dimensions
        h, w = self.frame_height, self.frame_width

        # Build poses_tquat_startend (1, 2, 7) from inherited sensor poses
        assert self.frames_sensor_start_pose is not None
        assert self.frames_sensor_end_pose is not None
        poses_tquat = torch.stack([self.frames_sensor_start_pose, self.frames_sensor_end_pose], dim=0).unsqueeze(
            0
        )  # (1, 2, 7)

        # Build timestamps_startend_us (1, 2)
        timestamps_us = torch.tensor(
            [[self.frame_start_timestamp, self.frame_end_timestamp]], dtype=torch.int64, device=self.device
        )

        assert self.frames_sensor_model is not None
        rendering_data = RenderingData(
            rays=self.rays,
            sensor_model_parameters=[self.frames_sensor_model],
            poses_tquat_startend=poses_tquat,
            timestamps_startend_us=timestamps_us,
            timestamps_startend_us_cpu=timestamps_us.cpu(),
            rays_timestamps_us=self.rays_timestamps_us,
        )

        # Build FrameMeta from sensor IDs
        assert self.frames_sensor_ids is not None
        frame_meta = [
            FrameMeta(
                unique_sensor_idx=int(self.frames_sensor_ids[0]),
                unique_frame_idx=int(self.frames_sensor_ids[1]),
                training_frame_idx=0,
                subsample=None,
                T_offset_nurec_startend=None,
            )
        ]

        # Run rendering with GSplatRenderer
        print("Rendering with GSplatRenderer...")
        gsplat_result = gsplat_renderer.render(
            rendering_data=rendering_data,
            gaussian_parameters=self.gaussian_parameters,
            n_active_features=self.n_active_features,
            extra_ray_signal_infos=self.camera_extra_ray_signal_infos,
            frame_meta=frame_meta,
        )

        # Validate core outputs (always required)
        assert gsplat_result.distance is not None, "GSplat ray distance output is None"
        assert gsplat_result.opacity is not None, "GSplat opacity output is None"

        # Extract enabled outputs from config (outputs.camera.*)
        outputs_config = self.renderer_config.get("outputs", {}).get("camera", {})
        enable_features = outputs_config.get("enable_features", True)
        enable_normals = outputs_config.get("enable_normals", True)
        enable_extended_features = outputs_config.get("enable_extended_features", True)

        # Verify RGB output presence matches config
        if enable_features:
            assert gsplat_result.rgb is not None, "GSplat RGB output is None but enable_features=True in config"
        else:
            assert gsplat_result.rgb is None, "GSplat RGB output is not None but enable_features=False in config"

        # enable_distance=False (with features) must skip the depth channel (CDIM 4 -> 3):
        # the renderer returns distance=None with 3-channel RGB, while opacity (alpha) is
        # still produced. Exercises the output-config gate that drives the inference 4->3.
        if enable_features:
            no_depth_config = OmegaConf.create(dict(self.renderer_config))
            no_depth_config.name = "3dgut-gsplat"
            no_depth_config.use_rays = use_rays
            OmegaConf.update(no_depth_config, "outputs.camera.enable_distance", False, force_add=True)
            no_depth_renderer = GSplatRenderer(GSplatRendererConfig.model_validate(no_depth_config), model_gsplat)
            no_depth_result = no_depth_renderer.render(
                rendering_data=rendering_data,
                gaussian_parameters=self.gaussian_parameters,
                n_active_features=self.n_active_features,
                extra_ray_signal_infos=self.camera_extra_ray_signal_infos,
                frame_meta=frame_meta,
            )
            assert no_depth_result.distance is None, (
                "enable_distance=False must yield distance=None (depth must not be rendered)"
            )
            assert no_depth_result.rgb is not None and no_depth_result.rgb.shape[-1] == 3, (
                "enable_distance=False must keep 3-channel RGB"
            )
            assert no_depth_result.opacity is not None, "opacity must still be produced (alpha is free)"

        # Verify normal output presence matches config
        if not enable_normals:
            assert gsplat_result.normal is None, "GSplat normal output is not None but enable_normals=False in config"

        # Verify extra signals output presence matches config
        if not enable_extended_features:
            assert gsplat_result.extra_ray_signals is None, (
                "GSplat extra signals output is not None but enable_extended_features=False in config"
            )

        # Extract GSplat outputs for property validation
        gsplat_opacity = gsplat_result.opacity.detach().cpu().numpy().reshape(-1)
        gsplat_depth_raw = gsplat_result.distance.detach().cpu().numpy().reshape(-1)
        # Normalize depth by opacity (opacity-weighted depth → mean depth per pixel)
        gsplat_depth = np.divide(
            gsplat_depth_raw, gsplat_opacity, out=np.zeros_like(gsplat_depth_raw), where=gsplat_opacity > 0
        )
        gsplat_rgb = None
        if gsplat_result.rgb is not None:
            gsplat_rgb_raw = gsplat_result.rgb.detach().cpu().numpy().reshape(-1, 3)
            gsplat_rgb = np.divide(
                gsplat_rgb_raw,
                gsplat_opacity[:, None],
                out=np.zeros_like(gsplat_rgb_raw),
                where=gsplat_opacity[:, None] > 0,
            )
        gsplat_normal = (
            gsplat_result.normal.detach().cpu().numpy().reshape(-1, 3) if gsplat_result.normal is not None else None
        )

        print(f"GSplatRenderer output statistics for {self.frame_id}:")
        print(
            f"  Config: enable_features={enable_features}, enable_normals={enable_normals}, enable_extended_features={enable_extended_features}"
        )
        if gsplat_rgb is not None:
            print(
                f"  RGB     - min: {gsplat_rgb.min():.6f}, max: {gsplat_rgb.max():.6f}, mean: {gsplat_rgb.mean():.6f}"
            )
        print(
            f"  Depth   - min: {gsplat_depth.min():.6f}, max: {gsplat_depth.max():.6f}, mean: {gsplat_depth.mean():.6f}"
        )
        print(
            f"  Opacity - min: {gsplat_opacity.min():.6f}, max: {gsplat_opacity.max():.6f}, mean: {gsplat_opacity.mean():.6f}"
        )

        # Save debug visualization
        if DEBUG_OUTPUT_ENABLED:
            output_dir = os.path.join(DEBUG_OUTPUT_DIR, self.test_name)
            os.makedirs(output_dir, exist_ok=True)

            gsplat_rgb_img = gsplat_rgb.reshape(h, w, 3) if gsplat_rgb is not None else None
            gsplat_opacity_img = gsplat_opacity.reshape(h, w)
            gsplat_depth_img = gsplat_depth.reshape(h, w)

            if gsplat_rgb_img is not None:
                _save_rgb(gsplat_rgb_img, os.path.join(output_dir, "gsplat_rgb.png"))
            _save_grayscale(gsplat_opacity_img, os.path.join(output_dir, "gsplat_opacity.png"))
            _save_colormap(
                gsplat_depth_img,
                os.path.join(output_dir, "gsplat_depth.png"),
                invert=True,
            )
            if gsplat_normal is not None:
                gsplat_normal_img = gsplat_normal.reshape(h, w, 3)
                _save_rgb((gsplat_normal_img + 1) / 2, os.path.join(output_dir, "gsplat_normals.png"))

            print(f"GSplatRenderer debug images saved to {output_dir}")

        # --- Property-based assertions ---

        # Opacity must be in [0, 1] (sigmoid-activated, alpha-blended)
        assert np.all(np.isfinite(gsplat_opacity)), "GSplat opacity contains non-finite values"
        np.testing.assert_array_less(-1e-5, gsplat_opacity, err_msg="GSplat opacity has values below 0")
        np.testing.assert_array_less(gsplat_opacity, 1.0 + 1e-5, err_msg="GSplat opacity has values above 1")

        # Depth: finite and non-negative (background pixels have depth=0)
        assert np.all(np.isfinite(gsplat_depth)), "GSplat depth contains non-finite values"
        np.testing.assert_array_less(-1e-5, gsplat_depth, err_msg="GSplat depth has negative values")

        if enable_features:
            assert gsplat_rgb is not None
            # RGB must be finite
            assert np.all(np.isfinite(gsplat_rgb)), "GSplat RGB contains non-finite values"
            # If saturate_radiance is enabled (default True), RGB must be in [0, 1]
            model_config = self.model.get("nurec_data", {}).get("config", {})
            if model_config.get("saturate_radiance", True):
                np.testing.assert_array_less(-1e-5, gsplat_rgb, err_msg="GSplat RGB (saturated) has values below 0")
                np.testing.assert_array_less(
                    gsplat_rgb, 1.0 + 1e-5, err_msg="GSplat RGB (saturated) has values above 1"
                )

        if gsplat_normal is not None:
            assert np.all(np.isfinite(gsplat_normal)), "GSplat normal contains non-finite values"

        torch.cuda.synchronize()


def _discover_test_asset_paths() -> list[str]:
    """Discover msgpack test asset files for GSplat testing.

    Returns:
        list of paths to .msgpack test files, or [] when the asset archive
        is not available (the parameterized tests are skipped in that case).

    TODO: replace @nrend_test_assets sentinel with @gsplat_render_test_assets
    once baked golden tensors are published (see D8 / Phase 6 migration plan).
    Until then this returns [] and the parametrized test_gsplat_renderer cases
    are skipped; test_render_mode_rgb_skips_depth_channel still runs.
    """
    from python.runfiles import runfiles

    RUNFILES = runfiles.Create()
    # Sentinel file from the (now-removed) nrend_test_assets archive.
    # When @gsplat_render_test_assets is published, update this path.
    test_assets_location = RUNFILES.Rlocation("nrend_test_assets/nrend_test_nerf_reference_0.2.334_250227.msgpack")
    if test_assets_location is None:
        # Asset archive not yet available — parametrized cases will be skipped.
        return []

    test_assets_dir = os.path.dirname(test_assets_location)
    if not os.path.exists(test_assets_dir):
        return []

    test_asset_paths = [
        os.path.join(test_assets_dir, f)
        for f in os.listdir(test_assets_dir)
        # only include 3dgut inputs (that's what we're testing), and "colmap", which avoids
        # dynamic objects that we don't support yet
        if os.path.isfile(os.path.join(test_assets_dir, f))
        and str(f).endswith(".msgpack")
        and "3dgut" in str(f)
        and "colmap" in str(f)
    ]

    return test_asset_paths


def _generate_test_id(asset_path: str) -> str:
    """Generate test ID from asset path.

    Args:
        asset_path: path to test asset file

    Returns:
        clean test ID (filename without extension)
    """
    filename = os.path.basename(asset_path)
    return os.path.splitext(filename)[0]


_DISCOVERED_ASSET_PATHS = _discover_test_asset_paths()

_ASSET_PATH_PARAMS: list[Any]
_ASSET_PATH_IDS: Union[Callable[[str], str], list[str]]

if _DISCOVERED_ASSET_PATHS:
    _ASSET_PATH_PARAMS = _DISCOVERED_ASSET_PATHS
    _ASSET_PATH_IDS = _generate_test_id
else:
    # Do NOT let this silently collect zero parametrized cases (pytest treats an
    # empty parametrize list as "no test exists", which is indistinguishable from
    # "everything passed" in CI). Emit a single, loudly-skipped case instead so the
    # gap is visible in test reports until @gsplat_render_test_assets is published.
    _ASSET_PATH_PARAMS = [
        pytest.param(
            "",
            marks=pytest.mark.skip(
                reason=(
                    "GSplat golden-output test assets are unavailable: "
                    "@gsplat_render_test_assets has not been published yet "
                    "(see TODO(D8) in nurec/models/gaussians/BUILD.bazel). "
                    "test_gsplat_renderer parametrized cases are being SKIPPED, "
                    "not executed -- this is a known gap, not a pass."
                )
            ),
        )
    ]
    _ASSET_PATH_IDS = ["MISSING_TEST_ASSETS"]


@pytest.mark.skipif(torch.cuda.get_device_capability() < (8, 0), reason="GSplat requires CUDA CC >= 8.0")
@pytest.mark.parametrize("asset_path", _ASSET_PATH_PARAMS, ids=_ASSET_PATH_IDS)
@pytest.mark.parametrize("use_rays", [True, False], ids=["extrays", "intrays"])
def test_gsplat_renderer(asset_path: str, use_rays: bool):
    """Test GSplatRenderer output properties against shape/dtype/range invariants.

    Infers which outputs to validate based on config.outputs.camera.* flags.

    Args:
        asset_path: path to msgpack test case file
        use_rays: If True, pass rays to GSplat renderer
    """
    test_case = GSplatRenderTestCase.from_file(asset_path, device=torch.device("cuda"))
    test_case.run(use_rays)


@pytest.mark.skipif(torch.cuda.get_device_capability() < (8, 0), reason="GSplat requires CUDA CC >= 8.0")
def test_render_mode_rgb_skips_depth_channel():
    """Rendering with render_mode="RGB" (vs "RGB+D") drops the depth channel (CDIM 4 -> 3).

    render_mode is the stable gsplat API the GSplatRenderer selects when the camera
    enable_distance output gate is off. Verifies "RGB" yields 3 channels vs 4 for "RGB+D",
    while the RGB channels and opacity (alpha) stay identical -- so the 4->3 change is
    quality-preserving.
    """
    import gsplat

    device = torch.device("cuda")
    torch.manual_seed(0)

    n = 256
    means = torch.randn(1, n, 3, device=device) * 0.4
    means[..., 2] += 4.0  # place gaussians in front of the camera (z > 0)
    quats = torch.zeros(1, n, 4, device=device)
    quats[..., 0] = 1.0  # identity rotation (w, x, y, z)
    scales = torch.full((1, n, 3), 0.05, device=device)
    opacities = torch.full((1, n), 0.6, device=device)
    colors = torch.rand(1, n, 3, device=device)

    width, height = 64, 48
    viewmats = torch.eye(4, device=device).reshape(1, 1, 4, 4)
    focal = 50.0
    ks = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    ).reshape(1, 1, 3, 3)

    shared = dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=ks,
        width=width,
        height=height,
        sh_degree=None,
    )

    # render_mode="RGB+D": RGB + depth -> 4 channels, distance present.
    rgbd, alpha_rgbd, _ = gsplat.rasterization(**shared, render_mode="RGB+D")
    assert rgbd.shape[-1] == 4, f"expected RGB+D (4 channels), got {tuple(rgbd.shape)}"

    # render_mode="RGB": depth skipped -> 3 channels (the mode GSplatRenderer uses when
    # the enable_distance gate is off).
    rgb, alpha_rgb, _ = gsplat.rasterization(**shared, render_mode="RGB")
    assert rgb.shape[-1] == 3, f"expected RGB (3 channels), got {tuple(rgb.shape)}"

    # Dropping the depth channel must not change the rendered color or opacity (depth is
    # an independent appended channel) -> confirms the 4->3 change is quality-preserving.
    torch.testing.assert_close(rgbd[..., :3], rgb, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(alpha_rgbd, alpha_rgb, rtol=1e-4, atol=1e-4)
