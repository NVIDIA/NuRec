# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Slang-based GPU implementation of the Gaussian parameter collector.

This module provides the concrete implementation of GaussianParameterCollector using
Slang GPU kernels for high-performance parallel processing.

Architecture:
-------------
The implementation uses a layered architecture:

1. Layer Handlers (_LayerHandler subclasses):
   - Each layer type (Base, SH, Rigid, Deformable) has a dedicated handler
   - Handlers generate Slang kernel configurations and manage kernel parameters
   - Hierarchy: Base -> SH -> Rigid -> Deformable (each extends the previous)

2. Autograd Functions (torch.autograd.Function):
   - CollectorFunction: Main collection of Gaussians with activations and transforms
   - TracksCalibFunction: Track pose calibration
   - InputEmbeddingFunction: Hash grid input embedding computation
   - TracksInterpolationFunction: Track pose interpolation over time

3. Slang Kernel Integration:
   - Kernels are defined in collector.slang and can be dynamically compiled, but
     pre-compiled kernels are provided for many typical layer configurations
   - Slang provides automatic differentiation for GPU kernels
   - Forward and backward passes are handled by the same kernel infrastructure

Key Optimizations:
------------------
- Contiguous memory layout for coalesced GPU memory access
- Fused operations to reduce kernel launches
- Gradient zeroing only when necessary (loadOnce vs loadEx semantics)
- Parallel processing across all Gaussians/tracks
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch

from nurec.libs.slang_gaussians.collector import (
    CollectorConfiguration,
    CollectorKernel,
    get_slang_kernels,
)
from nurec.libs.slang_utils.utils import profile
from nurec.models.gaussians.collect.collector import (
    CollectorResult,
    DensityActivation,
    DirectTracksCalibData,
    GaussianParameterCollector,
    HolisticRemapTimeInputEmbeddingConfig,
    IndividualRemapTimeInputEmbeddingConfig,
    IndividualRemapTimeInputEmbeddingData,
    IndividualStepTimeInputEmbeddingConfig,
    IndividualStepTimeInputEmbeddingData,
    InputEmbeddingData,
    LayerConfigBase,
    LayerConfigDeformable,
    LayerConfigRigid,
    LayerConfigSH,
    LayerDataBase,
    LayerDataDeformable,
    LayerDataRigid,
    LayerDataSH,
    LayersConfig,
    LayersData,
    LayerSegments,
    RotationActivation,
    ScaleActivation,
    TracksCalibData,
    TracksCalibResiduals,
    TracksFrustumCullData,
    TracksInterpolationData,
    TracksTimestampsEstimationData,
    TracksTimestampsGlobalData,
    TracksTimestampsPerTrackData,
    WeightedInstanceInputEmbeddingData,
    whole_layer_segments,
)
from nurec.models.tracks_calib_data import ElevationAwareUnicycleTracksCalibData


# Kernel parameters.
@dataclass(slots=True)
class KernelParameters:
    """Encapsulates a Slang kernel and its bound parameters.

    Attributes:
        kernel: The compiled Slang kernel to execute
        parameters: List of parameter structs/tensors to pass to the kernel
    """

    kernel: CollectorKernel
    parameters: List[Any]


_ZERO_GRAD_BUFFER_COUNT = 9
_ZERO_GRAD_BLOCK_WIDTH = 32
_ZERO_GRAD_NUM_THREADS = 256


def launch_zero_gradients_for_disabled_tracks(
    slang_module: Any,
    tracks_ids: torch.Tensor,
    keep_mask: torch.Tensor,
    grad_buffers: Tuple[torch.Tensor, ...],
) -> None:
    """Zero gradient rows belonging to disabled tracks.

    Args:
        slang_module: Loaded collector Slang module containing the 2D kernel.
        tracks_ids: Contiguous CUDA int32 tensor of shape ``[N]`` mapping each
            Gaussian row to a track.
        keep_mask: Contiguous CUDA bool tensor indexed by ``tracks_ids``.
        grad_buffers: Nine contiguous CUDA float tensors with leading dimension
            ``N``. All trailing dimensions are flattened for the kernel launch;
            empty buffers are allowed.

    Returns:
        None.

    Raises:
        ValueError: If the buffer count, rank, or leading dimensions violate
            the launch contract.

    The launch uses 32 channel threads by 8 Gaussian threads. One warp owns
    each row, allowing lane 0 to gather and broadcast the track mask.
    """
    n = tracks_ids.shape[0]
    if len(grad_buffers) != _ZERO_GRAD_BUFFER_COUNT:
        raise ValueError(f"Expected {_ZERO_GRAD_BUFFER_COUNT} gradient buffers, got {len(grad_buffers)}.")

    block_height = _ZERO_GRAD_NUM_THREADS // _ZERO_GRAD_BLOCK_WIDTH
    max_d = 1
    flat_grad_buffers = []
    for i, buf in enumerate(grad_buffers):
        if buf.dim() < 2:
            if buf.numel() != 0:
                raise ValueError(f"grad_buffers[{i}] must be rank >= 2 when non-empty.")
            flat_buf = buf
        elif int(buf.size(0)) != n:
            raise ValueError(f"grad_buffers[{i}].size(0) must equal tracks_ids.shape[0] ({n}).")
        elif buf.numel() == 0:
            flat_buf = buf
        else:
            flat_buf = buf.view(n, -1)
            max_d = max(max_d, int(flat_buf.size(1)))

        flat_grad_buffers.append(flat_buf)

    if n == 0:
        return

    block = (_ZERO_GRAD_BLOCK_WIDTH, block_height, 1)
    grid = (
        (n + block_height - 1) // block_height,
        (max_d + _ZERO_GRAD_BLOCK_WIDTH - 1) // _ZERO_GRAD_BLOCK_WIDTH,
        1,
    )
    # One warp owns each row; lane 0 gathers the track mask and the Slang
    # kernel broadcasts it across the channel lanes.
    slang_module.zero_gradients_for_disabled_tracks_2d.fn_handle(
        block,
        grid,
        tracks_ids,
        keep_mask,
        *flat_grad_buffers,
    )


#
#
# Layer-specific logic.
#
class _LayerHandler(ABC):
    """Abstract base class for handling layer-specific collection logic.

    Each layer type (Base, SH, Rigid, Deformable) has a concrete handler that:
    - Generates Slang kernel configurations
    - Validates input data
    - Extracts trainable tensors for autograd tracking
    - Prepares kernel parameters for forward/backward passes
    - Performs post-processing after backward pass (e.g., gradient masking)

    Attributes:
        kernels: Tuple of compiled Slang kernels for this layer type
    """

    kernels: Tuple[CollectorKernel, ...]

    @abstractmethod
    def get_collector_configurations(self) -> List[CollectorConfiguration]:
        """Generate Slang kernel configurations for this layer type.

        Returns:
            List of configurations specifying which Slang tasks to instantiate
        """
        ...

    @abstractmethod
    def validate(self, layers_data: LayersData, layer_data: LayerDataBase) -> None:
        """Validate input data shapes and types.

        Args:
            layers_data: Complete data for all layers
            layer_data: Data for this specific layer

        Raises:
            AssertionError: If validation fails
        """
        ...

    @abstractmethod
    def get_trainable_input_tensors(self, layer_data: LayerDataBase) -> List[torch.Tensor]:
        """Extract tensors that require gradients for autograd tracking.

        Args:
            layer_data: Data for this layer

        Returns:
            List of tensors to save for backward pass
        """
        ...

    @abstractmethod
    def get_kernel_parameters(
        self,
        layers_data: LayersData,
        layer_data: LayerDataBase,
        output_params: List[Any],
        tensor_func: Callable[[torch.Tensor, bool], Tuple[torch.Tensor, Tuple[torch.Tensor]]],
    ) -> KernelParameters:
        """Prepare parameters for kernel execution.

        Args:
            layers_data: Complete data for all layers
            layer_data: Data for this specific layer
            output_params: Pre-allocated output tensor parameters
            tensor_func: Function to wrap tensors for SlangTorch (handles forward/backward)

        Returns:
            KernelParameters with bound kernel and parameters
        """
        ...

    @abstractmethod
    def post_process_backward(self, kernel_parameters: KernelParameters) -> None:
        """Post-process results after backward pass.

        Used for operations like zeroing gradients for masked-out Gaussians.

        Args:
            kernel_parameters: Parameters used in backward pass
        """
        ...


# The base layer handler does not implement the full interface, because there are no
# layers which are just the base layer.  You always need at least spherical features.
class _LayerHandlerBase(_LayerHandler):
    """Handler for base layer parameters (positions, rotations, scales, densities, signals).

    This is a partial implementation that handles the common parameters shared by all layers.
    Concrete layer types extend this to add their specific functionality (e.g., features).

    Responsibilities:
    - Apply activations: normalize rotations, exp scales, sigmoid densities
    - Copy positions and extra signals
    - Generate kernel configurations for base collectors

    Note: This class doesn't implement the full _LayerHandler interface because base
    layers always require at least spherical features in practice.
    """

    def __init__(self, layers_config: LayersConfig, layer_index: int):
        """Initialize base layer handler.

        Args:
            layers_config: Complete configuration for all layers
            layer_index: Index of this layer in the layers list
        """
        self.layers_config = layers_config
        self.layer_config = layers_config.layers[layer_index]
        assert isinstance(self.layer_config, LayerConfigBase)

    def get_collector_configurations(self) -> List[CollectorConfiguration]:
        parameters = []

        layer_config = self.layer_config
        layers_config = self.layers_config

        parameters.append("Collector_Copy<3>")

        if layer_config.rotation_activation == RotationActivation.NORMALIZE:
            parameters.append("RotationsCollector_Normalize")
        else:
            raise ValueError(f"Unsupported rotation activation: {layer_config.rotation_activation}")

        if layer_config.scale_activation == ScaleActivation.EXP:
            parameters.append("ScalesCollector_Exp")
        else:
            raise ValueError(f"Unsupported scale activation: {layer_config.scale_activation}")

        if layer_config.density_activation == DensityActivation.SIGMOID:
            parameters.append("DensitiesCollector_Sigmoid")
        else:
            raise ValueError(f"Unsupported density activation: {layer_config.density_activation}")

        if layers_config.extra_signal_dim > 0:
            parameters.append(f"Collector_Copy<{layers_config.extra_signal_dim}>")

        if layers_config.camera_extra_signal_dim > 0:
            parameters.append(f"Collector_Copy<{layers_config.camera_extra_signal_dim}>")

        if layers_config.lidar_extra_signal_dim > 0:
            parameters.append(f"Collector_Copy<{layers_config.lidar_extra_signal_dim}>")

        return [CollectorConfiguration(parameters=tuple(parameters))]

    def validate(self, layers_data: LayersData, layer_data: LayerDataBase) -> None:
        # Validate the number of Gaussians.
        nb_gaussians = layer_data.positions.shape[0]
        assert nb_gaussians == layer_data.rotations.shape[0]
        assert nb_gaussians == layer_data.scales.shape[0]
        assert nb_gaussians == layer_data.densities.shape[0]
        assert nb_gaussians == layer_data.extra_signal.shape[0]
        assert nb_gaussians == layer_data.camera_extra_signal.shape[0]
        assert nb_gaussians == layer_data.lidar_extra_signal.shape[0]

        # Validate the component dimensions.
        assert layer_data.positions.shape[1] == 3
        assert layer_data.positions.dtype == torch.float32
        assert layer_data.rotations.shape[1] == 4
        assert layer_data.rotations.dtype == torch.float32
        assert layer_data.scales.shape[1] == 3
        assert layer_data.scales.dtype == torch.float32
        assert layer_data.densities.shape[1] == 1
        assert layer_data.densities.dtype == torch.float32
        assert layer_data.extra_signal.shape[1] == self.layers_config.extra_signal_dim
        assert layer_data.extra_signal.dtype == torch.float32
        assert layer_data.camera_extra_signal.shape[1] == self.layers_config.camera_extra_signal_dim
        assert layer_data.camera_extra_signal.dtype == torch.float32
        assert layer_data.lidar_extra_signal.shape[1] == self.layers_config.lidar_extra_signal_dim
        assert layer_data.lidar_extra_signal.dtype == torch.float32

    def get_trainable_input_tensors(self, layer_data: LayerDataBase) -> List[torch.Tensor]:
        input_tensors = [
            layer_data.positions,
            layer_data.rotations,
            layer_data.scales,
            layer_data.densities,
        ]

        if self.layers_config.extra_signal_dim > 0:
            input_tensors.append(layer_data.extra_signal)
        if self.layers_config.camera_extra_signal_dim > 0:
            input_tensors.append(layer_data.camera_extra_signal)
        if self.layers_config.lidar_extra_signal_dim > 0:
            input_tensors.append(layer_data.lidar_extra_signal)

        return input_tensors

    def get_kernel_parameters(
        self,
        layers_data: LayersData,
        layer_data: LayerDataBase,
        output_params: List[Any],
        tensor_func: Callable[[torch.Tensor, bool], Tuple[torch.Tensor, Tuple[torch.Tensor]]],
    ) -> KernelParameters:
        slang_module = self.kernels[0].slang_module

        # All tensors in this layer don't need to have their gradients zeroed.
        def tf(tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
            return tensor_func(tensor, False)

        positions = slang_module.Collector_Copy(input=tf(layer_data.positions), output=output_params[0])
        rotations = slang_module.RotationsCollector_Normalize(input=tf(layer_data.rotations), output=output_params[1])
        scales = slang_module.ScalesCollector_Exp(input=tf(layer_data.scales), output=output_params[2])
        densities = slang_module.DensitiesCollector_Sigmoid(input=tf(layer_data.densities), output=output_params[3])

        parameters = [
            positions,
            rotations,
            scales,
            densities,
        ]

        if self.layers_config.extra_signal_dim > 0:
            parameters.append(slang_module.Collector_Copy(input=tf(layer_data.extra_signal), output=output_params[4]))

        if self.layers_config.camera_extra_signal_dim > 0:
            parameters.append(
                slang_module.Collector_Copy(input=tf(layer_data.camera_extra_signal), output=output_params[5])
            )

        if self.layers_config.lidar_extra_signal_dim > 0:
            parameters.append(
                slang_module.Collector_Copy(input=tf(layer_data.lidar_extra_signal), output=output_params[6])
            )

        return KernelParameters(kernel=self.kernels[0], parameters=parameters)

    def post_process_backward(self, kernel_parameters: KernelParameters) -> None:
        pass


class _LayerHandlerSH(_LayerHandlerBase):
    """Handler for Spherical Harmonics layers with view-dependent features.

    Extends base layer with:
    - Albedo (diffuse) and specular (view-dependent) features
    - Optional Fourier-based temporal modulation of features
    - Time embedding support (individual or holistic remapping)

    When fourier_features_dim > 1:
    - Features are stored as [num_gaussians, fourier_dim, albedo_dim]
    - At collection time, Fourier transform is applied using frame timestamp
    - Output is time-varying appearance: f(t) = Σ fourier_basis(t) * features

    When fourier_features_dim == 1:
    - Features are static, simply copied to output
    """

    def __init__(self, layers_config: LayersConfig, layer_index: int):
        """Initialize SH layer handler with validation.

        Args:
            layers_config: Complete configuration for all layers
            layer_index: Index of this layer in the layers list
        """
        super().__init__(layers_config, layer_index)
        layer_config = self.layer_config
        assert isinstance(layer_config, LayerConfigSH)

        if layer_config.fourier_features_dim > 1:
            if type(layer_config.embed_config) is IndividualRemapTimeInputEmbeddingConfig:
                assert not layer_config.embed_config.timestamps_us_ranges.requires_grad
                assert layer_config.embed_config.timestamps_us_ranges.shape[1] == 2
                assert torch.all(
                    layer_config.embed_config.timestamps_us_ranges[:, 0]
                    < layer_config.embed_config.timestamps_us_ranges[:, 1]
                )
            elif type(layer_config.embed_config) is HolisticRemapTimeInputEmbeddingConfig:
                assert layer_config.embed_config.timestamps_us_min < layer_config.embed_config.timestamps_us_max
                assert layer_config.embed_config.remap_min < layer_config.embed_config.remap_max
            elif type(layer_config.embed_config) is IndividualStepTimeInputEmbeddingConfig:
                assert not layer_config.embed_config.timestamps_us_ranges.requires_grad
                assert layer_config.embed_config.timestamps_us_ranges.shape[1] == 2
                assert torch.all(
                    layer_config.embed_config.timestamps_us_ranges[:, 0]
                    < layer_config.embed_config.timestamps_us_ranges[:, 1]
                )
            else:
                raise ValueError(f"Unsupported embed config: {layer_config.embed_config}")
        else:
            assert layer_config.embed_config is None

    def get_collector_configurations(self) -> List[CollectorConfiguration]:
        configurations = super().get_collector_configurations()
        assert len(configurations) == 1
        parameters = list(configurations[0].parameters)

        layer_config = self.layer_config
        assert isinstance(layer_config, LayerConfigSH)
        # Pick the fp16 task struct sibling (suffixed _FP16) when SH output is half-precision.
        fp16_suffix = "_FP16" if self.layers_config.sh_features_dtype == "fp16" else ""
        if layer_config.fourier_features_dim > 1:
            if type(layer_config.embed_config) is IndividualStepTimeInputEmbeddingConfig:
                assert layer_config.embed_config.n_dims == 1
                parameters.append(
                    f"SphericalFeaturesCollector_Fourier_IndividualStep{fp16_suffix}<{self.layers_config.albedo_dim},{self.layers_config.specular_dim},{layer_config.fourier_features_dim},{layer_config.embed_config.n_steps}>"
                )
            else:
                if type(layer_config.embed_config) is IndividualRemapTimeInputEmbeddingConfig:
                    time_embedding_suffix = "Individual"
                elif type(layer_config.embed_config) is HolisticRemapTimeInputEmbeddingConfig:
                    time_embedding_suffix = "Holistic"
                else:
                    raise ValueError(f"Unsupported embed config: {layer_config.embed_config}")
                parameters.append(
                    f"SphericalFeaturesCollector_Fourier_{time_embedding_suffix}{fp16_suffix}<{self.layers_config.albedo_dim},{self.layers_config.specular_dim},{layer_config.fourier_features_dim}>"
                )
        else:
            parameters.append(
                f"SphericalFeaturesCollector_Copy{fp16_suffix}<{self.layers_config.albedo_dim},{self.layers_config.specular_dim}>"
            )

        return [CollectorConfiguration(parameters=tuple(parameters))]

    def validate(self, layers_data: LayersData, layer_data: LayerDataBase) -> None:
        super().validate(layers_data, layer_data)
        assert isinstance(layer_data, LayerDataSH)

        # Validate the number of Gaussians.
        nb_gaussians = layer_data.positions.shape[0]
        assert nb_gaussians == layer_data.features_albedo.shape[0]
        assert nb_gaussians == layer_data.features_specular.shape[0]

        # Validate the component dimensions.
        assert layer_data.features_albedo.shape[-1] == self.layers_config.albedo_dim
        assert layer_data.features_albedo.dtype == torch.float32
        assert layer_data.features_specular.shape[-1] == self.layers_config.specular_dim
        assert layer_data.features_specular.dtype == torch.float32

        layer_config = self.layer_config
        assert isinstance(layer_config, LayerConfigSH)
        if layer_config.fourier_features_dim > 1:
            if type(layer_config.embed_config) is IndividualRemapTimeInputEmbeddingConfig:
                assert type(layer_data.embed_data) is IndividualRemapTimeInputEmbeddingData
                assert not layer_data.embed_data.instance_idx.requires_grad
                assert layer_data.embed_data.instance_idx.shape[0] == layer_data.positions.shape[0]
            elif type(layer_config.embed_config) is HolisticRemapTimeInputEmbeddingConfig:
                assert layer_data.embed_data is None
            elif type(layer_config.embed_config) is IndividualStepTimeInputEmbeddingConfig:
                assert type(layer_data.embed_data) is IndividualStepTimeInputEmbeddingData
                assert not layer_data.embed_data.instance_idx.requires_grad
                assert layer_data.embed_data.instance_idx.shape[0] == layer_data.positions.shape[0]

            assert layer_data.features_albedo.shape[1] == layer_config.fourier_features_dim

            assert layers_data.frame_timestamp_us is not None

    def get_trainable_input_tensors(self, layer_data: LayerDataBase) -> List[torch.Tensor]:
        input_tensors = super().get_trainable_input_tensors(layer_data)

        assert isinstance(layer_data, LayerDataSH)

        assert isinstance(self.layer_config, LayerConfigSH)
        if type(self.layer_config.embed_config) is IndividualStepTimeInputEmbeddingConfig:
            assert isinstance(layer_data.embed_data, IndividualStepTimeInputEmbeddingData)
            input_tensors.extend([layer_data.embed_data.u])

        input_tensors.extend([layer_data.features_albedo, layer_data.features_specular])

        return input_tensors

    def get_kernel_parameters(
        self,
        layers_data: LayersData,
        layer_data: LayerDataBase,
        output_params: List[Any],
        tensor_func: Callable[[torch.Tensor, bool], Tuple[torch.Tensor, Tuple[torch.Tensor]]],
    ) -> KernelParameters:
        base_kernel_parameters = (
            super().get_kernel_parameters(layers_data, layer_data, output_params, tensor_func).parameters
        )

        slang_module = self.kernels[0].slang_module

        def tf(tensor: torch.Tensor, zero_gradients: bool = False) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
            return tensor_func(tensor, zero_gradients)

        assert isinstance(layer_data, LayerDataSH)

        layer_config = self.layer_config
        assert isinstance(layer_config, LayerConfigSH)
        fp16_suffix = "_FP16" if self.layers_config.sh_features_dtype == "fp16" else ""
        if layer_config.fourier_features_dim > 1:
            frame_timestamp_us = layers_data.frame_timestamp_us

            if type(layer_config.embed_config) is IndividualRemapTimeInputEmbeddingConfig:
                if type(layer_data.embed_data) is not IndividualRemapTimeInputEmbeddingData:
                    raise ValueError(
                        "embed_data should be IndividualRemapTimeInputEmbeddingData for IndividualRemapTimeInputEmbeddingConfig"
                    )

                time_embedding = slang_module.IndividualRemapTimeInputEmbedding(
                    instance_idx=layer_data.embed_data.instance_idx.contiguous(),
                    timestamps_ranges=layer_config.embed_config.timestamps_us_ranges.contiguous(),
                    remap_min=layer_config.embed_config.remap_min,
                    remap_max=layer_config.embed_config.remap_max,
                )
                task_type = getattr(slang_module, f"SphericalFeaturesCollector_Fourier_Individual{fp16_suffix}")
            elif type(layer_config.embed_config) is HolisticRemapTimeInputEmbeddingConfig:
                if layer_data.embed_data is not None:
                    raise ValueError("embed_data should be None for HolisticRemapTimeInputEmbeddingConfig")

                time_embedding = slang_module.HolisticRemapTimeInputEmbedding(
                    timestamps_us_min=layer_config.embed_config.timestamps_us_min,
                    timestamps_us_max=layer_config.embed_config.timestamps_us_max,
                    remap_min=layer_config.embed_config.remap_min,
                    remap_max=layer_config.embed_config.remap_max,
                )
                task_type = getattr(slang_module, f"SphericalFeaturesCollector_Fourier_Holistic{fp16_suffix}")
            elif type(layer_config.embed_config) is IndividualStepTimeInputEmbeddingConfig:
                if type(layer_data.embed_data) is not IndividualStepTimeInputEmbeddingData:
                    raise ValueError(
                        "embed_data should be IndividualStepTimeInputEmbeddingData for IndividualStepTimeInputEmbeddingConfig"
                    )

                time_embedding = slang_module.IndividualStepTimeInputEmbedding(
                    instance_idx=layer_data.embed_data.instance_idx.contiguous(),
                    timestamps_ranges=layer_config.embed_config.timestamps_us_ranges.contiguous(),
                    u=tf(layer_data.embed_data.u, True),
                    beta=layer_data.embed_data.beta,
                )
                task_type = getattr(slang_module, f"SphericalFeaturesCollector_Fourier_IndividualStep{fp16_suffix}")
            else:
                raise ValueError(f"Unsupported embed config: {layer_config.embed_config}")

            spherical_features = task_type(
                time_embedding=time_embedding,
                timestamp=frame_timestamp_us,
                albedo=tf(layer_data.features_albedo),
                specular=tf(layer_data.features_specular),
                output=output_params[7],
            )
        else:
            copy_task_type = getattr(slang_module, f"SphericalFeaturesCollector_Copy{fp16_suffix}")
            spherical_features = copy_task_type(
                albedo=tf(layer_data.features_albedo),
                specular=tf(layer_data.features_specular),
                output=output_params[7],
            )

        return KernelParameters(kernel=self.kernels[0], parameters=[*base_kernel_parameters, spherical_features])


class _LayerHandlerRigid(_LayerHandlerSH):
    """Handler for rigid layers attached to moving tracks.

    Extends SH layer with:
    - Track-based rigid transformations: Each Gaussian transforms with its track's pose
    - Visibility masking: Tracks can be disabled (keep_mask=False) trypically when the
      timestamp is outside of the track's range, setting density to 0
    - Fused kernel: Combines position/rotation/density collection with track transforms

    Key optimizations:
    - Single kernel handles positions, rotations, and densities together
    - Reduces redundant track pose loads (read once per Gaussian)
    - Post-backward gradient zeroing for invisible tracks

    Transformation pipeline:
    1. Read track pose from interpolated poses array
    2. Transform Gaussian position: p' = track_pose * p_local
    3. Transform Gaussian rotation: q' = track_rotation * q_local
    4. Apply activation to density (sigmoid), or set to 0 if track invisible
    """

    def __init__(self, layers_config: LayersConfig, layer_index: int):
        """Initialize rigid layer handler.

        Args:
            layers_config: Complete configuration for all layers
            layer_index: Index of this layer in the layers list
        """
        super().__init__(layers_config, layer_index)
        layer_config = self.layer_config
        assert isinstance(layer_config, LayerConfigRigid)

    def get_collector_configurations(self) -> List[CollectorConfiguration]:
        configurations = super().get_collector_configurations()
        assert len(configurations) == 1
        parameters = list(configurations[0].parameters)

        assert parameters[0] == "Collector_Copy<3>"
        assert parameters[1] == "RotationsCollector_Normalize"
        assert parameters[3] == "DensitiesCollector_Sigmoid"

        # Replace the positions, rotations and density collectors with one that does all three.
        del parameters[3]
        parameters[0:2] = ["PositionsRotationsDensitiesCollector_Tracks"]

        return [CollectorConfiguration(parameters=tuple(parameters))]

    def validate(self, layers_data: LayersData, layer_data: LayerDataBase) -> None:
        super().validate(layers_data, layer_data)
        assert isinstance(layer_data, LayerDataRigid)

        # Validate the number of Gaussians.
        nb_gaussians = layer_data.positions.shape[0]
        assert nb_gaussians == layer_data.tracks_ids.shape[0]

        # Validate the component dimensions.
        assert layer_data.poses.dim() == 2
        assert layer_data.poses.shape[0] == layer_data.keep_mask.shape[0]
        assert layer_data.poses.shape[1] == 7
        assert layer_data.poses.dtype == torch.float32
        assert layer_data.keep_mask.dim() == 1
        assert layer_data.keep_mask.dtype == torch.bool
        assert layer_data.tracks_ids.dim() == 1
        assert layer_data.tracks_ids.dtype == torch.int32

    def get_trainable_input_tensors(self, layer_data: LayerDataBase) -> List[torch.Tensor]:
        input_tensors = super().get_trainable_input_tensors(layer_data)

        assert isinstance(layer_data, LayerDataRigid)
        input_tensors.append(layer_data.poses)

        return input_tensors

    def get_kernel_parameters(
        self,
        layers_data: LayersData,
        layer_data: LayerDataBase,
        output_params: List[Any],
        tensor_func: Callable[[torch.Tensor, bool], Tuple[torch.Tensor, Tuple[torch.Tensor]]],
    ) -> KernelParameters:
        base_kernel_parameters = (
            super().get_kernel_parameters(layers_data, layer_data, output_params, tensor_func).parameters
        )

        slang_module = self.kernels[0].slang_module

        assert isinstance(layer_data, LayerDataRigid)

        # All tensors here will not have their gradients zeroed.
        # The only tricky part is the densities which will not have their gradient written back
        # for indices where keep_mask is False.
        # It doesn't matter, because we will zero the gradients for these indices in the backward pass
        # for all the parameters, including the densities.
        input_positions = base_kernel_parameters[0].input
        output_positions = base_kernel_parameters[0].output
        input_rotations = base_kernel_parameters[1].input
        output_rotations = base_kernel_parameters[1].output
        input_densities = base_kernel_parameters[3].input
        output_densities = base_kernel_parameters[3].output
        tracks_poses = tensor_func(layer_data.poses, True)
        tracks_ids = layer_data.tracks_ids.contiguous()
        keep_mask = layer_data.keep_mask.contiguous()

        tracks_collector = slang_module.PositionsRotationsDensitiesCollector_Tracks(
            input_positions=input_positions,
            input_rotations=input_rotations,
            input_densities=input_densities,
            output_positions=output_positions,
            output_rotations=output_rotations,
            output_densities=output_densities,
            tracks_poses=tracks_poses,
            tracks_ids=tracks_ids,
            keep_mask=keep_mask,
        )

        # Replace the positions, rotations and density collectors with one that does all three.
        del base_kernel_parameters[3]
        del base_kernel_parameters[1]
        base_kernel_parameters[0] = tracks_collector

        return KernelParameters(kernel=self.kernels[0], parameters=base_kernel_parameters)

    def post_process_backward(self, kernel_parameters: KernelParameters) -> None:
        # Tracks which are not visible have their gaussian's density set to 0.
        # To make sure they don't affect training, we zero the gradients for these gaussians as well.
        # It's unclear if it's actually necessary, or if setting the density to 0 is enough.
        with profile("post_process_backward"):
            super().post_process_backward(kernel_parameters)

            slang_module = kernel_parameters.kernel.slang_module
            parameters = kernel_parameters.parameters

            tracks_ids = parameters[0].tracks_ids
            keep_mask = parameters[0].keep_mask
            positions_grad = parameters[0].input_positions[1][0]
            rotations_grad = parameters[0].input_rotations[1][0]
            scales_grad = parameters[1].input[1][0]
            densities_grad = parameters[0].input_densities[1][0]

            empty_tensor = None
            param_index = 2

            def get_gradient(dim: int) -> torch.Tensor:
                nonlocal empty_tensor
                nonlocal param_index
                if dim > 0:
                    tensor = parameters[param_index].input[1][0]
                    param_index += 1
                else:
                    if empty_tensor is None:
                        empty_tensor = torch.empty(0, device=positions_grad.device)
                    tensor = empty_tensor
                return tensor

            extra_signal_grad = get_gradient(self.layers_config.extra_signal_dim)
            camera_extra_signal_grad = get_gradient(self.layers_config.camera_extra_signal_dim)
            lidar_extra_signal_grad = get_gradient(self.layers_config.lidar_extra_signal_dim)
            features_albedo_grad = parameters[param_index].albedo[1][0]
            features_specular_grad = parameters[param_index].specular[1][0]

            launch_zero_gradients_for_disabled_tracks(
                slang_module,
                tracks_ids,
                keep_mask,
                (
                    positions_grad,
                    rotations_grad,
                    scales_grad,
                    densities_grad,
                    extra_signal_grad,
                    camera_extra_signal_grad,
                    lidar_extra_signal_grad,
                    features_albedo_grad,
                    features_specular_grad,
                ),
            )


class _LayerHandlerDeformable(_LayerHandlerRigid):
    """Handler for deformable layers with learned per-Gaussian deformations.

    Extends rigid layer with:
    - Optional deformation offsets (positions and rotations)
    - Applied in local space before rigid track transformation
    - Dual kernel support: Use rigid kernel when deformations are None, deformable otherwise

    Deformation pipeline when active:
    1. Apply deformation: p_local' = p_local + deform_position
    2. Apply deformation: q_local' = deform_rotation * q_local
    3. Apply rigid track transform: p_world = track_pose * p_local'
    4. Apply rigid track transform: q_world = track_rotation * q_local'

    Note: Deformation rotations are stored as offsets from identity:
    - deform_rotation = [x, y, z, w_offset] where w_full = w_offset + 1.0

    This allows learning small deformations around the identity rotation.
    """

    def __init__(self, layers_config: LayersConfig, layer_index: int):
        """Initialize deformable layer handler.

        Args:
            layers_config: Complete configuration for all layers
            layer_index: Index of this layer in the layers list
        """
        super().__init__(layers_config, layer_index)
        layer_config = self.layer_config
        assert isinstance(layer_config, LayerConfigDeformable)

    def get_collector_configurations(self) -> List[CollectorConfiguration]:
        configurations = super().get_collector_configurations()
        assert len(configurations) == 1
        parameters = list(configurations[0].parameters)

        assert parameters[0] == "PositionsRotationsDensitiesCollector_Tracks"

        # Replace the tracks collector by the deformable one.
        parameters[0] = "PositionsRotationsDensitiesCollector_Deformable"

        # We want both the rigid and the deformable collector kernels so we can
        # run efficiently when deformation is not applied.
        return [*configurations, CollectorConfiguration(parameters=tuple(parameters))]

    def validate(self, layers_data: LayersData, layer_data: LayerDataBase) -> None:
        super().validate(layers_data, layer_data)
        assert isinstance(layer_data, LayerDataDeformable)

        if layer_data.deform_positions is None:
            assert layer_data.deform_rotations is None
        else:
            assert layer_data.deform_rotations is not None

            # Validate the number of Gaussians.
            nb_gaussians = layer_data.positions.shape[0]
            assert nb_gaussians == layer_data.deform_positions.shape[0]
            assert nb_gaussians == layer_data.deform_rotations.shape[0]

            # Validate the component dimensions.
            assert layer_data.deform_positions.shape[1] == 3
            assert layer_data.deform_positions.dtype == torch.float32
            assert layer_data.deform_rotations.shape[1] == 4
            assert layer_data.deform_rotations.dtype == torch.float32

    def get_trainable_input_tensors(self, layer_data: LayerDataBase) -> List[torch.Tensor]:
        input_tensors = super().get_trainable_input_tensors(layer_data)

        assert isinstance(layer_data, LayerDataDeformable)
        if layer_data.deform_positions is not None:
            input_tensors.append(layer_data.deform_positions)
        if layer_data.deform_rotations is not None:
            input_tensors.append(layer_data.deform_rotations)

        return input_tensors

    def get_kernel_parameters(
        self,
        layers_data: LayersData,
        layer_data: LayerDataBase,
        output_params: List[Any],
        tensor_func: Callable[[torch.Tensor, bool], Tuple[torch.Tensor, Tuple[torch.Tensor]]],
    ) -> KernelParameters:
        base_kernel_parameters = (
            super().get_kernel_parameters(layers_data, layer_data, output_params, tensor_func).parameters
        )

        assert isinstance(layer_data, LayerDataDeformable)

        tracks_collector = base_kernel_parameters[0]

        if layer_data.deform_positions is None:
            # Use the rigid collector kernel.
            kernel = self.kernels[0]
        else:
            # Use the deformable collector kernel.
            kernel = self.kernels[1]

            assert layer_data.deform_positions is not None
            assert layer_data.deform_rotations is not None
            deformable_collector = kernel.slang_module.PositionsRotationsDensitiesCollector_Deformable(
                input_positions=tracks_collector.input_positions,
                input_rotations=tracks_collector.input_rotations,
                input_densities=tracks_collector.input_densities,
                output_positions=tracks_collector.output_positions,
                output_rotations=tracks_collector.output_rotations,
                output_densities=tracks_collector.output_densities,
                tracks_poses=tracks_collector.tracks_poses,
                tracks_ids=tracks_collector.tracks_ids,
                keep_mask=tracks_collector.keep_mask,
                deform_positions=tensor_func(layer_data.deform_positions, False),
                deform_rotations=tensor_func(layer_data.deform_rotations, False),
            )

            # Replace the tracks collector.
            base_kernel_parameters[0] = deformable_collector

        return KernelParameters(kernel=kernel, parameters=base_kernel_parameters)


class CollectorFunction(torch.autograd.Function):
    """Autograd function for collecting and transforming Gaussian parameters.

    This is the main workhorse that orchestrates GPU kernel execution for
    all layers, applying activations and transformations in parallel.

    Forward pass:
    1. Allocate concatenated output tensors for all layers
    2. For each layer:
       - Get kernel parameters from layer handler
       - Launch GPU kernel to process all Gaussians in parallel
       - Write to corresponding slice of output tensors
    3. Save inputs and outputs for backward pass

    Backward pass:
    1. Make output gradients contiguous
    2. For each layer:
       - Get kernel parameters with gradient tensors
       - Launch backward GPU kernel
       - Propagate gradients to inputs
       - Post-process (e.g., zero gradients for invisible tracks)
    3. Return input gradients

    Thread configuration:
    - Forward: 256 threads/block (conservative for wide per-Gaussian signals)
    - Backward: 128 threads/block (uses more registers, needs lower occupancy)

    Memory layout:
    - All tensors are made contiguous before kernel launch for coalesced access
    - Outputs are pre-allocated as single contiguous blocks, layers write to slices
    """

    # We use fewer threads for the backward pass because the kernel uses more registers
    # (more intermediate values to store) and we can reach kernel resources limits and
    # not be able to launch the kernel.
    _FORWARD_THREADS_PER_BLOCK = 256
    _BACKWARD_THREADS_PER_BLOCK = 128

    @staticmethod
    def _div_up(a: int, b: int) -> int:
        return (a + b - 1) // b

    @staticmethod
    def _fwd_param_raw(tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        # SlangTorch integration passes pairs of tensors, not individual tensors, in the
        # code wrapping the Slang kernels to re-use code between forward and backward.
        # If a single tensor is passed in the forward pass, phony tensors are allocated.
        # To avoid this, we pass pairs of tensors in the forward pass.  The second of the pair
        # is never accessed in the forward pass.
        #
        # See https://github.com/shader-slang/slang-torch/blob/c917325a7a8de63714a36a430f9570dabe67037d/slangtorch/util/builtin_wrappers.py#L31

        # Things should pretty much already be contiguous.
        # FIXME: Figure out optimizations for non-contiguous tensors to avoid the copy.
        tensor = tensor.contiguous()
        return (tensor, (tensor,))

    @staticmethod
    def _bwd_param_raw(tensor: torch.Tensor, zero_gradients: bool) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        # Allocate a new tensor for the gradient.
        # Note that when tensors requiring gradients are read using loadOnce(),
        # we don't need to zero the gradients, they will be directly assigned instead
        # of accumulated.
        # This is only true if all values from the input tensors are read using loadOnce().
        # If some values are not read at all, then we need to zero the gradients.
        if zero_gradients:
            gradient = torch.zeros_like(tensor)
        else:
            gradient = torch.empty_like(tensor)
        return (tensor, (gradient,))

    @staticmethod
    def forward_tensor_func(tensor: torch.Tensor, zero_gradients: bool) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        return CollectorFunction._fwd_param_raw(tensor)

    @staticmethod
    def get_backward_tensor_func(
        input_grads: List[torch.Tensor],
    ) -> Callable[[torch.Tensor, bool], Tuple[torch.Tensor, Tuple[torch.Tensor]]]:
        def _backward_tensor_func(
            tensor: torch.Tensor, zero_gradients: bool
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
            param = CollectorFunction._bwd_param_raw(tensor, zero_gradients)
            input_grads.append(param[1][0])
            return param

        return _backward_tensor_func

    @staticmethod
    def forward(ctx, collector, layers_data, layer_indices, out, segments, *input_tensors):
        collector.validate(layers_data, layer_indices)

        device = input_tensors[0].device
        # A caller without a plan gets whole layers, which use the dense kernel.
        if segments is None:
            segments = [whole_layer_segments(ld.positions.shape[0]) for ld in layers_data.layers]
        else:
            assert len(segments) == len(layers_data.layers), (
                f"got {len(segments)} layer segment lists for {len(layers_data.layers)} layers"
            )

        sh_is_fp16 = collector.layers_config.sh_features_dtype == "fp16"
        with profile("allocate_output"):
            nb_gaussians = sum(layer_data.positions.shape[0] for layer_data in layers_data.layers)
            if out is not None:
                # Caller-owned destination: write the collected layers straight into
                # its buffers instead of allocating a second set to copy out of.
                # Inference-only, because the destination is an aliased view whose
                # gradients would not flow back to it; `collect` enforces that.
                (
                    output_positions,
                    output_rotations,
                    output_scales,
                    output_densities,
                    output_extra_signal,
                    output_camera_extra_signal,
                    output_lidar_extra_signal,
                    output_spherical_features,
                ) = (
                    out.positions,
                    out.rotations,
                    out.scales,
                    out.densities,
                    out.extra_signal,
                    out.camera_extra_signal,
                    out.lidar_extra_signal,
                    out.features,
                )
                for name, tensor in (
                    ("positions", output_positions),
                    ("rotations", output_rotations),
                    ("scales", output_scales),
                    ("densities", output_densities),
                    ("extra_signal", output_extra_signal),
                    ("camera_extra_signal", output_camera_extra_signal),
                    ("lidar_extra_signal", output_lidar_extra_signal),
                    ("features", output_spherical_features),
                ):
                    assert tensor.shape[0] >= nb_gaussians, (
                        f"collect destination '{name}' holds {tensor.shape[0]} rows, must have room for "
                        f"{nb_gaussians} collected rows"
                    )
                    assert tensor.is_contiguous(), f"collect destination '{name}' must be contiguous"
                    assert tensor.device == device, f"collect destination '{name}' is on {tensor.device}, not {device}"
                assert output_spherical_features.dtype == (torch.float16 if sh_is_fp16 else torch.float32)
            else:
                output_positions = torch.empty(nb_gaussians, 3, device=device)
                output_rotations = torch.empty(nb_gaussians, 4, device=device)
                output_scales = torch.empty(nb_gaussians, 3, device=device)
                output_densities = torch.empty(nb_gaussians, 1, device=device)
                output_extra_signal = torch.empty(nb_gaussians, collector.layers_config.extra_signal_dim, device=device)
                output_camera_extra_signal = torch.empty(
                    nb_gaussians, collector.layers_config.camera_extra_signal_dim, device=device
                )
                output_lidar_extra_signal = torch.empty(
                    nb_gaussians, collector.layers_config.lidar_extra_signal_dim, device=device
                )
                output_spherical_features = torch.empty(
                    nb_gaussians,
                    collector.layers_config.albedo_dim + collector.layers_config.specular_dim,
                    dtype=torch.float16 if sh_is_fp16 else torch.float32,
                    device=device,
                )

        # HalfDiffView (fp16) binds as a flat (primal, diff) 2-tuple; DiffTensorView (fp32) is
        # nested (primal, (diff,)). slangtorch decides this per type, so we shape it per dtype.
        sh_output_param = (
            (output_spherical_features, output_spherical_features)
            if sh_is_fp16
            else CollectorFunction._fwd_param_raw(output_spherical_features)
        )

        output_params = [
            CollectorFunction._fwd_param_raw(output_positions),
            CollectorFunction._fwd_param_raw(output_rotations),
            CollectorFunction._fwd_param_raw(output_scales),
            CollectorFunction._fwd_param_raw(output_densities),
            CollectorFunction._fwd_param_raw(output_extra_signal),
            CollectorFunction._fwd_param_raw(output_camera_extra_signal),
            CollectorFunction._fwd_param_raw(output_lidar_extra_signal),
            sh_output_param,
        ]

        with profile("forward"):
            threads_per_block = CollectorFunction._FORWARD_THREADS_PER_BLOCK

            # Getting slices from the output tensor has overhead,
            # so we compute the offset in the kernel instead.

            offset = 0
            for i in range(len(layers_data.layers)):
                layer_data = layers_data.layers[i]
                handler_idx = layer_indices[i] if layer_indices is not None else i
                layer_handler = collector.layer_handlers[handler_idx]

                count = layer_data.positions.shape[0]
                if count == 0:
                    continue

                # One thread per launched slot; the segments turn a slot into the layer-local
                # row it writes, which for a whole layer is the slot itself.
                layer_segments = segments[i]
                launched = layer_segments.active_count
                if launched == 0:
                    offset += count
                    continue

                params = layer_handler.get_kernel_parameters(
                    layers_data, layer_data, output_params, CollectorFunction.forward_tensor_func
                )

                blocks_per_grid = CollectorFunction._div_up(launched, threads_per_block)
                assert blocks_per_grid > 0

                with profile("slangtorch_call_direct"):
                    segmented = layer_segments.n_segments > 0
                    kernel = params.kernel.kernels[int(segmented)]
                    segment_params: tuple[Any, ...] = ()
                    if segmented:
                        segment_params = (
                            layer_segments.starts,
                            layer_segments.window_size_prefix,
                            layer_segments.n_segments,
                        )
                    kernel.fn_handle(
                        (threads_per_block, 1, 1),
                        (blocks_per_grid, 1, 1),
                        offset,
                        launched,
                        *segment_params,
                        *params.parameters,
                    )

                offset += count

        with profile("post"):
            output_tensors = [
                output_positions,
                output_rotations,
                output_scales,
                output_densities,
                output_extra_signal,
                output_camera_extra_signal,
                output_lidar_extra_signal,
                output_spherical_features,
            ]
            ctx.save_for_backward(*output_tensors, *input_tensors)
            ctx.collector = collector
            ctx.layers_data = layers_data
            ctx.layer_indices = layer_indices

        return tuple(output_tensors)

    @staticmethod
    def backward(ctx, *output_grads):
        collector = ctx.collector
        layers_data = ctx.layers_data
        layer_indices = ctx.layer_indices

        # Reset the context to avoid keeping references to the tensors.
        ctx.collector = None
        ctx.layers_data = None
        ctx.layer_indices = None

        nb_components = len(output_grads)
        assert nb_components == 8

        saved_tensors = ctx.saved_tensors
        with profile("contiguous"):
            output_grads = [output_grad.contiguous() for output_grad in output_grads]

        output_tensors = saved_tensors[0:nb_components]

        # Slot 7 (SH features) is flat (primal, diff) for HalfDiffView, nested for DiffTensorView.
        sh_is_fp16 = collector.layers_config.sh_features_dtype == "fp16"
        output_params = []
        for i in range(nb_components):
            if i == 7 and sh_is_fp16:
                output_params.append((output_tensors[i], output_grads[i]))
            else:
                output_params.append((output_tensors[i], (output_grads[i],)))

        with profile("backward"):
            threads_per_block = CollectorFunction._BACKWARD_THREADS_PER_BLOCK

            input_grads = []
            backward_tensor_func = CollectorFunction.get_backward_tensor_func(input_grads)

            offset = 0
            for i in range(len(layers_data.layers)):
                layer_data = layers_data.layers[i]
                handler_idx = layer_indices[i] if layer_indices is not None else i
                layer_handler = collector.layer_handlers[handler_idx]

                params = layer_handler.get_kernel_parameters(
                    layers_data, layer_data, output_params, backward_tensor_func
                )

                count = layer_data.positions.shape[0]
                if count == 0:
                    continue
                blocks_per_grid = CollectorFunction._div_up(count, threads_per_block)
                assert blocks_per_grid > 0

                # The backward only ever runs dense: a windowed collect requires no_grad.
                with profile("slangtorch_call_direct"):
                    params.kernel.kernels[0].bwd_wrapped_fn.fn_handle(
                        (threads_per_block, 1, 1),
                        (blocks_per_grid, 1, 1),
                        offset,
                        count,
                        *params.parameters,
                    )

                layer_handler.post_process_backward(params)

                offset += count

        return (None, None, None, None, None, *input_grads)


class TracksCalibFunction(torch.autograd.Function):
    """Autograd bridge for the supported track-calibration pose kernels."""

    _FORWARD_THREADS_PER_BLOCK = 512
    _BACKWARD_THREADS_PER_BLOCK = 512

    @staticmethod
    def forward(ctx, slang_module, tracks_poses: torch.Tensor, tracks_calib_data: TracksCalibData, *input_tensors):
        with profile("allocate_output"):
            output_poses = torch.empty_like(tracks_poses)

        count = tracks_poses.shape[0]
        # This will have been validated before.
        assert count > 0

        with profile("forward"):
            threads_per_block = (
                256
                if type(tracks_calib_data) is ElevationAwareUnicycleTracksCalibData
                else TracksCalibFunction._FORWARD_THREADS_PER_BLOCK
            )
            blocks_per_grid = CollectorFunction._div_up(count, threads_per_block)
            assert blocks_per_grid > 0

            if type(tracks_calib_data) is DirectTracksCalibData:
                tracks_calib = slang_module.DirectTracksCalib(
                    gradient_mask=tracks_calib_data.gradient_mask.contiguous(),
                    tracks_poses=tracks_poses.contiguous(),
                    tracks_delta_q=CollectorFunction._fwd_param_raw(tracks_calib_data.tracks_delta_q),
                    tracks_delta_t=CollectorFunction._fwd_param_raw(tracks_calib_data.tracks_delta_t),
                    left_multiply=tracks_calib_data.left_multiply,
                )

                slang_module.tracks_calib_direct.fn_handle(
                    (threads_per_block, 1, 1),
                    (blocks_per_grid, 1, 1),
                    count,
                    tracks_calib,
                    CollectorFunction._fwd_param_raw(output_poses),
                )
            elif type(tracks_calib_data) is ElevationAwareUnicycleTracksCalibData:
                tracks_calib = slang_module.ElevationAwareUnicycleTracksCalib(
                    xy=CollectorFunction._fwd_param_raw(tracks_calib_data.xy.contiguous()),
                    height=CollectorFunction._fwd_param_raw(tracks_calib_data.height.contiguous()),
                    theta=CollectorFunction._fwd_param_raw(tracks_calib_data.theta.contiguous()),
                    pitch=CollectorFunction._fwd_param_raw(tracks_calib_data.pitch.contiguous()),
                    gradient_mask=tracks_calib_data.pose_gradient_mask.contiguous(),
                    tracks_idx=tracks_calib_data.tracks_idx.contiguous(),
                    ground_t=tracks_calib_data.ground_t.contiguous(),
                    ground_q=tracks_calib_data.ground_q.contiguous(),
                    tclc=tracks_calib_data.Tclc_tquat.view(-1).contiguous(),
                )

                slang_module.tracks_calib_unicycle_pose.fn_handle(
                    (threads_per_block, 1, 1),
                    (blocks_per_grid, 1, 1),
                    count,
                    tracks_calib,
                    CollectorFunction._fwd_param_raw(output_poses),
                )
            else:
                raise ValueError(f"Unsupported tracks calib data: {tracks_calib_data}")

        with profile("post"):
            ctx.save_for_backward(output_poses, tracks_poses, *input_tensors)
            ctx.slang_module = slang_module
            ctx.tracks_calib_data = tracks_calib_data

        return output_poses

    @staticmethod
    def backward(ctx, *output_grads):
        slang_module = ctx.slang_module
        tracks_calib_data = ctx.tracks_calib_data

        # Reset the context to avoid keeping references to the tensors.
        ctx.slang_module = None
        ctx.tracks_calib_data = None

        saved_tensors = ctx.saved_tensors
        tracks_poses = saved_tensors[1]
        with profile("contiguous"):
            output_grads = [output_grad.contiguous() for output_grad in output_grads]

        output_tensor = saved_tensors[0]

        output_param = (output_tensor, (output_grads[0],))

        with profile("backward"):
            input_grads = []
            backward_tensor_func = CollectorFunction.get_backward_tensor_func(input_grads)

            # All tensors in this function don't need to have their gradients zeroed.
            def tf(tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
                return backward_tensor_func(tensor, False)

            if type(tracks_calib_data) is DirectTracksCalibData:
                tracks_calib = slang_module.DirectTracksCalib(
                    gradient_mask=tracks_calib_data.gradient_mask.contiguous(),
                    tracks_poses=tracks_poses.contiguous(),
                    tracks_delta_q=tf(tracks_calib_data.tracks_delta_q),
                    tracks_delta_t=tf(tracks_calib_data.tracks_delta_t),
                    left_multiply=tracks_calib_data.left_multiply,
                )

                count = tracks_poses.shape[0]
                # This will have been validated before.
                assert count > 0
                threads_per_block = TracksCalibFunction._BACKWARD_THREADS_PER_BLOCK
                blocks_per_grid = CollectorFunction._div_up(count, threads_per_block)
                assert blocks_per_grid > 0

                slang_module.tracks_calib_direct.bwd_wrapped_fn.fn_handle(
                    (threads_per_block, 1, 1),
                    (blocks_per_grid, 1, 1),
                    count,
                    tracks_calib,
                    output_param,
                )
            elif type(tracks_calib_data) is ElevationAwareUnicycleTracksCalibData:
                tracks_calib = slang_module.ElevationAwareUnicycleTracksCalib(
                    xy=tf(tracks_calib_data.xy.contiguous()),
                    height=tf(tracks_calib_data.height.contiguous()),
                    theta=tf(tracks_calib_data.theta.contiguous()),
                    pitch=tf(tracks_calib_data.pitch.contiguous()),
                    gradient_mask=tracks_calib_data.pose_gradient_mask.contiguous(),
                    tracks_idx=tracks_calib_data.tracks_idx.contiguous(),
                    ground_t=tracks_calib_data.ground_t.contiguous(),
                    ground_q=tracks_calib_data.ground_q.contiguous(),
                    tclc=tracks_calib_data.Tclc_tquat.view(-1).contiguous(),
                )
                threads_per_block = 128
                count = tracks_poses.shape[0]
                blocks_per_grid = CollectorFunction._div_up(count, threads_per_block)
                slang_module.tracks_calib_unicycle_pose.bwd_wrapped_fn.fn_handle(
                    (threads_per_block, 1, 1),
                    (blocks_per_grid, 1, 1),
                    count,
                    tracks_calib,
                    output_param,
                )
            else:
                raise ValueError(f"Unsupported tracks calib data: {tracks_calib_data}")

        return (None, None, None, *input_grads)


class UnicycleConsistencyFunction(torch.autograd.Function):
    """Autograd bridge for the unicycle motion-consistency residuals."""

    _FORWARD_THREADS_PER_BLOCK = 256
    _BACKWARD_THREADS_PER_BLOCK = 128

    @staticmethod
    def forward(
        ctx,
        slang_module,
        xy,
        height,
        theta,
        pitch,
        velocity,
        theta_velocity,
        pitch_velocity,
        keep,
        pair_start,
        time_delta_s,
        pitch_slope_clamp_rad,
    ):
        tensors = tuple(
            tensor.contiguous()
            for tensor in (
                xy,
                height,
                theta,
                pitch,
                velocity,
                theta_velocity,
                pitch_velocity,
                keep,
                pair_start,
                time_delta_s,
            )
        )
        (
            xy,
            height,
            theta,
            pitch,
            velocity,
            theta_velocity,
            pitch_velocity,
            keep,
            pair_start,
            time_delta_s,
        ) = tensors
        consistency = torch.empty((velocity.shape[0], 5), device=velocity.device, dtype=velocity.dtype)
        blocks = CollectorFunction._div_up(velocity.shape[0], UnicycleConsistencyFunction._FORWARD_THREADS_PER_BLOCK)
        if blocks > 0:
            slang_module.tracks_calib_unicycle_consistency.fn_handle(
                (UnicycleConsistencyFunction._FORWARD_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                CollectorFunction._fwd_param_raw(xy),
                CollectorFunction._fwd_param_raw(height),
                CollectorFunction._fwd_param_raw(theta),
                CollectorFunction._fwd_param_raw(pitch),
                CollectorFunction._fwd_param_raw(velocity),
                CollectorFunction._fwd_param_raw(theta_velocity),
                CollectorFunction._fwd_param_raw(pitch_velocity),
                keep,
                pair_start,
                time_delta_s,
                pitch_slope_clamp_rad,
                CollectorFunction._fwd_param_raw(consistency),
            )
        ctx.save_for_backward(*tensors, consistency)
        ctx.slang_module = slang_module
        ctx.pitch_slope_clamp_rad = pitch_slope_clamp_rad
        return consistency

    @staticmethod
    def backward(ctx, grad_output):
        (
            xy,
            height,
            theta,
            pitch,
            velocity,
            theta_velocity,
            pitch_velocity,
            keep,
            pair_start,
            time_delta_s,
            consistency,
        ) = ctx.saved_tensors
        grads = [
            torch.zeros_like(tensor) for tensor in (xy, height, theta, pitch, velocity, theta_velocity, pitch_velocity)
        ]
        blocks = CollectorFunction._div_up(velocity.shape[0], UnicycleConsistencyFunction._BACKWARD_THREADS_PER_BLOCK)
        if blocks > 0:
            diff_inputs = [
                (tensor, (grad,))
                for tensor, grad in zip((xy, height, theta, pitch, velocity, theta_velocity, pitch_velocity), grads)
            ]
            slang_module = ctx.slang_module
            slang_module.tracks_calib_unicycle_consistency.bwd_wrapped_fn.fn_handle(
                (UnicycleConsistencyFunction._BACKWARD_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                *diff_inputs,
                keep,
                pair_start,
                time_delta_s,
                ctx.pitch_slope_clamp_rad,
                (consistency, (grad_output.contiguous().clone(),)),
            )
        return (None, *grads, None, None, None, None)


class UnicycleSmoothnessFunction(torch.autograd.Function):
    """Autograd bridge for the unicycle acceleration residuals."""

    _FORWARD_THREADS_PER_BLOCK = 256
    _BACKWARD_THREADS_PER_BLOCK = 128

    @staticmethod
    def forward(
        ctx,
        slang_module,
        height,
        velocity,
        theta_velocity,
        pitch_velocity,
        keep,
        pair_start,
        triplet_start,
        time_delta_s,
        time_delta_v_s,
    ):
        tensors = tuple(
            tensor.contiguous()
            for tensor in (
                height,
                velocity,
                theta_velocity,
                pitch_velocity,
                keep,
                pair_start,
                triplet_start,
                time_delta_s,
                time_delta_v_s,
            )
        )
        (
            height,
            velocity,
            theta_velocity,
            pitch_velocity,
            keep,
            pair_start,
            triplet_start,
            time_delta_s,
            time_delta_v_s,
        ) = tensors
        smoothness = torch.empty((triplet_start.shape[0], 4), device=velocity.device, dtype=velocity.dtype)
        blocks = CollectorFunction._div_up(
            triplet_start.shape[0], UnicycleSmoothnessFunction._FORWARD_THREADS_PER_BLOCK
        )
        if blocks > 0:
            slang_module.tracks_calib_unicycle_smoothness.fn_handle(
                (UnicycleSmoothnessFunction._FORWARD_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                CollectorFunction._fwd_param_raw(height),
                CollectorFunction._fwd_param_raw(velocity),
                CollectorFunction._fwd_param_raw(theta_velocity),
                CollectorFunction._fwd_param_raw(pitch_velocity),
                keep,
                pair_start,
                triplet_start,
                time_delta_s,
                time_delta_v_s,
                CollectorFunction._fwd_param_raw(smoothness),
            )
        ctx.save_for_backward(*tensors, smoothness)
        ctx.slang_module = slang_module
        return smoothness

    @staticmethod
    def backward(ctx, grad_output):
        (
            height,
            velocity,
            theta_velocity,
            pitch_velocity,
            keep,
            pair_start,
            triplet_start,
            time_delta_s,
            time_delta_v_s,
            smoothness,
        ) = ctx.saved_tensors
        grads = [torch.zeros_like(tensor) for tensor in (height, velocity, theta_velocity, pitch_velocity)]
        blocks = CollectorFunction._div_up(
            triplet_start.shape[0], UnicycleSmoothnessFunction._BACKWARD_THREADS_PER_BLOCK
        )
        if blocks > 0:
            diff_inputs = [
                (tensor, (grad,)) for tensor, grad in zip((height, velocity, theta_velocity, pitch_velocity), grads)
            ]
            ctx.slang_module.tracks_calib_unicycle_smoothness.bwd_wrapped_fn.fn_handle(
                (UnicycleSmoothnessFunction._BACKWARD_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                *diff_inputs,
                keep,
                pair_start,
                triplet_start,
                time_delta_s,
                time_delta_v_s,
                (smoothness, (grad_output.contiguous().clone(),)),
            )
        return (None, *grads, None, None, None, None, None)


class UnicyclePoseRegularizationFunction(torch.autograd.Function):
    """Autograd bridge for the unicycle position regularizer."""

    _FORWARD_THREADS_PER_BLOCK = 256
    _BACKWARD_THREADS_PER_BLOCK = 128

    @staticmethod
    def forward(ctx, slang_module, xy, height, keep, initial_xy, initial_height):
        xy, height, keep, initial_xy, initial_height = (
            tensor.contiguous() for tensor in (xy, height, keep, initial_xy, initial_height)
        )
        regularization = torch.empty_like(height)
        blocks = CollectorFunction._div_up(
            height.shape[0], UnicyclePoseRegularizationFunction._FORWARD_THREADS_PER_BLOCK
        )
        if blocks > 0:
            slang_module.tracks_calib_unicycle_pose_regularization.fn_handle(
                (UnicyclePoseRegularizationFunction._FORWARD_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                CollectorFunction._fwd_param_raw(xy),
                CollectorFunction._fwd_param_raw(height),
                keep,
                initial_xy,
                initial_height,
                CollectorFunction._fwd_param_raw(regularization),
            )
        ctx.save_for_backward(xy, height, keep, initial_xy, initial_height, regularization)
        ctx.slang_module = slang_module
        return regularization

    @staticmethod
    def backward(ctx, grad_output):
        xy, height, keep, initial_xy, initial_height, regularization = ctx.saved_tensors
        grad_xy = torch.empty_like(xy)
        grad_height = torch.empty_like(height)
        blocks = CollectorFunction._div_up(
            height.shape[0], UnicyclePoseRegularizationFunction._BACKWARD_THREADS_PER_BLOCK
        )
        if blocks > 0:
            ctx.slang_module.tracks_calib_unicycle_pose_regularization.bwd_wrapped_fn.fn_handle(
                (UnicyclePoseRegularizationFunction._BACKWARD_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (xy, (grad_xy,)),
                (height, (grad_height,)),
                keep,
                initial_xy,
                initial_height,
                (regularization, (grad_output.contiguous().clone(),)),
            )
        return None, grad_xy, grad_height, None, None, None


class InputEmbeddingFunction(torch.autograd.Function):
    """Compute input embeddings (xyz + instance_emb + time_emb) for hash grid.

    Prepares concatenated feature vectors suitable for hash grid input, combining:
    1. Spatial coordinates (3D): Contracted and normalized to [0,1]³
    2. Instance embedding (1D): Learned per-instance features
    3. Time embedding (1D): Remapped timestamp to normalized range

    Embedding pipeline:
    1. Contract world coordinates using scene contractor (AABB + optional contraction)
    2. Look up instance embedding weights by instance ID
    3. Compute time embedding from timestamp using per-instance ranges
    4. Concatenate: [contracted_xyz(3), instance_emb(1), time_emb(1)]

    Optional timestamp delta mode:
    - If timestamps_delta is provided, compute 3 variants per Gaussian:
      * Embedding at time t
      * Embedding at time t - delta
      * Embedding at time t + delta
    - Output shape becomes [num_gaussians*3, base_dim] instead of [num_gaussians, base_dim]
    - Used for temporal finite differences in deformation networks

    Gradient handling:
    - Instance embedding weights accumulate gradients (multiple Gaussians may share instance)
    - XYZ gradients flow back through contraction transform
    - Time embeddings have no learnable parameters (remapping is fixed)
    """

    _FORWARD_THREADS_PER_BLOCK = 512
    _BACKWARD_THREADS_PER_BLOCK = 512

    @staticmethod
    def _make_timestamps_fetcher(
        slang_module,
        timestamps_us: int | torch.Tensor,
        device: torch.device,
    ):
        if isinstance(timestamps_us, int):
            return slang_module.TimestampsGlobalFetcher(timestamp=timestamps_us)
        assert timestamps_us.dtype == torch.int64
        assert timestamps_us.device == device
        return slang_module.TimestampsPerTrackFetcher(timestamps=timestamps_us.contiguous())

    @staticmethod
    def _select_kernel(slang_module, *, per_track_timestamps: bool, with_timestamps_delta: bool):
        if with_timestamps_delta:
            return (
                slang_module.prepare_input_embeddings_per_track_with_timestamps_delta
                if per_track_timestamps
                else slang_module.prepare_input_embeddings_with_timestamps_delta
            )
        return (
            slang_module.prepare_input_embeddings_per_track
            if per_track_timestamps
            else slang_module.prepare_input_embeddings
        )

    @staticmethod
    def _get_timestamps_delta(timestamps_delta: torch.Tensor, device: torch.device) -> torch.Tensor:
        assert timestamps_delta.dtype == torch.int64
        assert timestamps_delta.device == device
        return timestamps_delta.contiguous()

    @staticmethod
    def forward(ctx, slang_module, input_embedding_data: InputEmbeddingData, *input_tensors):
        device = input_tensors[0].device
        # Determine instance embedding dimension (must be == 1)
        if type(input_embedding_data.instance_embedding_weights) is WeightedInstanceInputEmbeddingData:
            # Infer dimension from weights tensor
            instance_embedding_dim = input_embedding_data.instance_embedding_weights.instance_embedding_weights.shape[1]
            assert instance_embedding_dim == 1, (
                "instance embedding dimension must be 1 for now (would require template modifications in Slang kernels)"
            )
        else:
            raise ValueError(
                f"Unsupported instance embedding data: {type(input_embedding_data.instance_embedding_weights)}"
            )

        with profile("allocate_output"):
            num_gaussians = input_embedding_data.instance_idx.shape[0]
            base_output_dim = 3 + instance_embedding_dim + 1  # xyz + instance + time (1D)

            # If timestamps_delta is provided, output 3 full embeddings stacked along batch dimension
            # Output shape: [num_gaussians * 3, base_dim] with rows ordered as [t, t-delta, t+delta]
            n_output_rows = num_gaussians * 3 if input_embedding_data.timestamps_delta is not None else num_gaussians

            output = torch.empty((n_output_rows, base_output_dim), dtype=torch.float32, device=device)

        count = num_gaussians
        # This will have been validated before.
        assert count > 0

        with profile("forward"):
            # Setup scene contractor
            sc = input_embedding_data.scene_contractor
            # When degree is None (no contraction), norm is never computed, so this value is unused.
            # Using NaN makes accidental usage obvious since NaN propagates through calculations.
            degree_value = float(sc.degree) if sc.degree is not None else float("nan")
            scene_contractor = slang_module.SceneContractor(
                aabb_blb=sc.aabb_blb.contiguous(),
                aabb_trf=sc.aabb_trf.contiguous(),
                degree=degree_value,
                is_merf=sc.is_merf,
                has_contraction=sc.degree is not None,
            )

            # Setup instance embedding
            if type(input_embedding_data.instance_embedding_weights) is WeightedInstanceInputEmbeddingData:
                # Infer dimension from weights tensor
                weights_tensor = input_embedding_data.instance_embedding_weights.instance_embedding_weights
                instance_embedding_dim = weights_tensor.shape[1]
                assert instance_embedding_dim == 1, (
                    "instance embedding dimension must be 1 for now (would require template modifications in Slang kernels)"
                )
                instance_emb = slang_module.WeightedInstanceInputEmbedding(
                    instance_idx=input_embedding_data.instance_idx.contiguous(),
                    embedding_weights=CollectorFunction._fwd_param_raw(weights_tensor),
                    output=CollectorFunction._fwd_param_raw(output),
                )
            else:
                raise ValueError(
                    f"Unsupported instance embedding data: {type(input_embedding_data.instance_embedding_weights)}"
                )

            # Setup time embedding and select appropriate kernel
            if type(input_embedding_data.time_embedding_config) is IndividualRemapTimeInputEmbeddingConfig:
                time_emb = slang_module.IndividualRemapTimeInputEmbedding(
                    instance_idx=input_embedding_data.instance_idx.contiguous(),
                    timestamps_ranges=input_embedding_data.time_embedding_config.timestamps_us_ranges.contiguous(),
                    remap_min=input_embedding_data.time_embedding_config.remap_min,
                    remap_max=input_embedding_data.time_embedding_config.remap_max,
                )
            else:
                raise ValueError(
                    f"Unsupported time embedding config for InputEmbeddingData: {input_embedding_data.time_embedding_config}. "
                    f"Please use IndividualRemapTimeInputEmbeddingConfig."
                )

            timestamps_fetcher = InputEmbeddingFunction._make_timestamps_fetcher(
                slang_module,
                input_embedding_data.timestamps_us,
                device,
            )

            threads_per_block = InputEmbeddingFunction._FORWARD_THREADS_PER_BLOCK
            blocks_per_grid = CollectorFunction._div_up(count, threads_per_block)
            assert blocks_per_grid > 0

            # Prepare common arguments (grid dimensions + kernel parameters)
            args = [
                (threads_per_block, 1, 1),  # grid_dim
                (blocks_per_grid, 1, 1),  # block_dim
                count,
                CollectorFunction._fwd_param_raw(input_embedding_data.xyzs),
                input_embedding_data.instance_idx.contiguous(),
                scene_contractor,
                instance_emb,
                timestamps_fetcher,
                time_emb,
            ]

            # Select kernel and build argument list based on whether we have timestamps_delta
            per_track_timestamps = not isinstance(input_embedding_data.timestamps_us, int)
            kernel = InputEmbeddingFunction._select_kernel(
                slang_module,
                per_track_timestamps=per_track_timestamps,
                with_timestamps_delta=input_embedding_data.timestamps_delta is not None,
            )
            if input_embedding_data.timestamps_delta is not None:
                timestamps_delta_tensor = InputEmbeddingFunction._get_timestamps_delta(
                    input_embedding_data.timestamps_delta, device
                )
                args.append(timestamps_delta_tensor)
            kernel.fn_handle(*args, CollectorFunction._fwd_param_raw(output))

        with profile("post"):
            ctx.save_for_backward(output, *input_tensors)
            ctx.slang_module = slang_module
            ctx.input_embedding_data = input_embedding_data

        return output

    @staticmethod
    def backward(ctx, *output_grads):
        slang_module = ctx.slang_module
        input_embedding_data = ctx.input_embedding_data

        # Reset the context to avoid keeping references to the tensors.
        ctx.slang_module = None
        ctx.input_embedding_data = None

        saved_tensors = ctx.saved_tensors
        with profile("contiguous"):
            output_grads = [output_grad.contiguous() for output_grad in output_grads]

        output_tensor = saved_tensors[0]
        output_param = (output_tensor, (output_grads[0],))

        with profile("backward"):
            device = input_embedding_data.xyzs.device
            input_grads = []
            backward_tensor_func = CollectorFunction.get_backward_tensor_func(input_grads)

            # instance_embedding_weights need gradient zeroing because they're
            # read multiple times with loadEx() (once per variant),
            # so gradients are accumulated.
            # Other tensors don't need zeroing.
            def tf(tensor: torch.Tensor, zero_grad: bool = False) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
                return backward_tensor_func(tensor, zero_grad)

            # Setup scene contractor
            sc = input_embedding_data.scene_contractor
            # When degree is None (no contraction), norm is never computed, so this value is unused.
            # Using NaN makes accidental usage obvious since NaN propagates through calculations.
            degree_value = float(sc.degree) if sc.degree is not None else float("nan")
            scene_contractor = slang_module.SceneContractor(
                aabb_blb=sc.aabb_blb.contiguous(),
                aabb_trf=sc.aabb_trf.contiguous(),
                degree=degree_value,
                is_merf=sc.is_merf,
                has_contraction=sc.degree is not None,
            )
            xyzs = tf(input_embedding_data.xyzs)

            # Setup instance embedding
            if type(input_embedding_data.instance_embedding_weights) is WeightedInstanceInputEmbeddingData:
                # Validate dimension
                weights_tensor = input_embedding_data.instance_embedding_weights.instance_embedding_weights
                instance_embedding_dim = weights_tensor.shape[1]
                assert instance_embedding_dim == 1, (
                    "instance embedding dimension must be 1 for now (would require template modifications in Slang kernels)"
                )
                instance_emb = slang_module.WeightedInstanceInputEmbedding(
                    instance_idx=input_embedding_data.instance_idx.contiguous(),
                    embedding_weights=tf(weights_tensor, zero_grad=True),
                    output=output_param,
                )
            else:
                raise ValueError(
                    f"Unsupported instance embedding data: {type(input_embedding_data.instance_embedding_weights)}"
                )

            # Setup time embedding and launch appropriate backward kernel
            if type(input_embedding_data.time_embedding_config) is IndividualRemapTimeInputEmbeddingConfig:
                time_emb = slang_module.IndividualRemapTimeInputEmbedding(
                    instance_idx=input_embedding_data.instance_idx.contiguous(),
                    timestamps_ranges=input_embedding_data.time_embedding_config.timestamps_us_ranges.contiguous(),
                    remap_min=input_embedding_data.time_embedding_config.remap_min,
                    remap_max=input_embedding_data.time_embedding_config.remap_max,
                )
            else:
                raise ValueError(
                    f"Unsupported time embedding config for InputEmbeddingData: {input_embedding_data.time_embedding_config}. "
                    f"Please use IndividualRemapTimeInputEmbeddingConfig."
                )

            timestamps_fetcher = InputEmbeddingFunction._make_timestamps_fetcher(
                slang_module,
                input_embedding_data.timestamps_us,
                device,
            )

            count = input_embedding_data.instance_idx.shape[0]
            assert count > 0
            threads_per_block = InputEmbeddingFunction._BACKWARD_THREADS_PER_BLOCK
            blocks_per_grid = CollectorFunction._div_up(count, threads_per_block)
            assert blocks_per_grid > 0

            # Prepare common arguments (grid dimensions + kernel parameters)
            args = [
                (threads_per_block, 1, 1),  # grid_dim
                (blocks_per_grid, 1, 1),  # block_dim
                count,
                xyzs,
                input_embedding_data.instance_idx.contiguous(),
                scene_contractor,
                instance_emb,
                timestamps_fetcher,
                time_emb,
            ]

            # Select kernel and build argument list based on whether we have timestamps_delta
            per_track_timestamps = not isinstance(input_embedding_data.timestamps_us, int)
            kernel = InputEmbeddingFunction._select_kernel(
                slang_module,
                per_track_timestamps=per_track_timestamps,
                with_timestamps_delta=input_embedding_data.timestamps_delta is not None,
            )
            if input_embedding_data.timestamps_delta is not None:
                timestamps_delta_tensor = InputEmbeddingFunction._get_timestamps_delta(
                    input_embedding_data.timestamps_delta, device
                )
                args.append(timestamps_delta_tensor)

            kernel.bwd_wrapped_fn.fn_handle(*args, output_param)

        return (None, None, *input_grads)


class TracksInterpolationFunction(torch.autograd.Function):
    """Autograd function for interpolating track poses at query timestamps.

    Given time-series of poses per track (packed format), interpolates to find
    poses at specified timestamps using SE(3) interpolation (SLERP for rotations).

    Data format:
    - Tracks are stored as packed sequences: all poses concatenated
    - tracks_packinfo[i] = [start_index, length] for track i
    - tracks_timestamps[start:start+length] = timestamps for track i's poses
    - Allows variable-length sequences per track

    Interpolation strategies:
    - Linear interpolation (default): SLERP for rotations, lerp for translations
    - Nearest neighbor: Select the closest annotated pose while retaining the
      query timestamp for deformation conditioning. Ties select the left pose.

    Timestamp selection modes:
    1. Global: Single timestamp for all tracks
    2. Per-track: Pre-specified timestamp array
    3. Estimation: Estimate per-track timestamp via ray-cuboid intersection
       - Finds which rays hit each track's bounding box
       - Averages timestamps of intersecting rays
       - Used for temporal super-resolution in dynamic scenes

    Output:
    - poses: Interpolated poses [num_tracks, 7]
    - inside: Whether the timestamp is accepted for the track, including endpoint
      clamping within cadence-plus-exposure tolerance [num_tracks] (bool)
      * True: Within the range or accepted at an endpoint
      * False: Outside the endpoint tolerance
    - resolved_timestamps: Effective deformation-conditioning timestamp for each
      track [num_tracks] (int64). Normal interpolation and nearest-neighbor pose
      lookup retain the query timestamp. An accepted out-of-range query uses the
      endpoint timestamp because its rigid pose is clamped to that endpoint;
      rejected queries retain the raw query and are masked inactive.

    Gradient flow:
    - Gradients flow back to poses that contributed to interpolation
    - Gradient buffer is zero-initialized (not all poses contribute)
    - No gradients to timestamps (discrete query points)
    """

    _FORWARD_THREADS_PER_BLOCK = 512
    _BACKWARD_THREADS_PER_BLOCK = 512
    # Cull pre-pass is 1-D over tracks (typically O(1k)).  Smaller block size
    # gives the SM scheduler more blocks to overlap and avoids a half-empty
    # tail block at 512 threads.
    _CULL_THREADS_PER_BLOCK = 128
    _TIMESTAMP_THREADS_PER_BLOCK = 256
    _TIMESTAMP_MAX_BLOCKS = 2048
    _TIMESTAMP_CACHE_PREPARE_MAX_BLOCKS = 1024
    # Bound temporary storage even for large image batches. Each entry is 7
    # float32 pose values plus one int32 validity flag. Exceeding the budget
    # retains the original kernel, without reading active_count on the CPU.
    _TIMESTAMP_CACHE_MAX_BYTES = 64 * 1024 * 1024

    @staticmethod
    def _timestamp_cache_layout(
        data: TracksTimestampsEstimationData, nb_tracks: int, nb_poses: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """Select a bounded cache from CPU layout/shape metadata, without GPU reads.

        Args:
            data: Estimation rays [B*H*W,6], with optional producer-certified layout
                and effective (B,H,W) dimensions after rectangular subsampling.
            nb_tracks: Nonnegative number of packed tracks.
            nb_poses: Nonnegative total number of packed poses.

        Returns:
            (height, width, layout_id, groups) for rows=0, columns=1, global=2;
            None for unknown/mismatched/empty layouts, short-track tensors or
            scratch storage exceeding the budget. The short-track condition is
            a performance heuristic, not a claim that every track is static.
        """
        shape = data.rays_grid_shape
        if shape is None or nb_tracks <= 0 or nb_poses <= 2 * nb_tracks:
            return None
        batch, height, width = shape
        if min(shape) <= 0 or batch * height * width != data.rays.shape[0]:
            return None
        if data.rays_timestamp_layout == "rows":
            layout, groups = 0, batch * height
        elif data.rays_timestamp_layout == "columns":
            layout, groups = 1, batch * width
        elif data.rays_timestamp_layout == "global":
            layout, groups = 2, batch
        else:
            return None
        if nb_tracks * groups * 32 > TracksInterpolationFunction._TIMESTAMP_CACHE_MAX_BYTES:
            return None
        return height, width, layout, groups

    @staticmethod
    def forward(ctx, slang_module, tracks_interpolation_data: TracksInterpolationData, *input_tensors):
        device = input_tensors[0].device
        nb_tracks = tracks_interpolation_data.tracks_packinfo.size(0)
        # This will have been validated before.
        assert nb_tracks > 0
        with profile("allocate_output"):
            output_poses = torch.empty((nb_tracks, tracks_interpolation_data.tracks_poses.shape[1]), device=device)
            output_inside = torch.empty((nb_tracks,), device=device, dtype=torch.bool)
            resolved_tracks_timestamps = torch.empty((nb_tracks,), device=device, dtype=torch.int64)

        with profile("contiguous"):
            tracks_poses = tracks_interpolation_data.tracks_poses.contiguous()
            tracks_timestamps = tracks_interpolation_data.tracks_timestamps.contiguous()
            tracks_packinfo = tracks_interpolation_data.tracks_packinfo.contiguous()
            max_positive_timestamp_gap_us = tracks_interpolation_data.max_positive_timestamp_gap_us.contiguous()

        # First get timestamps to use for interpolation.
        if type(tracks_interpolation_data.timestamps_data) is TracksTimestampsGlobalData:
            # Easy, just use the timestamp (singular) from the data.
            timestamp = tracks_interpolation_data.timestamps_data.timestamp
            timestamps_fetcher = slang_module.TimestampsGlobalFetcher(timestamp=timestamp)
            kernel_suffix = "global"
        elif type(tracks_interpolation_data.timestamps_data) is TracksTimestampsPerTrackData:
            # Easy, just use the timestamps (plural) from the data.
            timestamps = tracks_interpolation_data.timestamps_data.timestamps.to(
                device=device, dtype=torch.int64
            ).contiguous()
            timestamps_fetcher = slang_module.TimestampsPerTrackFetcher(timestamps=timestamps)
            kernel_suffix = "per_track"
        elif isinstance(tracks_interpolation_data.timestamps_data, TracksTimestampsEstimationData):
            with profile("estimate_track_timestamps"):
                tracks_timestamps_estimation_data = tracks_interpolation_data.timestamps_data

                # This operation is not differentiable, so we don't need a backward pass.
                # Just call the kernel directly.
                with profile("allocate_output"):
                    output_tracks_timestamps_sum_and_count = torch.zeros(
                        (nb_tracks, 2), device=device, dtype=torch.int64
                    )

                with profile("contiguous"):
                    rays = tracks_timestamps_estimation_data.rays.contiguous()
                    rays_timestamps = tracks_timestamps_estimation_data.rays_timestamps.contiguous()
                    cuboids_dimensions = tracks_timestamps_estimation_data.cuboids_dimensions.contiguous()

                num_threads = TracksInterpolationFunction._TIMESTAMP_THREADS_PER_BLOCK
                nb_rays = rays.shape[0]
                num_rays_blocks = CollectorFunction._div_up(nb_rays, num_threads)

                with profile("build_active_tracks_indices"):
                    # Output buffer sized to the worst case (every track
                    # active).  Only positions [0, count) are read after.
                    active_tracks_indices = torch.full((nb_tracks,), -1, device=device, dtype=torch.int32)
                    active_count = torch.zeros(1, device=device, dtype=torch.int32)
                    build_threads = TracksInterpolationFunction._CULL_THREADS_PER_BLOCK
                    build_blocks = CollectorFunction._div_up(nb_tracks, build_threads)
                    data = tracks_timestamps_estimation_data
                    if data.frustum_cull is None:
                        slang_module.build_active_tracks_indices_time_only.fn_handle(
                            (build_threads, 1, 1),
                            (build_blocks, 1, 1),
                            active_tracks_indices,
                            active_count,
                            tracks_timestamps,
                            tracks_packinfo,
                            data.frame_start_timestamp,
                            data.frame_end_timestamp,
                        )
                    else:
                        frustum_rays = data.frustum_cull.rays.contiguous()
                        slang_module.build_active_tracks_indices_time_frustum.fn_handle(
                            (build_threads, 1, 1),
                            (build_blocks, 1, 1),
                            active_tracks_indices,
                            active_count,
                            tracks_poses,
                            tracks_timestamps,
                            tracks_packinfo,
                            cuboids_dimensions,
                            frustum_rays,
                            data.frame_start_timestamp,
                            data.frame_end_timestamp,
                            int(data.frustum_cull.rays_height),
                            int(data.frustum_cull.rays_width),
                            data.frustum_cull.margin_rad,
                        )
                with profile("kernel_call"):
                    # Each block iterates over active track/ray-block pairs on-device.
                    # Bound the launch even when nearly every track is culled, without
                    # reading active_count back to the CPU.
                    timestamp_blocks = max(
                        1, min(nb_tracks * num_rays_blocks, TracksInterpolationFunction._TIMESTAMP_MAX_BLOCKS)
                    )
                    timestamp_args = (
                        output_tracks_timestamps_sum_and_count,
                        rays,
                        rays_timestamps,
                        tracks_poses,
                        tracks_timestamps,
                        tracks_packinfo,
                        cuboids_dimensions,
                        active_tracks_indices,
                        active_count,
                        tracks_timestamps_estimation_data.default_timestamp,
                    )
                    cache_layout = TracksInterpolationFunction._timestamp_cache_layout(
                        tracks_timestamps_estimation_data, nb_tracks, tracks_poses.shape[0]
                    )
                    if cache_layout is None:
                        slang_module.estimate_tracks_timestamps_masked.fn_handle(
                            (num_threads, 1, 1), (timestamp_blocks, 1, 1), *timestamp_args
                        )
                    else:
                        groups = cache_layout[3]
                        # Per-call scratch avoids stale learned poses and is safe for
                        # concurrent callers. The CUDA allocator reuses released storage.
                        cached_poses = torch.empty((nb_tracks * groups, 7), device=device, dtype=torch.float32)
                        cached_valid = torch.empty((nb_tracks * groups,), device=device, dtype=torch.int32)
                        prepare_blocks = max(
                            1,
                            min(
                                CollectorFunction._div_up(nb_tracks * groups, num_threads),
                                TracksInterpolationFunction._TIMESTAMP_CACHE_PREPARE_MAX_BLOCKS,
                            ),
                        )
                        # Layout is uniform across this batch; specialize both kernels once on the host.
                        height, width, layout, groups = cache_layout
                        suffix = ("rows", "columns", "global")[layout]
                        prepare_kernel = getattr(slang_module, f"prepare_track_timestamp_poses_{suffix}")
                        estimate_kernel = getattr(slang_module, f"estimate_tracks_timestamps_masked_cached_{suffix}")
                        prepare_kernel.fn_handle(
                            (num_threads, 1, 1),
                            (prepare_blocks, 1, 1),
                            cached_poses,
                            cached_valid,
                            rays_timestamps,
                            tracks_poses,
                            tracks_timestamps,
                            tracks_packinfo,
                            active_tracks_indices,
                            active_count,
                            height,
                            width,
                            groups,
                        )
                        estimate_kernel.fn_handle(
                            (num_threads, 1, 1),
                            (timestamp_blocks, 1, 1),
                            *timestamp_args,
                            cached_poses,
                            cached_valid,
                            height,
                            width,
                            groups,
                        )
            timestamps_fetcher = slang_module.TimestampsEstimationFetcher(
                tracks_timestamps_sum_and_count=output_tracks_timestamps_sum_and_count,
                default_timestamp=tracks_timestamps_estimation_data.default_timestamp,
            )
            kernel_suffix = "with_estimation"
        else:
            raise ValueError(f"Unsupported tracks timestamps data: {tracks_interpolation_data.timestamps_data}")

        # Then interpolate the poses.
        with profile("interpolation"):
            threads_per_block = TracksInterpolationFunction._FORWARD_THREADS_PER_BLOCK
            blocks_per_grid = CollectorFunction._div_up(nb_tracks, threads_per_block)
            assert blocks_per_grid > 0

            if type(tracks_interpolation_data) is TracksInterpolationData:
                if tracks_interpolation_data.nearest_neighbor:
                    kernel_prefix = "nearest_neighbor"
                else:
                    kernel_prefix = "interpolate"
                kernel_name = f"{kernel_prefix}_tracks_poses_{kernel_suffix}"
                interpolation_kernel = getattr(slang_module, kernel_name)

                interpolation_kernel.fn_handle(
                    (threads_per_block, 1, 1),
                    (blocks_per_grid, 1, 1),
                    nb_tracks,
                    CollectorFunction._fwd_param_raw(output_poses),
                    output_inside,
                    resolved_tracks_timestamps,
                    CollectorFunction._fwd_param_raw(tracks_poses),
                    tracks_timestamps,
                    tracks_packinfo,
                    timestamps_fetcher,
                    max_positive_timestamp_gap_us,
                    tracks_interpolation_data.frame_exposure_us,
                )
            else:
                raise ValueError(f"Unsupported tracks interpolation data: {tracks_interpolation_data}")

        with profile("post"):
            ctx.save_for_backward(output_poses, output_inside, resolved_tracks_timestamps, *input_tensors)
            ctx.tracks_interpolation_data = tracks_interpolation_data
            ctx.timestamps_fetcher = timestamps_fetcher
            ctx.interpolation_kernel = interpolation_kernel

        return output_poses, output_inside, resolved_tracks_timestamps

    @staticmethod
    def backward(ctx, output_poses_grad, output_inside_grad, output_timestamps_grad):
        tracks_interpolation_data = ctx.tracks_interpolation_data
        timestamps_fetcher = ctx.timestamps_fetcher
        interpolation_kernel = ctx.interpolation_kernel

        # Reset the context to avoid keeping references to the tensors.
        ctx.tracks_interpolation_data = None
        ctx.timestamps_fetcher = None
        ctx.interpolation_kernel = None

        saved_tensors = ctx.saved_tensors
        with profile("contiguous"):
            output_poses_grad = output_poses_grad.contiguous()
            # Ignore output_inside_grad and output_timestamps_grad. They are
            # non-differentiable interpolation metadata.

        output_poses = saved_tensors[0]
        output_inside = saved_tensors[1]
        resolved_tracks_timestamps = saved_tensors[2]

        output_poses_param = (output_poses, (output_poses_grad,))
        output_inside_param = output_inside

        with profile("interpolation"):
            input_grads = []
            backward_tensor_func = CollectorFunction.get_backward_tensor_func(input_grads)

            if type(tracks_interpolation_data) is TracksInterpolationData:
                # Even though track poses are read with loadOnce(), not all poses will be read,
                # therefore not all poses will get their gradients assigned in the backward pass,
                # so we need to zero the gradients.
                tracks_poses = backward_tensor_func(tracks_interpolation_data.tracks_poses, True)
                tracks_timestamps = tracks_interpolation_data.tracks_timestamps
                tracks_packinfo = tracks_interpolation_data.tracks_packinfo
                nb_tracks = tracks_interpolation_data.tracks_packinfo.shape[0]
                max_positive_timestamp_gap_us = tracks_interpolation_data.max_positive_timestamp_gap_us.contiguous()

                # This will have been validated before.
                assert nb_tracks > 0
                threads_per_block = TracksInterpolationFunction._BACKWARD_THREADS_PER_BLOCK
                blocks_per_grid = CollectorFunction._div_up(nb_tracks, threads_per_block)
                assert blocks_per_grid > 0

                interpolation_kernel.bwd_wrapped_fn.fn_handle(
                    (threads_per_block, 1, 1),
                    (blocks_per_grid, 1, 1),
                    nb_tracks,
                    output_poses_param,
                    output_inside_param,
                    resolved_tracks_timestamps,
                    tracks_poses,
                    tracks_timestamps,
                    tracks_packinfo,
                    timestamps_fetcher,
                    max_positive_timestamp_gap_us,
                    tracks_interpolation_data.frame_exposure_us,
                )
            else:
                raise ValueError(f"Unsupported tracks interpolation data: {tracks_interpolation_data}")

        return (None, None, *input_grads)


class SlangGaussianParameterCollector(GaussianParameterCollector):
    """Slang-based GPU implementation of GaussianParameterCollector.

    This is the main entry point for high-performance Gaussian collection using
    GPU kernels written in Slang and integrated via SlangTorch.

    Architecture:
    - One layer handler per layer in configuration
    - Handlers generate and manage Slang kernel configurations
    - Kernels are loaded from a cache at initialization; the cache is pre-filled with
      pre-compiled kernels for typical layer configurations, and only if a kernel is not
      found for the required layer configuration, it is compiled at runtime and cached
    - Forward/backward passes dispatched through torch.autograd.Function

    Validation:
    - When enabled (run_validation=True), validates all input shapes and types
    - Useful during development and debugging
    - Can be disabled for production to reduce overhead

    Attributes:
        layers_config: Configuration for all layers
        run_validation: Whether to validate inputs on each call
        layer_handlers: List of handlers, one per layer
    """

    layers_config: LayersConfig
    run_validation: bool
    layer_handlers: List[_LayerHandler]

    def __init__(self, layers_config: LayersConfig, run_validation: bool):
        """Initialize collector with layer configuration.

        Args:
            layers_config: Complete configuration for all layers
            run_validation: Whether to validate inputs (True for safety, False for speed)
        """
        self.layers_config = layers_config
        self.run_validation = run_validation

        self.layer_handlers = SlangGaussianParameterCollector._build_layer_handlers(layers_config)

    @staticmethod
    def _build_layer_handlers(layers_config) -> List[_LayerHandler]:
        """Build layer handlers and load/compile all required Slang kernels.

        This is done in two phases:
        1. Create handlers and collect all kernel configurations
        2. Load kernels from cache (pre-filled with typical configurations) or compile at runtime if needed

        Kernel loading/compilation process:
        - Kernels are loaded from a global cache that is pre-filled with pre-compiled kernels
          for typical layer configurations
        - If a kernel is not found for the required layer configuration, it is compiled at
          runtime and added to the cache
        - Cache is shared globally across collector instances

        Args:
            layers_config: Configuration for all layers

        Returns:
            List of initialized layer handlers with loaded/compiled kernels attached
        """
        layer_handlers: List[_LayerHandler] = []
        for i, layer_config in enumerate(layers_config.layers):
            handler_types = {
                LayerConfigSH: _LayerHandlerSH,
                LayerConfigRigid: _LayerHandlerRigid,
                LayerConfigDeformable: _LayerHandlerDeformable,
            }
            handler_type = handler_types[type(layer_config)]
            layer_handlers.append(handler_type(layers_config, i))

        # Get the kernels all in one go to group compilation together.
        configurations = []
        configuration_counts = []
        for layer_handler in layer_handlers:
            layer_configurations = layer_handler.get_collector_configurations()
            configurations.extend(layer_configurations)
            configuration_counts.append(len(layer_configurations))

        kernels = get_slang_kernels(configurations)
        assert len(kernels) == len(configurations)
        assert len(kernels) == sum(configuration_counts)

        kernel_index = 0
        for i, layer_handler in enumerate(layer_handlers):
            count = configuration_counts[i]
            layer_handler.kernels = tuple(kernels[kernel_index : kernel_index + count])
            kernel_index += count

        return layer_handlers

    def _get_default_slang_module(self) -> Any:
        """Get the Slang module for standalone kernel access.

        Used by track calibration, track interpolation, and input embedding
        operations that need direct kernel access outside the main collection
        flow.

        Note: These standalone kernels exist in all Slang modules, so we
        can get it from any kernel's Slang module.

        Returns:
            Slang module with compiled kernels

        Raises:
            ValueError: If no kernels are available
        """
        # We assume we can get the kernel from any layer in the collector.
        if not self.layer_handlers or not self.layer_handlers[0].kernels:
            raise ValueError("No layer handlers or kernels available")
        return self.layer_handlers[0].kernels[0].slang_module

    @staticmethod
    def _validate_unicycle_tracks_calib_data(data: ElevationAwareUnicycleTracksCalibData) -> None:
        """Validate the tensor contract shared by unicycle pose and residual kernels."""
        nb_poses = data.xy.shape[0]
        nb_pairs = data.velocity.shape[0]
        nb_tracks = data.ground_t.shape[0]

        # Validate metadata before any operation that reads tensor values.
        assert data.xy.shape == (nb_poses, 2)
        for tensor in (data.height, data.theta, data.pitch, data.pose_gradient_mask, data.tracks_idx):
            assert tensor.shape == (nb_poses,)
        for tensor in (data.velocity, data.theta_velocity, data.pitch_velocity, data.time_delta_s):
            assert tensor.shape == (nb_pairs,)
        assert data.pose_pair_start_indices.shape == (nb_pairs,)
        assert data.time_delta_v_s.shape == (nb_pairs,)
        assert data.initial_pose_xy.shape == (nb_poses, 2)
        assert data.initial_pose_height.shape == (nb_poses,)
        assert data.ground_t.shape == (nb_tracks, 3)
        assert data.ground_q.shape == (nb_tracks, 4)
        assert data.Tclc_tquat.numel() == 7
        assert data.motion_triplet_start_indices.dim() == 1

        float_tensors = (
            data.xy,
            data.height,
            data.theta,
            data.pitch,
            data.velocity,
            data.theta_velocity,
            data.pitch_velocity,
            data.time_delta_s,
            data.time_delta_v_s,
            data.initial_pose_xy,
            data.initial_pose_height,
            data.ground_t,
            data.ground_q,
            data.Tclc_tquat,
        )
        assert all(tensor.dtype == torch.float32 for tensor in float_tensors)
        assert data.pose_gradient_mask.dtype == torch.bool
        assert data.tracks_idx.dtype == torch.int64
        assert data.pose_pair_start_indices.dtype == torch.int64
        assert data.motion_triplet_start_indices.dtype == torch.int64

        all_tensors = (
            *float_tensors,
            data.pose_gradient_mask,
            data.tracks_idx,
            data.pose_pair_start_indices,
            data.motion_triplet_start_indices,
        )
        device = data.xy.device
        assert device.type == "cuda", "Unicycle Slang tensors must be CUDA-resident"
        assert all(tensor.device == device for tensor in all_tensors), (
            "Unicycle Slang tensors must share one CUDA device"
        )

        # The kernels trust these indices and perform unchecked adjacent-row access.
        assert nb_tracks > 0, "Unicycle calibration requires at least one track"
        assert bool(((data.tracks_idx >= 0) & (data.tracks_idx < nb_tracks)).all().item()), (
            "tracks_idx contains an out-of-range track index"
        )
        expected_tracks = torch.arange(nb_tracks, device=device, dtype=torch.int64)
        assert torch.equal(torch.unique_consecutive(data.tracks_idx), expected_tracks), (
            "tracks_idx must describe contiguous, non-empty packed tracks"
        )

        expected_pair_starts = torch.nonzero(
            data.tracks_idx[:-1] == data.tracks_idx[1:],
            as_tuple=False,
        ).flatten()
        assert torch.equal(data.pose_pair_start_indices, expected_pair_starts), (
            "pose_pair_start_indices does not match the packed track topology"
        )

        pair_starts = data.pose_pair_start_indices
        if pair_starts.numel() < 3:
            expected_triplet_starts = pair_starts.new_empty((0,))
        else:
            consecutive_triplets = (pair_starts[1:-1] == pair_starts[:-2] + 1) & (
                pair_starts[2:] == pair_starts[:-2] + 2
            )
            expected_triplet_starts = torch.nonzero(consecutive_triplets, as_tuple=False).flatten()
        assert torch.equal(data.motion_triplet_start_indices, expected_triplet_starts), (
            "motion_triplet_start_indices does not match the packed pair topology"
        )

    def collect(
        self,
        layers_data: LayersData,
        layer_indices: Optional[List[int]] = None,
        out: Optional[CollectorResult] = None,
        segments: Optional[List[LayerSegments]] = None,
    ) -> CollectorResult:
        # Both no_grad preconditions are read here rather than in CollectorFunction.forward,
        # which is the only place they are about: autograd runs a Function's forward with
        # grad mode disabled, so torch.is_grad_enabled() inside it is always False and an
        # assert there can never fire. This is the last frame that still sees the caller's
        # mode.
        #
        # A windowed collect has no backward -- rows outside the windows are never written,
        # so there is nothing to differentiate through for them -- and a caller-owned
        # destination is an aliased view whose gradients would not flow back to it.
        if torch.is_grad_enabled():
            if out is not None:
                raise ValueError("a caller-owned collect destination requires no_grad")
            if segments is not None and any(s.n_segments != 0 for s in segments):
                raise ValueError("a windowed collect requires no_grad")
        with profile("collect"):
            # Flatten the layer buffers into a single list of tensor so that Torch's autograd
            # can track the dependencies between the tensors.
            with profile("flatten_layer_buffers"):
                input_tensors = []
                for i, layer_data in enumerate(layers_data.layers):
                    handler_idx = layer_indices[i] if layer_indices is not None else i
                    layer_handler = self.layer_handlers[handler_idx]
                    input_tensors.extend(layer_handler.get_trainable_input_tensors(layer_data))

            with profile("apply"):
                output_tensors = CollectorFunction.apply(
                    self, layers_data, layer_indices, out, segments, *input_tensors
                )

            return CollectorResult(*output_tensors)

    # This function is an external stand-alone function, but eventually might be moved inside the merged
    # kernels or the merged autograd function.
    def calibrate_tracks_poses(
        self,
        tracks_poses: torch.Tensor,
        tracks_calib_data: TracksCalibData,
    ) -> torch.Tensor:
        if self.run_validation:
            with profile("validate"):
                assert not tracks_poses.requires_grad
                nb_poses = tracks_poses.shape[0]
                assert tracks_poses.shape == (nb_poses, 7)
                assert tracks_poses.dtype == torch.float32
                if type(tracks_calib_data) is DirectTracksCalibData:
                    assert not tracks_calib_data.gradient_mask.requires_grad
                    assert tracks_calib_data.gradient_mask.shape == (nb_poses,)
                    assert tracks_calib_data.gradient_mask.dtype == torch.bool
                    assert tracks_calib_data.tracks_delta_q.shape == (nb_poses, 4)
                    assert tracks_calib_data.tracks_delta_q.dtype == torch.float32
                    assert tracks_calib_data.tracks_delta_t.shape == (nb_poses, 3)
                    assert tracks_calib_data.tracks_delta_t.dtype == torch.float32
                elif type(tracks_calib_data) is ElevationAwareUnicycleTracksCalibData:
                    self._validate_unicycle_tracks_calib_data(tracks_calib_data)
                    assert tracks_calib_data.xy.shape[0] == nb_poses
                    assert tracks_poses.device == tracks_calib_data.xy.device
                else:
                    raise ValueError(f"Unsupported tracks calib data: {tracks_calib_data}")

        with profile("calibrate_tracks_poses"):
            if tracks_poses.shape[0] == 0:
                return torch.empty_like(tracks_poses)

            slang_module = self._get_default_slang_module()

            input_tensors = []
            if type(tracks_calib_data) is DirectTracksCalibData:
                input_tensors = [
                    tracks_calib_data.tracks_delta_q,
                    tracks_calib_data.tracks_delta_t,
                ]
            elif type(tracks_calib_data) is ElevationAwareUnicycleTracksCalibData:
                input_tensors = [
                    tracks_calib_data.xy,
                    tracks_calib_data.height,
                    tracks_calib_data.theta,
                    tracks_calib_data.pitch,
                ]
            else:
                raise ValueError(f"Unsupported tracks calib data: {tracks_calib_data}")
            return TracksCalibFunction.apply(slang_module, tracks_poses, tracks_calib_data, *input_tensors)

    def compute_tracks_calib_residuals(self, tracks_calib_data: TracksCalibData) -> TracksCalibResiduals:
        """Compute structured-calibrator residuals with the collector's Slang module."""
        if type(tracks_calib_data) is not ElevationAwareUnicycleTracksCalibData:
            raise ValueError(f"Unsupported tracks calib residual data: {tracks_calib_data}")
        if self.run_validation:
            self._validate_unicycle_tracks_calib_data(tracks_calib_data)

        slang_module = self._get_default_slang_module()
        consistency = UnicycleConsistencyFunction.apply(
            slang_module,
            tracks_calib_data.xy,
            tracks_calib_data.height,
            tracks_calib_data.theta,
            tracks_calib_data.pitch,
            tracks_calib_data.velocity,
            tracks_calib_data.theta_velocity,
            tracks_calib_data.pitch_velocity,
            tracks_calib_data.pose_gradient_mask,
            tracks_calib_data.pose_pair_start_indices,
            tracks_calib_data.time_delta_s,
            tracks_calib_data.pitch_slope_clamp_rad,
        )
        smoothness = UnicycleSmoothnessFunction.apply(
            slang_module,
            tracks_calib_data.height,
            tracks_calib_data.velocity,
            tracks_calib_data.theta_velocity,
            tracks_calib_data.pitch_velocity,
            tracks_calib_data.pose_gradient_mask,
            tracks_calib_data.pose_pair_start_indices,
            tracks_calib_data.motion_triplet_start_indices,
            tracks_calib_data.time_delta_s,
            tracks_calib_data.time_delta_v_s,
        )
        pose_regularization = UnicyclePoseRegularizationFunction.apply(
            slang_module,
            tracks_calib_data.xy,
            tracks_calib_data.height,
            tracks_calib_data.pose_gradient_mask,
            tracks_calib_data.initial_pose_xy,
            tracks_calib_data.initial_pose_height,
        )
        return TracksCalibResiduals(
            consistency=consistency,
            smoothness=smoothness,
            pose_regularization=pose_regularization,
        )

    def prepare_input_embeddings(self, input_embedding_data: InputEmbeddingData) -> torch.Tensor:
        if self.run_validation:
            with profile("validate"):
                assert input_embedding_data.instance_idx.shape[0] > 0
                assert input_embedding_data.instance_idx.dim() == 1
                assert input_embedding_data.instance_idx.dtype == torch.int32

                assert input_embedding_data.xyzs.dim() == 2
                assert input_embedding_data.xyzs.shape[1] == 3
                assert input_embedding_data.xyzs.shape[0] == input_embedding_data.instance_idx.shape[0]
                assert input_embedding_data.xyzs.dtype == torch.float32

                if not isinstance(input_embedding_data.timestamps_us, int):
                    assert input_embedding_data.timestamps_us.dim() == 1
                    assert input_embedding_data.timestamps_us.dtype == torch.int64
                    assert input_embedding_data.timestamps_us.device == input_embedding_data.xyzs.device

                if input_embedding_data.timestamps_delta is not None:
                    assert input_embedding_data.timestamps_delta.dim() == 1
                    assert input_embedding_data.timestamps_delta.shape[0] == input_embedding_data.xyzs.shape[0]
                    assert input_embedding_data.timestamps_delta.dtype == torch.int64
                    assert input_embedding_data.timestamps_delta.device == input_embedding_data.xyzs.device

                if type(input_embedding_data.instance_embedding_weights) is WeightedInstanceInputEmbeddingData:
                    # Infer dimension from weights tensor
                    weights_tensor = input_embedding_data.instance_embedding_weights.instance_embedding_weights
                    inst_dim = weights_tensor.shape[1]
                    assert inst_dim == 1, (
                        "instance embedding dimension must be 1 for now (would require template modifications in Slang kernels)"
                    )
                    assert weights_tensor.dtype == torch.float32
                    if not isinstance(input_embedding_data.timestamps_us, int):
                        assert input_embedding_data.timestamps_us.shape == (weights_tensor.shape[0],)
                else:
                    raise ValueError(
                        f"Unsupported instance embedding data: {type(input_embedding_data.instance_embedding_weights)}"
                    )

                assert input_embedding_data.time_embedding_config is not None, "time_embedding_config must be provided"
                if type(input_embedding_data.time_embedding_config) is IndividualRemapTimeInputEmbeddingConfig:
                    assert not input_embedding_data.time_embedding_config.timestamps_us_ranges.requires_grad
                    assert input_embedding_data.time_embedding_config.timestamps_us_ranges.shape[1] == 2
                    assert input_embedding_data.time_embedding_config.timestamps_us_ranges.dtype == torch.int64
                else:
                    raise ValueError(f"Unsupported time embedding config: {input_embedding_data.time_embedding_config}")

                sc = input_embedding_data.scene_contractor
                assert sc.aabb_blb.dim() == 2
                assert sc.aabb_blb.shape[1] == 3
                assert sc.aabb_trf.dim() == 2
                assert sc.aabb_trf.shape[1] == 3
                assert sc.aabb_blb.shape[0] == sc.aabb_trf.shape[0]
                assert sc.aabb_blb.dtype == torch.float32
                assert sc.aabb_trf.dtype == torch.float32

        with profile("prepare_input_embeddings"):
            count = input_embedding_data.xyzs.shape[0]
            assert count > 0, "count must be greater than 0"

            # Validate instance embedding data
            if type(input_embedding_data.instance_embedding_weights) is WeightedInstanceInputEmbeddingData:
                weights_tensor = input_embedding_data.instance_embedding_weights.instance_embedding_weights
                instance_embedding_dim = weights_tensor.shape[1]
                assert instance_embedding_dim == 1, (
                    "instance embedding dimension must be 1 for now (would require template modifications in Slang kernels)"
                )
            else:
                raise ValueError(
                    f"Unsupported instance embedding data: {type(input_embedding_data.instance_embedding_weights)}"
                )

            slang_module = self._get_default_slang_module()

            # Collect trainable input tensors
            input_tensors = []
            input_tensors.append(input_embedding_data.xyzs)
            # Extract the actual weights tensor from the data structure
            assert isinstance(input_embedding_data.instance_embedding_weights, WeightedInstanceInputEmbeddingData)
            input_tensors.append(input_embedding_data.instance_embedding_weights.instance_embedding_weights)

            return InputEmbeddingFunction.apply(slang_module, input_embedding_data, *input_tensors)

    # This function is an external stand-alone function, but eventually might be moved inside the merged
    # kernels.
    def interpolate_tracks_poses(
        self, tracks_interpolation_data: TracksInterpolationData
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.run_validation:
            if type(tracks_interpolation_data) is TracksInterpolationData:
                nb_poses = tracks_interpolation_data.tracks_poses.shape[0]
                assert tracks_interpolation_data.tracks_poses.shape == (nb_poses, 7)
                assert tracks_interpolation_data.tracks_poses.dtype == torch.float32
                assert tracks_interpolation_data.tracks_timestamps.shape == (nb_poses,)
                assert tracks_interpolation_data.tracks_timestamps.dtype == torch.int64
                nb_tracks = tracks_interpolation_data.tracks_packinfo.shape[0]
                assert tracks_interpolation_data.tracks_packinfo.shape == (nb_tracks, 2)
                assert tracks_interpolation_data.tracks_packinfo.dtype == torch.int32
                assert type(tracks_interpolation_data.nearest_neighbor) is bool
                assert tracks_interpolation_data.max_positive_timestamp_gap_us.shape == (nb_tracks,)
                assert tracks_interpolation_data.max_positive_timestamp_gap_us.dtype == torch.int64
                assert (
                    tracks_interpolation_data.max_positive_timestamp_gap_us.device
                    == tracks_interpolation_data.tracks_timestamps.device
                )
                assert type(tracks_interpolation_data.frame_exposure_us) is int
                assert tracks_interpolation_data.frame_exposure_us >= 0
                if type(tracks_interpolation_data.timestamps_data) is TracksTimestampsGlobalData:
                    assert type(tracks_interpolation_data.timestamps_data.timestamp) is int
                elif type(tracks_interpolation_data.timestamps_data) is TracksTimestampsPerTrackData:
                    assert tracks_interpolation_data.timestamps_data.timestamps.shape == (nb_tracks,)
                    assert tracks_interpolation_data.timestamps_data.timestamps.dtype == torch.int64
                elif isinstance(tracks_interpolation_data.timestamps_data, TracksTimestampsEstimationData):
                    nb_rays = tracks_interpolation_data.timestamps_data.rays.shape[0]
                    assert tracks_interpolation_data.timestamps_data.rays.shape == (nb_rays, 6)
                    assert tracks_interpolation_data.timestamps_data.rays.dtype == torch.float32
                    assert tracks_interpolation_data.timestamps_data.rays_timestamps.shape == (nb_rays,)
                    assert tracks_interpolation_data.timestamps_data.rays_timestamps.dtype == torch.int64
                    assert tracks_interpolation_data.timestamps_data.cuboids_dimensions.shape == (nb_tracks, 3)
                    assert tracks_interpolation_data.timestamps_data.cuboids_dimensions.dtype == torch.float32
                    assert type(tracks_interpolation_data.timestamps_data.default_timestamp) is int
                    assert type(tracks_interpolation_data.timestamps_data.frame_start_timestamp) is int
                    assert type(tracks_interpolation_data.timestamps_data.frame_end_timestamp) is int
                    if tracks_interpolation_data.timestamps_data.frustum_cull is not None:
                        assert type(tracks_interpolation_data.timestamps_data.frustum_cull) is TracksFrustumCullData
                        assert type(tracks_interpolation_data.timestamps_data.frustum_cull.rays_height) is int
                        assert type(tracks_interpolation_data.timestamps_data.frustum_cull.rays_width) is int
                        assert tracks_interpolation_data.timestamps_data.frustum_cull.rays_height >= 2
                        assert tracks_interpolation_data.timestamps_data.frustum_cull.rays_width >= 2
                        assert type(tracks_interpolation_data.timestamps_data.frustum_cull.margin_rad) is float
                        nb_frustum_rays = tracks_interpolation_data.timestamps_data.frustum_cull.rays.shape[0]
                        assert tracks_interpolation_data.timestamps_data.frustum_cull.rays.shape == (
                            nb_frustum_rays,
                            6,
                        )
                        assert (
                            nb_frustum_rays
                            == tracks_interpolation_data.timestamps_data.frustum_cull.rays_height
                            * tracks_interpolation_data.timestamps_data.frustum_cull.rays_width
                        )
                        assert tracks_interpolation_data.timestamps_data.frustum_cull.rays.dtype == torch.float32
                else:
                    raise ValueError(f"Unsupported tracks timestamps data: {tracks_interpolation_data.timestamps_data}")
            else:
                raise ValueError(f"Unsupported tracks interpolation data: {tracks_interpolation_data}")

        with profile("interpolate_tracks_poses"):
            if tracks_interpolation_data.tracks_packinfo.shape[0] == 0:
                poses = torch.empty_like(tracks_interpolation_data.tracks_poses)
                device = tracks_interpolation_data.tracks_poses.device
                inside = torch.empty((0,), dtype=torch.bool, device=device)
                timestamps = torch.empty((0,), dtype=torch.int64, device=device)
                return poses, inside, timestamps

            slang_module = self._get_default_slang_module()

            input_tensors = []
            if type(tracks_interpolation_data) is TracksInterpolationData:
                assert not tracks_interpolation_data.tracks_timestamps.requires_grad
                assert not tracks_interpolation_data.tracks_packinfo.requires_grad
                input_tensors = [
                    tracks_interpolation_data.tracks_poses,
                ]
            else:
                raise ValueError(f"Unsupported tracks interpolation data: {tracks_interpolation_data}")
            return TracksInterpolationFunction.apply(slang_module, tracks_interpolation_data, *input_tensors)

    def validate(self, layers_data: LayersData, layer_indices: Optional[List[int]] = None) -> None:
        if not self.run_validation:
            return

        with profile("validate"):
            if layer_indices is not None:
                assert len(layers_data.layers) == len(layer_indices)
                assert all(0 <= idx < len(self.layer_handlers) for idx in layer_indices)
            else:
                assert len(layers_data.layers) == len(self.layer_handlers)
            for i, layer_data in enumerate(layers_data.layers):
                handler_idx = layer_indices[i] if layer_indices is not None else i
                layer_handler = self.layer_handlers[handler_idx]
                layer_handler.validate(layers_data, layer_data)


def CreateSlangGaussianParameterCollector(layers_config: LayersConfig) -> SlangGaussianParameterCollector:
    """Factory function to create a Slang-based Gaussian parameter collector.

    This is the recommended way to instantiate a collector. It creates a collector
    with validation enabled by default, since its overhead is minimal and it helps
    catch errors early.

    Args:
        layers_config: Complete configuration specifying all layers and their properties

    Returns:
        Initialized collector ready for use
    """
    run_validation = True
    return SlangGaussianParameterCollector(layers_config, run_validation)
