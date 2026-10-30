# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for GSplatRenderer Phase 2: Training Support

Tests gradient flow, memory optimizations, and training integration
for the GSplatRenderer implementation.

Phase 2 Exit Criteria:
- Training step runs without errors
- Gradients flow to all Gaussian parameters
- Numerical gradient check passes
- Memory usage acceptable
- sparse_grad and absgrad work correctly
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from ncore.data import (
    FThetaCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ShutterType,
)
from nurec.config.model import GSplatRendererConfig
from nurec.libs.sensors.kernels.cameras import image_points_to_world_rays_shutter_pose
from nurec.libs.sensors.kernels.cameras.parameters import (
    NoExternalDistortion,
    OpenCVPinholeProjection,
)
from nurec.libs.sensors.kernels.cameras.parameters import ShutterType as SensorlibShutterType
from nurec.libs.sensors.kernels.common import DynamicPose, Pose
from nurec.models.gaussians.renderers import (
    FisheyeSensorParametersAdapter,
    FThetaSensorParametersAdapter,
    GSplatRenderer,
    PinholeSensorParametersAdapter,
    _concatenate_extra_signals,
)
from nurec.models.gaussians.segment_plan import SegmentPlan
from nurec.utils.batch import RenderingData
from nurec.utils.cached_device_tensor import CachedDeviceTensorPool


@pytest.fixture(autouse=True)
def seed_rng():
    """Reset RNG before each test in this file for reproducibility."""
    torch.manual_seed(42)


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_concatenate_single_extra_signal_matches_cat_forward_and_backward(noncontiguous):
    width = 40 if noncontiguous else 20
    values = torch.randn(7, width)
    reference_base = values.clone().requires_grad_()
    candidate_base = values.clone().requires_grad_()
    reference_part = reference_base[:, ::2] if noncontiguous else reference_base
    candidate_part = candidate_base[:, ::2] if noncontiguous else candidate_base
    candidate_before = candidate_part.detach().clone()
    candidate_version = candidate_part._version

    reference = torch.cat([reference_part], dim=-1)
    candidate = _concatenate_extra_signals([candidate_part])
    assert candidate is candidate_part
    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)

    # gsplat requires contiguous inputs and performs this conversion at its API
    # boundary, so include it when comparing the full old and new tensor paths.
    weights = torch.randn_like(reference)
    (reference.square() * weights).sum().backward()
    (candidate.contiguous().square() * weights).sum().backward()

    torch.testing.assert_close(candidate_base.grad, reference_base.grad, rtol=0, atol=0)
    torch.testing.assert_close(candidate_part, candidate_before, rtol=0, atol=0)
    assert candidate_part._version == candidate_version


def test_concatenate_multiple_extra_signals_preserves_order():
    parts = [
        torch.tensor([[1.0, 2.0], [6.0, 7.0]]),
        torch.tensor([[3.0], [8.0]]),
        torch.tensor([[4.0, 5.0], [9.0, 10.0]]),
    ]
    parts_before = [part.clone() for part in parts]
    versions_before = [part._version for part in parts]

    torch.testing.assert_close(
        _concatenate_extra_signals(parts),
        torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]),
        rtol=0,
        atol=0,
    )
    for part, before, version in zip(parts, parts_before, versions_before):
        torch.testing.assert_close(part, before, rtol=0, atol=0)
        assert part._version == version


@pytest.mark.parametrize(
    ("adapter_cls", "sensor_params", "expected_shapes"),
    [
        (
            FThetaSensorParametersAdapter,
            FThetaCameraModelParameters(
                resolution=np.array([1920, 1080], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([960.0, 540.0], dtype=np.float32),
                reference_poly=FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST,
                pixeldist_to_angle_poly=np.array([0.0, 0.005, 0.0, 0.0, 0.0], dtype=np.float32),
                angle_to_pixeldist_poly=np.array([0.0, 200.0, 0.0, 0.0, 0.0], dtype=np.float32),
                max_angle=1.2,
                linear_cde=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            ),
            {"Ks": (1, 1, 3, 3)},
        ),
        (
            PinholeSensorParametersAdapter,
            OpenCVPinholeCameraModelParameters(
                resolution=np.array([1920, 1080], dtype=np.uint64),
                focal_length=np.array([200.0, 200.0], dtype=np.float32),
                principal_point=np.array([960.0, 540.0], dtype=np.float32),
                radial_coeffs=np.zeros(6, dtype=np.float32),
                tangential_coeffs=np.zeros(2, dtype=np.float32),
                thin_prism_coeffs=np.zeros(4, dtype=np.float32),
                shutter_type=ShutterType.GLOBAL,
            ),
            {
                "Ks": (1, 1, 3, 3),
                "radial_coeffs": (1, 1, 6),
                "tangential_coeffs": (1, 1, 2),
            },
        ),
        (
            FisheyeSensorParametersAdapter,
            OpenCVFisheyeCameraModelParameters(
                resolution=np.array([1920, 1080], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([960.0, 540.0], dtype=np.float32),
                focal_length=np.array([200.0, 200.0], dtype=np.float32),
                radial_coeffs=np.zeros(4, dtype=np.float32),
                max_angle=1.2,
            ),
            {"Ks": (1, 1, 3, 3), "radial_coeffs": (1, 1, 4)},
        ),
    ],
)
def test_sensor_params_reuse_cached_device_storage(adapter_cls, sensor_params, expected_shapes):
    device = torch.device("cpu")
    tensor_pool = CachedDeviceTensorPool(maxsize=len(expected_shapes), name="test_sensor_tensors")
    first = adapter_cls(sensor_params, device, tensor_pool).get_gsplat_sensor_params()
    second = adapter_cls(sensor_params, device, tensor_pool).get_gsplat_sensor_params()

    tensor_keys = {name for name, value in first.items() if torch.is_tensor(value)}
    assert tensor_keys == set(expected_shapes)
    torch.testing.assert_close(
        first["Ks"],
        torch.tensor([[[[200.0, 0.0, 960.0], [0.0, 200.0, 540.0], [0.0, 0.0, 1.0]]]]),
        rtol=0,
        atol=0,
    )
    for name, shape in expected_shapes.items():
        assert first[name].shape == shape
        assert first[name].untyped_storage().data_ptr() == second[name].untyped_storage().data_ptr()


def test_cached_pinhole_camera_tensors_are_read_only_through_real_cuda_gsplat(mock_model, simple_gaussian_parameters):
    """gsplat must not write the cached Ks / distortion buffers.

    Cached tensors are shared for the rest of the session, so an in-place write
    during rasterization would corrupt every later render for that sensor.
    After one real CUDA rasterization, values and ``_version`` on the cached
    bases must be unchanged.

    3dgut-gsplat is tested so radial/tangential actually reach
    ``gsplat.rasterization``.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    h, w = 64, 64
    # Distinctive nonzero coeffs so a ``.zero_()`` cannot hide behind the
    # all-zero defaults the other fixtures use.
    sensor_params = OpenCVPinholeCameraModelParameters(
        resolution=np.array([w, h], dtype=np.uint64),
        focal_length=np.array([50.0, 50.0], dtype=np.float32),
        principal_point=np.array([32.0, 32.0], dtype=np.float32),
        radial_coeffs=np.array([0.12, -0.03, 0.008, -0.001, 0.0004, -0.0002], dtype=np.float32),
        tangential_coeffs=np.array([0.0015, -0.0007], dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        shutter_type=ShutterType.GLOBAL,
    )
    cached = PinholeSensorParametersAdapter(
        sensor_params,
        device,
        CachedDeviceTensorPool(maxsize=3, name="test_pinhole_sensor_tensors"),
    ).get_gsplat_sensor_params()
    assert set(cached) == {"Ks", "radial_coeffs", "tangential_coeffs"}
    before = {name: tensor.detach().clone() for name, tensor in cached.items()}
    versions = {name: tensor._version for name, tensor in cached.items()}

    timestamps = torch.tensor([[0, 1000]], device=device, dtype=torch.int64)
    pose_tquat = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], device=device)
    rendering_data = RenderingData(
        rays=torch.randn(1, h, w, 6, device=device),
        sensor_model_parameters=[sensor_params],
        poses_tquat_startend=pose_tquat.unsqueeze(1).repeat(1, 2, 1),
        timestamps_startend_us=timestamps,
        timestamps_startend_us_cpu=timestamps.cpu(),
        rays_timestamps_us=None,
        _rays_footprints=None,
    )
    renderer = GSplatRenderer(
        GSplatRendererConfig.model_validate(
            {
                "name": "3dgut-gsplat",
                "use_rays": True,
                "tiling": {"camera": {"tile_width": 16, "tile_height": 16}},
                "background": {"color": 0.0},
                "culling": {"near_clip_distance": 0.01, "far_clip_distance": 1000.0},
                "projection": {
                    "min_projected_ray_radius": 0.5477,
                    "ut_alpha": 0.1,
                    "ut_beta": 2.0,
                    "ut_kappa": 0.0,
                    "image_margin_factor": 0.1,
                },
                "rasterize_mode": "classic",
                "packed": False,
                "sparse_grad": False,
                "absgrad": False,
            }
        ),
        mock_model,
    )
    output = renderer.render(
        rendering_data,
        simple_gaussian_parameters,
        n_active_features=0,
        extra_ray_signal_infos=([], [], []),
    )
    assert output.rgb is not None

    for name, tensor in cached.items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
        assert tensor._version == versions[name], name


@pytest.fixture
def simple_config():
    """Simple renderer configuration for testing."""
    return GSplatRendererConfig.model_validate(
        {
            "name": "3dgs-gsplat",  # Use 3dgs for gradients support
            "tiling": {"camera": {"tile_width": 16, "tile_height": 16}},
            "background": {"color": 0.0},
            "culling": {
                "near_clip_distance": 0.01,
                "far_clip_distance": 1000.0,
            },
            "projection": {"min_projected_ray_radius": 0.5477},
            "rasterize_mode": "classic",
            "packed": True,
            "sparse_grad": False,
            "absgrad": False,
        }
    )


@pytest.fixture
def mock_model():
    """Mock model for renderer initialization."""

    class MockModel(nn.Module):
        def serialize_to_json_dict(self, with_state_dict=False):
            """Mock serialization method required by GSplatRenderer."""
            return {
                "type": "mock_model",
                "config": {},
            }

    return MockModel()


def test_renderer_owns_configured_tensor_pools(mock_model):
    config = GSplatRendererConfig.model_validate(
        {
            "name": "3dgs-gsplat",
            "max_camera_tensor_cache_entries": 7,
            "max_lidar_geometry_cache_entries": 11,
            "max_lidar_frame_context_cache_entries": 13,
        }
    )
    renderer = GSplatRenderer(config, mock_model)

    assert renderer._camera_sensor_tensors.maxsize == 7
    assert renderer._lidar_geometry_tensors.maxsize == 11
    assert renderer._lidar_frame_contexts._maxsize == 13
    assert renderer._camera_sensor_tensors.overflow_message is not None
    assert renderer._lidar_geometry_tensors.overflow_message is not None
    assert renderer._lidar_frame_contexts._overflow_message is not None
    assert "model.renderer.max_camera_tensor_cache_entries" in renderer._camera_sensor_tensors.overflow_message
    assert "model.renderer.max_lidar_geometry_cache_entries" in renderer._lidar_geometry_tensors.overflow_message
    assert "model.renderer.max_lidar_frame_context_cache_entries" in renderer._lidar_frame_contexts._overflow_message


@pytest.fixture
def simple_rendering_data(device="cuda"):
    """Create simple rendering data for testing."""
    h, w = 64, 64

    # Simple rays
    rays = torch.randn(1, h, w, 6, device=device)

    # Simple camera parameters
    sensor_params = OpenCVPinholeCameraModelParameters(
        resolution=np.array([w, h], dtype=np.uint64),
        focal_length=np.array([50.0, 50.0], dtype=np.float32),
        principal_point=np.array([32.0, 32.0], dtype=np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        shutter_type=ShutterType.GLOBAL,
    )

    # Simple pose (identity transform)
    pose_tquat = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])  # (1, 7)
    poses_tquat_startend = pose_tquat.unsqueeze(1).repeat(1, 2, 1)  # (1, 2, 7)

    # Timestamps
    timestamps = torch.tensor([[0, 1000]], device=device, dtype=torch.int64)
    timestamps_cpu = timestamps.cpu()

    return RenderingData(
        rays=rays,
        sensor_model_parameters=[sensor_params],
        poses_tquat_startend=poses_tquat_startend.to(device),
        timestamps_startend_us=timestamps,
        timestamps_startend_us_cpu=timestamps_cpu,
        rays_timestamps_us=None,
        _rays_footprints=None,
    )


@pytest.fixture
def simple_gaussian_parameters(device="cuda"):
    """Create simple Gaussian parameters for testing."""
    n_gaussians = 100

    # Create leaf tensors by generating values first, THEN enabling gradients
    positions = torch.randn(n_gaussians, 3, device=device)
    positions.requires_grad = True

    rotations = nn.functional.normalize(torch.randn(n_gaussians, 4, device=device), dim=-1)
    rotations.requires_grad = True

    scales = torch.exp(torch.randn(n_gaussians, 3, device=device))
    scales.requires_grad = True

    densities = torch.sigmoid(torch.randn(n_gaussians, device=device))
    densities.requires_grad = True

    features = torch.randn(n_gaussians, 3, device=device)
    features.requires_grad = True

    extra_signal = torch.zeros(n_gaussians, 0, device=device)
    camera_extra_signal = torch.zeros(n_gaussians, 0, device=device)

    return {
        "positions": positions,
        "rotations": rotations,
        "scales": scales,
        "densities": densities,
        "features": features,
        "extra_signal": extra_signal,
        "camera_extra_signal": camera_extra_signal,
    }


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_single_camera_extra_signal_real_cuda_matches_cat_and_is_read_only(
    simple_config,
    mock_model,
    simple_rendering_data,
    simple_gaussian_parameters,
    monkeypatch,
    noncontiguous,
):
    """The aliased singleton matches the former copy through the real gsplat path.

    Contiguous and strided layouts are both covered so gsplat's
    ``.contiguous()`` boundary (no-op vs copy) is exercised on the alias path.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    import nurec.models.gaussians.renderers as gaussian_renderers

    renderer = GSplatRenderer(simple_config, mock_model)
    n_gaussians = simple_gaussian_parameters["positions"].shape[0]
    # Strided [:, ::2] yields a non-contiguous [N, 4] view of an [N, 8] leaf.
    signal_width = 8 if noncontiguous else 4
    signal_values = torch.randn(n_gaussians, signal_width, device="cuda")

    def render_with(concatenate):
        parameters = {name: value.detach().clone() for name, value in simple_gaussian_parameters.items()}
        signal_base = signal_values.clone().requires_grad_()
        signal = signal_base[:, ::2] if noncontiguous else signal_base
        assert signal.is_contiguous() is (not noncontiguous)
        parameters["camera_extra_signal"] = signal
        signal_before = signal.detach().clone()
        signal_version = signal._version
        monkeypatch.setattr(gaussian_renderers, "_concatenate_extra_signals", concatenate)

        output = renderer.render(
            simple_rendering_data,
            parameters,
            n_active_features=0,
            extra_ray_signal_infos=(["semantic_logits"], [4], [lambda value: value]),
        )
        assert output.extra_ray_signals is not None
        rendered_signal = output.extra_ray_signals.semantic_logits
        assert rendered_signal is not None
        rendered_signal.sum().backward()
        assert signal_base.grad is not None
        signal_grad = signal_base.grad[:, ::2] if noncontiguous else signal_base.grad
        assert torch.count_nonzero(signal_grad) > 0
        torch.testing.assert_close(signal, signal_before, rtol=0, atol=0)
        assert signal._version == signal_version
        return rendered_signal.detach(), output.opacity.detach(), signal_grad.detach()

    optimized_concatenate = gaussian_renderers._concatenate_extra_signals
    reference = render_with(lambda parts: torch.cat(parts, dim=-1))
    candidate = render_with(optimized_concatenate)
    # Packed gsplat accumulation can vary by floating-point roundoff between
    # otherwise identical launches, so use the standard float32 tolerance here.
    for candidate_tensor, reference_tensor in zip(candidate, reference):
        torch.testing.assert_close(candidate_tensor, reference_tensor)


class TestGSplatRendererGradients:
    """Test gradient flow through GSplatRenderer."""

    def test_gradient_flow_basic(self, simple_config, mock_model, simple_rendering_data, simple_gaussian_parameters):
        """Test that gradients flow to all Gaussian parameters."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        renderer = GSplatRenderer(simple_config, mock_model)

        # Forward pass
        output = renderer.render(
            simple_rendering_data,
            simple_gaussian_parameters,
            n_active_features=0,  # DC only
            extra_ray_signal_infos=([], [], []),
            frame_meta=None,
        )

        # Compute simple loss
        loss = output.rgb.sum() + output.opacity.sum() + output.distance.sum()

        # Backward pass
        loss.backward()

        # Check gradients exist and are non-zero
        assert simple_gaussian_parameters["positions"].grad is not None
        assert torch.any(simple_gaussian_parameters["positions"].grad != 0)

        assert simple_gaussian_parameters["rotations"].grad is not None
        assert torch.any(simple_gaussian_parameters["rotations"].grad != 0)

        assert simple_gaussian_parameters["scales"].grad is not None
        assert torch.any(simple_gaussian_parameters["scales"].grad != 0)

        assert simple_gaussian_parameters["densities"].grad is not None
        assert torch.any(simple_gaussian_parameters["densities"].grad != 0)

        assert simple_gaussian_parameters["features"].grad is not None
        assert torch.any(simple_gaussian_parameters["features"].grad != 0)

        print("✓ Gradients flow to all Gaussian parameters")

    def test_gradient_shapes(self, simple_config, mock_model, simple_rendering_data, simple_gaussian_parameters):
        """Test that gradient shapes match parameter shapes."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        renderer = GSplatRenderer(simple_config, mock_model)

        # Forward + backward
        output = renderer.render(
            simple_rendering_data,
            simple_gaussian_parameters,
            n_active_features=0,
            extra_ray_signal_infos=([], [], []),
        )
        loss = output.rgb.sum()
        loss.backward()

        # Check gradient shapes (skip zero-size tensors that don't participate in the graph)
        for key, param in simple_gaussian_parameters.items():
            if param.grad is None:
                continue
            assert param.grad.shape == param.shape, (
                f"Gradient shape mismatch for {key}: grad.shape={param.grad.shape}, param.shape={param.shape}"
            )

        print("✓ Gradient shapes match parameter shapes")

    def test_no_nan_gradients(self, simple_config, mock_model, simple_rendering_data, simple_gaussian_parameters):
        """Test that gradients don't contain NaN or Inf values."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        renderer = GSplatRenderer(simple_config, mock_model)

        # Forward + backward
        output = renderer.render(
            simple_rendering_data,
            simple_gaussian_parameters,
            n_active_features=0,
            extra_ray_signal_infos=([], [], []),
        )
        loss = output.rgb.sum() + output.opacity.sum()
        loss.backward()

        # Check for NaN/Inf (skip zero-size tensors that don't participate in the graph)
        for key, param in simple_gaussian_parameters.items():
            if param.grad is None:
                continue
            assert not torch.isnan(param.grad).any(), f"NaN gradients in {key}"
            assert not torch.isinf(param.grad).any(), f"Inf gradients in {key}"

        print("✓ No NaN or Inf gradients")


class TestMemoryOptimizations:
    """Test sparse_grad and absgrad features."""

    def test_sparse_grad_config(self, mock_model, simple_rendering_data, simple_gaussian_parameters):
        """Test sparse_grad configuration.

        Note: sparse_grad in gsplat does not support batch dimensions.
        This is a known limitation of the gsplat library.
        """
        pytest.skip("sparse_grad does not support batch dimensions (gsplat expects batch_dims==())")

    def test_absgrad_storage(self, mock_model, simple_rendering_data, simple_gaussian_parameters):
        """Test that absgrad is stored in meta dict."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = GSplatRendererConfig.model_validate(
            {
                "name": "3dgs-gsplat",
                "tiling": {"camera": {"tile_width": 16, "tile_height": 16}},
                "background": {"color": 0.0},
                "culling": {"near_clip_distance": 0.01, "far_clip_distance": 1000.0},
                "projection": {"min_projected_ray_radius": 0.5477},
                "absgrad": True,  # Enable absolute gradients
            }
        )

        renderer = GSplatRenderer(config, mock_model)

        # Forward + backward
        output = renderer.render(
            simple_rendering_data,
            simple_gaussian_parameters,
            n_active_features=0,
            extra_ray_signal_infos=([], [], []),
        )
        loss = output.rgb.sum()
        loss.backward()

        # Check meta dict was stored
        assert renderer.last_rendering_meta is not None, "Meta dict not stored"
        assert "means2d" in renderer.last_rendering_meta, "means2d not in meta"

        # Check absgrad was computed (accessed after backward)
        means2d = renderer.last_rendering_meta["means2d"]
        assert hasattr(means2d, "absgrad"), "absgrad not computed"

        print("✓ absgrad stored in meta dict")

    def test_sparse_grad_with_3dgut_warning(self, mock_model, simple_rendering_data, simple_gaussian_parameters):
        """Test that sparse_grad with 3DGUT produces warning."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = GSplatRendererConfig.model_validate(
            {
                "name": "3dgut-gsplat",  # 3DGUT mode doesn't support sparse_grad
                "tiling": {"camera": {"tile_width": 16, "tile_height": 16}},
                "background": {"color": 0.0},
                "culling": {"near_clip_distance": 0.01, "far_clip_distance": 1000.0},
                "projection": {
                    "min_projected_ray_radius": 0.5477,
                    "ut_alpha": 0.1,
                    "ut_beta": 2.0,
                    "ut_kappa": 0.0,
                    "image_margin_factor": 0.1,
                },
                "sparse_grad": True,  # Should be disabled with warning
                "packed": True,
            }
        )

        renderer = GSplatRenderer(config, mock_model)

        # Should still work, just without sparse_grad
        output = renderer.render(
            simple_rendering_data,
            simple_gaussian_parameters,
            n_active_features=0,
            extra_ray_signal_infos=([], [], []),
        )

        assert output.rgb is not None
        print("✓ sparse_grad with 3DGUT handled correctly")


class TestPoseGradientLimitations:
    """Test pose gradient warnings and limitations."""

    def test_3dgut_pose_gradient_warning(self, mock_model, caplog):
        """Test that 3dgut-gsplat without ray input produces pose gradient warning."""
        import logging

        config = GSplatRendererConfig.model_validate(
            {
                "name": "3dgut-gsplat",  # Should trigger warning
                "use_rays": False,  # Warning only fires without ray input
                "tiling": {"camera": {"tile_width": 16, "tile_height": 16}},
                "background": {"color": 0.0},
            }
        )

        with caplog.at_level(logging.WARNING):
            GSplatRenderer(config, mock_model)

        # Check warning was logged
        assert any("does NOT support gradients" in record.message for record in caplog.records)
        print("✓ 3DGUT pose gradient warning produced")

    def test_3dgs_no_warning(self, mock_model, caplog):
        """Test that 3dgs-gsplat does NOT produce pose gradient warning."""
        import logging

        config = GSplatRendererConfig.model_validate(
            {
                "name": "3dgs-gsplat",  # Should NOT trigger warning
                "tiling": {"camera": {"tile_width": 16, "tile_height": 16}},
                "background": {"color": 0.0},
            }
        )

        with caplog.at_level(logging.WARNING):
            GSplatRenderer(config, mock_model)

        # Check NO warning was logged
        assert not any("does NOT support gradients" in record.message for record in caplog.records)
        print("✓ 3DGS (non-3DGUT) produces no warning")

    def test_3dgut_rejects_antialiased_rasterization(self):
        """Test that 3dgut-gsplat rejects antialiased rasterization at config validation time."""
        with pytest.raises(ValueError, match="match 3DGUT opacity behavior"):
            GSplatRendererConfig.model_validate(
                {
                    "name": "3dgut-gsplat",
                    "rasterize_mode": "antialiased",
                    "tiling": {"camera": {"tile_width": 16, "tile_height": 16}},
                    "background": {"color": 0.0},
                }
            )


class TestPoseGradientsThroughRays:
    """3dgut-gsplat WITH ray input (use_rays=True) supports pose gradients via ray gradients.

    Counterpart to TestPoseGradientLimitations, whose warning only fires for use_rays=False.
    With ray input the rasterizer returns dL/d(ray_o, ray_d) (``v_rays``); that gradient lands on
    the NuRec-calibrated rays and flows back to camera pose. ``test_3dgut_ray_gradients_flow`` guards
    the gsplat half of the chain (loss -> v_rays); ``test_3dgut_pose_gradients_flow_through_rays``
    guards the full pose -> rays (sensorlib) -> render -> pose chain. The standalone rays -> pose
    half is also covered by calib_test.py.
    """

    _H = 64
    _W = 64
    _FOCAL = 50.0

    @classmethod
    def _opencv_params(cls) -> OpenCVPinholeCameraModelParameters:
        return OpenCVPinholeCameraModelParameters(
            resolution=np.array([cls._W, cls._H], dtype=np.uint64),
            focal_length=np.array([cls._FOCAL, cls._FOCAL], dtype=np.float32),
            principal_point=np.array([cls._W / 2, cls._H / 2], dtype=np.float32),
            radial_coeffs=np.zeros(6, dtype=np.float32),
            tangential_coeffs=np.zeros(2, dtype=np.float32),
            thin_prism_coeffs=np.zeros(4, dtype=np.float32),
            shutter_type=ShutterType.GLOBAL,
        )

    @classmethod
    def _gen_world_rays(cls, device, *, translation_delta=None, rotation_delta=None):
        """Generate coherent world rays from an identity pinhole pose via the production sensorlib
        op (the same op NuRec's calibration path uses). Differentiable w.r.t. the camera translation
        when ``translation_delta`` requires grad, and w.r.t. the camera rotation when
        ``rotation_delta`` (a 3-vector perturbing the wxyz quaternion's xyz part) requires grad."""
        proj = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([cls._FOCAL, cls._FOCAL], device=device),
            principal_point=torch.tensor([cls._W / 2, cls._H / 2], device=device),
            radial_coeffs=torch.zeros(6, device=device),
            tangential_coeffs=torch.zeros(2, device=device),
            thin_prism_coeffs=torch.zeros(4, device=device),
            resolution=torch.tensor([cls._W, cls._H], device=device),
        )
        translation = torch.zeros(3, device=device)
        if translation_delta is not None:
            translation = translation + translation_delta
        rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # identity quaternion (wxyz)
        if rotation_delta is not None:
            rotation = nn.functional.normalize(torch.cat([torch.ones(1, device=device), rotation_delta]), dim=-1)
        pose = Pose(translation=translation, rotation=rotation)
        world_rays, *_ = image_points_to_world_rays_shutter_pose(
            image_points=None,
            projection=proj,
            external_distortion=NoExternalDistortion(),
            resolution=(cls._W, cls._H),
            shutter_type=SensorlibShutterType.GLOBAL,
            dynamic_pose=DynamicPose(start_pose=pose, end_pose=pose),
            return_timestamps=False,
        )
        return world_rays.reshape(cls._H, cls._W, 6).unsqueeze(0)  # (1, H, W, 6)

    @classmethod
    def _make_rendering_data(cls, rays, device):
        # Identity pose: translation (3) + quaternion xyzw (4) — the renderer's viewmat convention.
        pose_tquat = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], device=device)
        timestamps = torch.tensor([[0, 1000]], device=device, dtype=torch.int64)
        return RenderingData(
            rays=rays,
            sensor_model_parameters=[cls._opencv_params()],
            poses_tquat_startend=pose_tquat.unsqueeze(1).repeat(1, 2, 1),
            timestamps_startend_us=timestamps,
            timestamps_startend_us_cpu=timestamps.cpu(),
            rays_timestamps_us=None,
            _rays_footprints=None,
        )

    @staticmethod
    def _make_3dgut_config():
        return GSplatRendererConfig.model_validate(
            {
                "name": "3dgut-gsplat",
                "use_rays": True,
                "tiling": {"camera": {"tile_width": 16, "tile_height": 16}},
                "background": {"color": 0.0},
                "culling": {"near_clip_distance": 0.01, "far_clip_distance": 1000.0},
                "projection": {
                    "min_projected_ray_radius": 0.5477,
                    "ut_alpha": 0.1,
                    "ut_beta": 2.0,
                    "ut_kappa": 0.0,
                    "image_margin_factor": 0.1,
                },
                "rasterize_mode": "classic",
                "packed": False,
                "sparse_grad": False,
                "absgrad": False,
            }
        )

    def test_3dgut_ray_gradients_flow(self, mock_model, simple_gaussian_parameters):
        """Backward through 3dgut-gsplat (use_rays=True) produces usable gradients on the input
        rays — the gsplat half of the pose-gradient chain (loss -> v_rays)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")

        rays = self._gen_world_rays(device).detach().clone().requires_grad_(True)
        rendering_data = self._make_rendering_data(rays, device)
        renderer = GSplatRenderer(self._make_3dgut_config(), mock_model)

        output = renderer.render(
            rendering_data,
            simple_gaussian_parameters,
            n_active_features=0,
            extra_ray_signal_infos=([], [], []),
            frame_meta=None,
        )
        loss = output.rgb.sum() + output.opacity.sum() + output.distance.sum()
        loss.backward()

        assert rays.grad is not None, "No gradient on rays — gsplat did not return v_rays"
        assert torch.isfinite(rays.grad).all(), "Non-finite ray gradients"
        # rays are (1, H, W, 6): [..., :3] origins, [..., 3:] directions. Prove BOTH halves carry
        # gradient — a regression that drops direction grads can still leave origin grads nonzero.
        assert rays.grad[..., :3].abs().sum() > 0, "Ray-origin gradients are all zero"
        assert rays.grad[..., 3:].abs().sum() > 0, "Ray-direction gradients are all zero"
        print("✓ 3dgut-gsplat returns usable gradients on input rays")

    def test_3dgut_pose_gradients_flow_through_rays(self, mock_model, simple_gaussian_parameters):
        """End-to-end: a learnable camera-translation delta -> sensorlib ray generation -> 3dgut-gsplat
        render -> loss yields a nonzero gradient on the delta. This covers the renderer-side
        gradient path that pose-optimization workflows rely on when they pass rays
        (use_rays=True, the default for 3dgut-gsplat)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")

        translation_delta = torch.zeros(3, device=device, requires_grad=True)
        rays = self._gen_world_rays(device, translation_delta=translation_delta)
        rendering_data = self._make_rendering_data(rays, device)
        renderer = GSplatRenderer(self._make_3dgut_config(), mock_model)

        output = renderer.render(
            rendering_data,
            simple_gaussian_parameters,
            n_active_features=0,
            extra_ray_signal_infos=([], [], []),
            frame_meta=None,
        )
        loss = output.rgb.sum() + output.opacity.sum() + output.distance.sum()
        loss.backward()

        assert translation_delta.grad is not None, "No gradient reached the camera pose delta"
        assert torch.isfinite(translation_delta.grad).all(), "Non-finite pose gradients"
        assert translation_delta.grad.abs().sum() > 0, "Pose gradient is zero through 3dgut-gsplat"
        print("✓ 3dgut-gsplat propagates gradients to camera pose via rays")

    def test_3dgut_rotation_gradients_flow_through_rays(self, mock_model, simple_gaussian_parameters):
        """End-to-end rotation counterpart: a learnable camera-rotation delta -> sensorlib ray
        generation -> 3dgut-gsplat render -> loss yields a nonzero gradient on the delta. Guards the
        rotation half of the pose-gradient chain, which the translation-only variant leaves untested
        (a regression dropping ray-direction gradients can still pass the translation test)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")

        rotation_delta = torch.zeros(3, device=device, requires_grad=True)
        rays = self._gen_world_rays(device, rotation_delta=rotation_delta)
        rendering_data = self._make_rendering_data(rays, device)
        renderer = GSplatRenderer(self._make_3dgut_config(), mock_model)

        output = renderer.render(
            rendering_data,
            simple_gaussian_parameters,
            n_active_features=0,
            extra_ray_signal_infos=([], [], []),
            frame_meta=None,
        )
        loss = output.rgb.sum() + output.opacity.sum() + output.distance.sum()
        loss.backward()

        assert rotation_delta.grad is not None, "No gradient reached the camera rotation delta"
        assert torch.isfinite(rotation_delta.grad).all(), "Non-finite rotation gradients"
        assert rotation_delta.grad.abs().sum() > 0, "Rotation gradient is zero through 3dgut-gsplat"
        print("✓ 3dgut-gsplat propagates gradients to camera rotation via rays")


class TestSceneDataOutputs:
    """Test scene-level gsplat outputs returned by the renderer."""

    _H, _W, _N = 4, 5, 3

    @pytest.fixture
    def scene_test_data(self):
        """Shared small-resolution rendering data for scene-output tests."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")
        h, w, n = self._H, self._W, self._N
        sensor_params = OpenCVPinholeCameraModelParameters(
            resolution=np.array([w, h], dtype=np.uint64),
            focal_length=np.array([50.0, 50.0], dtype=np.float32),
            principal_point=np.array([w / 2, h / 2], dtype=np.float32),
            radial_coeffs=np.zeros(6, dtype=np.float32),
            tangential_coeffs=np.zeros(2, dtype=np.float32),
            thin_prism_coeffs=np.zeros(4, dtype=np.float32),
            shutter_type=ShutterType.GLOBAL,
        )
        pose_tquat = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
        timestamps = torch.tensor([[0, 1000]], dtype=torch.int64, device=device)
        rendering_data = RenderingData(
            rays=torch.randn(1, h, w, 6, device=device),
            sensor_model_parameters=[sensor_params],
            poses_tquat_startend=pose_tquat.unsqueeze(1).repeat(1, 2, 1),
            timestamps_startend_us=timestamps,
            timestamps_startend_us_cpu=timestamps.cpu(),
            rays_timestamps_us=None,
            _rays_footprints=None,
        )
        gaussian_parameters = {
            "positions": torch.randn(n, 3, device=device),
            "rotations": nn.functional.normalize(torch.randn(n, 4, device=device), dim=-1),
            "scales": torch.rand(n, 3, device=device),
            "densities": torch.sigmoid(torch.randn(n, device=device)),
            "features": torch.rand(n, 3, device=device),
            "extra_signal": torch.zeros(n, 0, device=device),
            "camera_extra_signal": torch.zeros(n, 0, device=device),
        }
        return rendering_data, gaussian_parameters, device

    def test_scene_data_is_flattened(self, simple_config, mock_model, scene_test_data, monkeypatch):
        """Singleton batch/view axes from gsplat meta should collapse to per-Gaussian vectors."""
        import nurec.models.gaussians.renderers as gaussian_renderers

        rendering_data, gaussian_parameters, device = scene_test_data
        h, w, n = self._H, self._W, self._N
        renderer = GSplatRenderer(simple_config, mock_model)

        def fake_rasterization(*args, **kwargs):
            render_colors = torch.zeros(1, 1, h, w, 4, dtype=torch.float32, device=device)
            render_alphas = torch.zeros(1, 1, h, w, 1, dtype=torch.float32, device=device)
            meta = {
                "radii": torch.tensor(
                    [[[[0.0, 0.0], [1.0, 0.5], [0.0, 2.0]]]],
                    dtype=torch.float32,
                    device=device,
                )
            }
            return render_colors, render_alphas, meta

        monkeypatch.setattr(gaussian_renderers.gsplat, "rasterization", fake_rasterization)

        output = renderer.render(
            rendering_data,
            gaussian_parameters,
            n_active_features=0,
            extra_ray_signal_infos=([], [], []),
        )

        assert output.visibility is not None
        assert output.visibility.shape == (n,)
        assert torch.equal(output.visibility, torch.tensor([0.0, 1.0, 1.0], device=device))

    def test_flat_radii_passed_through(self, simple_config, mock_model, scene_test_data, monkeypatch):
        """Flat (N,) radii from gsplat should produce (N,) visibility without amax."""
        import nurec.models.gaussians.renderers as gaussian_renderers

        rendering_data, gaussian_parameters, device = scene_test_data
        h, w, n = self._H, self._W, self._N
        renderer = GSplatRenderer(simple_config, mock_model)

        def fake_rasterization(*args, **kwargs):
            render_colors = torch.zeros(1, 1, h, w, 4, dtype=torch.float32, device=device)
            render_alphas = torch.zeros(1, 1, h, w, 1, dtype=torch.float32, device=device)
            meta = {
                "radii": torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32, device=device),
            }
            return render_colors, render_alphas, meta

        monkeypatch.setattr(gaussian_renderers.gsplat, "rasterization", fake_rasterization)

        output = renderer.render(
            rendering_data,
            gaussian_parameters,
            n_active_features=0,
            extra_ray_signal_infos=([], [], []),
        )

        assert output.visibility is not None
        assert output.visibility.shape == (n,)
        assert torch.equal(output.visibility, torch.tensor([0.0, 1.0, 1.0], device=device))


class TestUnstructuredLidarIntensity:
    """Unit coverage for the model-free LiDAR intensity bridge."""

    @staticmethod
    def _renderer() -> GSplatRenderer:
        renderer = object.__new__(GSplatRenderer)
        renderer._lidar_extra_signals_sh_degree = None
        return renderer

    def test_selects_intensity_from_active_lidar_extra_signal_layout(self):
        renderer = self._renderer()
        common = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
        lidar = torch.tensor([[0.25, 1.0, 2.0], [0.75, 3.0, 4.0]])

        intensity = renderer._get_unstructured_lidar_intensity(
            {"extra_signal": common, "lidar_extra_signal": lidar},
            enable_extended_features=True,
            enable_sensor_extended_features=True,
            extra_ray_signal_infos=(
                ["common_signal", "intensity", "raydrop"],
                [2, 1, 2],
                [lambda x: x, lambda x: x, lambda x: x],
            ),
        )

        assert intensity is not None
        torch.testing.assert_close(intensity, lidar[:, :1])

    def test_selects_lidar_intensity_when_common_output_gate_is_disabled(self):
        renderer = self._renderer()
        common = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
        lidar = torch.tensor([[0.25, 1.0, 2.0], [0.75, 3.0, 4.0]])

        intensity = renderer._get_unstructured_lidar_intensity(
            {"extra_signal": common, "lidar_extra_signal": lidar},
            enable_extended_features=False,
            enable_sensor_extended_features=True,
            extra_ray_signal_infos=(
                ["common_signal", "intensity", "raydrop"],
                [2, 1, 2],
                [lambda x: x, lambda x: x, lambda x: x],
            ),
        )

        assert intensity is not None
        torch.testing.assert_close(intensity, lidar[:, :1])

    def test_evaluates_sh_lidar_intensity_at_midframe_pose(self, monkeypatch):
        import nurec.models.gaussians.renderers as gaussian_renderers

        renderer = self._renderer()
        renderer._lidar_extra_signals_sh_degree = 1
        means = torch.tensor([[3.0, 1.0, 0.0], [5.0, 0.0, 0.0]], requires_grad=True)
        coefficients = torch.tensor(
            [
                [[0.10, 1.0, 2.0], [0.20, 3.0, 4.0], [0.30, 5.0, 6.0], [0.40, 7.0, 8.0]],
                [[0.50, 9.0, 10.0], [0.60, 11.0, 12.0], [0.70, 13.0, 14.0], [0.80, 15.0, 16.0]],
            ],
            requires_grad=True,
        )
        captured: dict[str, torch.Tensor | int] = {}

        def fake_spherical_harmonics(degree, dirs, coeffs, masks=None):
            captured["degree"] = degree
            captured["dirs"] = dirs
            captured["coeffs"] = coeffs
            return coeffs[:, 0, :] + coeffs[:, 1, :] * dirs

        monkeypatch.setattr(gaussian_renderers.gsplat, "spherical_harmonics", fake_spherical_harmonics)

        # This world->sensor transform places the sensor at x=2 in world space.
        viewmat_midframe = torch.eye(4)
        viewmat_midframe[0, 3] = -2.0

        intensity = renderer._get_unstructured_lidar_intensity(
            {"extra_signal": torch.zeros(2, 0), "lidar_extra_signal": coefficients.reshape(2, -1)},
            enable_extended_features=False,
            enable_sensor_extended_features=True,
            extra_ray_signal_infos=(
                ["intensity", "raydrop"],
                [1, 2],
                [lambda x: x, lambda x: x],
            ),
            means=means,
            viewmat_midframe=viewmat_midframe,
        )

        expected_dirs = torch.tensor([[1.0, 1.0, 0.0], [3.0, 0.0, 0.0]])
        expected = coefficients[:, 0, :1] + coefficients[:, 1, :1] * expected_dirs[:, :1] + 0.5
        assert intensity is not None
        assert captured["degree"] == 1
        torch.testing.assert_close(captured["dirs"], expected_dirs)
        torch.testing.assert_close(captured["coeffs"], coefficients)
        torch.testing.assert_close(intensity, expected)

        intensity.sum().backward()
        assert coefficients.grad is not None
        assert coefficients.grad.abs().sum() > 0
        assert means.grad is not None
        assert means.grad.abs().sum() > 0

    def test_unstructured_render_uses_public_rasterization_extra_signals(self, monkeypatch):
        import nurec.models.gaussians.renderers as gaussian_renderers

        class FakeFrame:
            tiling = SimpleNamespace(time_tiled=False)

            @classmethod
            def from_tensors(cls, **kwargs):
                captured["frame_tensors"] = kwargs
                return cls()

            def multi_visit_fraction(self) -> float:
                return 0.0

        captured: dict[str, Any] = {}

        def fake_rasterization(**kwargs):
            captured.update(kwargs)
            return (
                torch.tensor([[[[[3.0], [0.0], [7.0]]]]]),
                torch.tensor([[[[[0.6], [0.0], [0.8]]]]]),
                {"render_extra_signals": torch.tensor([[[[[0.3], [0.0], [0.9]]]]])},
            )

        monkeypatch.setattr(gaussian_renderers.gsplat, "UnstructuredLidarFrameContext", FakeFrame)
        monkeypatch.setattr(gaussian_renderers.gsplat, "rasterization", fake_rasterization)

        renderer = self._renderer()
        renderer._unstructured_logged = True
        renderer.config = SimpleNamespace(
            renderer_config="parallel_batch",
            culling=SimpleNamespace(
                cull_dead_pairs_lidar_unstructured=False,
                box_method_lidar_unstructured="ut",
                pbf_near_range_lidar_unstructured=8.0,
                rs_single_cover_lidar_unstructured=False,
            ),
        )
        rendering_data = SimpleNamespace(
            rays=[
                torch.tensor(
                    [
                        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
                        [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                    ]
                )
            ],
            rays_timestamps_us=[torch.tensor([[0], [5], [10]])],
            timestamps_startend_us_cpu=torch.tensor([[0, 10]]),
            # Real RenderingData always carries this; _render_unstructured rejects a
            # non-None mask (active tiling is a structured-grid concept).
            active_tile_mask=None,
            unstructured_lidar_tiling=SimpleNamespace(
                directions=torch.zeros(3, 3),
                az=torch.zeros(3),
                el=torch.zeros(3),
                relative_timestamps=torch.tensor([0.0, 0.5, 1.0]),
                occupancy=torch.ones(1, 1, dtype=torch.bool),
                tiles_pack_info=torch.zeros(1, 2, dtype=torch.int32),
                tiles_to_elements_map=torch.zeros(3, 2, dtype=torch.int32),
                ray_index=torch.arange(3),
                time_mask=torch.zeros(1, dtype=torch.int32),
                tile_mean_time=torch.zeros(1),
                meta={},
            ),
        )
        gaussian_params = SimpleNamespace(
            means=torch.zeros(4, 3),
            quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1),
            scales=torch.ones(4, 3),
            opacities=torch.full((4,), 0.5),
        )
        intensity = torch.tensor([[0.1], [0.2], [0.3], [0.4]])

        output = renderer._render_unstructured(
            rendering_data=rendering_data,
            gaussian_params=gaussian_params,
            viewmat_start=torch.eye(4),
            viewmat_end=torch.eye(4),
            near_plane=0.0,
            far_plane=100.0,
            enable_distance=True,
            intensity=intensity,
            device=torch.device("cpu"),
        )

        assert captured["extra_signals"].shape == (4, 1)
        torch.testing.assert_close(captured["extra_signals"], intensity)
        assert captured["lidar_frame_context"].__class__ is FakeFrame
        assert isinstance(captured["renderer_config"], gaussian_renderers.gsplat.RendererConfig_ParallelBatch)
        assert captured["renderer_config"].unstructured_lidar.arc_sampling == "endpoints"
        assert not captured["renderer_config"].unstructured_lidar.cull_dead_pairs
        torch.testing.assert_close(output.distance, torch.tensor([3.0, 0.0, 7.0]))
        assert output.extra_ray_signals is not None
        torch.testing.assert_close(
            output.extra_ray_signals.intensity,
            torch.tensor([[0.3], [0.0], [0.9]]),
        )
        torch.testing.assert_close(
            output.extra_ray_signals.raydrop,
            torch.tensor([[0.4], [1.0], [0.2]]),
        )

    def test_cuda_fused_unstructured_render_backpropagates_intensity(self, monkeypatch):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        import gsplat.cuda._torch_unstructured_render as unstructured_render_module

        import nurec.models.gaussians.renderers as gaussian_renderers

        def torch_reference_must_not_run(*_args, **_kwargs):
            raise AssertionError("unstructured intensity test must use the fused CUDA renderer")

        monkeypatch.setattr(unstructured_render_module, "_render_torch_reference", torch_reference_must_not_run)

        device = torch.device("cuda")
        renderer = self._renderer()
        renderer._unstructured_logged = True
        renderer.config = SimpleNamespace(
            renderer_config="parallel_batch",
            culling=SimpleNamespace(
                cull_dead_pairs_lidar_unstructured=False,
                box_method_lidar_unstructured="ut",
                pbf_near_range_lidar_unstructured=8.0,
                rs_single_cover_lidar_unstructured=False,
            ),
        )
        rays = torch.tensor(
            [
                [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
            ],
            device=device,
        )
        timestamps_us = torch.tensor([0, 5, 10], device=device)
        frame_context = gaussian_renderers.gsplat.UnstructuredLidarFrameContext(
            rays[:, 0, 3:],
            timestamps_us=timestamps_us,
        )
        rendering_data = SimpleNamespace(
            rays=[rays],
            rays_timestamps_us=[timestamps_us[:, None]],
            timestamps_startend_us_cpu=torch.tensor([[0, 10]]),
            active_tile_mask=None,
            unstructured_lidar_tiling=SimpleNamespace(**frame_context.to_tensors()),
        )
        gaussian_params = SimpleNamespace(
            means=torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], device=device),
            quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(3, 1),
            scales=torch.full((3, 3), 1.5, device=device),
            opacities=torch.full((3,), 0.8, device=device),
        )
        intensity = torch.tensor([[0.2], [0.5], [0.8]], device=device, requires_grad=True)

        output = renderer._render_unstructured(
            rendering_data=rendering_data,
            gaussian_params=gaussian_params,
            viewmat_start=torch.eye(4, device=device),
            viewmat_end=torch.eye(4, device=device),
            near_plane=0.0,
            far_plane=100.0,
            enable_distance=True,
            intensity=intensity,
            device=device,
        )

        assert output.extra_ray_signals is not None
        loss = output.extra_ray_signals.intensity.sum()
        loss.backward()

        assert intensity.grad is not None
        assert torch.isfinite(intensity.grad).all()
        assert intensity.grad.abs().sum() > 0


class TestTrainingIntegration:
    """Test integration with training workflow."""

    def test_training_step_completes(
        self, simple_config, mock_model, simple_rendering_data, simple_gaussian_parameters
    ):
        """Test that a full training step completes without error."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        renderer = GSplatRenderer(simple_config, mock_model)

        # Simulate training step
        optimizer = torch.optim.Adam(
            [
                simple_gaussian_parameters["positions"],
                simple_gaussian_parameters["rotations"],
                simple_gaussian_parameters["scales"],
                simple_gaussian_parameters["densities"],
                simple_gaussian_parameters["features"],
            ],
            lr=0.001,
        )

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = renderer.render(
            simple_rendering_data,
            simple_gaussian_parameters,
            n_active_features=0,
            extra_ray_signal_infos=([], [], []),
        )

        # Compute loss
        loss = output.rgb.sum() + output.opacity.sum()

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        print("✓ Training step completes successfully")

    def test_multiple_training_steps(
        self, simple_config, mock_model, simple_rendering_data, simple_gaussian_parameters
    ):
        """Test multiple training steps complete without error."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        renderer = GSplatRenderer(simple_config, mock_model)

        optimizer = torch.optim.Adam(
            [
                simple_gaussian_parameters["positions"],
                simple_gaussian_parameters["rotations"],
                simple_gaussian_parameters["scales"],
                simple_gaussian_parameters["densities"],
                simple_gaussian_parameters["features"],
            ],
            lr=0.001,
        )

        losses = []
        for step in range(5):
            optimizer.zero_grad()

            output = renderer.render(
                simple_rendering_data,
                simple_gaussian_parameters,
                n_active_features=0,
                extra_ray_signal_infos=([], [], []),
            )

            loss = output.rgb.sum() + output.opacity.sum()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should change over steps (parameters are updating)
        assert not all(loss == losses[0] for loss in losses), "Loss not changing (parameters not updating)"
        print(f"✓ Multiple training steps complete (losses: {losses})")


if __name__ == "__main__":
    # Run tests manually
    pytest.main([__file__, "-v", "-s"])


class TestBand0FoldEligibility:
    """The fold-eligibility predicate is the single source of truth shared by the
    collector and render(). These tests pin its terms directly, plus the fact that
    render() forwards the fourier_* arguments if and only if it answers True."""

    _H, _W, _N = 16, 16, 64

    @staticmethod
    def _config(name="3dgut-gsplat", camera_outputs=None, lidar_outputs=None, **overrides):
        outputs = {
            "camera": {"enable_features": True, "enable_normals": False, **(camera_outputs or {})},
            "lidar": {"enable_features": True, "enable_normals": False, **(lidar_outputs or {})},
        }
        return GSplatRendererConfig.model_validate(
            {
                "name": name,
                "use_rays": True,
                "tiling": {"camera": {"tile_width": 16, "tile_height": 16}},
                "background": {"color": 0.0},
                "culling": {"near_clip_distance": 0.01, "far_clip_distance": 1000.0},
                "projection": {"min_projected_ray_radius": 0.5477},
                "rasterize_mode": "classic",
                "packed": False,
                "sparse_grad": False,
                "absgrad": False,
                "outputs": outputs,
                **overrides,
            }
        )

    @staticmethod
    def _query(renderer, **overrides):
        kwargs = {
            "is_camera": True,
            "features_dim": 48,
            "extra_signal_dim": 0,
            "sensor_extra_signal_dim": 0,
        }
        kwargs.update(overrides)
        with torch.no_grad():
            return renderer.band0_fold_supported(**kwargs)

    def test_supported_for_an_sh_camera_render(self, mock_model):
        renderer = GSplatRenderer(self._config(), mock_model)
        assert self._query(renderer) is True

    def test_rgb_only_features_are_not_foldable(self, mock_model):
        # features of width 3 are the post-activation RGB path: gsplat renders those
        # with sh_degree=None and never applies the fold.
        renderer = GSplatRenderer(self._config(), mock_model)
        assert self._query(renderer, features_dim=3) is False

    def test_disabled_features_are_not_foldable(self, mock_model):
        # No features means a depth-only render mode, which has no SH colour to fold.
        renderer = GSplatRenderer(self._config(camera_outputs={"enable_features": False}), mock_model)
        assert self._query(renderer) is False

    def test_extra_signals_are_not_foldable(self, mock_model):
        renderer = GSplatRenderer(self._config(camera_outputs={"enable_extended_features": True}), mock_model)
        assert self._query(renderer, extra_signal_dim=20) is False
        renderer2 = GSplatRenderer(self._config(camera_outputs={"enable_sensor_features": True}), mock_model)
        assert self._query(renderer2, sensor_extra_signal_dim=20) is False

    def test_extra_signal_dims_are_ignored_when_their_gate_is_off(self, mock_model):
        renderer = GSplatRenderer(
            self._config(camera_outputs={"enable_extended_features": False, "enable_sensor_features": False}),
            mock_model,
        )
        assert self._query(renderer, extra_signal_dim=20, sensor_extra_signal_dim=20) is True

    def test_non_3dgut_camera_is_not_foldable(self, mock_model):
        renderer = GSplatRenderer(self._config(name="3dgs-gsplat"), mock_model)
        assert self._query(renderer) is False

    def test_lidar_is_foldable_because_its_render_is_always_3dgut(self, mock_model):
        # The lidar branch of render() hardcodes 3DGUT regardless of the config name,
        # so the predicate must too, or it would refuse a fold the render supports.
        renderer = GSplatRenderer(self._config(name="3dgs-gsplat"), mock_model)
        assert self._query(renderer, is_camera=False) is True

    def test_normals_do_not_block_the_fold(self, mock_model):
        # Normals come out of the eval3d rasterizer and never touch the fold branch,
        # so a normals-enabled render (what training uses) stays foldable.
        renderer = GSplatRenderer(self._config(camera_outputs={"enable_normals": True}), mock_model)
        assert self._query(renderer) is True

    def test_grad_mode_is_not_foldable(self, mock_model):
        renderer = GSplatRenderer(self._config(), mock_model)
        with torch.enable_grad():
            assert (
                renderer.band0_fold_supported(
                    is_camera=True, features_dim=48, extra_signal_dim=0, sensor_extra_signal_dim=0
                )
                is False
            )

    # ---- render() honors exactly what the predicate answered ----
    def _scene(self, device, features_dim):
        rays = torch.randn(1, self._H, self._W, 6, device=device)
        sensor_params = OpenCVPinholeCameraModelParameters(
            resolution=np.array([self._W, self._H], dtype=np.uint64),
            focal_length=np.array([50.0, 50.0], dtype=np.float32),
            principal_point=np.array([self._W / 2, self._H / 2], dtype=np.float32),
            radial_coeffs=np.zeros(6, dtype=np.float32),
            tangential_coeffs=np.zeros(2, dtype=np.float32),
            thin_prism_coeffs=np.zeros(4, dtype=np.float32),
            shutter_type=ShutterType.GLOBAL,
        )
        pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).unsqueeze(1).repeat(1, 2, 1)
        timestamps = torch.tensor([[0, 1000]], device=device, dtype=torch.int64)
        rendering_data = RenderingData(
            rays=rays,
            sensor_model_parameters=[sensor_params],
            poses_tquat_startend=pose.to(device),
            timestamps_startend_us=timestamps,
            timestamps_startend_us_cpu=timestamps.cpu(),
            rays_timestamps_us=None,
            _rays_footprints=None,
        )
        n = self._N
        gaussian_parameters = {
            "positions": torch.randn(n, 3, device=device),
            "rotations": nn.functional.normalize(torch.randn(n, 4, device=device), dim=-1),
            "scales": torch.exp(torch.randn(n, 3, device=device)),
            "densities": torch.sigmoid(torch.randn(n, device=device)),
            "features": torch.randn(n, features_dim, device=device),
            "extra_signal": torch.zeros(n, 0, device=device),
            "camera_extra_signal": torch.zeros(n, 0, device=device),
            "lidar_extra_signal": torch.zeros(n, 0, device=device),
        }
        return rendering_data, gaussian_parameters

    def _fold_ref(self, device):
        return {
            "albedo": torch.randn(self._N, 3, 3, device=device),
            "frame_timestamp_us": 5000,
            "timestamps_us_min": 1000,
            "timestamps_us_max": 9000,
            "remap_min": -1.0,
            "remap_max": 1.0,
            "start": 0,
            "size": self._N,
        }

    def _render_capturing_kwargs(
        self, renderer, rendering_data, gaussian_parameters, monkeypatch, n_active_features, segment_plan=None
    ):
        """Return the keyword dictionary passed to ``gsplat.rasterization``."""
        import nurec.models.gaussians.renderers as gaussian_renderers

        # Each key is a rasterization parameter name. Each value is its passed value.
        captured = {}

        def fake_rasterization(*args, **kwargs):
            captured.update(kwargs)
            device = gaussian_parameters["positions"].device
            return (
                torch.zeros(1, 1, self._H, self._W, 4, device=device),
                torch.zeros(1, 1, self._H, self._W, 1, device=device),
                {"radii": torch.zeros(1, 1, self._N, 2, device=device)},
            )

        monkeypatch.setattr(gaussian_renderers.gsplat, "rasterization", fake_rasterization)
        with torch.no_grad():
            renderer.render(
                rendering_data,
                gaussian_parameters,
                n_active_features=n_active_features,
                extra_ray_signal_infos=([], [], []),
                segment_plan=segment_plan,
            )
        return captured

    def test_sparse_render_without_a_plan_draws_dense(self, mock_model, monkeypatch):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")
        renderer = GSplatRenderer(self._config(sparse_segments=True), mock_model)
        rendering_data, gaussian_parameters = self._scene(device, features_dim=48)

        captured = self._render_capturing_kwargs(
            renderer,
            rendering_data,
            gaussian_parameters,
            monkeypatch,
            n_active_features=3,
            segment_plan=None,
        )
        assert "segments" not in captured, "a fully live scene must take the dense path"

    def test_sparse_render_rejects_extra_signals(self, mock_model, monkeypatch):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")
        renderer = GSplatRenderer(
            self._config(sparse_segments=True, camera_outputs={"enable_extended_features": True}),
            mock_model,
        )
        rendering_data, gaussian_parameters = self._scene(device, features_dim=48)
        gaussian_parameters["extra_signal"] = torch.ones(self._N, 1, device=device)
        # Shape [2, 1]: one segment selects every Gaussian row.
        plan = SegmentPlan(
            segments=torch.tensor([[0], [self._N]], dtype=torch.int32, device=device),
            segment_offsets=torch.tensor([0, self._N], dtype=torch.int32, device=device),
            active_count=self._N,
            total_rows=self._N,
        )

        with pytest.raises(ValueError, match="does not support extra signals"):
            self._render_capturing_kwargs(
                renderer,
                rendering_data,
                gaussian_parameters,
                monkeypatch,
                n_active_features=3,
                segment_plan=plan,
            )

    def test_sparse_render_forwards_the_explicit_segment_plan(self, mock_model, monkeypatch):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")
        renderer = GSplatRenderer(self._config(sparse_segments=True), mock_model)
        rendering_data, gaussian_parameters = self._scene(device, features_dim=48)
        # Shape [2, 2]: row 0 stores starts, and row 1 stores counts.
        plan = SegmentPlan(
            segments=torch.tensor([[0, 16], [7, 3]], dtype=torch.int32, device=device),
            segment_offsets=torch.tensor([0, 7, 10], dtype=torch.int32, device=device),
            active_count=10,
            total_rows=self._N,
        )
        captured = self._render_capturing_kwargs(
            renderer,
            rendering_data,
            gaussian_parameters,
            monkeypatch,
            n_active_features=3,
            segment_plan=plan,
        )
        assert torch.equal(captured["segments"], plan.segments)
        assert torch.equal(captured["segment_offsets"], plan.segment_offsets)
        assert captured["active_count"] == 10

    def test_empty_segment_plan_renders_no_gaussians(self, mock_model):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")
        renderer = GSplatRenderer(self._config(sparse_segments=True, use_rays=False), mock_model)
        rendering_data, gaussian_parameters = self._scene(device, features_dim=48)
        plan = SegmentPlan(
            segments=torch.empty((2, 0), dtype=torch.int32, device=device),
            segment_offsets=torch.zeros(1, dtype=torch.int32, device=device),
            active_count=0,
            total_rows=self._N,
        )

        with torch.no_grad():
            output = renderer.render(
                rendering_data,
                gaussian_parameters,
                n_active_features=3,
                extra_ray_signal_infos=([], [], []),
                segment_plan=plan,
            )

        assert torch.count_nonzero(output.opacity) == 0

    def test_partial_segments_match_dense_when_excluded_rows_have_zero_density(self, mock_model):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")
        rendering_data, gaussian_parameters = self._scene(device, features_dim=48)
        active_count = self._N // 2

        positions = torch.zeros(self._N, 3, device=device)
        positions[:, 0] = torch.linspace(-0.5, 0.5, self._N, device=device)
        positions[:, 1] = torch.linspace(0.5, -0.5, self._N, device=device)
        positions[:, 2] = 4.0
        gaussian_parameters["positions"] = positions
        gaussian_parameters["rotations"] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(self._N, -1)
        gaussian_parameters["scales"] = torch.full((self._N, 3), 0.08, device=device)
        gaussian_parameters["densities"] = torch.cat(
            (
                torch.full((active_count,), 0.8, device=device),
                torch.zeros(self._N - active_count, device=device),
            )
        )
        gaussian_parameters["features"] = torch.zeros(self._N, 48, device=device)
        gaussian_parameters["features"][:, :3] = 0.5
        # Shape [2, 1]: one start row and one row count select the visible prefix.
        plan = SegmentPlan(
            segments=torch.tensor([[0], [active_count]], dtype=torch.int32, device=device),
            segment_offsets=torch.tensor([0, active_count], dtype=torch.int32, device=device),
            active_count=active_count,
            total_rows=self._N,
        )

        # Key False stores the dense result. Key True stores the segment result.
        outputs = {}
        with torch.no_grad():
            for sparse_segments in (False, True):
                renderer = GSplatRenderer(self._config(sparse_segments=sparse_segments, use_rays=False), mock_model)
                outputs[sparse_segments] = renderer.render(
                    rendering_data,
                    gaussian_parameters,
                    n_active_features=3,
                    extra_ray_signal_infos=([], [], []),
                    segment_plan=plan if sparse_segments else None,
                )

        dense = outputs[False]
        sparse = outputs[True]
        assert sparse.opacity.max().item() > 0.0, "the selected camera rows must render"
        for name in ("rgb", "opacity", "distance", "normal"):
            dense_value = getattr(dense, name)
            sparse_value = getattr(sparse, name)
            assert (dense_value is None) == (sparse_value is None)
            if dense_value is not None:
                assert torch.equal(dense_value, sparse_value), f"partial segments changed {name}"

    def test_eligible_render_forwards_the_fourier_arguments(self, mock_model, monkeypatch):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")
        renderer = GSplatRenderer(self._config(), mock_model)
        rendering_data, gaussian_parameters = self._scene(device, features_dim=48)
        renderer._fourier_fold_ref = self._fold_ref(device)
        captured = self._render_capturing_kwargs(
            renderer, rendering_data, gaussian_parameters, monkeypatch, n_active_features=3
        )
        assert "fourier_theta" in captured
        assert "fourier_albedo" in captured
        assert captured["fourier_row_range"] == (0, self._N)

    def test_ineligible_render_refuses_rather_than_dropping_the_fold(self, mock_model, monkeypatch):
        # An rgb-only render cannot honor the fold. The collector would not have
        # folded here, so a ref being set means the two decisions diverged: fail loud
        # rather than silently render a stale band-0.
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")
        renderer = GSplatRenderer(self._config(), mock_model)
        rendering_data, gaussian_parameters = self._scene(device, features_dim=3)
        renderer._fourier_fold_ref = self._fold_ref(device)
        with pytest.raises(AssertionError, match="fold divergence"):
            self._render_capturing_kwargs(
                renderer, rendering_data, gaussian_parameters, monkeypatch, n_active_features=0
            )


class TestUnstructuredRejectsActiveTileMask:
    """The unstructured path must refuse an active tile mask rather than drop it.

    Active tiling is defined on gsplat's STRUCTURED spinning-lidar tile grid
    (build_camera_to_lidar_tile_mapping -> compute_lidar_element_to_tile, which needs
    a RowOffsetStructuredSpinningLidarModelParameters and reshapes to
    (n_rows, n_columns)). The model-free path has no such grid and
    render_unstructured takes no `masks=`, so a mask arriving here could only be
    silently ignored. NuRecConfig rejects the combination up front; this is the
    render-time backstop for a producer that sets the field directly.
    """

    def test_render_unstructured_raises_when_active_tile_mask_is_set(self):
        renderer = object.__new__(GSplatRenderer)
        renderer._lidar_extra_signals_sh_degree = None
        renderer._unstructured_logged = True
        renderer.config = SimpleNamespace(
            renderer_config="parallel_batch",
            culling=SimpleNamespace(
                cull_dead_pairs_lidar_unstructured=False,
                box_method_lidar_unstructured="ut",
                pbf_near_range_lidar_unstructured=8.0,
                rs_single_cover_lidar_unstructured=False,
            ),
        )
        rendering_data = SimpleNamespace(
            rays=[torch.zeros(3, 1, 6)],
            rays_timestamps_us=[torch.tensor([[0], [5], [10]])],
            timestamps_startend_us_cpu=torch.tensor([[0, 10]]),
            unstructured_lidar_tiling=SimpleNamespace(),
            active_tile_mask=torch.ones(1, 4, 8, dtype=torch.bool),
        )

        with pytest.raises(ValueError, match=r"active_tile_mask.*unstructured.*camera_invalid_only"):
            renderer._render_unstructured(
                rendering_data=rendering_data,
                gaussian_params=SimpleNamespace(),
                viewmat_start=torch.eye(4).unsqueeze(0),
                viewmat_end=torch.eye(4).unsqueeze(0),
                near_plane=0.1,
                far_plane=200.0,
                enable_distance=True,
                intensity=None,
                device=torch.device("cpu"),
                lidar_id="lidar_a",
            )
