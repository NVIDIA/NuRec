# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, auto
from functools import lru_cache
from unittest.mock import patch

import numpy as np
import pytest
import torch

from scipy.spatial.transform import Rotation

from ncore.data import (
    ConcreteCameraModelParametersUnion,
    FThetaCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    RowOffsetStructuredSpinningLidarModelParameters,
    ShutterType,
)
from ncore.sensors import RowOffsetStructuredSpinningLidarModel, camera_model_from_parameters
from nurec.libs.geometry.kernels.pose import se3pose_from_matrix
from nurec.libs.sensors.kernels.cameras import image_points_to_world_rays_shutter_pose
from nurec.libs.sensors.kernels.lidars import generate_spinning_lidar_rays
from nurec.utils.batch import generate_grid_2d_indices
from nurec.utils.prober import (
    ProberDataSet,
    ProberTestResult,
    prober_test_decorator,
)
from nurec.utils.sensors.ncore_sensors_converters import (
    CameraModelConverter,
    DynamicPose,
    LidarModelConverter,
    Pose,
)
from nurec.utils.sensors.sensors import CameraModelComputations, RectSubsampledSensor, SensorModelComputations
from nurec.utils.tests import WithTolerance


def create_lidar_model_parameters(param_file: str) -> RowOffsetStructuredSpinningLidarModelParameters:
    """
    Create RowOffsetStructuredSpinningLidarModelParameters from a JSON file.

    Args:
        param_file: Name of the parameter file (e.g., "row-offset-spinning-lidar-model-parameters.json")
                    Files should be located in nurec/libs/vren/test_data/

    Returns:
        RowOffsetStructuredSpinningLidarModelParameters instance
    """
    with open(f"nurec/libs/vren/test_data/{param_file}", "r") as fp:
        data = fp.read()

    return RowOffsetStructuredSpinningLidarModelParameters.from_json(data)


class _ImplementationTestMode(Enum):
    NCORE_IMPLEMENTATION = auto()
    SENSORLIB_IMPLEMENTATION = auto()
    SENSORLIB_GENRAY_IMPLEMENTATION = auto()


_TestCombinations = [
    (_ImplementationTestMode.NCORE_IMPLEMENTATION,),
    (_ImplementationTestMode.SENSORLIB_IMPLEMENTATION,),
    (_ImplementationTestMode.SENSORLIB_GENRAY_IMPLEMENTATION,),
]


@prober_test_decorator(
    snapshot_set_name="sensor_to_world_rays_shutter_pose",
    test_args_combinations=_TestCombinations,
    perf_test_args_combinations=_TestCombinations,
)
def test_pixels_to_world_rays_shutter_pose(data: ProberDataSet, test_mode: _ImplementationTestMode):
    """
    Test comparing old and new implementations of pixels_to_world_rays_shutter_pose.

    The old implementation used sensor_model.pixels_to_world_rays_shutter_pose and lidar_model.elements_to_world_rays_shutter_pose from NCore.
    The new implementation uses CameraModelComputations.pixels_to_world_rays_shutter_pose and LidarModelComputations.elements_to_world_rays_shutter_pose.

    This test ensures both implementations produce the same results.
    """
    """Generate test data for comparing the two implementations."""

    world_rays_map = {}
    timestamps_us_map = {}
    T_sensor_world_startend_grad_map = {}

    for camera_model_index in range(3):
        T_sensor_world_startend = data["T_sensor_world_startend"]
        timestamps_startend_us = data["timestamps_startend_us"]
        world_rays_grad = data["world_rays_grad"]

        camera_params: ConcreteCameraModelParametersUnion

        def create_camera_params(camera_model_index: int) -> ConcreteCameraModelParametersUnion:
            # Create a simple camera model
            if camera_model_index == 0:
                camera_params_json = """{"resolution": [960, 540], "shutter_type": "ROLLING_TOP_TO_BOTTOM", "external_distortion_parameters": {"reference_poly": "FORWARD",
                    "horizontal_poly": [-0.003537920070812106, 1.0016000270843506, 4.741529846796766e-05, -0.0027847199235111475, 0.0014640099834650755, 0.0019989400170743465],
                    "vertical_poly": [0.00510897021740675, 0.0020038599614053965, 0.0018203799845650792, -0.0011258500162512064,
                        0.0004649079928640276, 1.0136699676513672, -0.00010489000123925507, 0.009701469913125038, -0.0022726499009877443,
                        0.019545000046491623, -0.00022134000028017908, 0.013854499906301498, 0.01996220089495182, -0.003917189780622721, 0.008652609772980213],
                    "horizontal_poly_inverse": [0.0035187317989766598, 0.9984092116355896, -5.0759197620209306e-05, 0.002756733912974596, -0.001425599679350853, -0.002004376845434308],
                    "vertical_poly_inverse": [-0.005057216621935368, -0.0019700995180755854, -0.0017141086282208562, 0.0010707827750593424,
                        -0.00043124068179167807, 0.9866272807121277, 7.768073555780575e-05, -0.008865139447152615, 0.002170565305277705,
                        -0.01821419596672058, -0.0001051785729941912, -0.012822290882468224, -0.017289206385612488, 0.003648932557553053, -0.0069558690302073956]},
                    "principal_point": [477.8616638183594, 371.405029296875], "reference_poly": "ANGLE_TO_PIXELDIST",
                    "pixeldist_to_angle_poly": [0.0, 0.002108694054186344, 1.4248414004214283e-07, -8.064807710539412e-10, 2.485065538765374e-12, -1.7939406081105393e-15],
                    "angle_to_pixeldist_poly": [0.0, 473.61334228515625, -9.587550163269043, 24.94717788696289, -42.338558197021484, 15.140613555908203],
                    "max_angle": 1.3461724520632263, "linear_cde": [0.9997570514678955, 6.27807603450492e-05, -0.00010061755892820656]}"""
                return FThetaCameraModelParameters.from_json(camera_params_json)
            elif camera_model_index == 1:
                return OpenCVPinholeCameraModelParameters(
                    resolution=np.array([960, 540], dtype=np.uint64),
                    focal_length=np.array([50.0, 50.0], dtype=np.float32),
                    principal_point=np.array([32.0, 32.0], dtype=np.float32),
                    radial_coeffs=np.zeros(6, dtype=np.float32),
                    tangential_coeffs=np.zeros(2, dtype=np.float32),
                    thin_prism_coeffs=np.zeros(4, dtype=np.float32),
                    shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
                )

            else:
                return OpenCVFisheyeCameraModelParameters(
                    resolution=np.array([960, 540], dtype=np.uint64),
                    shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
                    principal_point=np.array([960.0, 540.0], dtype=np.float32),
                    focal_length=np.array([1000.0, 1000.0], dtype=np.float32),
                    radial_coeffs=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                    max_angle=np.deg2rad(140 / 2),
                )

        camera_params = create_camera_params(camera_model_index)
        # Generating them in the right order to be able to compare to the ray generation version (image_points = None in image_points_to_world_rays_shutter_pose)
        pixel_idxs = generate_grid_2d_indices(
            resolution=(camera_params.resolution[0], camera_params.resolution[1]),
            device=T_sensor_world_startend.device,
            order="xy",
        )

        camera_model = camera_model_from_parameters(camera_params, device=pixel_idxs.device, dtype=torch.float32)
        sensorlib_parameters = CameraModelConverter.convert(camera_model, device=pixel_idxs.device)
        image_points = 0.5 + pixel_idxs.to(camera_model.dtype)

        camera_rays = camera_model.image_points_to_camera_rays(image_points)
        camera_rays = camera_rays.reshape(-1, 3)

        if test_mode == _ImplementationTestMode.NCORE_IMPLEMENTATION:
            world_rays_return = camera_model.pixels_to_world_rays_shutter_pose(
                pixel_idxs,
                T_sensor_world_startend[0],
                T_sensor_world_startend[1],
                start_timestamp_us=int(timestamps_startend_us[0].item()),
                end_timestamp_us=int(timestamps_startend_us[1].item()),
                camera_rays=camera_rays,
                return_timestamps=True,
                return_T_sensor_worlds=False,
            )
            (world_rays, timestamps_us) = world_rays_return.world_rays, world_rays_return.timestamps_us
        elif (
            test_mode == _ImplementationTestMode.SENSORLIB_IMPLEMENTATION
            or test_mode == _ImplementationTestMode.SENSORLIB_GENRAY_IMPLEMENTATION
        ):
            (trans_startend, rot_startend) = se3pose_from_matrix(T_sensor_world_startend)

            image_points_to_use = (
                image_points.detach().clone() if test_mode == _ImplementationTestMode.SENSORLIB_IMPLEMENTATION else None
            )

            (world_rays, timestamps_us, _, _) = image_points_to_world_rays_shutter_pose(
                image_points=image_points_to_use,
                projection=sensorlib_parameters.projection,
                external_distortion=sensorlib_parameters.external_distortion,
                resolution=sensorlib_parameters.resolution,
                shutter_type=sensorlib_parameters.shutter_type,
                dynamic_pose=DynamicPose(
                    start_pose=Pose(translation=trans_startend[0], rotation=rot_startend[0]),
                    end_pose=Pose(translation=trans_startend[1], rotation=rot_startend[1]),
                ),
                start_timestamp_us=int(timestamps_startend_us[0].item()),
                end_timestamp_us=int(timestamps_startend_us[1].item()),
                return_timestamps=True,
            )

        world_rays.backward(world_rays_grad)

        world_rays_map[camera_model_index] = world_rays
        timestamps_us_map[camera_model_index] = timestamps_us
        T_sensor_world_startend_grad_map[camera_model_index] = T_sensor_world_startend.grad

    return ProberTestResult(
        str(test_mode),
        (
            world_rays_map[0],
            timestamps_us_map[0],
            T_sensor_world_startend_grad_map[0],
            world_rays_map[1],
            timestamps_us_map[1],
            T_sensor_world_startend_grad_map[1],
            world_rays_map[2],
            timestamps_us_map[2],
            T_sensor_world_startend_grad_map[2],
        ),
    )


@lru_cache(maxsize=1)
def _lidar_model(device: torch.device) -> RowOffsetStructuredSpinningLidarModel:
    lidar_params = create_lidar_model_parameters("row-offset-spinning-lidar-model-parameters.json")
    return RowOffsetStructuredSpinningLidarModel(
        lidar_params,
        angles_to_columns_map_init=True,
        device=device,
    )


@prober_test_decorator(
    snapshot_set_name="sensor_to_world_rays_shutter_pose",
    test_args_combinations=_TestCombinations,
    perf_test_args_combinations=_TestCombinations,
)
def test_elements_to_world_rays_shutter_pose(data: ProberDataSet, test_mode: _ImplementationTestMode):
    """
    Test comparing old and new implementations of elements_to_world_rays_shutter_pose.

    The old implementation used sensor_model.elements_to_world_rays_shutter_pose and lidar_model.elements_to_world_rays_shutter_pose from NCore.
    The new implementation uses CameraModelComputations.elements_to_world_rays_shutter_pose and LidarModelComputations.elements_to_world_rays_shutter_pose.

    This test ensures both implementations produce the same results.
    """
    """Generate test data for comparing the two implementations."""
    lidar_T_sensor_world_startend = data["lidar_T_sensor_world_startend"]
    lidar_timestamps_startend_us = data["lidar_timestamps_startend_us"]
    lidar_world_rays_grad = data["lidar_world_rays_grad"]

    # Create lidar model from parameters file
    lidar_model = _lidar_model(lidar_T_sensor_world_startend.device)
    # Generating them in the right order to be able to compare to the ray generation version (elements = None in elements_to_world_rays_shutter_pose)
    lidar_elements = generate_grid_2d_indices(
        resolution=(lidar_model.n_columns, lidar_model.n_rows),
        device=lidar_T_sensor_world_startend.device,
        order="yx",
    )
    lidar_sensor_rays = lidar_model.elements_to_sensor_rays(lidar_elements)
    sensorlib_parameters = LidarModelConverter.convert(lidar_model, device=lidar_elements.device)

    if test_mode == _ImplementationTestMode.NCORE_IMPLEMENTATION:
        world_rays_return = lidar_model.elements_to_world_rays_shutter_pose(
            lidar_elements,
            lidar_T_sensor_world_startend[0],
            lidar_T_sensor_world_startend[1],
            start_timestamp_us=int(lidar_timestamps_startend_us[0].item()),
            end_timestamp_us=int(lidar_timestamps_startend_us[1].item()),
            sensor_rays=lidar_sensor_rays,
            return_timestamps=True,
            return_T_sensor_worlds=False,
        )
        (world_rays, timestamps_us) = world_rays_return.world_rays, world_rays_return.timestamps_us
    elif (
        test_mode == _ImplementationTestMode.SENSORLIB_IMPLEMENTATION
        or test_mode == _ImplementationTestMode.SENSORLIB_GENRAY_IMPLEMENTATION
    ):
        [trans_startend, rot_startend] = se3pose_from_matrix(lidar_T_sensor_world_startend)

        elements_to_use = (
            lidar_elements.detach().clone() if test_mode == _ImplementationTestMode.SENSORLIB_IMPLEMENTATION else None
        )

        (world_rays, timestamps_us, _, _) = generate_spinning_lidar_rays(
            projection=sensorlib_parameters.projection,
            elements=elements_to_use,
            dynamic_pose=DynamicPose(
                start_pose=Pose(translation=trans_startend[0], rotation=rot_startend[0]),
                end_pose=Pose(translation=trans_startend[1], rotation=rot_startend[1]),
            ),
            start_timestamp_us=int(lidar_timestamps_startend_us[0].item()),
            end_timestamp_us=int(lidar_timestamps_startend_us[1].item()),
            return_timestamps=True,
        )

    world_rays.backward(lidar_world_rays_grad)

    return ProberTestResult(
        str(test_mode),
        # Gradient tolerance loosened from default (~1e-5) to 1e-2: elements were previously
        # randomly permuted but are now ordered, causing different floating-point accumulation order.
        (world_rays, timestamps_us, WithTolerance(lidar_T_sensor_world_startend.grad, atol=1e-2, rtol=1e-2)),
    )


def test_cpu_gpu_timestamp_consistency():
    """
    Test that CPU and GPU timestamps match for both Slang and compiled paths.

    This validates that the CPU-side timestamp computation (to avoid GPU sync)
    produces identical results to the GPU-computed timestamps.
    """
    device = torch.device("cuda")

    # Create test data
    n_frames = 5
    width, height = 1920, 1080

    # Create valid random pose matrices
    n_poses = n_frames * 2
    translation = torch.randn(n_poses, 3, device=device, dtype=torch.float32)
    rotation_matrices = torch.from_numpy(Rotation.random(n_poses).as_matrix()).to(device=device, dtype=torch.float32)

    T_sensor_world_startend_allviews = torch.zeros(n_poses, 4, 4, device=device, dtype=torch.float32)
    T_sensor_world_startend_allviews[:, :3, :3] = rotation_matrices
    T_sensor_world_startend_allviews[:, :3, 3] = translation
    T_sensor_world_startend_allviews[:, 3, 3] = 1.0
    T_sensor_world_startend_allviews = T_sensor_world_startend_allviews.reshape(n_frames, 2, 4, 4)

    # Create timestamps
    timestamps_start = torch.randint(0, 500000, (n_frames,), device=device, dtype=torch.int64)
    timestamps_end = timestamps_start + torch.randint(1, 500000, (n_frames,), device=device, dtype=torch.int64)
    timestamps_startend_us_allviews = torch.stack([timestamps_start, timestamps_end], dim=1)
    timestamps_startend_us_allviews_cpu = timestamps_startend_us_allviews.cpu()

    # Create a simple camera model
    camera_params = OpenCVPinholeCameraModelParameters(
        resolution=np.array([width, height], dtype=np.uint64),
        focal_length=np.array([500.0, 500.0], dtype=np.float32),
        principal_point=np.array([width / 2, height / 2], dtype=np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
    )
    camera_model = camera_model_from_parameters(camera_params, device=device, dtype=torch.float32)

    # Create sensor models dict
    sensor_models = torch.nn.ModuleDict({"camera_0": camera_model})

    # Create subsampling parameters
    subsample = RectSubsampledSensor(
        i=100,
        j=200,
        width=800,
        height=600,
        subsample_factor=1.0,
        original_width=width,
        original_height=height,
    ).to(device)

    # Test frame index
    unique_frame_idx = 2
    unique_frame_idx_tensor = torch.tensor([unique_frame_idx], dtype=torch.int32, device=device)

    # Test 1: CUDA path with embedding (Slang kernel)
    embedding = torch.nn.Embedding(n_frames, 9, device=device)
    torch.nn.init.zeros_(embedding.weight)

    result_slang = SensorModelComputations.get_poses_and_timestamps_startend(
        subsample=subsample,
        embeds=embedding,
        T_offset_nurec_startend=None,
        T_sensor_world_startend_allviews=T_sensor_world_startend_allviews,
        timestamps_startend_us_allviews=timestamps_startend_us_allviews,
        timestamps_startend_us_allviews_cpu=timestamps_startend_us_allviews_cpu,
        sensor_models=sensor_models,
        unique_frame_idx=unique_frame_idx,
        unique_frame_idx_tensor=unique_frame_idx_tensor,
        unique_sensor_idx_str="camera_0",
        enable_calib=True,
        is_lidar=False,
    )

    assert result_slang.timestamps_startend_us_cpu.device.type == "cpu"
    assert result_slang.timestamps_startend_us_gpu.device.type == "cuda"
    # Verify CPU and GPU timestamps match for Slang path
    torch.testing.assert_close(
        result_slang.timestamps_startend_us_cpu.cuda(),
        result_slang.timestamps_startend_us_gpu,
        atol=1e-5,
        rtol=1e-5,
        msg="Slang path: CPU and GPU timestamps should match",
    )

    # Test 2: Compiled path (no embedding, forces CPU fallback logic)
    result_compiled = SensorModelComputations.get_poses_and_timestamps_startend(
        subsample=subsample,
        embeds=None,
        T_offset_nurec_startend=None,
        T_sensor_world_startend_allviews=T_sensor_world_startend_allviews,
        timestamps_startend_us_allviews=timestamps_startend_us_allviews,
        timestamps_startend_us_allviews_cpu=timestamps_startend_us_allviews_cpu,
        sensor_models=sensor_models,
        unique_frame_idx=unique_frame_idx,
        unique_frame_idx_tensor=unique_frame_idx_tensor,
        unique_sensor_idx_str="camera_0",
        enable_calib=False,  # Must be False when embeds=None
        is_lidar=False,
    )

    assert result_compiled.timestamps_startend_us_cpu.device.type == "cpu"
    assert result_compiled.timestamps_startend_us_gpu.device.type == "cuda"
    # Verify CPU and GPU timestamps match for Compiled path
    torch.testing.assert_close(
        result_compiled.timestamps_startend_us_cpu.cuda(),
        result_compiled.timestamps_startend_us_gpu,
        atol=1e-5,
        rtol=1e-5,
        msg="Compiled path (CPU): CPU and GPU timestamps should match",
    )


def _make_camera_sensor_models(
    *, width: int, height: int, shutter_type: ShutterType, device: torch.device
) -> torch.nn.ModuleDict:
    camera_params = OpenCVPinholeCameraModelParameters(
        resolution=np.array([width, height], dtype=np.uint64),
        focal_length=np.array([500.0, 500.0], dtype=np.float32),
        principal_point=np.array([width / 2, height / 2], dtype=np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        shutter_type=shutter_type,
    )
    return torch.nn.ModuleDict(
        {"camera_0": camera_model_from_parameters(camera_params, device=device, dtype=torch.float32)}
    )


def _get_camera_poses_and_timestamps(
    *,
    subsample: RectSubsampledSensor,
    shutter_type: ShutterType,
    timestamps_startend_us_allviews_cpu: torch.Tensor,
) -> SensorModelComputations.PosesAndTimestampsStartendReturn:
    device = subsample.rect_points_lb.device
    n_frames = len(timestamps_startend_us_allviews_cpu)
    width, height = (int(value) for value in subsample.resolution_cpu.tolist())
    unique_frame_idx = 1
    return SensorModelComputations.get_poses_and_timestamps_startend(
        subsample=subsample,
        embeds=None,
        T_offset_nurec_startend=None,
        T_sensor_world_startend_allviews=torch.eye(4, device=device).repeat(n_frames, 2, 1, 1),
        timestamps_startend_us_allviews=timestamps_startend_us_allviews_cpu.to(device),
        timestamps_startend_us_allviews_cpu=timestamps_startend_us_allviews_cpu,
        sensor_models=_make_camera_sensor_models(
            width=width,
            height=height,
            shutter_type=shutter_type,
            device=device,
        ),
        unique_frame_idx=unique_frame_idx,
        unique_frame_idx_tensor=torch.tensor([unique_frame_idx], dtype=torch.int32, device=device),
        unique_sensor_idx_str="camera_0",
        enable_calib=False,
        is_lidar=False,
    )


@pytest.mark.parametrize(
    ("width", "height", "subsample_factor", "subsampled_width", "subsampled_height"),
    [
        (1920, 1080, 2.0, 960, 540),
        (1920, 1080, 2.4, 800, 450),
        (960, 540, 1.2, 800, 450),
    ],
)
def test_full_sensor_subsample_skips_cpu_timestamp_interpolation(
    width: int,
    height: int,
    subsample_factor: float,
    subsampled_width: int,
    subsampled_height: int,
):
    device = torch.device("cuda")
    timestamps_startend_us_allviews_cpu = torch.tensor([[100, 200], [300, 500]], dtype=torch.int64)
    subsample = RectSubsampledSensor(
        i=0,
        j=0,
        width=subsampled_width,
        height=subsampled_height,
        subsample_factor=subsample_factor,
        original_width=width,
        original_height=height,
    ).to(device)
    expected_source_timestamps = timestamps_startend_us_allviews_cpu.clone()
    for shutter_type in ShutterType:
        assert subsample.covers_full_shutter_range(shutter_type)
        expected_timestamps = CameraModelComputations.interpolate_rect_timestamps_cpu(
            subsample.rect_points_lb_cpu,
            subsample.resolution_cpu,
            shutter_type,
            timestamps_startend_us_allviews_cpu[1],
        ).unsqueeze(0)

        with patch.object(
            CameraModelComputations,
            "interpolate_rect_timestamps_cpu",
            side_effect=AssertionError("full-sensor timestamp interpolation should use the fast path"),
        ):
            result = _get_camera_poses_and_timestamps(
                subsample=subsample,
                shutter_type=shutter_type,
                timestamps_startend_us_allviews_cpu=timestamps_startend_us_allviews_cpu,
            )

        torch.testing.assert_close(result.timestamps_startend_us_cpu, expected_timestamps, atol=0, rtol=0)
        # Use the previous CPU helper as the compatibility oracle. The Slang
        # calculation is still exercised above, but its float32 reciprocal can
        # truncate a full-width horizontal endpoint by a few microseconds.
        result.timestamps_startend_us_cpu.add_(1)
        torch.testing.assert_close(timestamps_startend_us_allviews_cpu, expected_source_timestamps)


@pytest.mark.parametrize("frame_duration_us", [2**24 - 1, 2**24, 2**24 + 1, 20_000_001])
@pytest.mark.parametrize("shutter_type", list(ShutterType))
def test_full_sensor_timestamp_fast_path_preserves_float32_endpoint_contract(
    frame_duration_us: int,
    shutter_type: ShutterType,
):
    device = torch.device("cuda")
    frame_start_timestamp_us = 100
    timestamps_startend_us_allviews_cpu = torch.tensor(
        [
            [0, 1],
            [frame_start_timestamp_us, frame_start_timestamp_us + frame_duration_us],
        ],
        dtype=torch.int64,
    )
    subsample = RectSubsampledSensor(
        i=0,
        j=0,
        width=960,
        height=540,
        subsample_factor=2.0,
        original_width=1920,
        original_height=1080,
    ).to(device)
    assert subsample.covers_full_shutter_range(shutter_type)
    expected_timestamps = CameraModelComputations.interpolate_rect_timestamps_cpu(
        subsample.rect_points_lb_cpu,
        subsample.resolution_cpu,
        shutter_type,
        timestamps_startend_us_allviews_cpu[1],
    ).unsqueeze(0)

    with patch.object(
        CameraModelComputations,
        "interpolate_rect_timestamps_cpu",
        side_effect=AssertionError("full-sensor timestamp interpolation should use the fast path"),
    ):
        result = _get_camera_poses_and_timestamps(
            subsample=subsample,
            shutter_type=shutter_type,
            timestamps_startend_us_allviews_cpu=timestamps_startend_us_allviews_cpu,
        )

    torch.testing.assert_close(result.timestamps_startend_us_cpu, expected_timestamps, atol=0, rtol=0)
    # Use the previous CPU helper as the compatibility oracle. The Slang
    # calculation is still exercised above, but its float32 reciprocal can
    # truncate a full-width horizontal endpoint by a few microseconds.
    result.timestamps_startend_us_cpu.add_(1)
    torch.testing.assert_close(
        timestamps_startend_us_allviews_cpu[1],
        torch.tensor(
            [frame_start_timestamp_us, frame_start_timestamp_us + frame_duration_us],
            dtype=torch.int64,
        ),
    )


@pytest.mark.parametrize(
    ("subsampled_width", "subsampled_height", "covers_vertical", "covers_horizontal"),
    [
        (799, 450, True, False),
        (800, 449, False, True),
    ],
)
def test_subsample_coverage_follows_shutter_scan_axis(
    subsampled_width: int,
    subsampled_height: int,
    covers_vertical: bool,
    covers_horizontal: bool,
):
    subsample = RectSubsampledSensor(
        i=0,
        j=0,
        width=subsampled_width,
        height=subsampled_height,
        subsample_factor=2.4,
        original_width=1920,
        original_height=1080,
    )

    assert subsample.covers_full_shutter_range(ShutterType.ROLLING_TOP_TO_BOTTOM) == covers_vertical
    assert subsample.covers_full_shutter_range(ShutterType.ROLLING_BOTTOM_TO_TOP) == covers_vertical
    assert subsample.covers_full_shutter_range(ShutterType.ROLLING_LEFT_TO_RIGHT) == covers_horizontal
    assert subsample.covers_full_shutter_range(ShutterType.ROLLING_RIGHT_TO_LEFT) == covers_horizontal
    assert subsample.covers_full_shutter_range(ShutterType.GLOBAL)


@pytest.mark.parametrize(
    ("shutter_type", "i", "j", "subsampled_width", "subsampled_height"),
    [
        (ShutterType.ROLLING_TOP_TO_BOTTOM, 10, 0, 320, 540),
        (ShutterType.ROLLING_BOTTOM_TO_TOP, 10, 0, 320, 540),
        (ShutterType.ROLLING_LEFT_TO_RIGHT, 0, 10, 960, 180),
        (ShutterType.ROLLING_RIGHT_TO_LEFT, 0, 10, 960, 180),
        (ShutterType.GLOBAL, 10, 20, 320, 180),
    ],
)
def test_crop_orthogonal_to_shutter_scan_skips_cpu_timestamp_interpolation(
    shutter_type: ShutterType,
    i: int,
    j: int,
    subsampled_width: int,
    subsampled_height: int,
):
    device = torch.device("cuda")
    timestamps_startend_us_allviews_cpu = torch.tensor([[100, 200], [300, 500]], dtype=torch.int64)
    subsample = RectSubsampledSensor(
        i=i,
        j=j,
        width=subsampled_width,
        height=subsampled_height,
        subsample_factor=2.0,
        original_width=1920,
        original_height=1080,
    ).to(device)
    assert subsample.covers_full_shutter_range(shutter_type)
    expected_timestamps = CameraModelComputations.interpolate_rect_timestamps_cpu(
        subsample.rect_points_lb_cpu,
        subsample.resolution_cpu,
        shutter_type,
        timestamps_startend_us_allviews_cpu[1],
    ).unsqueeze(0)

    with patch.object(
        CameraModelComputations,
        "interpolate_rect_timestamps_cpu",
        side_effect=AssertionError("full shutter range should use the fast path"),
    ):
        result = _get_camera_poses_and_timestamps(
            subsample=subsample,
            shutter_type=shutter_type,
            timestamps_startend_us_allviews_cpu=timestamps_startend_us_allviews_cpu,
        )

    torch.testing.assert_close(result.timestamps_startend_us_cpu, expected_timestamps, atol=0, rtol=0)


def test_cropped_subsample_uses_cpu_timestamp_interpolation():
    device = torch.device("cuda")
    timestamps_startend_us_allviews_cpu = torch.tensor([[100, 200], [300, 500]], dtype=torch.int64)
    subsample = RectSubsampledSensor(
        i=10,
        j=20,
        width=320,
        height=180,
        original_width=1920,
        original_height=1080,
    ).to(device)
    assert not subsample.covers_full_shutter_range(ShutterType.ROLLING_TOP_TO_BOTTOM)
    expected = torch.tensor([321, 432], dtype=torch.int64)

    with patch.object(CameraModelComputations, "interpolate_rect_timestamps_cpu", return_value=expected) as interpolate:
        result = _get_camera_poses_and_timestamps(
            subsample=subsample,
            shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
            timestamps_startend_us_allviews_cpu=timestamps_startend_us_allviews_cpu,
        )

    interpolate.assert_called_once()
    torch.testing.assert_close(result.timestamps_startend_us_cpu, expected.unsqueeze(0), atol=0, rtol=0)
