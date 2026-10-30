# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest

import slangtorch
import torch

from nurec.libs.sensors.kernels.cameras import (
    FThetaPolynomialType,
    FThetaProjection,
    NoExternalDistortion,
    OpenCVFisheyeProjection,
    OpenCVPinholeProjection,
    ShutterType,
)
from nurec.libs.sensors.models.cameras import (
    FThetaCameraModel,
    OpenCVFisheyeCameraModel,
    OpenCVPinholeCameraModel,
)
from nurec.libs.slang_gaussians.collector import get_slang_module_path
from nurec.libs.slang_utils.utils import add_ninja_to_path


device = torch.device("cuda")
# Deterministic random for reproducibility
torch.manual_seed(123)

NB_GAUSSIANS = 1234
OFFSET = 567
THREADS_PER_BLOCK = 256
BLOCKS_PER_GRID = (NB_GAUSSIANS + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

KERNEL_PRELUDE = """
import collector;

"""

KERNEL_TEMPLATE = """
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void {kernel_name}(
    no_diff uint out_offset,
    no_diff uint count,
    {collector_type} collector,
) {{
    run_tasks_dense(out_offset, count, collector);
}}

"""


class TestSlangCollector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        kernels_to_generate = [
            ("test_positions_copy", "Collector_Copy<3>"),
            ("test_rotations_normalize", "RotationsCollector_Normalize"),
            ("test_scales_exp", "ScalesCollector_Exp"),
            ("test_densities_sigmoid", "DensitiesCollector_Sigmoid"),
            ("test_extra_signal_copy", "Collector_Copy<5>"),
            ("test_camera_extra_signal_copy", "Collector_Copy<20>"),
            ("test_lidar_extra_signal_copy", "Collector_Copy<10>"),
            ("test_spherical_features_copy", "SphericalFeaturesCollector_Copy<3,45>"),
            (
                "test_spherical_features_fourier_individual",
                "SphericalFeaturesCollector_Fourier_Individual<3,45,20>",
            ),
            (
                "test_spherical_features_fourier_holistic",
                "SphericalFeaturesCollector_Fourier_Holistic<3,45,20>",
            ),
            (
                "test_spherical_features_fourier_individual_step",
                "SphericalFeaturesCollector_Fourier_IndividualStep<3,45,20,15>",
            ),
        ]
        compiled_kernels = {}
        code = KERNEL_PRELUDE
        for kernel_name, collector_type in kernels_to_generate:
            code += KERNEL_TEMPLATE.format(kernel_name=kernel_name, collector_type=collector_type)
        with tempfile.NamedTemporaryFile("w", suffix=".slang", delete=False) as tf:
            slang_file_path = tf.name
            tf.write(code)
        try:
            add_ninja_to_path()
            slang_module_path = get_slang_module_path()
            include_paths = [os.path.dirname(slang_module_path)]
            slang_module = slangtorch.loadModule(slang_file_path, verbose=False, includePaths=include_paths)
        finally:
            os.unlink(slang_file_path)

        for kernel_name, _ in kernels_to_generate:
            compiled_kernels[kernel_name] = (slang_module, getattr(slang_module, kernel_name))
        cls.kernels = compiled_kernels
        cls.slang_module = slang_module

    @staticmethod
    def _launch(kernel, collector):
        """Launch a generated collector test kernel over the standard Gaussian test range."""
        return kernel(
            out_offset=OFFSET,
            count=NB_GAUSSIANS,
            collector=collector,
        ).launchRaw(blockSize=(THREADS_PER_BLOCK, 1, 1), gridSize=(BLOCKS_PER_GRID, 1, 1))

    @staticmethod
    def _timestamp_estimation_camera_models():
        """Create small camera models that exercise different ray projection implementations."""
        resolution = (640, 480)
        focal_length = 500.0
        resolution_tensor = torch.tensor(resolution, device=device, dtype=torch.float32)
        principal_point = torch.tensor(
            [(resolution[0] - 1) * 0.5, (resolution[1] - 1) * 0.5], device=device, dtype=torch.float32
        )

        pinhole_projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([focal_length, focal_length], device=device, dtype=torch.float32),
            principal_point=principal_point,
            radial_coeffs=torch.zeros(6, device=device, dtype=torch.float32),
            tangential_coeffs=torch.zeros(2, device=device, dtype=torch.float32),
            thin_prism_coeffs=torch.zeros(4, device=device, dtype=torch.float32),
            resolution=resolution_tensor,
        )

        fisheye_projection = OpenCVFisheyeProjection.from_components(
            principal_point=principal_point,
            focal_length=torch.tensor([focal_length, focal_length], device=device, dtype=torch.float32),
            forward_poly=torch.zeros(4, device=device, dtype=torch.float32),
            resolution=resolution_tensor,
            max_angle=1.2,
            newton_iterations=10,
            min_2d_norm=torch.tensor(1e-6, device=device, dtype=torch.float32),
        )

        ftheta_projection = FThetaProjection.from_components(
            principal_point=principal_point,
            fw_poly=torch.tensor([0.0, focal_length], device=device, dtype=torch.float32),
            bw_poly=torch.tensor([0.0, 1.0 / focal_length], device=device, dtype=torch.float32),
            A=torch.eye(2, device=device, dtype=torch.float32),
            Ainv=torch.eye(2, device=device, dtype=torch.float32),
            dfw_poly=torch.tensor([focal_length], device=device, dtype=torch.float32),
            dbw_poly=torch.tensor([1.0 / focal_length], device=device, dtype=torch.float32),
            reference_poly=FThetaPolynomialType.FORWARD,
            max_angle=1.2,
            newton_iterations=10,
            min_2d_norm=1e-6,
        )

        return [
            OpenCVPinholeCameraModel(
                projection=pinhole_projection,
                external_distortion=NoExternalDistortion(),
                resolution=resolution,
                shutter_type=ShutterType.GLOBAL,
            ),
            OpenCVFisheyeCameraModel(
                projection=fisheye_projection,
                external_distortion=NoExternalDistortion(),
                resolution=resolution,
                shutter_type=ShutterType.GLOBAL,
            ),
            FThetaCameraModel(
                projection=ftheta_projection,
                external_distortion=NoExternalDistortion(),
                resolution=resolution,
                shutter_type=ShutterType.GLOBAL,
            ),
        ]

    @staticmethod
    def _camera_model_rays(camera_model, frame_start_timestamp, frame_end_timestamp):
        """Generate flattened [origin, direction] rays and rolling-shutter-like timestamps."""
        width, height = camera_model.resolution
        pixels_y, pixels_x = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.int16),
            torch.arange(width, device=device, dtype=torch.int16),
            indexing="ij",
        )
        pixel_idxs = torch.stack([pixels_x.flatten(), pixels_y.flatten()], dim=1)
        camera_rays = torch.nn.functional.normalize(camera_model.pixels_to_camera_rays(pixel_idxs), dim=-1)
        rays = torch.cat([torch.zeros_like(camera_rays), camera_rays], dim=1).contiguous()

        row_ratio = pixel_idxs[:, 1].to(torch.float32) / float(height - 1)
        rays_timestamps = torch.round(
            frame_start_timestamp + row_ratio * (frame_end_timestamp - frame_start_timestamp)
        ).to(torch.int64)
        return rays, rays_timestamps

    @staticmethod
    def _timestamp_estimation_scene(camera_model):
        """Build a synthetic track/ray scene for masked-vs-unmasked timestamp estimation."""
        frame_start_timestamp = 1000
        frame_end_timestamp = 2000
        rays, rays_timestamps = TestSlangCollector._camera_model_rays(
            camera_model, frame_start_timestamp, frame_end_timestamp
        )

        width, height = camera_model.resolution
        # Rays are flattened row-major from a [height, width] pixel grid, so
        # flat_idx = y * width + x.  Pick one center-ish ray and one off-center
        # ray to cover different camera directions without relying on a
        # particular projection model.
        selected_ray_indices = [
            (height // 2) * width + (width // 2),
            (height // 4) * width + (width // 4),
        ]
        frame_start_ray_idx = width // 2

        def center_on_ray(ray_idx, depth):
            ray_origin = rays[ray_idx, :3]
            ray_direction = rays[ray_idx, 3:]
            return ray_origin + ray_direction * depth

        expected_active_tracks = []

        # Track 0: center on an actual generated ray so it is hittable for any
        # projection model, with timestamps overlapping the frame.
        track0_center = center_on_ray(selected_ray_indices[0], 3.0)
        track0_timestamps = [
            frame_start_timestamp,
            frame_start_timestamp + 250,
            frame_start_timestamp + 750,
            frame_end_timestamp,
        ]
        expected_active_tracks.append(0)  # visible position + in-frame timestamps

        # Track 1: another visible, in-frame track on a different ray direction.
        track1_center = center_on_ray(selected_ray_indices[1], 4.0)
        track1_timestamps = [frame_start_timestamp, frame_start_timestamp + 500, frame_end_timestamp]
        expected_active_tracks.append(1)  # visible position + in-frame timestamps

        # Track 2: visible position, outside time.  This isolates time culling:
        # the cuboid is in view, but all keyframes are after the frame window,
        # so the pre-pass should remove it before the frustum test matters.
        track2_center = track0_center.clone()
        track2_timestamps = [frame_end_timestamp + 500, frame_end_timestamp + 1000, frame_end_timestamp + 2000]

        # Track 3: in-frame time, outside frustum.  This isolates frustum
        # culling: the timestamps overlap the frame, but the cuboid is shifted
        # far sideways from a visible ray point, so the pre-pass should remove
        # it based on position.
        track3_center = center_on_ray(selected_ray_indices[0], 3.0) + torch.tensor([20.0, 0.0, 0.0], device=device)
        track3_timestamps = [
            frame_start_timestamp,
            frame_start_timestamp + 250,
            frame_start_timestamp + 500,
            frame_start_timestamp + 750,
            frame_end_timestamp,
        ]

        # Track 4: starts inside the frustum and ends outside.  The frustum
        # culler tests a swept sphere over the frame overlap, so this track must
        # stay active even though its frame-end pose is outside the view.
        track4_start_center = center_on_ray(frame_start_ray_idx, 3.0)
        track4_end_center = track4_start_center + torch.tensor([20.0, 0.0, 0.0], device=device)
        track4_centers = [track4_start_center, track4_end_center]
        track4_timestamps = [frame_start_timestamp, frame_end_timestamp]
        expected_active_tracks.append(4)  # swept sphere starts inside the frustum

        # Track 5: frame start and end both fall between the same two pose
        # keyframes. This should produce a single capsule between interpolated
        # frame endpoints with no interior keyframe split.
        track5_center = center_on_ray(frame_start_ray_idx, 3.5)
        track5_centers = [track5_center, track5_center]
        track5_timestamps = [frame_start_timestamp - 500, frame_end_timestamp + 500]
        expected_active_tracks.append(5)

        # Track 6: the frame crosses multiple pose keyframes. The raw poses just
        # outside the frame are outside the frustum on the same side, but the
        # interior keyframes are visible. The culler must split the sweep into
        # multiple capsules instead of testing only the outer endpoints.
        track6_visible_center = center_on_ray(selected_ray_indices[0], 3.5)
        track6_outside_center = track6_visible_center + torch.tensor([20.0, 0.0, 0.0], device=device)
        track6_centers = [
            track6_outside_center,
            track6_visible_center,
            track6_visible_center,
            track6_outside_center,
        ]
        track6_timestamps = [
            frame_start_timestamp - 500,
            frame_start_timestamp + 200,
            frame_end_timestamp - 200,
            frame_end_timestamp + 500,
        ]
        expected_active_tracks.append(6)

        track_pose_centers = [
            [track0_center] * len(track0_timestamps),
            [track1_center] * len(track1_timestamps),
            [track2_center] * len(track2_timestamps),
            [track3_center] * len(track3_timestamps),
            track4_centers,
            track5_centers,
            track6_centers,
        ]
        track_timestamps = [
            track0_timestamps,
            track1_timestamps,
            track2_timestamps,
            track3_timestamps,
            track4_timestamps,
            track5_timestamps,
            track6_timestamps,
        ]

        quaternion = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        # Tracks 0-3 keep a static pose across all keyframes to isolate their
        # hit-vs-cull behavior. Tracks 4-6 intentionally move to cover swept
        # frustum culling across different keyframe layouts.
        tracks_poses = torch.stack(
            [
                torch.cat([center, quaternion])
                for per_track_centers in track_pose_centers
                for center in per_track_centers
            ]
        ).contiguous()
        tracks_timestamps = torch.tensor(
            [timestamp for per_track_timestamps in track_timestamps for timestamp in per_track_timestamps],
            device=device,
            dtype=torch.int64,
        )
        packinfo = []
        start = 0
        for per_track_timestamps in track_timestamps:
            length = len(per_track_timestamps)
            packinfo.append((start, length))
            start += length
        tracks_packinfo = torch.tensor(packinfo, device=device, dtype=torch.int32)
        cuboids_dimensions = torch.full((len(track_timestamps), 3), 0.5, device=device, dtype=torch.float32)
        expected_active_tracks = torch.tensor(expected_active_tracks, device=device, dtype=torch.int32)

        return (
            rays,
            rays_timestamps,
            tracks_poses,
            tracks_timestamps,
            tracks_packinfo,
            cuboids_dimensions,
            frame_start_timestamp,
            frame_end_timestamp,
            expected_active_tracks,
        )

    def test_estimate_tracks_timestamps_masked_matches_unmasked_with_camera_frustum(self):
        slang_module = self.slang_module
        threads = 256
        build_threads = 128

        for camera_model in self._timestamp_estimation_camera_models():
            with self.subTest(camera_model=type(camera_model).__name__):
                (
                    rays,
                    rays_timestamps,
                    tracks_poses,
                    tracks_timestamps,
                    tracks_packinfo,
                    cuboids_dimensions,
                    frame_start_timestamp,
                    frame_end_timestamp,
                    expected_active_tracks,
                ) = self._timestamp_estimation_scene(camera_model)

                nb_tracks = tracks_packinfo.shape[0]
                nb_rays = rays.shape[0]
                ray_blocks = (nb_rays + threads - 1) // threads
                intersect_grid = (nb_tracks, ray_blocks, 1)

                unmasked = torch.zeros((nb_tracks, 2), device=device, dtype=torch.int64)
                slang_module.estimate_tracks_timestamps(
                    output_tracks_timestamps_sum_and_count=unmasked,
                    rays=rays,
                    rays_timestamps=rays_timestamps,
                    tracks_poses=tracks_poses,
                    tracks_timestamps=tracks_timestamps,
                    tracks_packinfo=tracks_packinfo,
                    cuboids_dimensions=cuboids_dimensions,
                    frame_timestamp=frame_start_timestamp,
                ).launchRaw(
                    (threads, 1, 1),
                    intersect_grid,
                )

                active_tracks_indices = torch.full((nb_tracks,), -1, device=device, dtype=torch.int32)
                active_count = torch.zeros(1, device=device, dtype=torch.int32)
                build_blocks = (nb_tracks + build_threads - 1) // build_threads
                slang_module.build_active_tracks_indices_time_frustum(
                    output_active_tracks_indices=active_tracks_indices,
                    output_active_count=active_count,
                    tracks_poses=tracks_poses,
                    tracks_timestamps=tracks_timestamps,
                    tracks_packinfo=tracks_packinfo,
                    cuboids_dimensions=cuboids_dimensions,
                    frustum_rays=rays,
                    frame_start_timestamp=frame_start_timestamp,
                    frame_end_timestamp=frame_end_timestamp,
                    frustum_rays_height=camera_model.resolution[1],
                    frustum_rays_width=camera_model.resolution[0],
                    frustum_margin_rad=0.0,
                ).launchRaw(
                    (build_threads, 1, 1),
                    (build_blocks, 1, 1),
                )

                masked = torch.zeros_like(unmasked)
                slang_module.estimate_tracks_timestamps_masked(
                    output_tracks_timestamps_sum_and_count=masked,
                    rays=rays,
                    rays_timestamps=rays_timestamps,
                    tracks_poses=tracks_poses,
                    tracks_timestamps=tracks_timestamps,
                    tracks_packinfo=tracks_packinfo,
                    cuboids_dimensions=cuboids_dimensions,
                    active_tracks_indices=active_tracks_indices,
                    active_count=active_count,
                    frame_timestamp=frame_start_timestamp,
                ).launchRaw(
                    (threads, 1, 1),
                    intersect_grid,
                )

                active = active_tracks_indices[: active_count.item()].sort().values
                self.assertTrue(torch.equal(active, expected_active_tracks))
                self.assertGreater(unmasked[0, 1].item(), 0)
                self.assertGreater(unmasked[1, 1].item(), 0)
                self.assertEqual(unmasked[2, 1].item(), 0)
                self.assertEqual(unmasked[3, 1].item(), 0)
                self.assertGreater(unmasked[4, 1].item(), 0)
                self.assertGreater(unmasked[5, 1].item(), 0)
                self.assertTrue(torch.equal(masked, unmasked))

    def _test_inputs_to_output(
        self, input_functions, output_nb_components, kernel_name, parameter_type, ground_truth_function, atol
    ):
        def compare(a, b):
            if atol is None:
                return torch.equal(a, b)
            else:
                return torch.allclose(a, b, atol=atol)

        slang_module, kernel = self.kernels[kernel_name]

        # Forward pass.
        inputs = {}
        for func in input_functions:
            name, shape = func()
            input = torch.nn.Parameter(torch.randn(*shape, device=device))
            inputs[name] = input
        output = torch.randn(OFFSET + NB_GAUSSIANS, output_nb_components, device=device)
        output_untouched = output[:OFFSET].clone()
        ground_truth = ground_truth_function(inputs)

        collector = getattr(slang_module, parameter_type)(**inputs, output=output)

        self.assertTrue(torch.equal(output[:OFFSET], output_untouched))
        self.assertFalse(torch.equal(output[OFFSET : OFFSET + NB_GAUSSIANS], ground_truth))
        self._launch(kernel, collector)
        self.assertTrue(torch.equal(output[:OFFSET], output_untouched))
        self.assertTrue(compare(output[OFFSET : OFFSET + NB_GAUSSIANS], ground_truth))

        # Backward pass.
        input_grads = {name: (input, torch.randn_like(input)) for name, input in inputs.items()}
        output_grad_ground_truth = torch.randn_like(output)

        collector = getattr(slang_module, parameter_type)(**input_grads, output=(output, output_grad_ground_truth))

        self._launch(kernel.bwd, collector)

        for name, (input, input_grad) in input_grads.items():
            input.grad = None
        torch.cat([output_untouched, ground_truth], dim=0).backward(output_grad_ground_truth)
        for name, (input, input_grad) in input_grads.items():
            self.assertTrue(compare(input.grad, input_grad))

    def _test_input_to_output(self, nb_components, kernel_name, parameter_type, ground_truth_function, atol):
        self._test_inputs_to_output(
            [lambda: ("input", (NB_GAUSSIANS, nb_components))],
            nb_components,
            kernel_name,
            parameter_type,
            lambda x: ground_truth_function(x["input"]),
            atol,
        )

    def test_positions_copy(self):
        self._test_input_to_output(3, "test_positions_copy", "Collector_Copy", lambda x: x.clone(), None)

    def test_rotations_normalize(self):
        self._test_input_to_output(
            4,
            "test_rotations_normalize",
            "RotationsCollector_Normalize",
            lambda x: torch.nn.functional.normalize(x),
            1e-6,
        )

    def test_scales_exp(self):
        self._test_input_to_output(3, "test_scales_exp", "ScalesCollector_Exp", lambda x: torch.exp(x), 1e-6)

    def test_densities_sigmoid(self):
        self._test_input_to_output(
            1, "test_densities_sigmoid", "DensitiesCollector_Sigmoid", lambda x: torch.sigmoid(x), 1e-6
        )

    def test_extra_signal_copy(self):
        self._test_input_to_output(5, "test_extra_signal_copy", "Collector_Copy", lambda x: x.clone(), None)

    def test_camera_extra_signal_copy(self):
        self._test_input_to_output(
            20,
            "test_camera_extra_signal_copy",
            "Collector_Copy",
            lambda x: x.clone(),
            None,
        )

    def test_lidar_extra_signal_copy(self):
        self._test_input_to_output(
            10,
            "test_lidar_extra_signal_copy",
            "Collector_Copy",
            lambda x: x.clone(),
            None,
        )

    def test_spherical_features_copy(self):
        ALBEDO_DIM = 3
        SPECULAR_DIM = 45
        self._test_inputs_to_output(
            [lambda: ("albedo", (NB_GAUSSIANS, ALBEDO_DIM)), lambda: ("specular", (NB_GAUSSIANS, SPECULAR_DIM))],
            ALBEDO_DIM + SPECULAR_DIM,
            "test_spherical_features_copy",
            "SphericalFeaturesCollector_Copy",
            lambda x: torch.cat([x["albedo"], x["specular"]], dim=1),
            None,
        )

    def test_spherical_features_fourier(self):
        atol = 1e-5

        ALBEDO_DIM = 3
        SPECULAR_DIM = 45
        FOURIER_DIM = 20

        def _get_individual_embedding():
            track_size = NB_GAUSSIANS // 3
            instance_idx = torch.zeros(NB_GAUSSIANS, dtype=torch.int32, device=device)
            instance_idx[track_size:] = 1
            timestamps_ranges = torch.tensor([[0, 1000000], [500000, 1500000]], dtype=torch.int64, device=device)
            remap_min = 0.0
            remap_max = 1.0
            time_embedding = slang_module.IndividualRemapTimeInputEmbedding(
                instance_idx=instance_idx,
                timestamps_ranges=timestamps_ranges,
                remap_min=remap_min,
                remap_max=remap_max,
            )
            return time_embedding

        def _evaluate_individual_embedding(time_embedding, timestamp):
            instance_idx = time_embedding.instance_idx
            timestamps_ranges = time_embedding.timestamps_ranges
            remap_min = time_embedding.remap_min
            remap_max = time_embedding.remap_max
            timestamps_us_ranges = timestamps_ranges[instance_idx]
            ratio = (timestamp - timestamps_us_ranges[:, 0]) / (timestamps_us_ranges[:, 1] - timestamps_us_ranges[:, 0])
            time_emb = (ratio.clamp(0, 1) * (remap_max - remap_min) + remap_min).unsqueeze(-1)
            return time_emb

        def _get_holistic_embedding():
            timestamps_us_min = 0
            timestamps_us_max = 1000000
            remap_min = 0.0
            remap_max = 1.0
            time_embedding = slang_module.HolisticRemapTimeInputEmbedding(
                timestamps_us_min=timestamps_us_min,
                timestamps_us_max=timestamps_us_max,
                remap_min=remap_min,
                remap_max=remap_max,
            )
            return time_embedding

        def _evaluate_holistic_embedding(time_embedding, timestamp):
            timestamps_us_min = time_embedding.timestamps_us_min
            timestamps_us_max = time_embedding.timestamps_us_max
            remap_min = time_embedding.remap_min
            remap_max = time_embedding.remap_max
            ratio = (timestamp - timestamps_us_min) / (timestamps_us_max - timestamps_us_min)
            ratio = torch.tensor([ratio], dtype=torch.float32, device=device)
            time_emb = (ratio.clamp(0, 1) * (remap_max - remap_min) + remap_min).unsqueeze(-1)
            return time_emb

        for kernel_name, get_embedding, evaluate_embedding, collector_type in [
            (
                "test_spherical_features_fourier_individual",
                _get_individual_embedding,
                _evaluate_individual_embedding,
                "SphericalFeaturesCollector_Fourier_Individual",
            ),
            (
                "test_spherical_features_fourier_holistic",
                _get_holistic_embedding,
                _evaluate_holistic_embedding,
                "SphericalFeaturesCollector_Fourier_Holistic",
            ),
        ]:
            slang_module, kernel = self.kernels[kernel_name]

            timestamp = 500000

            # Forward pass.
            time_embedding = get_embedding()
            albedo = torch.nn.Parameter(torch.randn(NB_GAUSSIANS, FOURIER_DIM, ALBEDO_DIM, device=device))
            specular = torch.nn.Parameter(torch.randn(NB_GAUSSIANS, SPECULAR_DIM, device=device))
            output = torch.randn(OFFSET + NB_GAUSSIANS, ALBEDO_DIM + SPECULAR_DIM, device=device)
            output_untouched = output[:OFFSET].clone()

            # Compute ground truth.
            time_embed = evaluate_embedding(time_embedding, timestamp)

            t = time_embed.view(-1, 1).float()
            dim = FOURIER_DIM
            idft = torch.zeros(t.shape[0], dim, device=device)
            indices = torch.arange(dim, device=device)
            even_indices = indices[::2]
            odd_indices = indices[1::2]
            idft[:, even_indices] = torch.cos(torch.pi * t * even_indices)
            idft[:, odd_indices] = torch.sin(torch.pi * t * (odd_indices + 1))

            ground_truth = torch.cat(
                [
                    torch.sum(albedo * idft.unsqueeze(-1), dim=1),
                    specular,
                ],
                -1,
            )

            collector = getattr(slang_module, collector_type)(
                time_embedding=time_embedding,
                timestamp=timestamp,
                albedo=albedo,
                specular=specular,
                output=output,
            )

            self.assertTrue(torch.equal(output[:OFFSET], output_untouched))
            self.assertFalse(torch.equal(output[OFFSET : OFFSET + NB_GAUSSIANS], ground_truth))
            self._launch(kernel, collector)
            self.assertTrue(torch.equal(output[:OFFSET], output_untouched))
            self.assertTrue(
                torch.allclose(
                    output[OFFSET : OFFSET + NB_GAUSSIANS, 0:ALBEDO_DIM], ground_truth[:, 0:ALBEDO_DIM], atol=atol
                )
            )
            self.assertTrue(
                torch.equal(output[OFFSET : OFFSET + NB_GAUSSIANS, ALBEDO_DIM:], ground_truth[:, ALBEDO_DIM:])
            )

            # Backward pass.
            albedo_grad = (albedo, torch.randn_like(albedo))
            specular_grad = (specular, torch.randn_like(specular))
            output_grad_ground_truth = torch.randn_like(output)

            collector = getattr(slang_module, collector_type)(
                time_embedding=time_embedding,
                timestamp=timestamp,
                albedo=albedo_grad,
                specular=specular_grad,
                output=(output, output_grad_ground_truth),
            )

            self._launch(kernel.bwd, collector)

            albedo.grad = None
            specular.grad = None
            torch.cat([output_untouched, ground_truth], dim=0).backward(output_grad_ground_truth)
            self.assertTrue(torch.allclose(albedo.grad, albedo_grad[1], atol=atol))
            self.assertTrue(torch.allclose(specular.grad, specular_grad[1], atol=atol))

    def test_spherical_features_fourier_individual_step(self):
        """Test IndividualStepTimeInputEmbedding following IndividualStepTimeInputEmbedding from input_embedding.py."""
        atol = 1e-5

        ALBEDO_DIM = 3
        SPECULAR_DIM = 45
        FOURIER_DIM = 20
        N_STEPS = 15
        N_TRACKS = 2

        slang_module, kernel = self.kernels["test_spherical_features_fourier_individual_step"]

        # Setup track assignments: first third -> track 0, rest -> track 1
        track_size = NB_GAUSSIANS // 3
        instance_idx = torch.zeros(NB_GAUSSIANS, dtype=torch.int32, device=device)
        instance_idx[track_size:] = 1

        # Per-track timestamp ranges
        timestamps_ranges = torch.tensor([[0, 1000000], [500000, 1500000]], dtype=torch.int64, device=device)

        # Learnable u parameter: [n_tracks, 1, n_steps]
        # Initialize similar to Python: linspace then reshape
        u_init = torch.linspace(0.0, 1.0, N_STEPS + 2)[1:-1].view(1, 1, N_STEPS)
        u = torch.nn.Parameter(u_init.repeat(N_TRACKS, 1, 1).to(device))

        # Beta parameter (stepness in microseconds)
        beta = torch.tensor(100000.0, dtype=torch.float32, device=device)

        timestamp = 500000

        def _evaluate_individual_step_embedding(instance_idx, timestamps_ranges, u, beta, timestamp):
            """Python ground truth following IndividualStepTimeInputEmbedding.forward() exactly."""
            timestamps_us_ranges = timestamps_ranges[instance_idx]  # [N, 2]

            # ratio = (timestamp - range_min) / (range_max - range_min)
            ratio = (timestamp - timestamps_us_ranges[:, 0].float()) / (
                timestamps_us_ranges[:, 1].float() - timestamps_us_ranges[:, 0].float()
            )

            # beta_scaled = beta / time_range, shape [N, 1, n_steps]
            time_range = (timestamps_us_ranges[:, 1] - timestamps_us_ranges[:, 0]).float()
            beta_scaled = (beta / time_range)[:, None, None].repeat(1, 1, N_STEPS)

            # output = ratio[:, None, None] - u[instance_idx]
            # u[instance_idx] has shape [N, 1, n_steps]
            output = ratio[:, None, None] - u[instance_idx]

            # Step function:
            #   if diff <= 0: 0.5 * exp(diff / clamp(|beta_scaled|, 1e-3))
            #   else: 1 - 0.5 * exp(-diff / clamp(|beta_scaled|, 1e-3))
            abs_beta = torch.clamp_min(torch.abs(beta_scaled), 1e-3)
            msk = output <= 0.0
            result = torch.empty_like(output)
            result[msk] = 0.5 * torch.exp(output[msk] / abs_beta[msk])
            result[~msk] = 1 - 0.5 * torch.exp(-output[~msk] / abs_beta[~msk])

            # Mean over steps: [N, 1]
            time_emb = result.mean(dim=-1)
            return time_emb

        # Forward pass
        albedo = torch.nn.Parameter(torch.randn(NB_GAUSSIANS, FOURIER_DIM, ALBEDO_DIM, device=device))
        specular = torch.nn.Parameter(torch.randn(NB_GAUSSIANS, SPECULAR_DIM, device=device))
        output = torch.randn(OFFSET + NB_GAUSSIANS, ALBEDO_DIM + SPECULAR_DIM, device=device)
        output_untouched = output[:OFFSET].clone()

        # Compute ground truth time embedding
        time_emb = _evaluate_individual_step_embedding(instance_idx, timestamps_ranges, u, beta, timestamp)

        # Compute Fourier features ground truth
        t = time_emb.view(-1, 1).float()
        dim = FOURIER_DIM
        idft = torch.zeros(t.shape[0], dim, device=device)
        indices = torch.arange(dim, device=device)
        even_indices = indices[::2]
        odd_indices = indices[1::2]
        idft[:, even_indices] = torch.cos(torch.pi * t * even_indices)
        idft[:, odd_indices] = torch.sin(torch.pi * t * (odd_indices + 1))

        ground_truth = torch.cat(
            [
                torch.sum(albedo * idft.unsqueeze(-1), dim=1),
                specular,
            ],
            -1,
        )

        # Create Slang time embedding struct
        time_embedding = slang_module.IndividualStepTimeInputEmbedding(
            instance_idx=instance_idx,
            timestamps_ranges=timestamps_ranges,
            u=u,
            beta=beta,
        )

        collector = slang_module.SphericalFeaturesCollector_Fourier_IndividualStep(
            time_embedding=time_embedding,
            timestamp=timestamp,
            albedo=albedo,
            specular=specular,
            output=output,
        )

        self.assertTrue(torch.equal(output[:OFFSET], output_untouched))
        self.assertFalse(torch.equal(output[OFFSET : OFFSET + NB_GAUSSIANS], ground_truth))
        self._launch(kernel, collector)
        self.assertTrue(torch.equal(output[:OFFSET], output_untouched))
        self.assertTrue(
            torch.allclose(
                output[OFFSET : OFFSET + NB_GAUSSIANS, 0:ALBEDO_DIM], ground_truth[:, 0:ALBEDO_DIM], atol=atol
            )
        )
        self.assertTrue(torch.equal(output[OFFSET : OFFSET + NB_GAUSSIANS, ALBEDO_DIM:], ground_truth[:, ALBEDO_DIM:]))

        # Backward pass
        u_grad = (u, torch.zeros_like(u))  # Zero-init because loadEx accumulates
        albedo_grad = (albedo, torch.randn_like(albedo))
        specular_grad = (specular, torch.randn_like(specular))
        output_grad_ground_truth = torch.randn_like(output)

        time_embedding_bwd = slang_module.IndividualStepTimeInputEmbedding(
            instance_idx=instance_idx,
            timestamps_ranges=timestamps_ranges,
            u=u_grad,
            beta=beta,
        )

        collector = slang_module.SphericalFeaturesCollector_Fourier_IndividualStep(
            time_embedding=time_embedding_bwd,
            timestamp=timestamp,
            albedo=albedo_grad,
            specular=specular_grad,
            output=(output, output_grad_ground_truth),
        )

        self._launch(kernel.bwd, collector)

        # Compute ground truth gradients
        u.grad = None
        albedo.grad = None
        specular.grad = None
        torch.cat([output_untouched, ground_truth], dim=0).backward(output_grad_ground_truth)

        self.assertTrue(torch.allclose(albedo.grad, albedo_grad[1], atol=atol))
        self.assertTrue(torch.allclose(specular.grad, specular_grad[1], atol=atol))
        # u gradient uses larger tolerance because:
        # 1. loadEx atomic accumulation can introduce small ordering-dependent errors
        # 2. Gradient flows through step function -> Fourier basis -> output (error amplification)
        # 3. Many Gaussians (1234) accumulate to few u elements (2 tracks * 15 steps = 30)
        u_atol = 5e-4
        self.assertTrue(torch.allclose(u.grad, u_grad[1], atol=u_atol))


if __name__ == "__main__":
    unittest.main()
