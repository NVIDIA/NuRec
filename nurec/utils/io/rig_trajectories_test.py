# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
import warnings

from collections import OrderedDict

import numpy as np
import torch

from pxr import Gf, Usd, UsdGeom

from ncore.data import (
    FThetaCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ShutterType,
)
from nurec.utils.io.rig_trajectories import (
    _author_lens_distortion,
    _author_rolling_shutter,
    add_default_opencv_fisheye_camera,
    add_default_opencv_pinhole_camera,
    serialize_rig_trajectories_usd,
)
from nurec.utils.types import FrameConversion, RigTrajectories


def _api_schemas(prim: Usd.Prim) -> list[str]:
    # the lens distortion schemas are not registered with the USD runtime, so read the raw metadata
    api_schemas = prim.GetMetadata("apiSchemas")
    return list(api_schemas.GetAddedOrExplicitItems()) if api_schemas else []


class LensDistortionExportTest(unittest.TestCase):
    def test_default_opencv_pinhole(self):
        stage = Usd.Stage.CreateInMemory()
        add_default_opencv_pinhole_camera(stage, "/World", "cam", Gf.Matrix4d(1))
        prim = stage.GetPrimAtPath("/World/cam")
        self.assertIn("OmniLensDistortionOpenCvPinholeAPI", _api_schemas(prim))
        self.assertEqual(prim.GetAttribute("omni:lensdistortion:model").Get(), "opencvPinhole")
        self.assertEqual(prim.GetAttribute("omni:lensdistortion:opencvPinhole:imageSize").Get(), Gf.Vec2i(1920, 1280))
        self.assertAlmostEqual(prim.GetAttribute("omni:lensdistortion:opencvPinhole:fx").Get(), 2059.047, places=2)
        self.assertAlmostEqual(prim.GetAttribute("omni:lensdistortion:opencvPinhole:k2").Get(), -0.3416567, places=5)
        self.assertAlmostEqual(prim.GetAttribute("omni:lensdistortion:opencvPinhole:p1").Get(), 0.0018055, places=6)
        self.assertIsNotNone(prim.GetAttribute("omni:lensdistortion:opencvPinhole:s4").Get())
        # the principal point is exported as is, in the same corner-based convention as NCore
        cx = prim.GetAttribute("omni:lensdistortion:opencvPinhole:cx").Get()
        cy = prim.GetAttribute("omni:lensdistortion:opencvPinhole:cy").Get()
        self.assertAlmostEqual(cx, 935.1248, places=3)
        self.assertAlmostEqual(cy, 635.0525, places=3)
        # standard USD camera attributes matching the intrinsics, for the viewers ignoring the lens distortion schema
        horizontal_aperture = prim.GetAttribute("horizontalAperture").Get()
        self.assertAlmostEqual(prim.GetAttribute("focalLength").Get(), 2059.0471 * horizontal_aperture / 1920, places=4)
        horizontal_offset = prim.GetAttribute("horizontalApertureOffset").Get()
        self.assertAlmostEqual(horizontal_offset, (960 - 935.1248) * horizontal_aperture / 1920, places=4)
        self.assertFalse(prim.HasAttribute("cameraProjectionType"))
        self.assertFalse(prim.HasAttribute("fthetaWidth"))

    def test_default_opencv_fisheye(self):
        stage = Usd.Stage.CreateInMemory()
        add_default_opencv_fisheye_camera(stage, "/World", "cam", Gf.Matrix4d(1))
        prim = stage.GetPrimAtPath("/World/cam")
        self.assertIn("OmniLensDistortionOpenCvFisheyeAPI", _api_schemas(prim))
        self.assertEqual(prim.GetAttribute("omni:lensdistortion:model").Get(), "opencvFisheye")
        self.assertAlmostEqual(prim.GetAttribute("omni:lensdistortion:opencvFisheye:cx").Get(), 1928.1845, places=3)
        self.assertAlmostEqual(prim.GetAttribute("omni:lensdistortion:opencvFisheye:k4").Get(), 0.001079542, places=6)

    def test_opencv_principal_point_is_exported_as_is_for_even_and_odd_image_sizes(self):
        stage = Usd.Stage.CreateInMemory()
        for name, (width, height) in {"even": (1920, 1080), "odd": (1921, 1081), "odd_width": (1921, 1080)}.items():
            for model_cls, api, model_name in (
                (OpenCVPinholeCameraModelParameters, "OmniLensDistortionOpenCvPinholeAPI", "opencvPinhole"),
                (OpenCVFisheyeCameraModelParameters, "OmniLensDistortionOpenCvFisheyeAPI", "opencvFisheye"),
            ):
                kwargs = dict(
                    resolution=np.array([width, height], dtype=np.uint64),
                    shutter_type=ShutterType.GLOBAL,
                    principal_point=np.array([960.25, 540.75], dtype=np.float32),
                    focal_length=np.array([1000.0, 1000.0], dtype=np.float32),
                    radial_coeffs=np.zeros(6 if model_cls is OpenCVPinholeCameraModelParameters else 4, np.float32),
                )
                if model_cls is OpenCVPinholeCameraModelParameters:
                    kwargs.update(
                        tangential_coeffs=np.zeros(2, np.float32),
                        thin_prism_coeffs=np.zeros(4, np.float32),
                    )
                else:
                    kwargs.update(max_angle=float(np.deg2rad(70)))
                prim = stage.DefinePrim(f"/{name}_{model_name}", "Camera")
                self.assertTrue(_author_lens_distortion(prim, model_cls(**kwargs)))
                self.assertIn(api, _api_schemas(prim))
                cx = prim.GetAttribute(f"omni:lensdistortion:{model_name}:cx").Get()
                cy = prim.GetAttribute(f"omni:lensdistortion:{model_name}:cy").Get()
                self.assertAlmostEqual(cx, 960.25, places=4)
                self.assertAlmostEqual(cy, 540.75, places=4)

    def test_ftheta(self):
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/cam", "Camera")
        model = FThetaCameraModelParameters(
            resolution=np.array([1920, 1080], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([960, 540], dtype=np.float32),
            reference_poly=FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE,
            pixeldist_to_angle_poly=np.array([0, 1e-3, 0, 0, 0, 0], dtype=np.float32),
            angle_to_pixeldist_poly=np.array([0, 1e3, 0, 0, 0, 0], dtype=np.float32),
            max_angle=np.deg2rad(100),
        )
        self.assertTrue(_author_lens_distortion(prim, model))
        self.assertIn("OmniLensDistortionFthetaAPI", _api_schemas(prim))
        self.assertEqual(prim.GetAttribute("omni:lensdistortion:model").Get(), "ftheta")
        self.assertEqual(prim.GetAttribute("omni:lensdistortion:ftheta:opticalCenter").Get(), Gf.Vec2f(960.5, 540.5))
        self.assertAlmostEqual(prim.GetAttribute("omni:lensdistortion:ftheta:maxFov").Get(), 200.0, places=3)
        self.assertAlmostEqual(prim.GetAttribute("omni:lensdistortion:ftheta:k1").Get(), 1e-3, places=6)

    def test_ftheta_k5_only_when_not_zero_and_reference_poly_warning(self):
        stage = Usd.Stage.CreateInMemory()

        def model(poly, reference):
            return FThetaCameraModelParameters(
                resolution=np.array([1920, 1080], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([960, 540], dtype=np.float32),
                reference_poly=reference,
                pixeldist_to_angle_poly=np.array(poly, dtype=np.float32),
                angle_to_pixeldist_poly=np.array([0, 1e3, 0, 0, 0, 0], dtype=np.float32),
                max_angle=np.deg2rad(100),
            )

        pixel_to_angle = FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE
        angle_to_pixel = FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            zero = stage.DefinePrim("/zero", "Camera")
            _author_lens_distortion(zero, model([0, 1e-3, 0, 0, 0, 0], pixel_to_angle))
            nonzero = stage.DefinePrim("/nonzero", "Camera")
            _author_lens_distortion(nonzero, model([0, 1e-3, 0, 0, 0, 1e-15], angle_to_pixel))
        self.assertFalse(zero.HasAttribute("omni:lensdistortion:ftheta:k5"))
        self.assertTrue(zero.HasAttribute("omni:lensdistortion:ftheta:k4"))
        self.assertTrue(nonzero.HasAttribute("omni:lensdistortion:ftheta:k5"))
        self.assertEqual(sum("reference polynomial" in str(w.message) for w in caught), 1)

    def test_ftheta_linear_cde_warns_when_not_identity(self):
        stage = Usd.Stage.CreateInMemory()
        model = FThetaCameraModelParameters(
            resolution=np.array([1920, 1080], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([960, 540], dtype=np.float32),
            reference_poly=FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE,
            pixeldist_to_angle_poly=np.array([0, 1e-3, 0, 0, 0, 0], dtype=np.float32),
            angle_to_pixeldist_poly=np.array([0, 1e3, 0, 0, 0, 0], dtype=np.float32),
            max_angle=np.deg2rad(100),
            linear_cde=np.array([1.0, 0.01, 0.0], dtype=np.float32),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _author_lens_distortion(stage.DefinePrim("/cam", "Camera"), model)
        self.assertTrue(any("linear_cde" in str(w.message) for w in caught))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.linear_cde = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            _author_lens_distortion(stage.DefinePrim("/cam2", "Camera"), model)
        self.assertFalse(any("linear_cde" in str(w.message) for w in caught))


class PinholeCameraAttributesTest(unittest.TestCase):
    def test_standard_camera_matches_the_opencv_pinhole_projection(self):
        width, height, fx, fy, cx, cy = 1920, 1200, 852.4, 860.1, 946.4, 556.8
        model = OpenCVPinholeCameraModelParameters(
            resolution=np.array([width, height], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([cx, cy], dtype=np.float32),
            focal_length=np.array([fx, fy], dtype=np.float32),
            radial_coeffs=np.zeros(6, dtype=np.float32),
            tangential_coeffs=np.zeros(2, dtype=np.float32),
            thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        )
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/cam", "Camera")
        self.assertTrue(_author_lens_distortion(prim, model))
        projection = np.array(UsdGeom.Camera(prim).GetCamera(0).frustum.ComputeProjectionMatrix())
        fx, fy, cx, cy = (float(v) for v in (*model.focal_length, *model.principal_point))
        for x, y, z in ((0.3, -0.2, -2.0), (-0.5, 0.4, -3.0), (0.7, 0.5, -4.0)):
            clip = np.array([x, y, z, 1.0]) @ projection
            ndc = clip[:2] / clip[3]
            u, v = (ndc[0] + 1) / 2 * width, (1 - ndc[1]) / 2 * height
            self.assertAlmostEqual(u, fx * x / -z + cx, delta=1e-2)
            self.assertAlmostEqual(v, cy - fy * y / -z, delta=1e-2)

    def test_other_camera_models_keep_the_default_standard_attributes(self):
        stage = Usd.Stage.CreateInMemory()
        fisheye = stage.DefinePrim("/fisheye", "Camera")
        default_focal_length = fisheye.GetAttribute("focalLength").Get()
        add_default_opencv_fisheye_camera(stage, "/World", "default_fisheye", Gf.Matrix4d(1))
        default_fisheye = stage.GetPrimAtPath("/World/default_fisheye")
        self.assertEqual(default_fisheye.GetAttribute("focalLength").Get(), default_focal_length)
        self.assertEqual(default_fisheye.GetAttribute("horizontalApertureOffset").Get(), 0.0)


class RollingShutterExportTest(unittest.TestCase):
    @staticmethod
    def _model(shutter_type: ShutterType) -> OpenCVFisheyeCameraModelParameters:
        return OpenCVFisheyeCameraModelParameters(
            resolution=np.array([1921, 1081], dtype=np.uint64),
            shutter_type=shutter_type,
            principal_point=np.array([960, 540], dtype=np.float32),
            focal_length=np.array([1000, 1000], dtype=np.float32),
            radial_coeffs=np.zeros(4, dtype=np.float32),
            max_angle=1.0,
        )

    def test_vertical_scan(self):
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/cam", "Camera")
        _author_rolling_shutter(prim, self._model(ShutterType.ROLLING_TOP_TO_BOTTOM), 33_000.0)
        self.assertIn("OmniRtxRollingShutterAPI", _api_schemas(prim))
        self.assertEqual(prim.GetAttribute("omni:rtx:rollingShutter:scanOrder").Get(), "topToBottom")
        offset = prim.GetAttribute("omni:rtx:rollingShutter:lineOffset").Get()
        # 33 ms over (1081 - 1) lines
        self.assertAlmostEqual(offset[0] / offset[1], 0.033 / 1080, places=12)

    def test_horizontal_scan_uses_width(self):
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/cam", "Camera")
        _author_rolling_shutter(prim, self._model(ShutterType.ROLLING_RIGHT_TO_LEFT), 19_200.0)
        self.assertEqual(prim.GetAttribute("omni:rtx:rollingShutter:scanOrder").Get(), "rightToLeft")
        offset = prim.GetAttribute("omni:rtx:rollingShutter:lineOffset").Get()
        self.assertAlmostEqual(offset[0] / offset[1], 0.0192 / 1920, places=12)

    def test_global_shutter_or_unknown_duration_is_skipped(self):
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/cam", "Camera")
        _author_rolling_shutter(prim, self._model(ShutterType.GLOBAL), 33_000.0)
        _author_rolling_shutter(prim, self._model(ShutterType.ROLLING_TOP_TO_BOTTOM), None)
        self.assertEqual(_api_schemas(prim), [])
        self.assertFalse(prim.HasAttribute("omni:rtx:rollingShutter:scanOrder"))


class SerializeRigTrajectoriesTest(unittest.TestCase):
    def test_rolling_shutter_is_only_exported_on_request(self):
        rig = self._rig_trajectories(
            self._fisheye(ShutterType.ROLLING_TOP_TO_BOTTOM), [[0, 30_000], [100_000, 136_000], [200_000, 233_000]]
        )
        default_stage = serialize_rig_trajectories_usd(rig, add_default_cameras=False)
        default_prim = default_stage.GetPrimAtPath("/World/sensor_rig_0/camera_front")
        self.assertNotIn("OmniRtxRollingShutterAPI", _api_schemas(default_prim))
        self.assertFalse(default_prim.HasAttribute("omni:rtx:rollingShutter:scanOrder"))
        self.assertIn("OmniLensDistortionOpenCvFisheyeAPI", _api_schemas(default_prim))
        requested_stage = serialize_rig_trajectories_usd(rig, add_default_cameras=False, export_rolling_shutter=True)
        requested_prim = requested_stage.GetPrimAtPath("/World/sensor_rig_0/camera_front")
        self.assertIn("OmniRtxRollingShutterAPI", _api_schemas(requested_prim))

    @staticmethod
    def _rig_trajectories(camera_model, frame_timestamps_us: list[list[int]]) -> RigTrajectories:
        camera_id = "camera_front"
        return RigTrajectories(
            T_world_base=torch.eye(4, dtype=torch.float64),
            world_to_nurec=FrameConversion(matrix=np.eye(4, dtype=np.float32)),
            rig_trajectories=[
                RigTrajectories.RigTrajectory(
                    sequence_id="seq",
                    rig_bbox=None,
                    cameras_frame_timestamps_us={camera_id: torch.tensor(frame_timestamps_us, dtype=torch.int64)},
                    lidars_frame_timestamps_us={},
                    T_rig_worlds=torch.eye(4, dtype=torch.float64).repeat(2, 1, 1),
                    T_rig_world_timestamps_us=torch.tensor([0, 100_000], dtype=torch.int64),
                )
            ],
            camera_calibrations=OrderedDict(
                {
                    camera_id: RigTrajectories.CameraCalibration(
                        sequence_id="seq",
                        logical_sensor_name="camera_front",
                        unique_sensor_idx=0,
                        T_sensor_rig=torch.eye(4, dtype=torch.float32),
                        camera_model_parameters=camera_model,
                    )
                }
            ),
            lidar_calibrations=OrderedDict(),
        )

    @staticmethod
    def _fisheye(shutter_type: ShutterType) -> OpenCVFisheyeCameraModelParameters:
        return RollingShutterExportTest._model(shutter_type)

    def test_rolling_shutter_uses_median_frame_duration(self):
        # frame durations 30 ms, 36 ms, 33 ms -> median 33 ms (resolution 1921 x 1081 -> 1080 line intervals)
        rig = self._rig_trajectories(
            self._fisheye(ShutterType.ROLLING_TOP_TO_BOTTOM), [[0, 30_000], [100_000, 136_000], [200_000, 233_000]]
        )
        stage = serialize_rig_trajectories_usd(rig, add_default_cameras=False, export_rolling_shutter=True)
        prim = stage.GetPrimAtPath("/World/sensor_rig_0/camera_front")
        self.assertIn("OmniLensDistortionOpenCvFisheyeAPI", _api_schemas(prim))
        self.assertIn("OmniRtxRollingShutterAPI", _api_schemas(prim))
        offset = prim.GetAttribute("omni:rtx:rollingShutter:lineOffset").Get()
        self.assertAlmostEqual(offset[0] / offset[1], 0.033 / 1080, places=12)

    def test_global_shutter_has_no_rolling_shutter(self):
        rig = self._rig_trajectories(self._fisheye(ShutterType.GLOBAL), [[0, 30_000]])
        stage = serialize_rig_trajectories_usd(rig, add_default_cameras=False, export_rolling_shutter=True)
        prim = stage.GetPrimAtPath("/World/sensor_rig_0/camera_front")
        self.assertNotIn("OmniRtxRollingShutterAPI", _api_schemas(prim))

    def test_unsupported_model_falls_back_with_warning(self):
        rig = self._rig_trajectories(object(), [[0, 30_000]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stage = serialize_rig_trajectories_usd(rig, add_default_cameras=False)
        self.assertTrue(any("Unsupported camera model" in str(w.message) for w in caught))
        prim = stage.GetPrimAtPath("/World/sensor_rig_0/camera_front")
        self.assertEqual(_api_schemas(prim), [])
        self.assertAlmostEqual(prim.GetAttribute("focalLength").Get(), 24.0)


if __name__ == "__main__":
    unittest.main()
