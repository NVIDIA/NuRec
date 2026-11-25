# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import warnings

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from ncore.data import (
    FThetaCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ShutterType,
)
from nurec.utils.io.utils import initialize_usd_stage, nurec_tf_to_usd_tf
from nurec.utils.types import ArtifactContents, HalfClosedInterval, NamedSerialized, NamedUSDStage, RigTrajectories


_LENS_DISTORTION_PREFIX = "omni:lensdistortion"
_FTHETA_SCHEMA_ORDER = 5  # the OmniLensDistortionFthetaAPI schema declares k0..k4


_PINHOLE_HORIZONTAL_APERTURE = 20.955  # USD default (mm), the focal length and vertical aperture are derived from it


def _author_pinhole_camera_attributes(camera_prim: Usd.Prim, camera_model: OpenCVPinholeCameraModelParameters) -> None:
    """
    Author the standard USD camera attributes (focal length, apertures and aperture offsets) matching the pinhole
    intrinsics, ignoring the distortion, for viewers that do not read the lens distortion schemas (e.g. the Kit
    viewport, which would otherwise show a default 50 mm perspective camera). RTX renders use the lens distortion
    schema instead. Only pinhole cameras can be described this way: the fisheye and FTheta cameras keep the default
    perspective attributes.

    The aperture offsets use the NCore principal point as is (pixel-corner convention, which is what the USD
    projection expects), whereas the lens distortion schema is shifted by half a pixel to RTX's pixel-center convention.
    """
    width, height = (float(v) for v in camera_model.resolution.tolist())
    fx, fy = (float(v) for v in camera_model.focal_length.tolist())
    cx, cy = (float(v) for v in camera_model.principal_point.tolist())
    horizontal_aperture = _PINHOLE_HORIZONTAL_APERTURE
    focal_length = fx * horizontal_aperture / width
    vertical_aperture = focal_length * height / fy
    values = {
        "focalLength": focal_length,
        "horizontalAperture": horizontal_aperture,
        "verticalAperture": vertical_aperture,
        "horizontalApertureOffset": (width / 2.0 - cx) * horizontal_aperture / width,
        "verticalApertureOffset": (cy - height / 2.0) * vertical_aperture / height,
    }
    for name, value in values.items():
        camera_prim.GetAttribute(name).Set(value)


def _author_lens_distortion(camera_prim: Usd.Prim, camera_model) -> bool:
    """
    Author the camera intrinsics as ``omni:lensdistortion:*`` attributes, applying the matching
    ``OmniLensDistortion*API`` schema. Returns False for unsupported camera models.
    """

    def author(api: str, model: str, values: dict[str, tuple[Sdf.ValueTypeName, object]]) -> None:
        camera_prim.AddAppliedSchema(api)
        camera_prim.CreateAttribute(f"{_LENS_DISTORTION_PREFIX}:model", Sdf.ValueTypeNames.Token).Set(Vt.Token(model))
        for name, (value_type, value) in values.items():
            camera_prim.CreateAttribute(f"{_LENS_DISTORTION_PREFIX}:{model}:{name}", value_type).Set(value)

    float_type = Sdf.ValueTypeNames.Float

    def opencv_values(resolution, principal_point, focal_length, radial_coeffs):
        # NCore's OpenCV principal point is relative to the top-left corner of the image, like the RTX one
        values = {
            "imageSize": (Sdf.ValueTypeNames.Int2, Gf.Vec2i(*(int(v) for v in resolution.tolist()))),
            "fx": (float_type, focal_length.tolist()[0]),
            "fy": (float_type, focal_length.tolist()[1]),
            "cx": (float_type, principal_point.tolist()[0]),
            "cy": (float_type, principal_point.tolist()[1]),
        }
        values.update({f"k{i + 1}": (float_type, v) for i, v in enumerate(radial_coeffs.tolist())})
        return values

    match camera_model:
        case OpenCVPinholeCameraModelParameters(
            resolution=resolution,
            principal_point=principal_point,
            focal_length=focal_length,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
        ):
            values = opencv_values(resolution, principal_point, focal_length, radial_coeffs)
            values.update({f"p{i + 1}": (float_type, v) for i, v in enumerate(tangential_coeffs.tolist())})
            values.update({f"s{i + 1}": (float_type, v) for i, v in enumerate(thin_prism_coeffs.tolist())})
            author("OmniLensDistortionOpenCvPinholeAPI", "opencvPinhole", values)
            _author_pinhole_camera_attributes(camera_prim, camera_model)

        case OpenCVFisheyeCameraModelParameters(
            resolution=resolution,
            principal_point=principal_point,
            focal_length=focal_length,
            radial_coeffs=radial_coeffs,
        ):
            values = opencv_values(resolution, principal_point, focal_length, radial_coeffs)
            author("OmniLensDistortionOpenCvFisheyeAPI", "opencvFisheye", values)

        case FThetaCameraModelParameters(
            resolution=resolution,
            principal_point=principal_point,
            pixeldist_to_angle_poly=pixeldist_to_angle_poly,
            max_angle=max_angle,
            linear_cde=linear_cde,
            reference_poly=reference_poly,
        ):
            if not np.allclose(linear_cde, (1.0, 0.0, 0.0)):
                # the RTX ftheta schema has no linear (skew) term
                warnings.warn(f"FTheta linear_cde {linear_cde.tolist()} is not exported")
            if reference_poly != FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE:
                warnings.warn(
                    f"FTheta reference polynomial is {reference_poly.name}, the exported pixel-distance-to-angle "
                    "polynomial is only an approximation of the model"
                )
            # the schema declares k0..k4 only: higher order coefficients are authored (RTX reads k5) when not zero
            poly = pixeldist_to_angle_poly.tolist()
            values = {
                "nominalWidth": (float_type, float(resolution.tolist()[0])),
                "nominalHeight": (float_type, float(resolution.tolist()[1])),
                # NCore's FTheta principal point has pixel indices at the pixel centers, RTX's optical center is
                # relative to the top-left corner of the image
                "opticalCenter": (Sdf.ValueTypeNames.Float2, Gf.Vec2f(*(principal_point + 0.5).tolist())),
                "maxFov": (float_type, 2.0 * float(np.rad2deg(max_angle))),
            }
            values.update(
                {f"k{i}": (float_type, v) for i, v in enumerate(poly) if i < _FTHETA_SCHEMA_ORDER or v != 0.0}
            )
            author("OmniLensDistortionFthetaAPI", "ftheta", values)

        case _:
            return False
    return True


_ROLLING_SHUTTER_SCAN_ORDER = {
    ShutterType.ROLLING_TOP_TO_BOTTOM: "topToBottom",
    ShutterType.ROLLING_BOTTOM_TO_TOP: "bottomToTop",
    ShutterType.ROLLING_LEFT_TO_RIGHT: "leftToRight",
    ShutterType.ROLLING_RIGHT_TO_LEFT: "rightToLeft",
}


def _author_rolling_shutter(camera_prim: Usd.Prim, camera_model, frame_duration_us: float | None) -> None:
    """
    Author the camera rolling shutter as ``omni:rtx:rollingShutter:*`` attributes (``OmniRtxRollingShutterAPI``).

    NCore spreads the scan lines uniformly over the frame start-to-end interval, i.e. line ``i`` is exposed at
    ``i / (num_lines - 1)`` of the frame duration, hence ``lineOffset = frame_duration / (num_lines - 1)``.
    Global shutter cameras, and cameras without a known frame duration, are left untouched.
    """
    scan_order = _ROLLING_SHUTTER_SCAN_ORDER.get(camera_model.shutter_type)
    if scan_order is None or frame_duration_us is None or frame_duration_us <= 0:
        return

    horizontal_scan = camera_model.shutter_type in (
        ShutterType.ROLLING_LEFT_TO_RIGHT,
        ShutterType.ROLLING_RIGHT_TO_LEFT,
    )
    num_lines = int(camera_model.resolution.tolist()[0 if horizontal_scan else 1])
    if num_lines < 2:
        return

    # rational seconds : (numerator, denominator), kept exact as (duration in us, 1e6 us/s * (num_lines - 1))
    camera_prim.AddAppliedSchema("OmniRtxRollingShutterAPI")
    camera_prim.CreateAttribute("omni:rtx:rollingShutter:scanOrder", Sdf.ValueTypeNames.Token).Set(Vt.Token(scan_order))
    camera_prim.CreateAttribute("omni:rtx:rollingShutter:lineOffset", Sdf.ValueTypeNames.Double2).Set(
        Gf.Vec2d(frame_duration_us, 1e6 * (num_lines - 1))
    )
    camera_prim.CreateAttribute("omni:rtx:rollingShutter:lineExposure", Sdf.ValueTypeNames.Double2).Set(Gf.Vec2d(0, 1))


def add_default_opencv_pinhole_camera(stage: Usd.Stage, path: str, camera_name: str, extrinsics: Gf.Matrix4d):
    """
    Add a default fisheye camera to the stage from the following NCore test model :
    OpenCVPinholeCameraModelParameters(
        resolution=np.array([1920, 1280], dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        principal_point=np.array([935.1248081874216, 635.052474560227], dtype=np.float32),
        focal_length=np.array(
            [
                2059.0471439559833,
                2059.0471439559833,
            ],
            dtype=np.float32,
        ),
        radial_coeffs=np.array(
            [
                0.04239636827428756,
                -0.34165672675852826,
                0,
                0,
                0,
                0,
            ],
            dtype=np.float32,
        ),
        tangential_coeffs=np.array([0.001805535524580487, -0.00005530628187935031], dtype=np.float32),
        thin_prism_coeffs=np.array([0, 0, 0, 0], dtype=np.float32),
    )
    """
    camera_prim = stage.DefinePrim(f"{path}/{camera_name}", "Camera")
    camera_prim.GetAttribute("clippingRange").Set(Gf.Vec2f([0.001, 10000000]))
    _author_lens_distortion(
        camera_prim,
        OpenCVPinholeCameraModelParameters(
            resolution=np.array([1920, 1280], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([935.1248081874216, 635.052474560227], dtype=np.float32),
            focal_length=np.array([2059.0471439559833, 2059.0471439559833], dtype=np.float32),
            radial_coeffs=np.array([0.04239636827428756, -0.34165672675852826, 0, 0, 0, 0], dtype=np.float32),
            tangential_coeffs=np.array([0.001805535524580487, -0.00005530628187935031], dtype=np.float32),
            thin_prism_coeffs=np.array([0, 0, 0, 0], dtype=np.float32),
        ),
    )
    #
    # extrinsics
    #
    camera_transform = UsdGeom.Xformable(camera_prim).AddTransformOp()
    camera_transform.Set(extrinsics)
    #
    # visibility
    #
    UsdGeom.Imageable(camera_prim).CreateVisibilityAttr().Set("invisible")


def add_default_opencv_fisheye_camera(stage: Usd.Stage, path: str, camera_name: str, extrinsics: Gf.Matrix4d):
    """
    Add a default fisheye camera to the stage from the following NCore test model :
    OpenCVFisheyeCameraModelParameters(
        resolution=np.array([3840, 2160], dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        principal_point=np.array([1928.184506, 1083.862789], dtype=np.float32),
        focal_length=np.array(
            [
                1913.76478,
                1913.99708,
            ],
            dtype=np.float32,
        ),
        radial_coeffs=np.array(
            [
                -0.030093122,
                -0.005103817,
                -0.000849622,
                0.001079542,
            ],
            dtype=np.float32,
        ),
        max_angle=np.deg2rad(140 / 2),

    )
    """
    camera_prim = stage.DefinePrim(f"{path}/{camera_name}", "Camera")
    camera_prim.GetAttribute("clippingRange").Set(Gf.Vec2f([0.001, 10000000]))
    _author_lens_distortion(
        camera_prim,
        OpenCVFisheyeCameraModelParameters(
            resolution=np.array([3840, 2160], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([1928.184506, 1083.862789], dtype=np.float32),
            focal_length=np.array([1913.76478, 1913.99708], dtype=np.float32),
            radial_coeffs=np.array([-0.030093122, -0.005103817, -0.000849622, 0.001079542], dtype=np.float32),
            max_angle=np.deg2rad(140 / 2),
        ),
    )
    #
    # extrinsics
    #
    camera_transform = UsdGeom.Xformable(camera_prim).AddTransformOp()
    camera_transform.Set(extrinsics)
    #
    # visibility
    #
    UsdGeom.Imageable(camera_prim).CreateVisibilityAttr().Set("invisible")


def _median_frame_duration_us(frame_timestamps_us: np.ndarray) -> float | None:
    """Median start-to-end frame duration of the Nx2 frame timestamps, or None if there are no frames"""
    if frame_timestamps_us.shape[0] == 0:
        return None
    return float(np.median(frame_timestamps_us[:, 1] - frame_timestamps_us[:, 0]))


def rig_trajectories_time_range(rig_trajectories: RigTrajectories) -> HalfClosedInterval:
    """Returns time range as a half close interval, containing every frame of the rig trajectories associated with the instance."""
    rigs_time_ranges = [
        HalfClosedInterval.from_series(rig.T_rig_world_timestamps_us) for rig in rig_trajectories.rig_trajectories
    ]
    total_time_range = (
        HalfClosedInterval(
            min([range.start for range in rigs_time_ranges]), max([range.end for range in rigs_time_ranges])
        )
        if rigs_time_ranges
        else HalfClosedInterval(0, 0)
    )
    # 53 bits of precision, a bit less than 16 digits (15 digits => ~ 32 years in microseconds)
    double_floating_precision_limit = 1e15
    if total_time_range.length > double_floating_precision_limit:
        warnings.warn(
            f"RigTrajectories span a time range which may not be representable in double precision floating point : {total_time_range.length * 1e-06} seconds"
        )
    return total_time_range


def serialize_rig_trajectories_usd(
    rig_trajectories: RigTrajectories,
    usd_timestamp_offset: int = 0,
    add_default_cameras: bool = True,
    export_rolling_shutter: bool = False,
) -> Usd.Stage:
    """
    Serialize the rig trajectories and the camera intrinsics to a USD stage.

    The rolling shutter of the cameras is only authored with ``export_rolling_shutter``: RTX applies the rolling
    shutter to the animated geometry / materials of a scene but not to the motion of the camera, and the Gaussian
    renderers ignore it, so it is not useful (and has been measured to degrade renders) for Gaussian assets.
    """
    # Get the rig transforms and camera sensor parameters
    rig_transforms: dict[str, dict[float, Gf.Matrix4d]] = {}
    camera_params: dict[str, dict] = {}
    camera_usd = nurec_tf_to_usd_tf([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # invert Y/Z axis
    camera_calibrations = rig_trajectories.camera_calibrations
    for i, rig_trajectory in enumerate(rig_trajectories.rig_trajectories):
        rig_name = f"sensor_rig_{i}"
        rig_transforms[rig_name] = {}
        T_rig_timestamps_us = rig_trajectory.T_rig_world_timestamps_us.numpy()  # timestamp in microseconds
        T_rig_transform = rig_trajectory.T_rig_worlds.numpy().astype(np.double)  # rig to world transform
        assert T_rig_timestamps_us.shape[0] == T_rig_transform.shape[0]
        for t in range(T_rig_timestamps_us.shape[0]):
            rig_transforms[rig_name][T_rig_timestamps_us[t]] = nurec_tf_to_usd_tf(T_rig_transform[t])

        for camera_unique_name in rig_trajectory.cameras_frame_timestamps_us:
            camera_data = camera_calibrations[camera_unique_name]
            camera_params[camera_data.logical_sensor_name] = {
                "rig_name": rig_name,
                "extrinsics": camera_usd * nurec_tf_to_usd_tf(camera_data.T_sensor_rig.numpy().astype(np.double)),
                "intrinsics": camera_data.camera_model_parameters,
                # start-/end-of-frame timestamps in microseconds, used for the rolling shutter
                "frame_timestamps_us": rig_trajectory.cameras_frame_timestamps_us[camera_unique_name].numpy(),
            }

    stage = initialize_usd_stage()

    # Define an xform containing the rig transforms
    usd_time_code_per_second = stage.GetTimeCodesPerSecond()
    usd_timestamp_scale = usd_time_code_per_second * 1e-06
    usd_start_time_code = np.inf
    usd_end_time_code = 0.0
    for rig_name, rig_transform in rig_transforms.items():
        rig_prim = stage.DefinePrim(f"/World/{rig_name}", "Xform")
        rig_xform = UsdGeom.Xformable(rig_prim)
        rig_transform_op = rig_xform.AddTransformOp()
        for timestamps_us, transform in rig_transform.items():
            usd_time_code = usd_timestamp_scale * (timestamps_us - usd_timestamp_offset)
            rig_transform_op.Set(transform, usd_time_code)
            usd_start_time_code = min(usd_start_time_code, usd_time_code)
            usd_end_time_code = max(usd_end_time_code, usd_time_code)

    # time related metadata
    if usd_start_time_code <= usd_end_time_code:
        stage.SetMetadata("startTimeCode", usd_start_time_code)
        stage.SetMetadata("endTimeCode", usd_end_time_code)
    stage.SetMetadataByDictKey("customLayerData", "absoluteTimeOffsetMicroSec", usd_timestamp_offset)

    # Define camera prims
    for name, params in camera_params.items():
        camera_model = params["intrinsics"]
        camera_prim = stage.DefinePrim(f"/World/{params['rig_name']}/{name}", "Camera")
        camera_prim.GetAttribute("clippingRange").Set(Gf.Vec2f([0.001, 10000000]))
        if _author_lens_distortion(camera_prim, camera_model):
            if export_rolling_shutter:
                frame_duration_us = _median_frame_duration_us(params["frame_timestamps_us"])
                _author_rolling_shutter(camera_prim, camera_model, frame_duration_us)
        else:
            # unsupported camera model : fallback to default (intrinsics and rolling shutter are not exported)
            warnings.warn(f"Unsupported camera model {type(camera_model).__name__} for camera '{name}'")
            camera_prim.GetAttribute("focalLength").Set(24.0)

        camera_transform = UsdGeom.Xformable(camera_prim).AddTransformOp()
        camera_transform.Set(params["extrinsics"])

        if add_default_cameras:
            add_default_opencv_pinhole_camera(
                stage,
                f"/World/{params['rig_name']}",
                f"{name}_ocv_pinhole",
                params["extrinsics"],
            )
            add_default_opencv_fisheye_camera(
                stage,
                f"/World/{params['rig_name']}",
                f"{name}_ocv_fisheye",
                params["extrinsics"],
            )

        # Hide camera from view since it will be composed with the NeRF by default
        imageable = UsdGeom.Imageable(camera_prim)
        imageable.CreateVisibilityAttr().Set("invisible")

    return stage


def serialize_rig_trajectories(
    rig_trajectories: RigTrajectories,
    filename: str = "rig_trajectories",
    formats: list[str] = ["json", "usda"],
    usd_timestamp_offset: int = 0,
    add_default_cameras: bool = False,
    export_rolling_shutter: bool = False,
) -> ArtifactContents:
    res: ArtifactContents = []
    for file_format in formats:
        filename_with_suffix = filename + "." + file_format
        match file_format:
            case "json":
                res.append(
                    NamedSerialized(
                        filename=filename_with_suffix, serialized=json.dumps(rig_trajectories.to_dict(), indent=4)
                    )
                )
            case "usd" | "usda":
                res.append(
                    NamedUSDStage(
                        filename=filename_with_suffix,
                        stage=serialize_rig_trajectories_usd(
                            rig_trajectories,
                            usd_timestamp_offset=usd_timestamp_offset,
                            add_default_cameras=add_default_cameras,
                            export_rolling_shutter=export_rolling_shutter,
                        ),
                    )
                )
            case _:
                raise ValueError(f"The following rig trajectory format is not supported: {file_format}")
    return res
