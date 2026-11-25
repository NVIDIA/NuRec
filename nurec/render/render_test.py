# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import os
import unittest

from types import SimpleNamespace
from typing import Any, cast
from unittest import mock
from unittest.mock import patch

import pytest
import torch

from omegaconf import OmegaConf

import nurec.render.render as render_module

from nurec.render.actors import ActorsSnapshot
from nurec.render.render import (
    CameraRenderOptions,
    CpuPreparationUnsupportedError,
    LidarRayBundle,
    RayBundle,
    RenderableModel,
)
from nurec.utils.batch import FrameMeta, RenderingData
from nurec.utils.types import GaussiansCompositeReturn, GaussiansRenderReturn


class TestSensorRenderExecutionMode(unittest.TestCase):
    def test_camera_and_lidar_use_the_same_execution_mode(self):
        observed_modes = {}

        class ObservingGaussian:
            def parameters(self):
                yield torch.empty(0)

            def __call__(self, *, batch, **_kwargs):
                sensor = "camera" if batch.data.camera is not None else "lidar"
                observed_modes[sensor] = (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
                rendered_cam = GaussiansRenderReturn(opacity=torch.empty(0)) if sensor == "camera" else None
                return GaussiansCompositeReturn(rendered_cam=rendered_cam)

        renderable = RenderableModel(cast(Any, ObservingGaussian()))
        timestamps = torch.tensor([[10, 20]], dtype=torch.int64)
        ray_bundle = RayBundle(
            rendering_data=RenderingData(
                rays=torch.zeros(1, 1, 1, 6),
                sensor_model_parameters=[None],
                poses_tquat_startend=torch.zeros(1, 2, 7),
                timestamps_startend_us=timestamps,
                timestamps_startend_us_cpu=timestamps,
            ),
            frame_meta=FrameMeta(unique_sensor_idx=0, unique_frame_idx=-1, training_frame_idx=-1),
        )

        with mock.patch("nurec.render.render.GaussiansComposite", ObservingGaussian):
            renderable._render_volume_from_ray_bundle(cast(Any, ray_bundle))
            renderable._lidar_rays_inference(cast(Any, ray_bundle))

        self.assertEqual(observed_modes["camera"], observed_modes["lidar"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_camera_render_keeps_cpu_timestamps_and_tensor_dtypes() -> None:
    timestamps = torch.tensor([[10, 20]], dtype=torch.int64)
    rendering = RenderingData(
        rays=torch.zeros(1, 1, 1, 6, dtype=torch.float64),
        sensor_model_parameters=[None],
        poses_tquat_startend=torch.zeros(1, 2, 7, dtype=torch.float64),
        timestamps_startend_us=timestamps,
        timestamps_startend_us_cpu=timestamps,
    )

    class ObservingGaussian:
        def parameters(self):
            yield torch.empty(0, device="cuda")

        def __call__(self, *, batch, **_kwargs):
            camera = batch.rendering.camera
            assert camera.timestamps_startend_us_cpu is timestamps
            assert camera.timestamps_startend_us_cpu.device.type == "cpu"
            assert camera.timestamps_startend_us.device.type == "cuda"
            assert camera.timestamps_startend_us.dtype == torch.int64
            assert camera.rays.device.type == camera.poses_tquat_startend.device.type == "cuda"
            assert camera.rays.dtype == camera.poses_tquat_startend.dtype == torch.float32
            torch.testing.assert_close(camera.timestamps_startend_us.cpu(), timestamps)
            return GaussiansCompositeReturn(
                rendered_cam=GaussiansRenderReturn(
                    rgb=torch.zeros(1, 3, device="cuda"), opacity=torch.ones(1, 1, device="cuda")
                )
            )

    renderable = RenderableModel(cast(Any, ObservingGaussian()))
    bundle = RayBundle(rendering, FrameMeta(unique_sensor_idx=0, unique_frame_idx=-1, training_frame_idx=-1))
    with mock.patch("nurec.render.render.GaussiansComposite", ObservingGaussian):
        frame = renderable.render_camera_frame_from_ray_bundle(bundle, CameraRenderOptions(color_image=True))
    assert frame.color_image is not None and frame.color_image.shape == (1, 1, 3)
    assert rendering.timestamps_startend_us is timestamps


class TestCompactLidarRendering(unittest.TestCase):
    def test_rejects_compact_lidar_extra_signal_outputs(self):
        renderable = RenderableModel(
            cast(Any, object()),
            _compact_lidar_extra_signal_tensors_missing=("lidar_extra_signal",),
        )

        with self.assertRaisesRegex(ValueError, "omit requested LiDAR extra-signal tensors: lidar_extra_signal"):
            renderable.render_lidar_frame_from_ray_bundle(cast(LidarRayBundle, object()))

    def test_rejects_compact_semantic_outputs_without_camera_signals(self):
        renderable = RenderableModel(
            cast(Any, object()),
            _compact_camera_extra_signals_missing=True,
        )

        with self.assertRaisesRegex(ValueError, "--preserve-extra-signal-tensor camera_extra_signal"):
            renderable.render_camera_frame_from_ray_bundle(
                cast(Any, object()),
                CameraRenderOptions(semantic_image=True),
            )

    def test_lidar_output_signal_tensors_enabled_by_default(self):
        config = OmegaConf.create({})

        self.assertEqual(
            RenderableModel._requested_lidar_output_signal_tensors(config),
            ("extra_signal", "lidar_extra_signal"),
        )

    def test_lidar_output_signal_tensors_can_be_disabled(self):
        config = OmegaConf.create(
            {
                "model": {
                    "renderer": {
                        "outputs": {
                            "lidar": {
                                "enable_extended_features": False,
                                "enable_sensor_features": False,
                            },
                        },
                    },
                },
            }
        )

        self.assertEqual(RenderableModel._requested_lidar_output_signal_tensors(config), ())

    def test_lidar_output_signal_tensors_follow_common_gate(self):
        config = OmegaConf.create(
            {
                "model": {
                    "renderer": {
                        "outputs": {
                            "lidar": {
                                "enable_extended_features": True,
                                "enable_sensor_features": False,
                            },
                        },
                    },
                },
            }
        )

        self.assertEqual(RenderableModel._requested_lidar_output_signal_tensors(config), ("extra_signal",))

    def test_compact_lidar_output_signals_missing_without_preserved_tensor(self):
        artifact = cast(Any, SimpleNamespace(is_compact=True, source="scene.compact.usdz"))

        self.assertEqual(
            RenderableModel._compact_lidar_output_signal_tensors_missing(artifact, OmegaConf.create({}), frozenset()),
            ("extra_signal", "lidar_extra_signal"),
        )

    def test_compact_lidar_output_signals_not_missing_with_preserved_tensor(self):
        artifact = cast(Any, SimpleNamespace(is_compact=True, source="scene.compact.usdz"))

        self.assertEqual(
            RenderableModel._compact_lidar_output_signal_tensors_missing(
                artifact, OmegaConf.create({}), frozenset(("extra_signal", "lidar_extra_signal"))
            ),
            (),
        )

    def test_compact_lidar_output_signals_missing_when_common_signal_omitted(self):
        artifact = cast(Any, SimpleNamespace(is_compact=True, source="scene.compact.usdz"))

        self.assertEqual(
            RenderableModel._compact_lidar_output_signal_tensors_missing(
                artifact, OmegaConf.create({}), frozenset(("lidar_extra_signal",))
            ),
            ("extra_signal",),
        )


def test_camera_batch_uses_rendering_interval_for_actor_edits() -> None:
    timestamps = torch.tensor([[10, 20], [10, 20]], dtype=torch.int64)
    rendering = RenderingData(
        rays=torch.zeros(2, 2, 3, 6),
        sensor_model_parameters=[None, None],
        poses_tquat_startend=torch.zeros(2, 2, 7),
        timestamps_startend_us=timestamps.clone(),
        timestamps_startend_us_cpu=timestamps,
    )
    frame_meta = [FrameMeta(unique_sensor_idx=index, unique_frame_idx=-1, training_frame_idx=-1) for index in range(2)]
    rendered = GaussiansRenderReturn(rgb=torch.zeros(12, 3), opacity=torch.zeros(12, 1))
    model = mock.MagicMock(return_value=GaussiansCompositeReturn(rendered_cam=rendered))
    renderable = RenderableModel(cast(Any, model))
    actors = ActorsSnapshot.empty()

    with (
        mock.patch.object(renderable, "supports_shared_camera_batch", return_value=True),
        mock.patch.object(renderable, "_edit_actors", return_value=None) as edit_actors,
    ):
        frames = renderable.render_camera_color_batch(rendering, frame_meta, actors)
        edit_actors.assert_called_once_with(actors, 10, 20)
        assert len(frames) == 2

        rendering.timestamps_startend_us_cpu[1, 1] = 21
        with pytest.raises(ValueError, match="same frame interval"):
            renderable.render_camera_color_batch(rendering, frame_meta, actors)
        edit_actors.assert_called_once()


def test_checkpoint_loaders_honor_cpu_device() -> None:
    checkpoint: dict[str, Any] = {"state_dict": {}}
    normal = cast(Any, SimpleNamespace(is_compact=False, open_checkpoint=io.BytesIO))
    with mock.patch("nurec.render.render.torch.load", return_value=checkpoint) as load:
        RenderableModel._load_gaussian_checkpoint(normal, torch.device("cpu"))
    assert load.call_args.kwargs["map_location"] == "cpu"

    compact = cast(Any, SimpleNamespace(is_compact=True, source="scene.usdz"))
    with mock.patch("nurec.render.render.compact_usdz.load_compact_checkpoint", return_value=checkpoint) as load:
        RenderableModel._load_gaussian_checkpoint(compact, torch.device("cpu"))
    assert load.call_args.kwargs["device"] == torch.device("cpu")


def test_instant_nurec_rejects_cpu_preparation() -> None:
    config = OmegaConf.create({"model": {"name": "celsius"}})
    with pytest.raises(CpuPreparationUnsupportedError, match="Instant NuRec artifacts"):
        RenderableModel._load_model_and_upgrade(config, cast(Any, object()), config, torch.device("cpu"))


@pytest.mark.parametrize(
    ("parameter_devices", "track_devices"),
    [(["cuda"], []), (["cpu"], ["cuda"])],
)
def test_cpu_preparation_rejects_non_cpu_model_state(parameter_devices: list[str], track_devices: list[str]) -> None:
    config = OmegaConf.create({"model": {}, "trainer": {}})
    artifact = mock.MagicMock(datasource_summary={})
    model = mock.MagicMock()
    model.parameters.return_value = [mock.MagicMock(device=torch.device(device)) for device in parameter_devices]
    model.buffers.return_value = []
    model.cuboid_track_devices.return_value = {torch.device(device) for device in track_devices}

    with (
        mock.patch("nurec.render.render.ModelConfig.model_validate") as model_config,
        mock.patch("nurec.render.render.TrainerConfig.model_validate"),
        mock.patch("nurec.render.render.DataSourceSummary.from_dict"),
        mock.patch("nurec.render.render.GaussiansComposite", return_value=model),
        mock.patch.object(RenderableModel, "_load_gaussian_checkpoint", return_value={"state_dict": {}}),
        mock.patch("nurec.render.render.upgrade_model", return_value={}),
        pytest.raises(CpuPreparationUnsupportedError, match="retained non-CPU model tensors"),
    ):
        model_config.return_value.renderer.name = "gsplat"
        RenderableModel._load_model_gaussians_composite_and_upgrade(config, artifact, config, torch.device("cpu"))


# --- Inference-time per-track timestamp-estimation ray-stride knob ---
# Covers the guarantees the feature must keep:
#   1. it is a no-op unless explicitly opted in (default behavior unchanged),
#   2. when enabled it produces the intended downsampling effect (every Nth ray,
#      fanned out to the dynamic track layers), and
#   3. it only targets track layers actually present in the artifact, so a static or
#      single-dynamic-layer config never gets a phantom partial layer injected.

BOTH_LAYERS = ["background", "road", "dynamic_rigids", "dynamic_deformables"]


def test_stride_one_is_a_noop():
    # Default stride must add no config overrides, so existing callers are unaffected.
    assert RenderableModel.tracks_timestamp_ray_stride_overrides(1, BOTH_LAYERS) == ()


@pytest.mark.parametrize("stride", [2, 4, 16, 64, 256])
def test_stride_overrides_fan_out_to_present_track_layers(stride):
    # Enabling the knob sets the stride on both dynamic track layers, nothing else.
    assert RenderableModel.tracks_timestamp_ray_stride_overrides(stride, BOTH_LAYERS) == (
        f"model.layers.dynamic_rigids.tracks_timestamp_estimation_ray_stride={stride}",
        f"model.layers.dynamic_deformables.tracks_timestamp_estimation_ray_stride={stride}",
    )


def test_stride_skips_absent_track_layers():
    # Single-dynamic-layer config: only the present layer gets an override (no phantom
    # dynamic_deformables entry that would fail typed-layer validation).
    assert RenderableModel.tracks_timestamp_ray_stride_overrides(64, ["background", "dynamic_rigids"]) == (
        "model.layers.dynamic_rigids.tracks_timestamp_estimation_ray_stride=64",
    )


def test_stride_on_static_config_emits_nothing():
    # Static-only artifact (no track layers): the knob is a no-op, not a phantom layer.
    assert RenderableModel.tracks_timestamp_ray_stride_overrides(64, ["background", "road"]) == ()


@pytest.mark.parametrize("bad_stride", [0, -1, -64])
def test_stride_below_one_is_rejected(bad_stride):
    with pytest.raises(ValueError):
        RenderableModel.tracks_timestamp_ray_stride_overrides(bad_stride, BOTH_LAYERS)


STRIDE_KEY = "model.layers.dynamic_rigids.tracks_timestamp_estimation_ray_stride"


def _merged_stride(stride: int, caller_overrides: tuple[str, ...]) -> int:
    """Resolve the stride the model would actually see, through the real dotlist merge."""
    stride_overrides = RenderableModel.tracks_timestamp_ray_stride_overrides(stride, BOTH_LAYERS)
    ordered = RenderableModel._ordered_config_overrides(stride_overrides, caller_overrides)
    return OmegaConf.from_dotlist(list(ordered)).model.layers.dynamic_rigids.tracks_timestamp_estimation_ray_stride


def test_cli_stride_reaches_the_track_layers():
    assert _merged_stride(16, ()) == 16


def test_explicit_hydra_override_outranks_the_cli_stride():
    """An explicit override is a deliberate request and must survive the CLI-level default.

    Regression: the stride used to be appended after the caller's overrides, so a caller asking
    for full estimation via Hydra was silently upgraded to the default stride.
    """
    assert _merged_stride(16, (f"{STRIDE_KEY}=1",)) == 1
    assert _merged_stride(16, (f"{STRIDE_KEY}=64",)) == 64


INFERENCE_DEFAULT_KEYS = tuple(entry.split("=", 1)[0] for entry in RenderableModel._INFERENCE_DEFAULT_OVERRIDES)


@pytest.mark.parametrize("key", INFERENCE_DEFAULT_KEYS)
def test_inference_defaults_stay_overridable_by_the_caller(key):
    """An explicit caller override outranks every inference default (the stride must not displace that)."""
    ordered = RenderableModel._ordered_config_overrides((), (f"{key}=True",))
    assert OmegaConf.select(OmegaConf.from_dotlist(list(ordered)), key) is True


def _grid_stride_factors(stride: int) -> tuple[int, int]:
    # Mirror of the RigidGaussianModel estimation dispatch: symmetric 2D-grid decimation
    # (~sqrt(N) per axis). Symmetric minimizes the worst-case per-axis stride, which measured
    # best for keeping small tracks (small in both axes); it beat an aspect-weighted split.
    import math

    row_stride = math.isqrt(stride)
    return row_stride, -(-stride // row_stride)  # ceil: row_stride * col_stride >= stride


@pytest.mark.parametrize("stride", [4, 16, 64])
def test_grid_subsample_keeps_about_one_over_stride(stride):
    # The 2D grid decimation keeps ~1/stride of a (B, H, W, 6) sensor grid.
    b, h, w = 1, 2160, 3840
    rays = torch.zeros(b, h, w, 6)
    sr, sc = _grid_stride_factors(stride)
    frac = rays[:, ::sr, ::sc, :].reshape(-1, 6).shape[0] / (b * h * w)
    assert 1.0 / (stride * 1.6) <= frac <= 1.0 / (stride / 1.6)


def test_grid_subsample_hits_narrow_tall_track_that_1d_stride_would_miss():
    # A thin, tall object spanning columns [70, 120) at 4K width (3840). A flat rays[::64]
    # only ever samples columns 0,64,128,... (3840 % 64 == 0) -> none inside [70,120) ->
    # the whole object gets zero samples. The 2D grid strides columns by ~8, so several
    # sampled columns land inside the band. Guards the aliasing fix.
    w, stride = 3840, 64
    col_lo, col_hi = 70, 120
    flat_cols = {(k * stride) % w for k in range((2160 * w) // stride)}
    assert not any(col_lo <= c < col_hi for c in flat_cols), "1D stride should miss the band (aliasing)"
    _, sc = _grid_stride_factors(stride)
    grid_cols = set(range(0, w, sc))
    assert any(col_lo <= c < col_hi for c in grid_cols), "2D grid stride must sample inside the band"


def _nrend_artifact_config(renderer_name: str):
    # Minimal shape of an old artifact config: only the version stamp and the trained
    # renderer name matter to the NRend migration upgrader.
    return OmegaConf.create(
        {
            "version": {"version_major": 26, "version_minor": 7, "version_patch": 9},
            "model": {"renderer": {"name": renderer_name}},
        }
    )


def test_renderer_override_rewrites_only_an_existing_renderer_name():
    config = _nrend_artifact_config("3dgrt-optix-nrend")

    overridden = RenderableModel._with_renderer_override(config, ("model.renderer.name=3dgut-gsplat",))

    assert overridden.model.renderer.name == "3dgut-gsplat"
    # The caller's config is the record of what the artifact was trained with.
    assert config.model.renderer.name == "3dgrt-optix-nrend"

    no_renderer = OmegaConf.create({"model": {}})
    assert RenderableModel._with_renderer_override(no_renderer, ("model.renderer.name=3dgut-gsplat",)) is no_renderer
    assert RenderableModel._with_renderer_override(config, ()) is config


def test_ray_traced_nrend_artifact_needs_an_explicit_gsplat_renderer():
    from nurec.config.version import Version
    from nurec.utils.upgrade import upgrade_config

    config = _nrend_artifact_config("3dgrt-optix-nrend")
    target = Version(version_major=26, version_minor=7, version_patch=10)

    # Default load: no automatic migration exists for the ray-traced NRend renderers.
    with pytest.raises(ValueError, match="3dgrt-optix-nrend"):
        upgrade_config(config, target)

    # Both the `--renderer gsplat` CLI flag and the gRPC GSplat backend funnel their choice
    # through `config_overrides`, and it must reach the upgraders for recovery to work.
    upgraded = upgrade_config(
        RenderableModel._with_renderer_override(config, ("model.renderer.name=3dgut-gsplat",)), target
    )
    assert upgraded.model.renderer.name == "3dgut-gsplat"


@pytest.mark.parametrize(
    ("is_compact", "preserved", "camera_features"),
    [(False, frozenset(), True), (True, frozenset(), False), (True, frozenset({"camera_extra_signal"}), True)],
)
def test_camera_outputs_follow_compact_signal_availability(
    monkeypatch: pytest.MonkeyPatch, is_compact: bool, preserved: frozenset[str], camera_features: bool
) -> None:
    outputs = {
        "enable_extended_features": True,
        "enable_sensor_features": True,
        "enable_distance": True,
        "enable_normals": True,
    }
    artifact = cast(
        Any,
        SimpleNamespace(
            parsed_config={
                "model": {
                    "renderer": {
                        "name": "3dgut-gsplat",
                        "outputs": {"camera": outputs, "lidar": outputs.copy()},
                    }
                }
            },
            is_compact=is_compact,
            source="scene.usdz",
        ),
    )

    def load_model(config, loaded_artifact, original_config, device):
        assert loaded_artifact is artifact
        assert device == torch.device("cpu")
        assert config.model.renderer.name == original_config.model.renderer.name == "3dgut-gsplat"
        assert OmegaConf.to_container(config.model.renderer.outputs.camera) == {
            **outputs,
            "enable_sensor_features": camera_features,
        }
        assert OmegaConf.to_container(config.model.renderer.outputs.lidar) == outputs
        assert OmegaConf.to_container(original_config.model.renderer.outputs.camera) == outputs
        raise RuntimeError("Reached model loading")

    monkeypatch.setattr(render_module.compact_usdz, "compact_preserved_extra_signal_tensors", lambda _: preserved)
    monkeypatch.setattr(RenderableModel, "_load_model_and_upgrade", staticmethod(load_model))
    overrides = (
        "model.renderer.name=3dgrt-optix-nrend",
        "model.renderer.name=3dgut-gsplat",
        "model.renderer.outputs.camera.enable_extended_features=True",
        "model.renderer.outputs.camera.enable_sensor_features=True",
        "model.renderer.outputs.camera.enable_distance=True",
        "model.renderer.outputs.camera.enable_normals=True",
        "model.renderer.outputs.lidar.enable_normals=True",
    )
    with pytest.raises(RuntimeError, match="Reached model loading"):
        RenderableModel.load_from_artifact(artifact, config_overrides=overrides, device="cpu")
    assert artifact.parsed_config["model"]["renderer"]["outputs"]["camera"] == outputs


def test_renderer_override_pins_matching_interpolation_before_upgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = cast(
        Any,
        SimpleNamespace(
            parsed_config={
                "chosen_renderer": "3dgut-gsplat",
                "model": {"renderer": {"name": "${chosen_renderer}"}},
            },
            is_compact=False,
        ),
    )

    def upgrade(config):
        config = config.copy()
        config.chosen_renderer = "3dgrt-optix-nrend"
        assert config.model.renderer.name == "3dgut-gsplat"
        return config

    def load_model(config, loaded_artifact, original_config, device):
        assert config.model.renderer.name == original_config.model.renderer.name == "3dgut-gsplat"
        assert original_config.chosen_renderer == "3dgut-gsplat"
        raise RuntimeError("Reached model loading")

    monkeypatch.setattr(render_module, "upgrade_config", upgrade)
    monkeypatch.setattr(RenderableModel, "_load_model_and_upgrade", staticmethod(load_model))
    with pytest.raises(RuntimeError, match="Reached model loading"):
        RenderableModel.load_from_artifact(
            artifact, config_overrides=("model.renderer.name=3dgut-gsplat",), device="cpu"
        )
    assert artifact.parsed_config["model"]["renderer"]["name"] == "${chosen_renderer}"


@pytest.mark.parametrize("with_callback", [False, True])
@pytest.mark.parametrize("override_split_size", [None, 512])
def test_model_load_callback_runs_after_effective_cuda_configuration(
    monkeypatch: pytest.MonkeyPatch, with_callback: bool, override_split_size: int | None
) -> None:
    """The default loader and capture hook share upgraded, caller-overridden CUDA settings."""
    calls: list[str] = []
    artifact = cast(
        Any,
        SimpleNamespace(
            parsed_config={"model": {}, "system": {"max_split_size_mb": 128}},
            is_compact=False,
        ),
    )
    expected_split_size = override_split_size or 256
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    def upgrade(config):
        return OmegaConf.merge(config, {"system": {"max_split_size_mb": 256}})

    def check_environment() -> None:
        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == (
            f"expandable_segments:True,max_split_size_mb:{expected_split_size}"
        )

    def before_model_load() -> None:
        check_environment()
        calls.append("callback")

    def load_model(config, loaded_artifact, original_config, device):
        check_environment()
        assert device == torch.device("cuda")
        assert config.system.max_split_size_mb == expected_split_size
        assert original_config.system.max_split_size_mb == 128
        assert loaded_artifact is artifact
        calls.append("load")
        raise RuntimeError("Reached model loading")

    monkeypatch.setattr(render_module, "upgrade_config", upgrade)
    monkeypatch.setattr(RenderableModel, "_load_model_and_upgrade", staticmethod(load_model))
    options: dict[str, Any] = {}
    if with_callback:
        options["before_model_load"] = before_model_load
    if override_split_size is not None:
        options["config_overrides"] = (f"system.max_split_size_mb={override_split_size}",)

    # Stop at the loader boundary; this regression needs no checkpoint or CUDA device.
    with pytest.raises(RuntimeError, match="Reached model loading"):
        RenderableModel.load_from_artifact(artifact, **options)

    assert calls == (["callback", "load"] if with_callback else ["load"])
    assert artifact.parsed_config["system"]["max_split_size_mb"] == 128


@pytest.mark.parametrize("neighbor_source", ["rangeview", "angle"])
@pytest.mark.parametrize("point_cloud", [False, True])
def test_template_filter_uses_sensor_angles_during_platform_rotation(neighbor_source, point_cloud):
    # The middle ray is a flying return. Platform rotation spreads the world rays
    # beyond the maximum neighbour gap, so using end-pose-compensated angles would
    # incorrectly keep it. Filtering must follow the original firing geometry.
    sensor_angles = torch.tensor([-0.2, 0.0, 0.2])
    firing_yaws = torch.tensor([0.0, 0.5, 1.0])

    def directions(angles):
        return torch.stack((angles.cos(), angles.sin(), torch.zeros_like(angles)), dim=1)

    end_yaw = firing_yaws[-1]
    end_inverse = torch.eye(4)
    end_inverse[:2, :2] = torch.stack(
        (torch.stack((end_yaw.cos(), end_yaw.sin())), torch.stack((-end_yaw.sin(), end_yaw.cos())))
    )
    bundle = SimpleNamespace(
        rendering_data=SimpleNamespace(
            rays=torch.cat((torch.zeros(3, 3), directions(sensor_angles + firing_yaws)), dim=1).reshape(1, 3, 1, 6),
            sensor_model_parameters=[None],
            unstructured_lidar_tiling=SimpleNamespace(directions=directions(sensor_angles)),
        ),
        T_nurec_sensor_end=end_inverse,
        model_elements=torch.zeros(3, 2),
    )
    model = RenderableModel(cast(Any, torch.nn.Linear(1, 1)))
    model._lidar_rendering_settings.enable_lidar_post_filter = True
    predicted = GaussiansCompositeReturn(
        rendered_lidar=GaussiansRenderReturn(distance=torch.tensor([10.0, 1.0, 10.0]), opacity=torch.ones(3))
    )
    method = model.render_lidar_frame_from_ray_bundle if point_cloud else model.render_lidar_ray_frame_from_ray_bundle
    with patch.object(model, "_lidar_rays_inference", return_value=predicted):
        frame = method(
            cast(LidarRayBundle, bundle),
            neighbor_source=neighbor_source,
            distance_filter_threshold=0.5,
            elevation_resolution_deg=1.0,
            azimuth_max_gap_deg=20.0,
        )

    if point_cloud:
        assert frame.point_positions.shape == (2, 3)
    else:
        torch.testing.assert_close(frame.ray_valid, torch.tensor([True, False, True]))
        torch.testing.assert_close(frame.ray_distances, predicted.rendered_lidar.distance)


def test_structured_bundle_keeps_rangeview_filter():
    bundle = SimpleNamespace(
        rendering_data=SimpleNamespace(
            rays=torch.zeros(1, 1, 3, 6),
            sensor_model_parameters=[object()],
            unstructured_lidar_tiling=None,
        ),
        T_nurec_sensor_end=torch.eye(4),
        model_elements=torch.zeros(3, 2),
    )
    model = RenderableModel(cast(Any, torch.nn.Linear(1, 1)))
    predicted = GaussiansCompositeReturn(
        rendered_lidar=GaussiansRenderReturn(distance=torch.ones(3), opacity=torch.ones(3))
    )
    with (
        patch.object(model, "_lidar_rays_inference", return_value=predicted),
        patch("nurec.render.render.angle_gradient_filter") as angle_filter,
        patch(
            "nurec.render.render.distance_based_filter", return_value=torch.tensor([False, True, False])
        ) as grid_filter,
    ):
        frame = model.render_lidar_ray_frame_from_ray_bundle(cast(LidarRayBundle, bundle), enable_distance_filter=True)

    grid_filter.assert_called_once()
    angle_filter.assert_not_called()
    torch.testing.assert_close(frame.ray_valid, torch.tensor([True, False, True]))
