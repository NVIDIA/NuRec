# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch

from omegaconf import OmegaConf

from nurec.render.render import CameraRenderOptions, LidarRayBundle, RenderableModel
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
        ray_bundle = SimpleNamespace(rendering_data=object(), frame_meta=object())

        with patch("nurec.render.render.GaussiansComposite", ObservingGaussian):
            renderable._render_volume_from_ray_bundle(cast(Any, ray_bundle))
            renderable._lidar_rays_inference(cast(Any, ray_bundle))

        self.assertEqual(observed_modes["camera"], observed_modes["lidar"])


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


def test_inference_defaults_stay_overridable_by_the_caller():
    """The stride must not have displaced the pre-existing caller-wins property."""
    key = "model.renderer.outputs.camera.enable_distance"
    ordered = RenderableModel._ordered_config_overrides((), (f"{key}=True",))
    assert OmegaConf.from_dotlist(list(ordered)).model.renderer.outputs.camera.enable_distance is True


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


@pytest.mark.parametrize("with_callback", [False, True])
@pytest.mark.parametrize("override_split_size", [None, 512])
def test_model_load_callback_runs_after_effective_cuda_configuration(
    monkeypatch: pytest.MonkeyPatch, with_callback: bool, override_split_size: int | None
) -> None:
    """The default loader and capture hook share upgraded, caller-overridden CUDA settings."""
    import nurec.render.render as render_module

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

    def load_model(config, loaded_artifact, original_config):
        check_environment()
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
