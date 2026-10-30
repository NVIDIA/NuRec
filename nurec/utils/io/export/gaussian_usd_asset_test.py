# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight tests for USD Gaussian Asset export utilities.
"""

import io
import tempfile
import unittest
import zipfile

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, cast
from unittest.mock import Mock, patch

import imageio.v3 as iio
import numpy as np
import torch

from pxr import Sdf, Usd, UsdGeom

from nurec.config.model import SkyEnvMapBackgroundConfig
from nurec.utils.io.export import gaussian_usd_asset as gaussian_usd_asset_module
from nurec.utils.io.export.gaussian_usd_asset import (
    USD_GAUSSIAN_ROOT_PATH,
    USD_GAUSSIAN_SHADER_PATH,
    CameraRenderProductInfo,
    USDGaussianExportCache,
    apply_gaussian_subsampling,
    compute_runtime_post_processing,
    create_background_domelight,
    create_gaussian_material,
    create_gaussian_model_root,
    resample_timestamps,
    update_animation_settings,
)
from nurec.utils.io.export.ppisp_export.export_integration import (
    PPISP_INTEGRATION_MODE_SH_OPTIMIZED,
    PPISP_INTEGRATION_MODE_SPG_RUNTIME,
)
from nurec.utils.io.export.ppisp_export.usd_writer import (
    LDR_COLOR_RENDER_VAR,
    PPISP_INPUT_RENDER_VAR,
)
from nurec.utils.io.usd_nrend import _disable_runtime_post_processing_config, _normalize_nrend_renderer_config
from nurec.utils.types import NamedSerialized, NamedUSDStage


class TestUtilityFunctions(unittest.TestCase):
    def test_normalize_nrend_renderer_config(self) -> None:
        model_json_dict = {"nre_data": {"config": {"renderer": {"name": "3dgut-gsplat"}}}}

        _normalize_nrend_renderer_config(model_json_dict)

        self.assertNotIn("nre_data", model_json_dict)
        self.assertEqual(model_json_dict["nurec_data"]["config"]["renderer"]["name"], "3dgut-nrend")

    def test_disable_runtime_post_processing_config(self) -> None:
        model_json_dict = {
            "nre_data": {
                "config": {
                    "saturate_radiance": True,
                    "background": {"saturate_radiance": True},
                    "post_processing": {"ppisp": {"name": "ppisp-post-processing"}},
                    "nested": [{"saturate_radiance": True}],
                }
            }
        }

        _disable_runtime_post_processing_config(model_json_dict)

        self.assertNotIn("nre_data", model_json_dict)
        config = cast(Dict[str, Any], model_json_dict["nurec_data"]["config"])
        self.assertEqual(config["post_processing"], {})
        self.assertFalse(config["saturate_radiance"])
        self.assertFalse(config["background"]["saturate_radiance"])
        self.assertFalse(config["nested"][0]["saturate_radiance"])

    def test_resample_timestamps_basic(self):
        timestamps_us = [100, 200, 300, 400, 500]
        reference_timestamps_us = [0, 50, 150, 250, 350, 450, 550, 600]
        resampled = resample_timestamps(timestamps_us, reference_timestamps_us)
        self.assertGreaterEqual(resampled[0], 0)
        self.assertLessEqual(resampled[0], timestamps_us[0])
        self.assertGreaterEqual(resampled[-1], timestamps_us[-1])
        self.assertLessEqual(resampled[-1], reference_timestamps_us[-1])

    def test_resample_timestamps_empty(self):
        self.assertEqual(resample_timestamps([], [1, 2, 3]), [])
        self.assertEqual(resample_timestamps([1, 2, 3], []), [1, 2, 3])

    def test_apply_gaussian_subsampling_full(self):
        indices = torch.arange(1000)
        result = apply_gaussian_subsampling(indices, 100.0)
        self.assertEqual(len(result), 1000)
        torch.testing.assert_close(result, indices)


class TestUSDStageUtilities(unittest.TestCase):
    def test_create_cubemap_domelight_uses_shared_sampler(self):
        background = Mock(spec=gaussian_usd_asset_module.SkyEnvMapBackground)
        background.envmap_type = gaussian_usd_asset_module.EnvMapType.CUBEMAP
        background.textures = torch.zeros((1, 6, 4, 4, 3))
        background.flattened_textures.return_value = background.textures
        background.maybe_inpaint.return_value = None
        background.config = SimpleNamespace(saturate_radiance=False)
        sampled = torch.linspace(0.0, 1.0, 8 * 12 * 3).reshape(8 * 12, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(gaussian_usd_asset_module, "sample_cubemap_seamless", return_value=sampled) as sampler:
                result = create_background_domelight(background, Path(tmpdir))

        self.assertIsNotNone(result)
        assert result is not None
        named_stage, named_hdr = result
        self.assertTrue(named_stage.stage.GetPrimAtPath(gaussian_usd_asset_module.SKY_DOME_LIGHT_PATH).IsValid())
        self.assertEqual(named_hdr.filename, "sky_envmap.hdr")
        self.assertGreater(len(named_hdr.serialized), 0)
        sampler.assert_called_once()
        self.assertEqual(tuple(sampler.call_args.args[0].shape), (6, 4, 4, 3))
        self.assertEqual(tuple(sampler.call_args.args[1].shape), (8 * 12, 3))

    @unittest.skipUnless(torch.cuda.is_available(), "cubemap sampling requires CUDA")
    def test_create_cubemap_domelight_samples_constant_cubemap_on_cuda(self):
        device = torch.device("cuda", torch.cuda.current_device())
        cubemap_size = 4
        expected_color = torch.tensor([0.125, 0.375, 0.625], device=device)
        config = SkyEnvMapBackgroundConfig(
            name="sky-env-map",
            width=cubemap_size,
            height=cubemap_size,
            envmap_type="cubemap",
            should_inpaint=False,
            inpaint_threshold=0.0,
            inpaint_kernel_size=1,
            min_grad_updates=0,
            saturate_radiance=False,
        )
        trainer_config = cast(Any, SimpleNamespace(relative_schedule=False))
        background = gaussian_usd_asset_module.SkyEnvMapBackground(config, trainer_config).to(device)
        with torch.no_grad():
            background.textures.copy_(expected_color.expand_as(background.textures))

        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_background_domelight(background, Path(tmpdir))

        self.assertIsNotNone(result)
        assert result is not None
        _, named_hdr = result
        decoded_hdr = iio.imread(io.BytesIO(named_hdr.serialized), extension=".hdr")
        expected_hdr = np.broadcast_to(expected_color.cpu().numpy(), decoded_hdr.shape)
        self.assertEqual(decoded_hdr.shape, (cubemap_size * 2, cubemap_size * 3, 3))
        np.testing.assert_allclose(decoded_hdr.astype(np.float32) / 255.0, expected_hdr, atol=1.0 / 255.0, rtol=0)

    def test_create_gaussian_model_root_basic(self):
        stage = Usd.Stage.CreateInMemory()
        root_path = create_gaussian_model_root(
            stage, flip_x_axis=False, flip_y_axis=False, flip_z_axis=False, dataset_offset=None
        )
        self.assertEqual(root_path, USD_GAUSSIAN_ROOT_PATH)
        xform_prim = stage.GetPrimAtPath(root_path)
        self.assertTrue(xform_prim.IsValid())
        self.assertTrue(xform_prim.IsA(UsdGeom.Xform))

    def test_create_gaussian_material(self):
        stage = Usd.Stage.CreateInMemory()
        material_prim = create_gaussian_material(stage)
        self.assertTrue(material_prim.IsValid())
        self.assertTrue(material_prim.IsA(Usd.Typed))

    def test_update_animation_settings(self):
        root_stage = Usd.Stage.CreateInMemory()
        ref_stage = Usd.Stage.CreateInMemory()
        ref_layer = ref_stage.GetRootLayer()
        ref_layer.startTimeCode = 10.0
        ref_layer.endTimeCode = 100.0
        ref_layer.timeCodesPerSecond = 30.0
        ref_layer.customLayerData = {"absoluteTimeOffsetMicroSec": 500000}
        update_animation_settings(root_stage, ref_layer)
        self.assertEqual(root_stage.GetStartTimeCode(), 10.0)
        self.assertEqual(root_stage.GetEndTimeCode(), 100.0)
        self.assertEqual(root_stage.GetTimeCodesPerSecond(), 30.0)
        self.assertEqual(root_stage.GetMetadataByDictKey("customLayerData", "absoluteTimeOffsetMicroSec"), 500000)


class TestUSDZPackaging(unittest.TestCase):
    def test_usdz_cache_initialization(self):
        cache = USDGaussianExportCache()
        self.assertEqual(len(cache.usd_stages), 0)
        self.assertEqual(len(cache.hdr_files), 0)
        self.assertEqual(len(cache.serialized_files), 0)

    def test_usdz_cache_add_stage(self):
        cache = USDGaussianExportCache()
        stage = Usd.Stage.CreateInMemory()
        named_stage = NamedUSDStage(filename="test.usda", stage=stage)
        cache.add_usd_stage(named_stage)
        self.assertEqual(len(cache.usd_stages), 1)
        self.assertEqual(cache.usd_stages[0], named_stage)

    def test_usdz_cache_write_to_usdz(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = USDGaussianExportCache()
            stage = Usd.Stage.CreateInMemory()
            named_stage = NamedUSDStage(filename="test.usda", stage=stage)
            cache.add_usd_stage(named_stage)
            output_path = Path(tmpdir) / "test.usdz"
            cache.write_to_usdz(output_path)
            self.assertTrue(output_path.exists())
            self.assertTrue(zipfile.is_zipfile(output_path))
            with zipfile.ZipFile(output_path, "r") as zf:
                names = zf.namelist()
                self.assertIn("default.usda", names)
                self.assertIn("test.usda", names)

    def test_usdz_cache_packages_serialized_sidecars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = USDGaussianExportCache()
            cache.add_usd_stage(NamedUSDStage(filename="test.usda", stage=Usd.Stage.CreateInMemory()))
            cache.add_serialized_files([NamedSerialized(filename="volume.nurec", serialized=b"nrend")])

            output_path = Path(tmpdir) / "test.usdz"
            cache.write_to_usdz(output_path)

            with zipfile.ZipFile(output_path, "r") as zf:
                self.assertEqual(zf.read("volume.nurec"), b"nrend")

    def test_default_stage_disables_registered_compositing_inversion_for_nrend_ppisp(self):
        cache = USDGaussianExportCache()
        cache.add_usd_stage(NamedUSDStage(filename="volume.usda", stage=Usd.Stage.CreateInMemory()))
        cache.add_serialized_files([NamedSerialized(filename="volume.nurec", serialized=b"nrend")])
        cache.set_camera_render_products([CameraRenderProductInfo("cam0", "rig0", 32, 24)])
        cache.set_ppisp(
            ppisp_module=object(),
            camera_name_to_index={"cam0": 0},
            camera_frame_mappings={"cam0": [0]},
        )

        with patch("nurec.utils.io.export.gaussian_usd_asset.add_ppisp_to_all_render_products"):
            stage = cache.compose_default_usd_stage()

        render_settings = stage.GetMetadataByDictKey("customLayerData", "renderSettings")
        self.assertFalse(render_settings["rtx:rtpt:gaussian:skipTonemapping:enabled"])
        self.assertFalse(render_settings["rtx:post:registeredCompositing:invertToneMap"])
        self.assertFalse(render_settings["rtx:post:registeredCompositing:invertColorCorrection"])


class TestRenderProductAndSpgExport(unittest.TestCase):
    """Material/post-processing tests for the USD export."""

    def test_material_post_processing_disables_mdl_color_correction(self):
        """Verify MDL shader params are set correctly based on post-processing flag."""
        # Without post-processing: no MDL param overrides
        stage_no_pp = Usd.Stage.CreateInMemory()
        create_gaussian_material(stage_no_pp, has_post_processing=False)
        shader_no_pp = stage_no_pp.GetPrimAtPath(USD_GAUSSIAN_SHADER_PATH)
        self.assertFalse(shader_no_pp.GetAttribute("inputs:apply_srgb_linear").IsValid())
        self.assertFalse(shader_no_pp.GetAttribute("inputs:apply_inverse_tonemap").IsValid())

        # With post-processing: MDL params disabled
        stage_pp = Usd.Stage.CreateInMemory()
        create_gaussian_material(stage_pp, has_post_processing=True)
        shader_pp = stage_pp.GetPrimAtPath(USD_GAUSSIAN_SHADER_PATH)
        self.assertEqual(shader_pp.GetAttribute("inputs:apply_srgb_linear").Get(), False)
        self.assertEqual(shader_pp.GetAttribute("inputs:apply_inverse_tonemap").Get(), False)


class TestExportCachePpispFlags(unittest.TestCase):
    """``USDGaussianExportCache.set_ppisp`` stores the new flags.

    The forwarding into ``add_ppisp_to_all_render_products`` is
    verified separately by patching the writer in
    ``TestExportCachePpispDispatch`` so the test stays hermetic and
    does not depend on a real PPISP module.
    """

    def test_defaults_when_set_ppisp_not_called(self) -> None:
        cache = USDGaussianExportCache()
        self.assertEqual(cache.ppisp_responsivity, 1.0)
        self.assertIsNone(cache.ppisp_reference_camera_id)
        self.assertIsNone(cache.ppisp_reference_frame_id)
        self.assertTrue(cache.ppisp_disable_controller)

    def test_set_ppisp_stores_new_flags(self) -> None:
        cache = USDGaussianExportCache()
        cache.set_ppisp(
            ppisp_module=Mock(),
            camera_name_to_index={"cam0": 0},
            camera_frame_mappings={"cam0": [0, 1]},
            responsivity=2.5,
            ppisp_reference_camera_id=3,
            ppisp_reference_frame_id=7,
            ppisp_disable_controller=False,
        )
        self.assertEqual(cache.ppisp_responsivity, 2.5)
        self.assertEqual(cache.ppisp_reference_camera_id, 3)
        self.assertEqual(cache.ppisp_reference_frame_id, 7)
        self.assertFalse(cache.ppisp_disable_controller)


class TestExportCachePpispDispatch(unittest.TestCase):
    """``compose_default_usd_stage`` forwards the new flags verbatim.

    Patches ``add_ppisp_to_all_render_products`` so we can inspect the
    kwargs without exercising the real PPISP USD writer (which needs
    a fully-formed PPISP module).
    """

    def test_compose_forwards_flags_to_writer(self) -> None:
        cache = USDGaussianExportCache()
        cache.set_camera_render_products([CameraRenderProductInfo("cam0", "rig", 32, 24)])
        ppisp_sentinel = Mock()
        cache.set_ppisp(
            ppisp_module=ppisp_sentinel,
            camera_name_to_index={"cam0": 0},
            camera_frame_mappings={"cam0": [0]},
            responsivity=1.75,
            ppisp_reference_camera_id=2,
            ppisp_reference_frame_id=4,
            ppisp_disable_controller=False,
        )
        with patch("nurec.utils.io.export.gaussian_usd_asset.add_ppisp_to_all_render_products") as mock_writer:
            cache.compose_default_usd_stage()

        mock_writer.assert_called_once()
        kwargs = mock_writer.call_args.kwargs
        self.assertIs(kwargs["ppisp"], ppisp_sentinel)
        self.assertEqual(kwargs["responsivity"], 1.75)
        self.assertEqual(kwargs["ppisp_reference_camera_id"], 2)
        self.assertEqual(kwargs["ppisp_reference_frame_id"], 4)
        self.assertFalse(kwargs["ppisp_disable_controller"])

    def test_compose_requests_ldr_color_when_no_runtime_ppisp(self) -> None:
        cache = USDGaussianExportCache()
        cache.set_camera_render_products([CameraRenderProductInfo("cam0", "rig", 32, 24)])
        stage = cache.compose_default_usd_stage()
        render_product = stage.GetPrimAtPath("/Render/cam0")

        self.assertEqual(
            render_product.GetRelationship("orderedVars").GetTargets(),
            [Sdf.Path(f"/Render/cam0/{LDR_COLOR_RENDER_VAR}")],
        )
        self.assertTrue(stage.GetPrimAtPath(f"/Render/cam0/{LDR_COLOR_RENDER_VAR}").IsValid())
        self.assertFalse(stage.GetPrimAtPath(f"/Render/cam0/{PPISP_INPUT_RENDER_VAR}").IsValid())
        render_settings = stage.GetRootLayer().customLayerData["renderSettings"]
        self.assertEqual(render_settings, {"rtx:post:tonemap:op": 2})
        self.assertNotIn("rtx:rtpt:gaussian:skipTonemapping:enabled", render_settings)

    def test_compose_requests_hdr_color_and_sets_skip_tonemapping_for_runtime_ppisp(self) -> None:
        cache = USDGaussianExportCache()
        cache.set_camera_render_products([CameraRenderProductInfo("cam0", "rig", 32, 24)])
        cache.set_ppisp(
            ppisp_module=Mock(),
            camera_name_to_index={"cam0": 0},
            camera_frame_mappings={"cam0": [0]},
        )

        with patch("nurec.utils.io.export.gaussian_usd_asset.add_ppisp_to_all_render_products"):
            stage = cache.compose_default_usd_stage()

        render_product = stage.GetPrimAtPath("/Render/cam0")
        self.assertEqual(
            render_product.GetRelationship("orderedVars").GetTargets(),
            [Sdf.Path(f"/Render/cam0/{PPISP_INPUT_RENDER_VAR}")],
        )
        self.assertTrue(stage.GetPrimAtPath(f"/Render/cam0/{PPISP_INPUT_RENDER_VAR}").IsValid())
        self.assertFalse(stage.GetPrimAtPath(f"/Render/cam0/{LDR_COLOR_RENDER_VAR}").IsValid())
        render_settings = stage.GetRootLayer().customLayerData["renderSettings"]
        self.assertEqual(render_settings["rtx:post:tonemap:op"], 2)
        self.assertFalse(render_settings["rtx:rtpt:gaussian:skipTonemapping:enabled"])

    def test_compose_skips_writer_when_no_ppisp_module(self) -> None:
        cache = USDGaussianExportCache()
        cache.set_camera_render_products([CameraRenderProductInfo("cam0", "rig", 32, 24)])
        with patch("nurec.utils.io.export.gaussian_usd_asset.add_ppisp_to_all_render_products") as mock_writer:
            cache.compose_default_usd_stage()
        mock_writer.assert_not_called()


class ComputeRuntimePostProcessingTest(unittest.TestCase):
    """Pin the orchestrator-level mapping from model state + integration
    mode to the ``runtime_post_processing`` flag that drives MDL color
    correction and SH packing color space.

    The flag is ``True`` iff a PPISP SPG shader will be authored on
    each RenderProduct, i.e. ``spg-runtime`` mode AND the model
    exposes a PPISP module. The ``sh-optimized`` mode folds PPISP
    into the SH coefficients during export and therefore has no
    runtime PPISP stage downstream of MDL, regardless of the source
    model carrying a PPISP collection.
    """

    def test_no_model_postprocessing_returns_false_in_spg_runtime(self) -> None:
        self.assertFalse(
            compute_runtime_post_processing(
                model_has_postprocessing=False,
                ppisp_module=None,
                ppisp_integration_mode=PPISP_INTEGRATION_MODE_SPG_RUNTIME,
            )
        )

    def test_no_model_postprocessing_returns_false_in_sh_optimized(self) -> None:
        self.assertFalse(
            compute_runtime_post_processing(
                model_has_postprocessing=False,
                ppisp_module=None,
                ppisp_integration_mode=PPISP_INTEGRATION_MODE_SH_OPTIMIZED,
            )
        )

    def test_sh_optimized_returns_false_even_with_postprocessing(self) -> None:
        # Regression: sh-optimized bakes PPISP into the SH, so the
        # runtime pipeline is SH -> MDL with no downstream PPISP stage.
        # If this returns True, MDL would skip its color correction and
        # the asset would render incorrectly (corrupted sRGB / tonemap).
        self.assertFalse(
            compute_runtime_post_processing(
                model_has_postprocessing=True,
                ppisp_module=Mock(),
                ppisp_integration_mode=PPISP_INTEGRATION_MODE_SH_OPTIMIZED,
            )
        )

    def test_spg_runtime_with_postprocessing_returns_true(self) -> None:
        self.assertTrue(
            compute_runtime_post_processing(
                model_has_postprocessing=True,
                ppisp_module=Mock(),
                ppisp_integration_mode=PPISP_INTEGRATION_MODE_SPG_RUNTIME,
            )
        )

    def test_spg_runtime_without_ppisp_module_returns_false(self) -> None:
        # Defense-in-depth: an orchestrator that decouples the
        # ``model_has_postprocessing`` flag from the ``ppisp_module``
        # acquisition (e.g. a future refactor) must not produce
        # ``runtime_post_processing=True`` without a real PPISP module
        # actually present to drive the runtime stage.
        self.assertFalse(
            compute_runtime_post_processing(
                model_has_postprocessing=True,
                ppisp_module=None,
                ppisp_integration_mode=PPISP_INTEGRATION_MODE_SPG_RUNTIME,
            )
        )


class _FakePPISPModule:
    def __init__(self) -> None:
        self.num_cameras = 3
        self.exposure_params = torch.zeros(5)


class _FakePPISPModuleWithController(_FakePPISPModule):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_controller=True)
        self.controllers = [object() for _ in range(self.num_cameras)]


class _FakeModel:
    def __init__(self) -> None:
        self.post_processings = [object()]
        self.background = None
        self.gaussian_process_group = None
        self.config = SimpleNamespace(renderer=SimpleNamespace(global_z_order=False))


class _FakeGaussiansSystem:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self.resume = None
        self.loaded_state_dict: object = None

    def load_state_dict(self, state_dict: object, *, strict: bool, assign: bool) -> None:
        del strict, assign
        self.loaded_state_dict = state_dict

    def cuda(self) -> None:
        return None


class _FakeDataset:
    def __init__(self, datasource: object) -> None:
        self._datasource = datasource

    def get_datasource(self) -> object:
        return self._datasource


class _FakeDatasource:
    aabb: object = object()

    def get_offset(self) -> None:
        return None


class _FakeNCOREDataSource(_FakeDatasource):
    def get_rig_trajectories(self) -> object:
        return object()

    def get_global_to_training_camera_frame_mapping(self) -> np.ndarray:
        return np.array([0, 1, 2], dtype=np.int64)


_DEFAULT_PPISP_MODULE = object()


class _RecordingExportCache:
    instances: List["_RecordingExportCache"] = []

    def __init__(self) -> None:
        self.usd_stages: List[NamedUSDStage] = []
        self.hdr_files: List[NamedSerialized] = []
        self.serialized_files: List[NamedSerialized] = []
        self.spg_files: List[NamedSerialized] = []
        self.camera_render_products: List[CameraRenderProductInfo] = []
        self.set_ppisp_kwargs: Dict[str, object] | None = None
        self.write_to_usdz_path: Path | None = None
        _RecordingExportCache.instances.append(self)

    def add_usd_stage(self, stage: NamedUSDStage) -> None:
        self.usd_stages.append(stage)

    def add_spg_files(self, spg_files: List[NamedSerialized]) -> None:
        self.spg_files.extend(spg_files)

    def add_serialized_files(self, serialized_files: List[NamedSerialized]) -> None:
        self.serialized_files.extend(serialized_files)

    def set_camera_render_products(self, camera_render_products: List[CameraRenderProductInfo]) -> None:
        self.camera_render_products = camera_render_products

    def set_ppisp(self, **kwargs: object) -> None:
        self.set_ppisp_kwargs = kwargs

    def write_to_usdz(self, output_path: Path) -> None:
        self.write_to_usdz_path = output_path


class ExportGaussianUsdAssetPPISPOrchestrationTest(unittest.TestCase):
    """Mocked orchestration coverage for PPISP export mode decisions."""

    def _config(self, tmpdir: str) -> SimpleNamespace:
        ckpt_dir = Path(tmpdir) / "ckpt"
        ckpt_dir.mkdir()
        (ckpt_dir / "last.ckpt").write_bytes(b"mock checkpoint")
        return SimpleNamespace(
            ckpt_dir=str(ckpt_dir),
            resume=None,
            system=SimpleNamespace(name="fake_system"),
            dataset=SimpleNamespace(name="fake_dataset", path=str(Path(tmpdir) / "dataset.ncore")),
        )

    def _invoke_export(
        self,
        *,
        tmpdir: str,
        datasource: object,
        ppisp_module: object | None = _DEFAULT_PPISP_MODULE,
        **overrides: object,
    ) -> Mock:
        config = self._config(tmpdir)
        callback = gaussian_usd_asset_module.export_gaussian_usd_asset.callback
        assert callback is not None
        if ppisp_module is _DEFAULT_PPISP_MODULE:
            ppisp_module = _FakePPISPModule()

        kwargs: Dict[str, object] = dict(
            config_name="fake_config",
            output_dir=str(Path(tmpdir) / "usd_out"),
            percentage_gaussians=100.0,
            usd_format="usdz",
            usd_schema="geompoints",
            force_sh_0=False,
            apply_activation=True,
            export_parsed_extra_signals=False,
            export_rig_trajectories=False,
            flip_axis="",
            do_not_cast_shadows=False,
            skip_gaussian_deformation=False,
            resample_animation=False,
            half_precision=False,
            export_summary=False,
            log_level="info",
            ppisp_integration_mode=PPISP_INTEGRATION_MODE_SH_OPTIMIZED,
            scene_radiance_scale=2.0,
            ppisp_responsivity=1.0,
            ppisp_reference_camera_id=1,
            ppisp_reference_frame_id=2,
            enable_ppisp_controller_export=None,
            sh_optimization_num_iterations=None,
            hydra_args=[],
        )
        kwargs.update(overrides)

        with (
            patch.object(gaussian_usd_asset_module, "parse_typed_config", return_value=config) as parse_config,
            patch.object(gaussian_usd_asset_module, "make_system", return_value=_FakeGaussiansSystem()),
            patch.object(gaussian_usd_asset_module.torch, "load", return_value={"state_dict": {}}),
            patch.object(gaussian_usd_asset_module, "GaussiansSystem", _FakeGaussiansSystem),
            patch.object(gaussian_usd_asset_module, "make_dataset", return_value=_FakeDataset(datasource)),
            patch.object(gaussian_usd_asset_module, "get_ppisp_from_model", return_value=ppisp_module),
            patch.object(gaussian_usd_asset_module, "USDGaussianExportCache", _RecordingExportCache),
        ):
            callback(**kwargs)
        return parse_config

    def test_forwards_config_parser_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(gaussian_usd_asset_module, "run_sh_match_optimization"),
                patch.object(gaussian_usd_asset_module, "iter_gaussian_nodes", return_value=[]),
            ):
                parse_config = self._invoke_export(
                    tmpdir=tmpdir,
                    datasource=_FakeDatasource(),
                    config_name="forwarded_config",
                    hydra_args=["dataset.path=/tmp/dataset.ncore"],
                    scene_radiance_scale=1.0,
                )

        parse_config.assert_called_once_with(
            config_name="forwarded_config",
            hydra_args=["dataset.path=/tmp/dataset.ncore", "mode=val"],
        )

    def test_sh_optimized_runs_sh_match_before_radiance_scale_and_skips_spg(self) -> None:
        call_order: List[str] = []
        _RecordingExportCache.instances = []

        def record_sh_match(**_: object) -> None:
            call_order.append("sh_match")

        def record_radiance_scale(*_: object) -> None:
            call_order.append("radiance_scale")

        def record_export_gaussians(**_: object) -> List[Any]:
            call_order.append("export_gaussians")
            return []

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(
                    gaussian_usd_asset_module,
                    "run_sh_match_optimization",
                    side_effect=record_sh_match,
                ) as sh_match,
                patch.object(
                    gaussian_usd_asset_module,
                    "apply_radiance_scale_to_sh_nodes",
                    side_effect=record_radiance_scale,
                ) as radiance_scale,
                patch.object(gaussian_usd_asset_module, "iter_gaussian_nodes", return_value=[("gaussians", object())]),
                patch.object(gaussian_usd_asset_module, "GaussianExportAccessor") as accessor_cls,
                patch.object(
                    gaussian_usd_asset_module,
                    "export_gaussians_as_usd_asset",
                    side_effect=record_export_gaussians,
                ),
                patch.object(gaussian_usd_asset_module, "select_spg_files_for_export") as select_spg,
            ):
                accessor_cls.return_value.get_capabilities.return_value = Mock()
                self._invoke_export(tmpdir=tmpdir, datasource=_FakeDatasource())

        self.assertEqual(call_order, ["sh_match", "radiance_scale", "export_gaussians"])
        sh_match.assert_called_once()
        radiance_scale.assert_called_once()
        select_spg.assert_not_called()
        self.assertEqual(_RecordingExportCache.instances[-1].spg_files, [])
        self.assertIsNone(_RecordingExportCache.instances[-1].set_ppisp_kwargs)

    def test_spg_runtime_forwards_resolved_ids_and_sidecars(self) -> None:
        _RecordingExportCache.instances = []
        sidecars = [NamedSerialized(filename="ppisp_usd_spg.slang", serialized=b"spg")]

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(gaussian_usd_asset_module, "RigTrajectoriesProvider", _FakeNCOREDataSource),
                patch.object(gaussian_usd_asset_module, "NCOREDataSource", _FakeNCOREDataSource),
                patch.object(
                    gaussian_usd_asset_module,
                    "create_rig_trajectories_stage",
                    return_value=(
                        NamedUSDStage(filename="rig.usda", stage=Usd.Stage.CreateInMemory()),
                        0,
                        None,
                        [],
                    ),
                ),
                patch.object(gaussian_usd_asset_module, "build_camera_name_to_index_mapping", return_value={"cam0": 0}),
                patch.object(
                    gaussian_usd_asset_module,
                    "extract_camera_render_products",
                    return_value=[CameraRenderProductInfo("cam0", "rig0", 32, 24)],
                ),
                patch.object(gaussian_usd_asset_module, "build_camera_frame_mappings", return_value={"cam0": [0, 2]}),
                patch.object(gaussian_usd_asset_module, "run_sh_match_optimization") as sh_match,
                patch.object(gaussian_usd_asset_module, "apply_radiance_scale_to_sh_nodes") as radiance_scale,
                patch.object(gaussian_usd_asset_module, "select_spg_files_for_export", return_value=sidecars),
                patch.object(gaussian_usd_asset_module, "iter_gaussian_nodes", return_value=[("gaussians", object())]),
                patch.object(gaussian_usd_asset_module, "GaussianExportAccessor") as accessor_cls,
                patch.object(gaussian_usd_asset_module, "export_gaussians_as_usd_asset", return_value=[]),
            ):
                accessor_cls.return_value.get_capabilities.return_value = Mock()
                self._invoke_export(
                    tmpdir=tmpdir,
                    datasource=_FakeNCOREDataSource(),
                    export_rig_trajectories=True,
                    ppisp_integration_mode=PPISP_INTEGRATION_MODE_SPG_RUNTIME,
                    scene_radiance_scale=1.0,
                    ppisp_responsivity=1.75,
                    ppisp_reference_camera_id=1,
                    ppisp_reference_frame_id=2,
                    enable_ppisp_controller_export=False,
                )

        cache = _RecordingExportCache.instances[-1]
        self.assertEqual(cache.spg_files, sidecars)
        assert cache.set_ppisp_kwargs is not None
        self.assertEqual(cache.set_ppisp_kwargs["ppisp_reference_camera_id"], 1)
        self.assertEqual(cache.set_ppisp_kwargs["ppisp_reference_frame_id"], 2)
        self.assertEqual(cache.set_ppisp_kwargs["responsivity"], 1.75)
        self.assertTrue(cache.set_ppisp_kwargs["ppisp_disable_controller"])
        sh_match.assert_not_called()
        radiance_scale.assert_not_called()

    def test_spg_runtime_defaults_to_controller_when_present(self) -> None:
        _RecordingExportCache.instances = []
        ppisp_module = _FakePPISPModuleWithController()
        sidecars = [NamedSerialized(filename="ppisp_controller_0.cu", serialized=b"cu")]

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(gaussian_usd_asset_module, "RigTrajectoriesProvider", _FakeNCOREDataSource),
                patch.object(gaussian_usd_asset_module, "NCOREDataSource", _FakeNCOREDataSource),
                patch.object(
                    gaussian_usd_asset_module,
                    "create_rig_trajectories_stage",
                    return_value=(
                        NamedUSDStage(filename="rig.usda", stage=Usd.Stage.CreateInMemory()),
                        0,
                        None,
                        [],
                    ),
                ),
                patch.object(gaussian_usd_asset_module, "build_camera_name_to_index_mapping", return_value={"cam0": 0}),
                patch.object(
                    gaussian_usd_asset_module,
                    "extract_camera_render_products",
                    return_value=[CameraRenderProductInfo("cam0", "rig0", 32, 24)],
                ),
                patch.object(gaussian_usd_asset_module, "build_camera_frame_mappings", return_value={"cam0": [0, 2]}),
                patch.object(gaussian_usd_asset_module, "run_sh_match_optimization"),
                patch.object(gaussian_usd_asset_module, "apply_radiance_scale_to_sh_nodes"),
                patch.object(
                    gaussian_usd_asset_module, "select_spg_files_for_export", return_value=sidecars
                ) as select_spg,
                patch.object(gaussian_usd_asset_module, "iter_gaussian_nodes", return_value=[("gaussians", object())]),
                patch.object(gaussian_usd_asset_module, "GaussianExportAccessor") as accessor_cls,
                patch.object(gaussian_usd_asset_module, "export_gaussians_as_usd_asset", return_value=[]),
            ):
                accessor_cls.return_value.get_capabilities.return_value = Mock()
                self._invoke_export(
                    tmpdir=tmpdir,
                    datasource=_FakeNCOREDataSource(),
                    ppisp_module=ppisp_module,
                    export_rig_trajectories=True,
                    ppisp_integration_mode=PPISP_INTEGRATION_MODE_SPG_RUNTIME,
                    scene_radiance_scale=1.0,
                )

        cache = _RecordingExportCache.instances[-1]
        self.assertEqual(cache.spg_files, sidecars)
        assert cache.set_ppisp_kwargs is not None
        self.assertFalse(cache.set_ppisp_kwargs["ppisp_disable_controller"])
        select_spg.assert_called_once_with(
            enable_ppisp_controller_export=True,
            ppisp_module=ppisp_module,
            camera_indices=[0],
        )

    def test_nrend_volume_schema_serializes_volume_and_skips_gaussian_stages(self) -> None:
        _RecordingExportCache.instances = []
        volume_stage = NamedUSDStage(filename="volume.usda", stage=Usd.Stage.CreateInMemory())
        volume_blob = NamedSerialized(filename="volume.nurec", serialized=b"nrend")

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(
                    gaussian_usd_asset_module,
                    "serialize_nrend_usd",
                    return_value=[volume_blob, volume_stage],
                ) as serialize_volume,
                patch.object(gaussian_usd_asset_module, "iter_gaussian_nodes") as iter_nodes,
                patch.object(gaussian_usd_asset_module, "export_gaussians_as_usd_asset") as export_gaussians,
            ):
                self._invoke_export(
                    tmpdir=tmpdir,
                    datasource=_FakeDatasource(),
                    ppisp_module=None,
                    usd_schema="nrendvol",
                    ppisp_integration_mode=PPISP_INTEGRATION_MODE_SPG_RUNTIME,
                    scene_radiance_scale=1.0,
                    ppisp_reference_camera_id=None,
                    ppisp_reference_frame_id=None,
                )

        cache = _RecordingExportCache.instances[-1]
        self.assertIn(volume_stage, cache.usd_stages)
        self.assertEqual(cache.serialized_files, [volume_blob])
        serialize_volume.assert_called_once()
        self.assertTrue(serialize_volume.call_args.kwargs["invert_registered_compositing"])
        self.assertFalse(serialize_volume.call_args.kwargs["accept_unsupported_ppisp"])
        self.assertFalse(serialize_volume.call_args.kwargs["disable_runtime_post_processing"])
        iter_nodes.assert_not_called()
        export_gaussians.assert_not_called()

    def test_nrend_volume_radiance_scale_applies_to_gaussian_nodes_before_serialization(self) -> None:
        _RecordingExportCache.instances = []
        volume_stage = NamedUSDStage(filename="volume.usda", stage=Usd.Stage.CreateInMemory())
        volume_blob = NamedSerialized(filename="volume.nurec", serialized=b"nrend")
        node_a = object()
        node_b = object()

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(
                    gaussian_usd_asset_module,
                    "serialize_nrend_usd",
                    return_value=[volume_blob, volume_stage],
                ) as serialize_volume,
                patch.object(
                    gaussian_usd_asset_module,
                    "iter_gaussian_nodes",
                    return_value=[("static", node_a), ("dynamic", node_b)],
                ) as iter_nodes,
                patch.object(gaussian_usd_asset_module, "apply_radiance_scale_to_sh_nodes") as radiance_scale,
                patch.object(gaussian_usd_asset_module, "export_gaussians_as_usd_asset") as export_gaussians,
            ):
                self._invoke_export(
                    tmpdir=tmpdir,
                    datasource=_FakeDatasource(),
                    ppisp_module=None,
                    usd_schema="nrendvol",
                    ppisp_integration_mode=PPISP_INTEGRATION_MODE_SPG_RUNTIME,
                    scene_radiance_scale=2.5,
                    ppisp_reference_camera_id=None,
                    ppisp_reference_frame_id=None,
                )

        iter_nodes.assert_called_once()
        radiance_scale.assert_called_once()
        scaled_nodes, scaled_value = radiance_scale.call_args.args
        self.assertEqual(scaled_nodes, [("static", node_a), ("dynamic", node_b)])
        self.assertEqual(scaled_value, 2.5)
        serialize_volume.assert_called_once()
        export_gaussians.assert_not_called()

    def test_nrend_volume_with_runtime_ppisp_disables_registered_compositing_inversion(self) -> None:
        _RecordingExportCache.instances = []
        volume_stage = NamedUSDStage(filename="volume.usda", stage=Usd.Stage.CreateInMemory())
        volume_blob = NamedSerialized(filename="volume.nurec", serialized=b"nrend")

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(gaussian_usd_asset_module, "RigTrajectoriesProvider", _FakeNCOREDataSource),
                patch.object(gaussian_usd_asset_module, "NCOREDataSource", _FakeNCOREDataSource),
                patch.object(
                    gaussian_usd_asset_module,
                    "create_rig_trajectories_stage",
                    return_value=(
                        NamedUSDStage(filename="rig.usda", stage=Usd.Stage.CreateInMemory()),
                        0,
                        None,
                        [],
                    ),
                ),
                patch.object(gaussian_usd_asset_module, "build_camera_name_to_index_mapping", return_value={"cam0": 0}),
                patch.object(
                    gaussian_usd_asset_module,
                    "extract_camera_render_products",
                    return_value=[CameraRenderProductInfo("cam0", "rig0", 32, 24)],
                ),
                patch.object(gaussian_usd_asset_module, "build_camera_frame_mappings", return_value={"cam0": [0, 2]}),
                patch.object(gaussian_usd_asset_module, "select_spg_files_for_export", return_value=[]),
                patch.object(
                    gaussian_usd_asset_module,
                    "serialize_nrend_usd",
                    return_value=[volume_blob, volume_stage],
                ) as serialize_volume,
                patch.object(gaussian_usd_asset_module, "iter_gaussian_nodes") as iter_nodes,
                patch.object(gaussian_usd_asset_module, "export_gaussians_as_usd_asset") as export_gaussians,
            ):
                self._invoke_export(
                    tmpdir=tmpdir,
                    datasource=_FakeNCOREDataSource(),
                    usd_schema="nrendvol",
                    export_rig_trajectories=True,
                    ppisp_integration_mode=PPISP_INTEGRATION_MODE_SPG_RUNTIME,
                    scene_radiance_scale=1.0,
                )

        cache = _RecordingExportCache.instances[-1]
        self.assertIn(volume_stage, cache.usd_stages)
        self.assertEqual(cache.serialized_files, [volume_blob])
        self.assertFalse(serialize_volume.call_args.kwargs["invert_registered_compositing"])
        self.assertTrue(serialize_volume.call_args.kwargs["accept_unsupported_ppisp"])
        self.assertTrue(serialize_volume.call_args.kwargs["disable_runtime_post_processing"])
        iter_nodes.assert_not_called()
        export_gaussians.assert_not_called()


if __name__ == "__main__":
    unittest.main()
