# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical equivalence tests for PPISP USD SPG exports.

These tests run the real ``ppisp`` torch/CUDA implementation and compare
it to the USD-authored SPG export path:

* the controller path compiles and launches the exported
  generated ``ppisp_controller_<cam>.cu`` sidecar through a PyTorch
  CUDA extension, using the same embedded-weight CUDA source and launch
  dimensions authored for SPG;
* the static and automatic PPISP image paths compile and launch the
  exported ``ppisp_usd_spg*.cu`` sidecars through a PyTorch CUDA
  extension, reading parameters from the UsdShade shader prims authored
  by the export writers.

The scene input is a procedurally generated 2D Gaussian radiance field
with many brightness variations across the image plane. This keeps the
controller CNN and the vignetting/color/CRF stages exercised without
requiring a full Gaussian renderer in the unit test.
"""

from __future__ import annotations

import unittest

from typing import Tuple

import torch

from pxr import Sdf, Usd

from nurec.utils.io.export.ppisp_export.controller.writer import (
    CONTROLLER_PARAMS_RENDER_VAR,
    CONTROLLER_PRIM_NAME_TEMPLATE,
    PPISP_AUTO_PRIM_NAME,
    add_ppisp_auto_shader_to_render_product,
    get_ppisp_embedded_controller_spg_files,
)
from nurec.utils.io.export.ppisp_export.reference_selection import ResolvedPPISPReferenceIds
from nurec.utils.io.export.ppisp_export.spg_cuda_controller_runtime import ExportedEmbeddedCudaController
from nurec.utils.io.export.ppisp_export.spg_cuda_ppisp_runtime import (
    ExportedCudaPPISP,
    pack_auto_ppisp_params,
    pack_static_ppisp_params,
)
from nurec.utils.io.export.ppisp_export.usd_writer import add_ppisp_shader_to_render_product


_CONTROLLER_RTOL = 1.0e-4
_CONTROLLER_ATOL = 1.0e-4


def _make_gaussian_radiance_scene(width: int, height: int, *, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    image = torch.zeros((height, width, 3), device=device, dtype=torch.float32)

    for _ in range(90):
        cx = torch.rand((), generator=generator, device=device) * float(width - 1)
        cy = torch.rand((), generator=generator, device=device) * float(height - 1)
        sx = 1.5 + torch.rand((), generator=generator, device=device) * 10.0
        sy = 1.5 + torch.rand((), generator=generator, device=device) * 10.0
        amp = 0.03 + torch.rand((), generator=generator, device=device) * 0.55
        color = 0.25 + torch.rand((3,), generator=generator, device=device) * 1.5
        blob = torch.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
        image = image + blob.unsqueeze(-1) * amp * color

    wave = 1.0 + 0.18 * torch.sin(x * 0.31) + 0.12 * torch.cos(y * 0.23) + 0.06 * torch.sin((x + y) * 0.17)
    image = image * wave.unsqueeze(-1)
    return torch.clamp_min(image + 0.02, 0.0)


def _build_minimal_render_stage(camera_name: str = "cam0") -> Tuple[Usd.Stage, str]:
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Render", "Scope")
    stage.DefinePrim("/World", "Scope")
    stage.DefinePrim("/World/Cameras", "Scope")

    camera_path = f"/World/Cameras/{camera_name}"
    stage.DefinePrim(camera_path, "Camera")

    render_product_path = f"/Render/{camera_name}"
    render_product = stage.DefinePrim(render_product_path, "RenderProduct")
    render_product.CreateRelationship("camera").SetTargets([Sdf.Path(camera_path)])
    render_product.CreateRelationship("orderedVars")
    stage.DefinePrim(f"{render_product_path}/HdrColor", "RenderVar")
    return stage, render_product_path


def _randomize_controller(
    controller: torch.nn.Module,
    *,
    seed: int,
    weight_scale: float = 0.006,
    bias_scale: float = 0.015,
) -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    with torch.no_grad():
        for name, param in controller.named_parameters():
            scale = weight_scale if "weight" in name else bias_scale
            param.copy_(torch.randn(param.shape, generator=generator, device=param.device) * scale)


def _randomize_ppisp_params(ppisp: torch.nn.Module, *, seed: int) -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    with torch.no_grad():
        ppisp.exposure_params.copy_(torch.empty_like(ppisp.exposure_params).uniform_(-0.35, 0.35, generator=generator))
        ppisp.color_params.copy_(torch.empty_like(ppisp.color_params).normal_(0.0, 0.35, generator=generator))
        ppisp.vignetting_params[..., :2].copy_(
            torch.empty_like(ppisp.vignetting_params[..., :2]).normal_(0.0, 0.04, generator=generator)
        )
        ppisp.vignetting_params[..., 2:].copy_(
            torch.empty_like(ppisp.vignetting_params[..., 2:]).uniform_(-0.35, 0.02, generator=generator)
        )
        ppisp.crf_params.add_(torch.empty_like(ppisp.crf_params).normal_(0.0, 0.08, generator=generator))


def _set_crf_center_values(ppisp: torch.nn.Module, values: Tuple[float, float, float]) -> None:
    centers = ppisp.crf_params[..., 3]
    pattern = torch.tensor(values, device=centers.device, dtype=centers.dtype)
    with torch.no_grad():
        centers.copy_(pattern.expand_as(centers))


def _make_endpoint_radiance_scene(width: int, height: int, *, device: torch.device) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device, dtype=torch.float32),
        torch.linspace(0.0, 1.0, width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    image = torch.stack((x, y, 0.5 * (x + y)), dim=-1)
    image[0, 0, :] = 0.0
    image[-1, -1, :] = 1.0
    return image


class PPISPSPGEquivalenceTest(unittest.TestCase):
    def test_exported_embedded_cuda_controller_matches_torch_controller(self) -> None:
        from ppisp import PPISP, PPISPConfig  # type: ignore[import-not-found]

        device = torch.device("cuda")
        ppisp = PPISP(num_cameras=1, num_frames=1, config=PPISPConfig(use_controller=True))
        ppisp.eval()
        controller = ppisp.controllers[0]
        _randomize_controller(controller, seed=11)

        hdr = _make_gaussian_radiance_scene(73, 67, seed=101, device=device)
        prior = torch.tensor([0.23], device=device, dtype=torch.float32)
        controller_source = get_ppisp_embedded_controller_spg_files(ppisp, [0])[0].serialized
        runtime = ExportedEmbeddedCudaController(controller_source, device=device)

        with torch.no_grad():
            torch_exposure, torch_color = controller(hdr, prior)
            spg_exposure, spg_color = runtime(hdr, prior)

        torch.testing.assert_close(spg_exposure, torch_exposure, rtol=_CONTROLLER_RTOL, atol=_CONTROLLER_ATOL)
        torch.testing.assert_close(spg_color, torch_color, rtol=_CONTROLLER_RTOL, atol=_CONTROLLER_ATOL)

    def test_controller_responsivity_scales_input_like_prescaled_hdr(self) -> None:
        # The controller scales its HDR input by responsivity before feature
        # extraction (mirroring the image PPISP shader). Running the controller
        # with responsivity=k must therefore match running it on an HDR scene
        # pre-scaled by k with responsivity=1.
        from ppisp import PPISP, PPISPConfig  # type: ignore[import-not-found]

        device = torch.device("cuda")
        ppisp = PPISP(num_cameras=1, num_frames=1, config=PPISPConfig(use_controller=True))
        ppisp.eval()
        # Use an input-sensitive controller (larger weights, smaller bias) so
        # responsivity actually changes the prediction; the default near-zero
        # weights make the controller insensitive to input magnitude, which
        # would make this test pass vacuously.
        _randomize_controller(ppisp.controllers[0], seed=11, weight_scale=0.15, bias_scale=0.005)

        hdr = _make_gaussian_radiance_scene(73, 67, seed=101, device=device)
        prior = torch.tensor([0.23], device=device, dtype=torch.float32)
        controller_source = get_ppisp_embedded_controller_spg_files(ppisp, [0])[0].serialized
        runtime = ExportedEmbeddedCudaController(controller_source, device=device)

        responsivity = 1.7
        scaled_exposure, scaled_color = runtime(hdr, prior, responsivity=responsivity)
        prescaled_exposure, prescaled_color = runtime(hdr * responsivity, prior)

        torch.testing.assert_close(scaled_exposure, prescaled_exposure, rtol=_CONTROLLER_RTOL, atol=_CONTROLLER_ATOL)
        torch.testing.assert_close(scaled_color, prescaled_color, rtol=_CONTROLLER_RTOL, atol=_CONTROLLER_ATOL)

        # Sanity: a non-unit responsivity must actually change the prediction.
        base_exposure, _ = runtime(hdr, prior)
        self.assertGreater((scaled_exposure - base_exposure).abs().item(), 1.0e-4)

    def test_static_ppisp_usd_shader_simulation_matches_torch_ppisp(self) -> None:
        from ppisp import PPISP, PPISPConfig  # type: ignore[import-not-found]

        device = torch.device("cuda")
        camera_id = 1
        frame_id = 2
        ppisp = PPISP(num_cameras=2, num_frames=4, config=PPISPConfig(use_controller=False))
        ppisp.eval()
        _randomize_ppisp_params(ppisp, seed=23)

        hdr = _make_gaussian_radiance_scene(65, 59, seed=202, device=device)
        stage, render_product_path = _build_minimal_render_stage()
        shader_prim = add_ppisp_shader_to_render_product(
            stage=stage,
            render_product_path=render_product_path,
            camera_index=camera_id,
            ppisp=ppisp,
            frame_indices=[frame_id],
            responsivity=1.0,
            resolved_reference_ids=ResolvedPPISPReferenceIds(
                camera_id=camera_id,
                frame_id=frame_id,
                camera_was_user_set=True,
                frame_was_user_set=True,
            ),
        )

        params = pack_static_ppisp_params(shader_prim).to(device=device)
        spg = ExportedCudaPPISP(device=device).run_static(hdr, params)

        with torch.no_grad():
            torch_out = ppisp(hdr, resolution=(hdr.shape[1], hdr.shape[0]), camera_idx=camera_id, frame_idx=frame_id)

        torch.testing.assert_close(spg, torch_out, rtol=5.0e-3, atol=5.0e-3)

    def test_auto_ppisp_usd_graph_simulation_matches_torch_ppisp_novel_view(self) -> None:
        from ppisp import PPISP, PPISPConfig  # type: ignore[import-not-found]

        device = torch.device("cuda")
        camera_id = 0
        ppisp = PPISP(
            num_cameras=1,
            num_frames=3,
            config=PPISPConfig(use_controller=True, controller_activation_ratio=0.0),
        )
        ppisp.eval()
        _randomize_ppisp_params(ppisp, seed=31)
        _randomize_controller(ppisp.controllers[camera_id], seed=37)

        hdr = _make_gaussian_radiance_scene(71, 62, seed=303, device=device)
        prior = torch.tensor([0.19], device=device, dtype=torch.float32)
        stage, render_product_path = _build_minimal_render_stage()
        add_ppisp_auto_shader_to_render_product(
            stage=stage,
            render_product_path=render_product_path,
            camera_index=camera_id,
            ppisp=ppisp,
            controller=ppisp.controllers[camera_id],
            responsivity=1.0,
            prior_exposure=float(prior.item()),
        )

        controller_prim = stage.GetPrimAtPath(
            f"{render_product_path}/{CONTROLLER_PRIM_NAME_TEMPLATE.format(camera_index=camera_id)}"
        )
        self.assertTrue(controller_prim.IsValid())
        self.assertEqual(controller_prim.GetAttribute("info:spg:sourceAsset").Get().path, "ppisp_controller_0.cu")
        self.assertEqual(controller_prim.GetAttribute("info:spg:sourceAsset:subIdentifier").Get(), "controllerProcess")
        self.assertFalse(controller_prim.GetAttribute("inputs:weights").IsValid())

        controller_source = get_ppisp_embedded_controller_spg_files(ppisp, [camera_id])[0]
        runtime = ExportedEmbeddedCudaController(controller_source.serialized, device=device)
        prior_from_usd = torch.tensor(
            [float(controller_prim.GetAttribute("inputs:priorExposure").Get())],
            device=device,
            dtype=torch.float32,
        )
        exposure, color = runtime(hdr, prior_from_usd)

        auto_prim = stage.GetPrimAtPath(f"{render_product_path}/{PPISP_AUTO_PRIM_NAME}")
        self.assertTrue(auto_prim.IsValid())
        controller_input = auto_prim.GetAttribute(f"inputs:{CONTROLLER_PARAMS_RENDER_VAR}")
        self.assertEqual(
            [str(p) for p in controller_input.GetConnections()],
            [f"{render_product_path}/{CONTROLLER_PARAMS_RENDER_VAR}.omni:rtx:aov"],
        )
        controller_params = torch.cat((exposure.reshape(1), color.reshape(-1)), dim=0)
        params = pack_auto_ppisp_params(auto_prim).to(device=device)
        spg = ExportedCudaPPISP(device=device).run_auto(hdr, controller_params, params)

        with torch.no_grad():
            torch_out = ppisp(
                hdr,
                resolution=(hdr.shape[1], hdr.shape[0]),
                camera_idx=camera_id,
                frame_idx=-1,
                exposure_prior=prior,
            )

        torch.testing.assert_close(spg, torch_out, rtol=5.0e-3, atol=5.0e-3)

    def test_static_ppisp_tiled_atlas_matches_per_tile_untiled(self) -> None:
        """Tiled static shader: each tile of a 2x1 atlas equals the untiled tile.

        SPG runs once on the full atlas; with tileCountX=2 the vignetting
        UV is evaluated per tile, so an atlas made of two identical tiles
        must reproduce the untiled single-tile output in each half, while
        an untiled (1x1) launch on the same atlas must differ (its UV spans
        the whole 2x-wide atlas).
        """
        from ppisp import PPISP, PPISPConfig  # type: ignore[import-not-found]

        device = torch.device("cuda")
        camera_id = 0
        ppisp = PPISP(num_cameras=1, num_frames=2, config=PPISPConfig(use_controller=False))
        ppisp.eval()
        _randomize_ppisp_params(ppisp, seed=51)

        hdr = _make_gaussian_radiance_scene(48, 40, seed=404, device=device)
        stage, render_product_path = _build_minimal_render_stage()
        shader_prim = add_ppisp_shader_to_render_product(
            stage=stage,
            render_product_path=render_product_path,
            camera_index=camera_id,
            ppisp=ppisp,
            frame_indices=[0],
            responsivity=1.0,
            resolved_reference_ids=ResolvedPPISPReferenceIds(
                camera_id=camera_id,
                frame_id=0,
                camera_was_user_set=True,
                frame_was_user_set=True,
            ),
        )
        params = pack_static_ppisp_params(shader_prim).to(device=device)
        runtime = ExportedCudaPPISP(device=device)

        width = hdr.shape[1]
        single = runtime.run_static(hdr, params)
        atlas = torch.cat((hdr, hdr), dim=1)
        tiled = runtime.run_static(atlas, params, tile_count_x=2, tile_count_y=1)

        torch.testing.assert_close(tiled[:, :width], single, rtol=0.0, atol=0.0)
        torch.testing.assert_close(tiled[:, width:], single, rtol=0.0, atol=0.0)

        # Sanity: an untiled launch on the atlas evaluates UV over the full
        # 2x-wide image, so its left half must NOT match the tile-local result
        # (guards against tileCount being silently ignored).
        untiled_atlas = runtime.run_static(atlas, params)
        self.assertFalse(torch.allclose(untiled_atlas[:, :width], single, rtol=0.0, atol=1.0 / 255.0))

    def test_auto_ppisp_tiled_atlas_selects_per_tile_controller_params(self) -> None:
        """Tiled auto shader: each tile reads its own 9-float ControllerParams group."""
        from ppisp import PPISP, PPISPConfig  # type: ignore[import-not-found]

        device = torch.device("cuda")
        camera_id = 0
        ppisp = PPISP(
            num_cameras=1,
            num_frames=2,
            config=PPISPConfig(use_controller=True, controller_activation_ratio=0.0),
        )
        ppisp.eval()
        _randomize_ppisp_params(ppisp, seed=53)

        hdr = _make_gaussian_radiance_scene(44, 38, seed=505, device=device)
        stage, render_product_path = _build_minimal_render_stage()
        add_ppisp_auto_shader_to_render_product(
            stage=stage,
            render_product_path=render_product_path,
            camera_index=camera_id,
            ppisp=ppisp,
            controller=ppisp.controllers[camera_id],
            responsivity=1.0,
            prior_exposure=0.0,
        )
        auto_prim = stage.GetPrimAtPath(f"{render_product_path}/{PPISP_AUTO_PRIM_NAME}")
        params = pack_auto_ppisp_params(auto_prim).to(device=device)
        runtime = ExportedCudaPPISP(device=device)

        # Two distinct per-tile controller groups (exposure + 8 color latents).
        cp0 = torch.tensor([0.30, 0.10, -0.05, 0.20, 0.02, -0.12, 0.07, 0.15, -0.09], device=device)
        cp1 = torch.tensor([-0.40, -0.18, 0.22, -0.03, 0.11, 0.25, -0.14, 0.04, 0.19], device=device)

        width = hdr.shape[1]
        single0 = runtime.run_auto(hdr, cp0, params)
        single1 = runtime.run_auto(hdr, cp1, params)

        atlas = torch.cat((hdr, hdr), dim=1)
        cp_tiled = torch.cat((cp0, cp1), dim=0)
        tiled = runtime.run_auto(atlas, cp_tiled, params, tile_count_x=2, tile_count_y=1)

        torch.testing.assert_close(tiled[:, :width], single0, rtol=0.0, atol=0.0)
        torch.testing.assert_close(tiled[:, width:], single1, rtol=0.0, atol=0.0)

    def test_auto_ppisp_tiled_atlas_end_to_end_matches_per_view_torch(self) -> None:
        """Full tiled pipeline vs per-view torch PPISP+controller.

        Renders N distinct views, assembles their HDR radiance into a
        tile_count_y x tile_count_x atlas, then runs the exported CUDA
        pipeline once on that atlas: the controller pool+MLP produce one
        9-float group per tile, and the auto image shader applies the
        per-tile group with tile-local UV. Each disassembled tile must
        match the untiled torch ``ppisp`` output (controller + image
        stages) computed on that view alone.
        """
        from ppisp import PPISP, PPISPConfig  # type: ignore[import-not-found]

        device = torch.device("cuda")
        camera_id = 0
        tile_count_x = 4
        tile_count_y = 2
        tile_count = tile_count_x * tile_count_y
        tile_w = 40
        tile_h = 32

        ppisp = PPISP(
            num_cameras=1,
            num_frames=3,
            config=PPISPConfig(use_controller=True, controller_activation_ratio=0.0),
        )
        ppisp.eval()
        _randomize_ppisp_params(ppisp, seed=61)
        # A near-zero random controller is bias-dominated and emits the same
        # params for every input; that would let a global-vs-per-tile pool bug
        # pass. Scale the weights up so distinct tile statistics drive distinct
        # per-tile params (asserted below).
        _randomize_controller(ppisp.controllers[camera_id], seed=67, weight_scale=0.15, bias_scale=0.005)

        prior = torch.tensor([0.19], device=device, dtype=torch.float32)
        views = [_make_gaussian_radiance_scene(tile_w, tile_h, seed=900 + i, device=device) for i in range(tile_count)]

        with torch.no_grad():
            torch_refs = [
                ppisp(
                    view,
                    resolution=(tile_w, tile_h),
                    camera_idx=camera_id,
                    frame_idx=-1,
                    exposure_prior=prior,
                )
                for view in views
            ]

        # Assemble the atlas row-major so tile (tx, ty) lands at linear
        # index ty * tile_count_x + tx, matching the CUDA tile ordering.
        atlas = torch.cat(
            [
                torch.cat([views[ty * tile_count_x + tx] for tx in range(tile_count_x)], dim=1)
                for ty in range(tile_count_y)
            ],
            dim=0,
        )

        stage, render_product_path = _build_minimal_render_stage()
        add_ppisp_auto_shader_to_render_product(
            stage=stage,
            render_product_path=render_product_path,
            camera_index=camera_id,
            ppisp=ppisp,
            controller=ppisp.controllers[camera_id],
            responsivity=1.0,
            prior_exposure=float(prior.item()),
        )
        auto_prim = stage.GetPrimAtPath(f"{render_product_path}/{PPISP_AUTO_PRIM_NAME}")
        params = pack_auto_ppisp_params(auto_prim).to(device=device)

        controller_source = get_ppisp_embedded_controller_spg_files(ppisp, [camera_id])[0]
        controller_runtime = ExportedEmbeddedCudaController(controller_source.serialized, device=device)
        image_runtime = ExportedCudaPPISP(device=device)

        controller_params = controller_runtime.run_tiled_params(
            atlas, prior, tile_count_x=tile_count_x, tile_count_y=tile_count_y
        )

        # The per-view scenes have distinct statistics, so the per-tile
        # controller groups must differ. This guards against a degenerate
        # (input-insensitive) controller masking a global-vs-per-tile pool
        # bug: if every tile read the whole atlas, the groups would collapse
        # and the per-tile comparison below would pass for the wrong reason.
        per_tile_params = controller_params.reshape(tile_count, 9)
        self.assertGreater(per_tile_params.std(dim=0).max().item(), 1.0e-3)

        atlas_out = image_runtime.run_auto(
            atlas, controller_params, params, tile_count_x=tile_count_x, tile_count_y=tile_count_y
        )

        # Sanity: PPISP must actually transform the image, otherwise matching
        # torch would be meaningless (a no-op equals a no-op). Compare the
        # torch PPISP output against a no-ISP passthrough (clamped raw
        # radiance) and require a sizeable difference (>> the 5e-3 tolerance).
        no_isp = torch.stack([torch.clamp(view, 0.0, 1.0) for view in views])
        with_isp = torch.stack(torch_refs)
        ppisp_effect = (with_isp - no_isp).abs().mean().item()
        self.assertGreater(ppisp_effect, 0.02)

        # Sanity: the controller must affect the result. Re-run the image
        # shader with zeroed controller params and require a meaningful
        # difference from the real per-tile params.
        neutral_params = torch.zeros_like(controller_params)
        atlas_out_neutral = image_runtime.run_auto(
            atlas, neutral_params, params, tile_count_x=tile_count_x, tile_count_y=tile_count_y
        )
        controller_effect = (atlas_out - atlas_out_neutral).abs().mean().item()
        self.assertGreater(controller_effect, 0.02)

        for ty in range(tile_count_y):
            for tx in range(tile_count_x):
                index = ty * tile_count_x + tx
                tile_out = atlas_out[ty * tile_h : (ty + 1) * tile_h, tx * tile_w : (tx + 1) * tile_w]
                with self.subTest(tile=(tx, ty)):
                    torch.testing.assert_close(tile_out, torch_refs[index], rtol=5.0e-3, atol=5.0e-3)

    def test_auto_ppisp_usd_graph_simulation_handles_extreme_crf_centers(self) -> None:
        from ppisp import PPISP, PPISPConfig  # type: ignore[import-not-found]

        device = torch.device("cuda")
        camera_id = 0
        ppisp = PPISP(
            num_cameras=1,
            num_frames=3,
            config=PPISPConfig(use_controller=True, controller_activation_ratio=0.0),
        )
        ppisp.eval()
        ppisp.to(device)
        _randomize_ppisp_params(ppisp, seed=41)
        _randomize_controller(ppisp.controllers[camera_id], seed=43)

        hdr = _make_endpoint_radiance_scene(23, 19, device=device)
        prior = torch.tensor([0.19], device=device, dtype=torch.float32)
        controller_source = get_ppisp_embedded_controller_spg_files(ppisp, [camera_id])[0]
        controller_runtime = ExportedEmbeddedCudaController(controller_source.serialized, device=device)
        auto_runtime = ExportedCudaPPISP(device=device)

        with torch.no_grad():
            exposure, color = controller_runtime(hdr, prior)
        controller_params = torch.cat((exposure.reshape(1), color.reshape(-1)), dim=0)

        for center_values in ((-10.0, 10.0, -10.0), (10.0, -10.0, 10.0)):
            with self.subTest(center_values=center_values):
                _set_crf_center_values(ppisp, center_values)
                stage, render_product_path = _build_minimal_render_stage()
                add_ppisp_auto_shader_to_render_product(
                    stage=stage,
                    render_product_path=render_product_path,
                    camera_index=camera_id,
                    ppisp=ppisp,
                    controller=ppisp.controllers[camera_id],
                    responsivity=1.0,
                    prior_exposure=float(prior.item()),
                )

                auto_prim = stage.GetPrimAtPath(f"{render_product_path}/{PPISP_AUTO_PRIM_NAME}")
                self.assertTrue(auto_prim.IsValid())
                params = pack_auto_ppisp_params(auto_prim).to(device=device)
                spg = auto_runtime.run_auto(hdr, controller_params, params)

                with torch.no_grad():
                    torch_out = ppisp(
                        hdr,
                        resolution=(hdr.shape[1], hdr.shape[0]),
                        camera_idx=camera_id,
                        frame_idx=-1,
                        exposure_prior=prior,
                    )

                self.assertTrue(torch.isfinite(spg).all().item())
                torch.testing.assert_close(spg, torch_out, rtol=5.0e-3, atol=5.0e-3)


if __name__ == "__main__":
    unittest.main()
