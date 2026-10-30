# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression and performance tests for exported PPISP CUDA sidecars."""

from __future__ import annotations

import json
import os
import statistics
import time
import unittest

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch

from pxr import Sdf, Usd

from nurec.utils.io.export.ppisp_export.controller.writer import (
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
_DEFAULT_PERF_SIZES = "64x36,128x72,256x144"
_DEFAULT_PERF_WARMUP = 2
_DEFAULT_PERF_ITERATIONS = 10
# Tiled-atlas scaling sweep. A fixed atlas budget (capped at 4K) is packed
# with a growing number of envs/tiles (up to 1024), so per-tile resolution
# shrinks as the env count grows -- the realistic multi-view atlas scenario.
_DEFAULT_PERF_ATLAS_SIZES = "1920x1080,3840x2160"
_DEFAULT_PERF_ENV_COUNTS = "1,4,16,64,256,1024"
# Mirrors INPUT_DOWNSAMPLING in ppisp_controller.cu: a tile must be at least
# this many pixels per side for the controller pool to sample meaningfully.
_CONTROLLER_INPUT_DOWNSAMPLING = 3


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


def _randomize_controller(controller: torch.nn.Module, *, seed: int) -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    with torch.no_grad():
        for name, param in controller.named_parameters():
            scale = 0.006 if "weight" in name else 0.015
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


def _read_int_env(name: str, default: int, *, min_value: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = int(raw)
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def _parse_perf_sizes() -> Tuple[Tuple[int, int], ...]:
    raw = os.environ.get("NUREC_PPISP_SPG_PERF_SIZES", _DEFAULT_PERF_SIZES)
    sizes: List[Tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        width_str, separator, height_str = item.partition("x")
        if separator == "":
            raise ValueError(f"invalid size '{item}', expected WIDTHxHEIGHT")
        width = int(width_str)
        height = int(height_str)
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid size '{item}', dimensions must be positive")
        sizes.append((width, height))
    if not sizes:
        raise ValueError("NUREC_PPISP_SPG_PERF_SIZES did not contain any sizes")
    return tuple(sizes)


def _parse_perf_atlas_sizes() -> Tuple[Tuple[int, int], ...]:
    raw = os.environ.get("NUREC_PPISP_SPG_PERF_ATLAS_SIZES", _DEFAULT_PERF_ATLAS_SIZES)
    sizes: List[Tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        width_str, separator, height_str = item.partition("x")
        if separator == "":
            raise ValueError(f"invalid atlas size '{item}', expected WIDTHxHEIGHT")
        width = int(width_str)
        height = int(height_str)
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid atlas size '{item}', dimensions must be positive")
        sizes.append((width, height))
    if not sizes:
        raise ValueError("NUREC_PPISP_SPG_PERF_ATLAS_SIZES did not contain any sizes")
    return tuple(sizes)


def _parse_perf_env_counts() -> Tuple[int, ...]:
    raw = os.environ.get("NUREC_PPISP_SPG_PERF_ENV_COUNTS", _DEFAULT_PERF_ENV_COUNTS)
    counts: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"invalid env count '{item}', must be positive")
        counts.append(value)
    if not counts:
        raise ValueError("NUREC_PPISP_SPG_PERF_ENV_COUNTS did not contain any counts")
    return tuple(counts)


def _square_tile_grid(env_count: int) -> Tuple[int, int]:
    """Factor env_count into the most-square (tileCountX, tileCountY) pair.

    Returns tileCountX >= tileCountY with tileCountX * tileCountY == env_count
    so each env maps to exactly one atlas tile.
    """
    best = (env_count, 1)
    divisor = 1
    while divisor * divisor <= env_count:
        if env_count % divisor == 0:
            best = (env_count // divisor, divisor)
        divisor += 1
    return best


def _time_cuda_wall_ms(fn: Callable[[], object], *, warmup: int, iterations: int) -> Dict[str, float]:
    result: object | None = None
    with torch.no_grad():
        for _ in range(warmup):
            result = fn()
        torch.cuda.synchronize()

        timings_ms: List[float] = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = fn()
            torch.cuda.synchronize()
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    del result
    return {
        "median_ms": statistics.median(timings_ms),
        "mean_ms": statistics.fmean(timings_ms),
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
    }


def _format_perf_table(rows: List[Dict[str, float | str]]) -> str:
    headers = [
        "size",
        "torch_static",
        "cuda_static",
        "static_ratio",
        "torch_ctrl",
        "cuda_ctrl",
        "ctrl_ratio",
        "torch_auto",
        "cuda_auto",
        "auto_ratio",
    ]
    lines = [
        "PPISP exported CUDA performance, median wall time in ms",
        " ".join(f"{header:>20}" for header in headers),
    ]
    for row in rows:
        lines.append(
            " ".join(
                [
                    f"{str(row['size']):>20}",
                    f"{float(row['torch_static_ms']):20.3f}",
                    f"{float(row['exported_cuda_static_ms']):20.3f}",
                    f"{float(row['exported_cuda_static_over_torch']):20.2f}x",
                    f"{float(row['torch_controller_ms']):20.3f}",
                    f"{float(row['exported_cuda_controller_ms']):20.3f}",
                    f"{float(row['exported_cuda_over_torch']):20.2f}x",
                    f"{float(row['torch_auto_ms']):20.3f}",
                    f"{float(row['exported_cuda_auto_ms']):20.3f}",
                    f"{float(row['exported_cuda_auto_over_torch']):20.2f}x",
                ]
            )
        )
    return "\n".join(lines)


def _format_tiled_perf_table(rows: List[Dict[str, float | str]]) -> str:
    headers = [
        "atlas",
        "envs",
        "grid",
        "tile",
        "cuda_static",
        "cuda_ctrl",
        "cuda_auto",
        "pipeline",
        "pipeline_per_env",
    ]
    lines = [
        "PPISP exported CUDA tiled-atlas performance, median wall time in ms",
        " ".join(f"{header:>18}" for header in headers),
    ]
    for row in rows:
        lines.append(
            " ".join(
                [
                    f"{str(row['atlas']):>18}",
                    f"{int(row['envs']):>18d}",
                    f"{str(row['grid']):>18}",
                    f"{str(row['tile']):>18}",
                    f"{float(row['static_ms']):18.3f}",
                    f"{float(row['controller_ms']):18.3f}",
                    f"{float(row['auto_ms']):18.3f}",
                    f"{float(row['pipeline_ms']):18.3f}",
                    f"{float(row['pipeline_per_env_ms']):18.4f}",
                ]
            )
        )
    return "\n".join(lines)


class PPISPExportedCudaRuntimeTest(unittest.TestCase):
    def _make_ppisp_and_runtime(self):
        from ppisp import PPISP, PPISPConfig  # type: ignore[import-not-found]

        device = torch.device("cuda")
        ppisp = PPISP(num_cameras=1, num_frames=1, config=PPISPConfig(use_controller=True))
        ppisp.eval()
        ppisp.to(device)
        _randomize_controller(ppisp.controllers[0], seed=71)
        controller_source = get_ppisp_embedded_controller_spg_files(ppisp, [0])[0].serialized
        runtime = ExportedEmbeddedCudaController(controller_source, device=device)
        return ppisp, runtime

    def _make_static_ppisp_runtime(self):
        from ppisp import PPISP, PPISPConfig  # type: ignore[import-not-found]

        device = torch.device("cuda")
        ppisp = PPISP(num_cameras=1, num_frames=1, config=PPISPConfig(use_controller=False))
        ppisp.eval()
        ppisp.to(device)
        _randomize_ppisp_params(ppisp, seed=73)

        stage, render_product_path = _build_minimal_render_stage()
        shader_prim = add_ppisp_shader_to_render_product(
            stage=stage,
            render_product_path=render_product_path,
            camera_index=0,
            ppisp=ppisp,
            frame_indices=[0],
            responsivity=1.0,
            resolved_reference_ids=ResolvedPPISPReferenceIds(
                camera_id=0,
                frame_id=0,
                camera_was_user_set=True,
                frame_was_user_set=True,
            ),
        )
        params = pack_static_ppisp_params(shader_prim).to(device=device)
        return ppisp, ExportedCudaPPISP(device=device), params

    def _make_auto_ppisp_runtime(self):
        from ppisp import PPISP, PPISPConfig  # type: ignore[import-not-found]

        device = torch.device("cuda")
        ppisp = PPISP(
            num_cameras=1,
            num_frames=1,
            config=PPISPConfig(use_controller=True, controller_activation_ratio=0.0),
        )
        ppisp.eval()
        ppisp.to(device)
        _randomize_ppisp_params(ppisp, seed=79)
        _randomize_controller(ppisp.controllers[0], seed=83)

        stage, render_product_path = _build_minimal_render_stage()
        add_ppisp_auto_shader_to_render_product(
            stage=stage,
            render_product_path=render_product_path,
            camera_index=0,
            ppisp=ppisp,
            controller=ppisp.controllers[0],
            responsivity=1.0,
            prior_exposure=0.19,
        )
        auto_prim = stage.GetPrimAtPath(f"{render_product_path}/{PPISP_AUTO_PRIM_NAME}")
        params = pack_auto_ppisp_params(auto_prim).to(device=device)
        controller_source = get_ppisp_embedded_controller_spg_files(ppisp, [0])[0].serialized
        return (
            ppisp,
            ExportedEmbeddedCudaController(controller_source, device=device),
            ExportedCudaPPISP(device=device),
            params,
        )

    def test_exported_embedded_cuda_controller_matches_torch_controller(self) -> None:
        ppisp, runtime = self._make_ppisp_and_runtime()
        device = torch.device("cuda")
        hdr = _make_gaussian_radiance_scene(73, 67, seed=701, device=device)
        prior = torch.tensor([0.23], device=device, dtype=torch.float32)

        with torch.no_grad():
            torch_exposure, torch_color = ppisp.controllers[0](hdr, prior)
            cuda_exposure, cuda_color = runtime(hdr, prior)

        torch.testing.assert_close(cuda_exposure, torch_exposure, rtol=_CONTROLLER_RTOL, atol=_CONTROLLER_ATOL)
        torch.testing.assert_close(cuda_color, torch_color, rtol=_CONTROLLER_RTOL, atol=_CONTROLLER_ATOL)

    @unittest.skipUnless(
        os.environ.get("NUREC_RUN_PPISP_EXPORTED_CUDA_PERF") == "1",
        "set NUREC_RUN_PPISP_EXPORTED_CUDA_PERF=1 to run exported CUDA performance sweep",
    )
    def test_exported_cuda_performance_sweep(self) -> None:
        static_ppisp, static_runtime, static_params = self._make_static_ppisp_runtime()
        ppisp, runtime, auto_runtime, auto_params = self._make_auto_ppisp_runtime()
        device = torch.device("cuda")
        prior = torch.tensor([0.19], device=device, dtype=torch.float32)
        warmup = _read_int_env("NUREC_PPISP_SPG_PERF_WARMUP", _DEFAULT_PERF_WARMUP, min_value=0)
        iterations = _read_int_env("NUREC_PPISP_SPG_PERF_ITERATIONS", _DEFAULT_PERF_ITERATIONS, min_value=1)

        rows: List[Dict[str, float | str]] = []
        for idx, (width, height) in enumerate(_parse_perf_sizes()):
            hdr = _make_gaussian_radiance_scene(width, height, seed=800 + idx, device=device)

            def run_torch_static() -> torch.Tensor:
                return static_ppisp(hdr, resolution=(width, height), camera_idx=0, frame_idx=0)

            def run_exported_cuda_static() -> torch.Tensor:
                return static_runtime.run_static(hdr, static_params)

            def run_torch_controller() -> Tuple[torch.Tensor, torch.Tensor]:
                return ppisp.controllers[0](hdr, prior)

            def run_exported_cuda_controller() -> Tuple[torch.Tensor, torch.Tensor]:
                return runtime(hdr, prior)

            def run_torch_auto() -> torch.Tensor:
                return ppisp(hdr, resolution=(width, height), camera_idx=0, frame_idx=-1, exposure_prior=prior)

            def run_exported_cuda_auto() -> torch.Tensor:
                exposure, color = runtime(hdr, prior)
                controller_params = torch.cat((exposure.reshape(1), color.reshape(-1)), dim=0)
                return auto_runtime.run_auto(hdr, controller_params, auto_params)

            with torch.no_grad():
                torch_static = run_torch_static()
                cuda_static = run_exported_cuda_static()
                torch.testing.assert_close(cuda_static, torch_static, rtol=5.0e-3, atol=5.0e-3)
                torch_exposure, torch_color = run_torch_controller()
                cuda_exposure, cuda_color = run_exported_cuda_controller()
                torch.testing.assert_close(cuda_exposure, torch_exposure, rtol=_CONTROLLER_RTOL, atol=_CONTROLLER_ATOL)
                torch.testing.assert_close(cuda_color, torch_color, rtol=_CONTROLLER_RTOL, atol=_CONTROLLER_ATOL)
                torch_auto = run_torch_auto()
                cuda_auto = run_exported_cuda_auto()
                torch.testing.assert_close(cuda_auto, torch_auto, rtol=5.0e-3, atol=5.0e-3)

            torch_static = _time_cuda_wall_ms(run_torch_static, warmup=warmup, iterations=iterations)
            exported_cuda_static = _time_cuda_wall_ms(run_exported_cuda_static, warmup=warmup, iterations=iterations)
            torch_controller = _time_cuda_wall_ms(run_torch_controller, warmup=warmup, iterations=iterations)
            exported_cuda_controller = _time_cuda_wall_ms(
                run_exported_cuda_controller,
                warmup=warmup,
                iterations=iterations,
            )
            torch_auto = _time_cuda_wall_ms(run_torch_auto, warmup=warmup, iterations=iterations)
            exported_cuda_auto = _time_cuda_wall_ms(run_exported_cuda_auto, warmup=warmup, iterations=iterations)

            rows.append(
                {
                    "size": f"{width}x{height}",
                    "pixels": float(width * height),
                    "torch_static_ms": torch_static["median_ms"],
                    "exported_cuda_static_ms": exported_cuda_static["median_ms"],
                    "exported_cuda_static_over_torch": exported_cuda_static["median_ms"]
                    / max(torch_static["median_ms"], 1.0e-9),
                    "torch_controller_ms": torch_controller["median_ms"],
                    "exported_cuda_controller_ms": exported_cuda_controller["median_ms"],
                    "exported_cuda_over_torch": exported_cuda_controller["median_ms"]
                    / max(torch_controller["median_ms"], 1.0e-9),
                    "torch_auto_ms": torch_auto["median_ms"],
                    "exported_cuda_auto_ms": exported_cuda_auto["median_ms"],
                    "exported_cuda_auto_over_torch": exported_cuda_auto["median_ms"]
                    / max(torch_auto["median_ms"], 1.0e-9),
                }
            )

        print(
            f"PPISP exported CUDA perf config: warmup={warmup}, iterations={iterations}. "
            "This target compiles and runs the exported controller, static PPISP, and auto PPISP CUDA sidecars.",
            flush=True,
        )
        print(_format_perf_table(rows), flush=True)
        output_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if output_dir is not None:
            output_path = Path(output_dir) / "ppisp_exported_cuda_performance.json"
            output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    @unittest.skipUnless(
        os.environ.get("NUREC_RUN_PPISP_EXPORTED_CUDA_PERF") == "1",
        "set NUREC_RUN_PPISP_EXPORTED_CUDA_PERF=1 to run exported CUDA performance sweep",
    )
    def test_exported_cuda_tiled_performance_sweep(self) -> None:
        """Tiled-atlas scaling perf: a fixed atlas budget (up to 4K) packed
        with a growing number of envs/tiles (up to 1024). Per-tile resolution
        shrinks as the env count grows, so the per-env column exposes how the
        static, controller, and auto sidecars amortize across many envs."""
        _, static_runtime, static_params = self._make_static_ppisp_runtime()
        _, runtime, auto_runtime, auto_params = self._make_auto_ppisp_runtime()
        device = torch.device("cuda")
        prior = torch.tensor([0.19], device=device, dtype=torch.float32)
        warmup = _read_int_env("NUREC_PPISP_SPG_PERF_WARMUP", _DEFAULT_PERF_WARMUP, min_value=0)
        iterations = _read_int_env("NUREC_PPISP_SPG_PERF_ITERATIONS", _DEFAULT_PERF_ITERATIONS, min_value=1)

        env_counts = _parse_perf_env_counts()
        rows: List[Dict[str, float | str]] = []
        for size_idx, (width, height) in enumerate(_parse_perf_atlas_sizes()):
            atlas = _make_gaussian_radiance_scene(width, height, seed=900 + size_idx, device=device)
            for env_count in env_counts:
                tcx, tcy = _square_tile_grid(env_count)
                tile_w = width // tcx
                tile_h = height // tcy
                if tile_w < _CONTROLLER_INPUT_DOWNSAMPLING or tile_h < _CONTROLLER_INPUT_DOWNSAMPLING:
                    print(
                        f"skip atlas {width}x{height} envs={env_count}: tile {tile_w}x{tile_h} "
                        f"below {_CONTROLLER_INPUT_DOWNSAMPLING}px controller downsample floor",
                        flush=True,
                    )
                    continue

                def run_static_tiled(tcx=tcx, tcy=tcy) -> torch.Tensor:
                    return static_runtime.run_static(atlas, static_params, tile_count_x=tcx, tile_count_y=tcy)

                def run_controller_tiled(tcx=tcx, tcy=tcy) -> torch.Tensor:
                    return runtime.run_tiled_params(atlas, prior, tile_count_x=tcx, tile_count_y=tcy)

                controller_params = runtime.run_tiled_params(atlas, prior, tile_count_x=tcx, tile_count_y=tcy)

                def run_auto_tiled(cp=controller_params, tcx=tcx, tcy=tcy) -> torch.Tensor:
                    return auto_runtime.run_auto(atlas, cp, auto_params, tile_count_x=tcx, tile_count_y=tcy)

                static = _time_cuda_wall_ms(run_static_tiled, warmup=warmup, iterations=iterations)
                controller = _time_cuda_wall_ms(run_controller_tiled, warmup=warmup, iterations=iterations)
                auto = _time_cuda_wall_ms(run_auto_tiled, warmup=warmup, iterations=iterations)
                pipeline_ms = controller["median_ms"] + auto["median_ms"]

                rows.append(
                    {
                        "atlas": f"{width}x{height}",
                        "envs": float(env_count),
                        "grid": f"{tcx}x{tcy}",
                        "tile": f"{tile_w}x{tile_h}",
                        "static_ms": static["median_ms"],
                        "controller_ms": controller["median_ms"],
                        "auto_ms": auto["median_ms"],
                        "pipeline_ms": pipeline_ms,
                        "pipeline_per_env_ms": pipeline_ms / env_count,
                    }
                )

        print(
            f"PPISP exported CUDA tiled perf config: warmup={warmup}, iterations={iterations}. "
            "pipeline = controller (pool+MLP) + auto image, both launched once on the full atlas.",
            flush=True,
        )
        print(_format_tiled_perf_table(rows), flush=True)
        output_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if output_dir is not None:
            output_path = Path(output_dir) / "ppisp_exported_cuda_tiled_performance.json"
            output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
