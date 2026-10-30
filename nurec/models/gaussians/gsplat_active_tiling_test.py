# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executable pins for the GSplat half of active tiling.

The camera/lidar tile geometry that active tiling relies on is an *agreement* between
``GSplatRenderer`` and gsplat's own internals: the mask must land on exactly the tile grid
gsplat resolves for a render, or tiles get skipped at the wrong place. That agreement was
hand-verified once; these tests make it executable so a gsplat pin bump or a refactor on
either side cannot silently break it.

Four groups, each pinning one link of the chain:

1. ``active_tile_mask_tile_size`` == gsplat's own ``_resolve_tile_size`` (RESOLUTION), and the
   configured square tile (CUSTOM).
2. ``compute_lidar_element_to_tile`` == gsplat's own ``create_elements`` ->
   ``elements_to_sensor_angles`` -> ``angles_to_tile_indices`` chain, in gsplat's
   elevation-major flat-index convention.
3. ``render()``'s lidar branch consumes ``lidar_tiling_params()`` as its single tiling source
   (structural pin: mask grid and render grid cannot drift apart).
4. End-to-end mask honoring through the public ``rendering_data.active_tile_mask`` path:
   all-True == no-mask, all-False == background-only, checkerboard == the per-tile mix, plus
   the ``visibility`` / ``frustum_visibility`` narrowing that SelectiveAdam depends on and the
   ``packed=True`` branch where that narrowing degrades to a one-shot warning.

The lidar end-to-end (group 4) variant is deliberately not covered -- see
``test_lidar_end_to_end_mask_honoring_is_not_covered`` for the reasoning.
"""

from __future__ import annotations

import dataclasses
import json
import logging

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from omegaconf import DictConfig, OmegaConf

import ncore.data

from ncore.data import OpenCVPinholeCameraModelParameters, ShutterType
from nurec.config.model import GSplatRendererConfig
from nurec.config.parse import parse_untyped_config
from nurec.models.gaussians.renderers import GSplatRenderer
from nurec.utils.batch import RenderingData


_TESTDATA_CONFIG_DIR = Path(__file__).resolve().parent / "testdata"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="GSplat requires CUDA CC >= 8.0",
)

DEVICE = torch.device("cuda")

# Camera scene dimensions. 64x64 sits below the 1080p threshold, so the RESOLUTION strategy
# resolves an 8px tile -> an 8x8 tile grid, which is big enough for a meaningful checkerboard
# and small enough to render in milliseconds.
CAM_H = 64
CAM_W = 64
CAM_FOCAL = 50.0

# Lidar model used for the tile-grid pins. Mirrors gsplat_lidar_test's sensor selection;
# AT128 is the smaller of the two grids (128 rows x 1200 columns).
LIDAR_SENSOR = "HESAI_AT128"


# --------------------------------------------------------------------------------------
# Shared fixtures (modelled on gsplat_renderer_training_test.py / gsplat_lidar_test.py)
# --------------------------------------------------------------------------------------


class _MockModel(nn.Module):
    """Minimal model stand-in: GSplatRenderer only reads ``nurec_data.config`` out of it.

    Same shape as gsplat_renderer_training_test.py's ``mock_model`` fixture.
    """

    def serialize_to_json_dict(self, with_state_dict: bool = False) -> dict:
        return {"type": "mock_model", "nurec_data": {"config": {}}}


@pytest.fixture(scope="module")
def mock_model() -> _MockModel:
    return _MockModel()


def _make_config(name: str = "3dgut-gsplat", **overrides) -> GSplatRendererConfig:
    """Build a validated GSplatRendererConfig. ``packed`` stays False so the mask-aware
    visibility narrowing (``tiles_per_gauss``) is available -- the 3dgut path forces it anyway."""
    config: dict = {
        "name": name,
        # Projection path (no external rays): the renderer derives them from Ks/viewmat, which
        # keeps the synthetic scene self-consistent. Same choice as the training test's
        # `simple_config`.
        "use_rays": False,
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
    config.update(overrides)
    return GSplatRendererConfig.model_validate(config)


# --------------------------------------------------------------------------------------
# 1. active_tile_mask_tile_size agrees with gsplat's own tile-size resolution
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("tile", [8, 16])
@pytest.mark.parametrize("name", ["3dgut-gsplat", "3dgs-gsplat"])
def test_custom_strategy_forwards_configured_tile_size(mock_model, name: str, tile: int):
    """CUSTOM: the configured square tile size is what the mask must be built at, verbatim,
    because render() passes that same ``self.tile_size`` to gsplat.rasterization."""
    renderer = GSplatRenderer(
        _make_config(
            name,
            tiling={"strategy": "custom", "camera": {"tile_width": tile, "tile_height": tile}},
        ),
        mock_model,
    )
    # Resolution must not influence CUSTOM: both sides of the 1080p threshold give the same tile.
    for width, height in [(1920, 1080), (1280, 720), (64, 64)]:
        assert renderer.active_tile_mask_tile_size(width, height) == (tile, tile), (
            f"CUSTOM must forward the configured tile size for {width}x{height}"
        )


@pytest.mark.parametrize("width,height", [(1920, 1080), (1280, 720), (CAM_W, CAM_H)])
@pytest.mark.parametrize("name", ["3dgut-gsplat", "3dgs-gsplat"])
def test_resolution_strategy_matches_gsplat_resolve_tile_size(mock_model, name: str, width: int, height: int):
    """RESOLUTION: the mask tile size must equal what gsplat itself resolves for the render.

    Compared against the imported ``gsplat.rendering._resolve_tile_size`` rather than a
    reimplementation -- that agreement *is* the invariant. The 3dgut/eval3d path gates on
    ``min(W, H) >= 1080``, the non-eval3d path is flat, so both names and both sides of the
    threshold are exercised.
    """
    from gsplat.rendering import _resolve_tile_size

    renderer = GSplatRenderer(_make_config(name, tiling={"strategy": "resolution"}), mock_model)
    use_3dgut = name == "3dgut-gsplat"
    expected = _resolve_tile_size(None, use_3dgut, width, height)

    assert renderer.active_tile_mask_tile_size(width, height) == (expected, expected), (
        f"mask tile size must match gsplat's resolved tile size ({expected}) for {name} at {width}x{height}"
    )


def test_resolution_strategy_ignores_stale_configured_tile_dims(mock_model):
    """RESOLUTION ignores the configured tile dims entirely (legacy configs carry stale ones),
    so the mask still lands on gsplat's resolved grid."""
    from gsplat.rendering import _resolve_tile_size

    renderer = GSplatRenderer(
        _make_config(
            "3dgut-gsplat",
            tiling={"strategy": "resolution", "camera": {"tile_width": 16, "tile_height": 16}},
        ),
        mock_model,
    )
    expected = _resolve_tile_size(None, True, CAM_W, CAM_H)
    assert renderer.active_tile_mask_tile_size(CAM_W, CAM_H) == (expected, expected)


# --------------------------------------------------------------------------------------
# 2. compute_lidar_element_to_tile lands on gsplat's own lidar tile grid
# --------------------------------------------------------------------------------------


def _load_lidar_models_config() -> DictConfig:
    """Load the sensor config YAML via parse_untyped_config (same helper as gsplat_lidar_test)."""
    # An absolute config path is parent-rooted (flat package), so the lidar models
    # land at .lidar_models rather than .sensor.lidar_models.
    return parse_untyped_config(str(_TESTDATA_CONFIG_DIR / "sensor/lidar_model"), hydra_args=[]).lidar_models


@pytest.fixture(scope="module")
def ncore_lidar_params() -> ncore.data.RowOffsetStructuredSpinningLidarModelParameters:
    """ncore spinning-lidar parameters, built exactly like gsplat_lidar_test builds them."""
    plain = OmegaConf.to_container(_load_lidar_models_config()[LIDAR_SENSOR], resolve=True)
    return ncore.data.RowOffsetStructuredSpinningLidarModelParameters.from_json(json.dumps(plain))


def _gsplat_lidar_ext(
    ncore_params: ncore.data.RowOffsetStructuredSpinningLidarModelParameters,
    tiling_params: dict[str, int],
):
    """Drive gsplat's own lidar-extension construction independently of the renderer.

    Deliberately does *not* go through ``GSplatRenderer._build_lidar_params`` -- the point is
    to reach the same tile grid from gsplat's public constructors alone, so a divergence in
    the renderer's construction shows up as a test failure rather than cancelling out.
    """
    import gsplat

    spinning_direction = (
        gsplat.SpinningDirection.CLOCKWISE
        if ncore_params.spinning_direction == "cw"
        else gsplat.SpinningDirection.COUNTER_CLOCKWISE
    )
    params = gsplat.RowOffsetStructuredSpinningLidarModelParameters(
        row_elevations_rad=torch.tensor(
            np.asarray(ncore_params.row_elevations_rad), dtype=torch.float32, device=DEVICE
        ),
        column_azimuths_rad=torch.tensor(
            np.asarray(ncore_params.column_azimuths_rad), dtype=torch.float32, device=DEVICE
        ),
        row_azimuth_offsets_rad=torch.tensor(
            np.asarray(ncore_params.row_azimuth_offsets_rad), dtype=torch.float32, device=DEVICE
        ),
        spinning_frequency_hz=ncore_params.spinning_frequency_hz,
        spinning_direction=spinning_direction,
        # Matches _build_lidar_params; the FOV span feeds the azimuth bin quantization.
        fov_eps_factor=4,
    )
    angles_to_columns_map = gsplat.compute_lidar_angles_to_columns_map(params)
    tiling = gsplat.compute_lidar_tiling(params, **tiling_params)
    return gsplat.RowOffsetStructuredSpinningLidarModelParametersExt.from_params(
        params,
        angles_to_columns_map=angles_to_columns_map,
        tiling=tiling,
    )


def test_compute_lidar_element_to_tile_matches_gsplat_chain(mock_model, ncore_lidar_params):
    """The per-element tile ids the mask is built from must be gsplat's own tile assignment.

    Reference chain (gsplat's, driven here directly): create_elements ->
    elements_to_sensor_angles -> angles_to_tile_indices with the tiling's own bin counts.
    """
    from gsplat.cuda._lidar import angles_to_tile_indices

    renderer = GSplatRenderer(_make_config("3dgut-gsplat"), mock_model)
    element_to_tile, n_bins_az, n_bins_el = renderer.compute_lidar_element_to_tile(ncore_lidar_params, DEVICE)

    # Bin counts must be plain ints (they size the host-side mask) and consistent with the
    # extension's own tiling.
    assert isinstance(n_bins_az, int) and isinstance(n_bins_el, int), (
        f"bin counts must be ints, got {type(n_bins_az)} / {type(n_bins_el)}"
    )
    ext = _gsplat_lidar_ext(ncore_lidar_params, renderer.lidar_tiling_params())
    assert (n_bins_az, n_bins_el) == (int(ext.tiling.n_bins_azimuth), int(ext.tiling.n_bins_elevation)), (
        "returned bin counts must match the gsplat tiling's own bin counts"
    )

    elements = ext.create_elements()
    angles = ext.elements_to_sensor_angles(elements)
    reference = angles_to_tile_indices(
        ext,
        angles,
        n_bins_azimuth=n_bins_az,
        n_bins_elevation=n_bins_el,
        cdf_elevation=ext.tiling.cdf_elevation,
    ).reshape(ext.n_rows, ext.n_columns)

    assert element_to_tile.shape == (ext.n_rows, ext.n_columns), (
        f"element_to_tile must be (n_rows, n_columns), got {tuple(element_to_tile.shape)}"
    )
    assert torch.equal(element_to_tile, reference.long()), (
        f"element_to_tile diverges from gsplat's own tile assignment: "
        f"{(element_to_tile != reference.long()).sum().item()} / {element_to_tile.numel()} mismatches"
    )


def test_compute_lidar_element_to_tile_is_elevation_major(mock_model, ncore_lidar_params):
    """Flat tile ids follow gsplat's elevation-major convention: ``el * n_bins_azimuth + az``.

    The composite lays the lidar mask out as (n_bins_elevation, n_bins_azimuth) on the strength
    of this convention, so decompose the ids and re-derive the azimuth bin from gsplat's own
    relative-angle quantization to prove the packing order.
    """
    from gsplat.cuda._lidar import relative_sensor_angles

    renderer = GSplatRenderer(_make_config("3dgut-gsplat"), mock_model)
    element_to_tile, n_bins_az, n_bins_el = renderer.compute_lidar_element_to_tile(ncore_lidar_params, DEVICE)

    flat = element_to_tile.reshape(-1)
    assert int(flat.min()) >= 0 and int(flat.max()) < n_bins_az * n_bins_el, (
        "tile ids must be inside the (n_bins_elevation x n_bins_azimuth) grid"
    )

    el_bin = flat // n_bins_az
    az_bin = flat % n_bins_az
    assert int(el_bin.max()) < n_bins_el, "elevation bin out of range under the elevation-major reading"

    # Independently recompute the azimuth bin from the sensor angles. If the flat index were
    # azimuth-major, `flat % n_bins_az` would not reproduce it.
    ext = _gsplat_lidar_ext(ncore_lidar_params, renderer.lidar_tiling_params())
    angles = ext.elements_to_sensor_angles(ext.create_elements())
    relative = relative_sensor_angles(ext, angles)
    # Mirror gsplat's own quantization: the closing edge normalizes to exactly n_bins_az and
    # is clamped into the last bin, not wrapped into the first.
    expected_az = (relative[..., 0] / ext.fov_horiz_rad.span * n_bins_az).int().clamp(max=n_bins_az - 1)
    assert torch.equal(az_bin.int(), expected_az), (
        "low-order factor of the flat tile id is not the azimuth bin -- the elevation-major "
        "convention the lidar mask layout assumes has changed"
    )

    # Both axes must actually be exercised, otherwise the decomposition above is vacuous.
    assert int(az_bin.max()) > 0 and int(el_bin.max()) > 0, "degenerate tile grid; the pin would be vacuous"


# --------------------------------------------------------------------------------------
# 3. render()'s lidar branch consumes lidar_tiling_params() (structural pin)
# --------------------------------------------------------------------------------------


class _ShortCircuit(Exception):
    """Raised from the patched ``_build_lidar_params`` once its kwargs have been captured."""


def _lidar_rendering_data(ncore_lidar_params) -> RenderingData:
    """Minimal lidar RenderingData. Only needs to be well-formed enough to reach the lidar
    branch's ``_build_lidar_params`` call -- the render is short-circuited there."""
    pose_tquat = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], device=DEVICE)
    timestamps = torch.tensor([[0, 1000]], device=DEVICE, dtype=torch.int64)
    return RenderingData(
        rays=torch.zeros(1, 4, 4, 6, device=DEVICE),
        sensor_model_parameters=[ncore_lidar_params],
        poses_tquat_startend=pose_tquat.unsqueeze(1).repeat(1, 2, 1),
        timestamps_startend_us=timestamps,
        timestamps_startend_us_cpu=timestamps.cpu(),
    )


def _capture_lidar_params_kwargs(renderer, rendering_data, monkeypatch) -> dict:
    captured: dict = {}

    def _capture(ncore_params, **kwargs):
        captured.update(kwargs)
        raise _ShortCircuit

    monkeypatch.setattr(renderer, "_build_lidar_params", _capture)
    with pytest.raises(_ShortCircuit):
        renderer.render(
            rendering_data,
            {},
            n_active_features=0,
            extra_ray_signal_infos=([], [], []),
            frame_meta=None,
        )
    captured.pop("device", None)
    return captured


def test_lidar_render_uses_the_tiling_params_accessor(mock_model, ncore_lidar_params, monkeypatch):
    """The lidar render must build its tiling from exactly what ``lidar_tiling_params()``
    reports -- the same accessor the camera->lidar mask build uses.

    This matters because histogram equalization derives ``n_bins_azimuth`` from
    ``max_pts_per_tile``: any drift between the two callers silently puts the mask on a
    different grid than the one gsplat rasterizes.
    """
    renderer = GSplatRenderer(_make_config("3dgut-gsplat"), mock_model)
    rendering_data = _lidar_rendering_data(ncore_lidar_params)

    # Baseline: unpatched, the render must consume exactly what the accessor reports.
    assert _capture_lidar_params_kwargs(renderer, rendering_data, monkeypatch) == renderer.lidar_tiling_params()


def test_lidar_render_tiling_params_come_from_the_accessor_not_the_config(mock_model, ncore_lidar_params, monkeypatch):
    """Sentinel variant of the invariant above: values that appear nowhere in the config must
    still reach ``_build_lidar_params`` if the render truly reads the accessor."""
    renderer = GSplatRenderer(_make_config("3dgut-gsplat"), mock_model)
    rendering_data = _lidar_rendering_data(ncore_lidar_params)

    sentinel = {
        "n_bins_elevation": 7,
        "max_pts_per_tile": 33,
        "resolution_elevation": 999,
        "densification_factor_azimuth": 5,
    }
    assert sentinel != renderer.lidar_tiling_params(), "sentinel must differ from the config-derived values"
    monkeypatch.setattr(renderer, "lidar_tiling_params", lambda: dict(sentinel))

    assert _capture_lidar_params_kwargs(renderer, rendering_data, monkeypatch) == sentinel, (
        "render() recomputed the lidar tiling instead of consuming lidar_tiling_params()"
    )


# --------------------------------------------------------------------------------------
# 4. End-to-end mask honoring through rendering_data.active_tile_mask (camera path)
# --------------------------------------------------------------------------------------


def _opencv_camera_params() -> OpenCVPinholeCameraModelParameters:
    return OpenCVPinholeCameraModelParameters(
        resolution=np.array([CAM_W, CAM_H], dtype=np.uint64),
        focal_length=np.array([CAM_FOCAL, CAM_FOCAL], dtype=np.float32),
        principal_point=np.array([CAM_W / 2, CAM_H / 2], dtype=np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        shutter_type=ShutterType.GLOBAL,
    )


@pytest.fixture(scope="module")
def camera_rendering_data() -> RenderingData:
    """Identity-pose pinhole frame. With ``use_rays=False`` the rays tensor only carries the
    frame dimensions and device (same as the training test's ``simple_rendering_data``)."""
    pose_tquat = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], device=DEVICE)
    timestamps = torch.tensor([[0, 1000]], device=DEVICE, dtype=torch.int64)
    return RenderingData(
        rays=torch.zeros(1, CAM_H, CAM_W, 6, device=DEVICE),
        sensor_model_parameters=[_opencv_camera_params()],
        poses_tquat_startend=pose_tquat.unsqueeze(1).repeat(1, 2, 1),
        timestamps_startend_us=timestamps,
        timestamps_startend_us_cpu=timestamps.cpu(),
    )


@pytest.fixture(scope="module")
def camera_gaussian_parameters() -> dict[str, torch.Tensor]:
    """A small cloud of tile-local gaussians spread over the whole frame.

    Scales are kept small (well under one tile at this focal length) so that a checkerboard
    mask genuinely removes gaussians from the active set -- which is what the visibility
    narrowing assertion needs.
    """
    torch.manual_seed(0)
    n = 1024
    positions = torch.empty(n, 3, device=DEVICE)
    positions[:, 0].uniform_(-1.1, 1.1)
    positions[:, 1].uniform_(-1.1, 1.1)
    positions[:, 2].uniform_(2.0, 4.0)
    rotations = nn.functional.normalize(torch.randn(n, 4, device=DEVICE), dim=-1)
    scales = torch.full((n, 3), 0.05, device=DEVICE)
    densities = torch.full((n,), 0.9, device=DEVICE)
    features = torch.rand(n, 3, device=DEVICE)
    return {
        "positions": positions,
        "rotations": rotations,
        "scales": scales,
        "densities": densities,
        "features": features,
        "extra_signal": torch.zeros(n, 0, device=DEVICE),
        "camera_extra_signal": torch.zeros(n, 0, device=DEVICE),
    }


def _render(renderer, rendering_data, gaussian_parameters, mask: torch.Tensor | None):
    data = dataclasses.replace(rendering_data, active_tile_mask=mask)
    return renderer.render(
        data,
        gaussian_parameters,
        n_active_features=0,
        extra_ray_signal_infos=([], [], []),
        frame_meta=None,
    )


def _pixel_mask(tile_mask: torch.Tensor, tile_h: int, tile_w: int) -> torch.Tensor:
    """Expand a (tiles_y, tiles_x) tile mask to a (H*W,) per-pixel mask."""
    per_pixel = tile_mask.repeat_interleave(tile_h, dim=0).repeat_interleave(tile_w, dim=1)
    return per_pixel[:CAM_H, :CAM_W].reshape(-1)


@pytest.fixture(scope="module")
def masked_render_results(mock_model, camera_rendering_data, camera_gaussian_parameters):
    """Render the same tiny scene four ways: no mask, all-True, all-False, checkerboard.

    Grouped into one fixture so the four renders (and the shared renderer) happen once; each
    assertion below is still an independent test.
    """
    renderer = GSplatRenderer(_make_config("3dgut-gsplat"), mock_model)

    # The mask MUST be built on the grid the renderer reports, not on a guessed tile size --
    # that accessor is the whole contract from group 1.
    tile_h, tile_w = renderer.active_tile_mask_tile_size(CAM_W, CAM_H)
    tiles_y = (CAM_H + tile_h - 1) // tile_h
    tiles_x = (CAM_W + tile_w - 1) // tile_w
    assert tiles_y > 1 and tiles_x > 1, "need a non-trivial tile grid for the checkerboard case"

    all_true = torch.ones(1, tiles_y, tiles_x, dtype=torch.bool, device=DEVICE)
    all_false = torch.zeros(1, tiles_y, tiles_x, dtype=torch.bool, device=DEVICE)
    ys = torch.arange(tiles_y, device=DEVICE).unsqueeze(1)
    xs = torch.arange(tiles_x, device=DEVICE).unsqueeze(0)
    checker = ((ys + xs) % 2 == 0).unsqueeze(0)

    results = {
        "no_mask": _render(renderer, camera_rendering_data, camera_gaussian_parameters, None),
        "all_true": _render(renderer, camera_rendering_data, camera_gaussian_parameters, all_true),
        "all_false": _render(renderer, camera_rendering_data, camera_gaussian_parameters, all_false),
        "checker": _render(renderer, camera_rendering_data, camera_gaussian_parameters, checker),
    }
    return results, checker[0], (tile_h, tile_w)


def test_scene_is_non_degenerate(masked_render_results):
    """Guard the guards: an empty render would make every mask assertion below pass trivially."""
    results, checker, (tile_h, tile_w) = masked_render_results
    no_mask = results["no_mask"]
    assert no_mask.opacity.max() > 0.1, "scene rendered nothing; the mask assertions would be vacuous"
    # Both halves of the checkerboard must carry signal, otherwise the per-tile mix is untested.
    active_pixels = _pixel_mask(checker, tile_h, tile_w)
    assert no_mask.opacity[active_pixels].max() > 0.0
    assert no_mask.opacity[~active_pixels].max() > 0.0


def test_all_true_mask_matches_the_unmasked_render(masked_render_results):
    """An all-active mask must be a no-op: same pixels as rendering without a mask at all."""
    results, _, _ = masked_render_results
    no_mask, all_true = results["no_mask"], results["all_true"]
    torch.testing.assert_close(all_true.rgb, no_mask.rgb, rtol=0, atol=0)
    torch.testing.assert_close(all_true.opacity, no_mask.opacity, rtol=0, atol=0)
    torch.testing.assert_close(all_true.distance, no_mask.distance, rtol=0, atol=0)


def test_all_false_mask_yields_the_background_only_render(masked_render_results):
    """An all-inactive mask must contribute no gaussian anywhere.

    The renderer passes no ``backgrounds=`` to gsplat, so the background-only render is
    identically zero in colour, alpha and depth.
    """
    results, _, _ = masked_render_results
    all_false = results["all_false"]
    assert torch.equal(all_false.rgb, torch.zeros_like(all_false.rgb)), "masked-out tiles still received colour"
    assert torch.equal(all_false.opacity, torch.zeros_like(all_false.opacity)), "masked-out tiles accumulated alpha"
    assert torch.equal(all_false.distance, torch.zeros_like(all_false.distance)), "masked-out tiles accumulated depth"


def test_checkerboard_mask_splits_pixels_per_tile(masked_render_results):
    """Per-tile: active tiles must be pixel-identical to the unmasked render, inactive tiles
    identical to the background-only render. This is what pins the mask's tile grid to
    gsplat's -- an off-by-one grid would smear the boundary."""
    results, checker, (tile_h, tile_w) = masked_render_results
    no_mask, checker_result = results["no_mask"], results["checker"]
    active = _pixel_mask(checker, tile_h, tile_w)

    torch.testing.assert_close(checker_result.rgb[active], no_mask.rgb[active], rtol=0, atol=0)
    torch.testing.assert_close(checker_result.opacity[active], no_mask.opacity[active], rtol=0, atol=0)
    torch.testing.assert_close(checker_result.distance[active], no_mask.distance[active], rtol=0, atol=0)

    inactive = ~active
    assert torch.equal(checker_result.rgb[inactive], torch.zeros_like(checker_result.rgb[inactive]))
    assert torch.equal(checker_result.opacity[inactive], torch.zeros_like(checker_result.opacity[inactive]))
    assert torch.equal(checker_result.distance[inactive], torch.zeros_like(checker_result.distance[inactive]))


def test_checkerboard_mask_narrows_visibility_but_not_frustum_visibility(masked_render_results):
    """Active tiling narrows ``visibility`` to gaussians that hit an active tile (so SelectiveAdam
    can skip the rest) while ``frustum_visibility`` keeps the pre-mask projection result (so MCMC
    invisible-step counting is unaffected)."""
    results, _, _ = masked_render_results
    no_mask, checker_result = results["no_mask"], results["checker"]

    assert no_mask.visibility is not None
    assert no_mask.frustum_visibility is None, "no mask -> nothing to narrow, so no separate frustum visibility"
    assert checker_result.visibility is not None
    assert checker_result.frustum_visibility is not None, (
        "a mask must produce the pre-mask frustum visibility alongside the narrowed visibility"
    )

    torch.testing.assert_close(checker_result.frustum_visibility, no_mask.visibility, rtol=0, atol=0)

    narrowed = int((checker_result.visibility > 0).sum())
    frustum = int((checker_result.frustum_visibility > 0).sum())
    assert narrowed < frustum, (
        f"checkerboard mask did not narrow visibility ({narrowed} vs {frustum}); "
        "mask-aware visibility (SelectiveAdam skipping) is not being applied"
    )
    # Narrowing only, never widening.
    assert torch.all((checker_result.visibility > 0) <= (checker_result.frustum_visibility > 0))


def test_packed_mode_honors_the_mask_but_degrades_visibility_with_one_warning(
    mock_model, camera_rendering_data, camera_gaussian_parameters, caplog
):
    """``packed=True``: rendering still honors the mask, but mask-aware visibility is gone.

    Under packed mode gsplat's ``tiles_per_gauss`` is ``(nnz,)`` rather than per-Gaussian, so
    the narrowing block cannot AND it into ``visibility``. That degradation must be (a)
    invisible in the pixels -- gsplat applies the mask at intersection either way -- (b)
    visible in ``visibility``/``frustum_visibility``, which stay at the unnarrowed projection
    result, and (c) announced exactly once, not once per render. Uses ``3dgs-gsplat``
    because the 3dgut path forces packed off.
    """
    renderer = GSplatRenderer(_make_config("3dgs-gsplat", packed=True), mock_model)
    assert renderer.config.packed, "3dgs-gsplat must keep packed=True; only the 3dgut path forces it off"

    tile_h, tile_w = renderer.active_tile_mask_tile_size(CAM_W, CAM_H)
    tiles_y = (CAM_H + tile_h - 1) // tile_h
    tiles_x = (CAM_W + tile_w - 1) // tile_w
    assert tiles_y > 1 and tiles_x > 1, "need a non-trivial tile grid for the checkerboard case"
    ys = torch.arange(tiles_y, device=DEVICE).unsqueeze(1)
    xs = torch.arange(tiles_x, device=DEVICE).unsqueeze(0)
    checker = ((ys + xs) % 2 == 0).unsqueeze(0)

    no_mask = _render(renderer, camera_rendering_data, camera_gaussian_parameters, None)
    assert no_mask.opacity.max() > 0.1, "scene rendered nothing; every assertion below would be vacuous"

    # Two masked renders: the second one is what proves the warning is latched, not repeated.
    with caplog.at_level(logging.WARNING, logger=GSplatRenderer.__module__):
        first = _render(renderer, camera_rendering_data, camera_gaussian_parameters, checker)
        second = _render(renderer, camera_rendering_data, camera_gaussian_parameters, checker)

    # (a) The mask is still honored at render time -- the degradation is visibility-only.
    active = _pixel_mask(checker[0], tile_h, tile_w)
    assert float(no_mask.opacity[~active].max()) > 0.0, "inactive half carried no signal; (a) would be vacuous"
    torch.testing.assert_close(first.rgb[active], no_mask.rgb[active], rtol=0, atol=0)
    assert torch.equal(first.rgb[~active], torch.zeros_like(first.rgb[~active])), (
        "packed mode must still skip inactive tiles"
    )
    assert torch.equal(first.opacity[~active], torch.zeros_like(first.opacity[~active]))

    # (b) visibility is NOT narrowed, and no pre-mask copy is advertised.
    assert first.visibility is not None
    assert first.frustum_visibility is None, (
        "packed mode cannot narrow visibility, so it must not advertise a separate pre-mask copy"
    )
    assert torch.equal(first.visibility, no_mask.visibility), (
        "packed-mode visibility must stay exactly the unnarrowed projection result"
    )
    assert second.frustum_visibility is None

    # (c) The degradation is announced exactly once, however many masked renders follow.
    packed_warnings = [record for record in caplog.records if "runs with packed=True" in record.getMessage()]
    assert len(packed_warnings) == 1, (
        f"expected exactly one packed-mode active-tile-mask warning across two masked renders, "
        f"got {len(packed_warnings)}: {[record.getMessage() for record in packed_warnings]}"
    )
    assert renderer._warned_packed_active_tile_mask, "the one-shot latch must be set after warning"


def test_lidar_end_to_end_mask_honoring_is_not_covered():
    """Documents a deliberate gap rather than leaving it silent.

    The lidar tile *geometry* is pinned by the ``compute_lidar_element_to_tile`` tests above and
    the mask/render grid agreement by the ``lidar_tiling_params`` structural pin. A lidar
    end-to-end mask render is not added here because no existing fixture drives
    ``GSplatRenderer.render()`` on the lidar path -- ``gsplat_lidar_test`` only exercises the
    standalone tiling helpers -- so it would mean building a new parallel lidar render harness
    (sensor poses, per-element rays, a 128 x n_bins_azimuth grid) rather than reusing one.
    """
    pytest.skip("lidar end-to-end mask render intentionally out of scope; see docstring")
