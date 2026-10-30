# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Equivalence tests for gsplat lidar functions against vren/ncore references.

Validates that gsplat's compute_angles_to_columns_map() and compute_tiling()
produce identical results to the vren/ncore implementations, treating
vren/ncore as the reference (ground truth).
"""

import json

from pathlib import Path

import gsplat
import pytest
import torch

from omegaconf import DictConfig, OmegaConf

import ncore.data

from ncore.impl.sensors.lidar import RowOffsetStructuredSpinningLidarModel
from nurec.config.parse import parse_untyped_config
from nurec.libs.vren.lidars import preprocess_lidar


DEVICE = torch.device("cuda")
_TESTDATA_CONFIG_DIR = Path(__file__).resolve().parent / "testdata"
RESOLUTION_FACTOR = 4
TILING_N_BINS_ELEVATION = 16
TILING_MAX_PTS_PER_TILE = 16 * 16
TILING_RESOLUTION_ELEVATION = 1600
TILING_DENSIFICATION_FACTOR_AZIMUTH = 8


def _load_lidar_models_config() -> DictConfig:
    """Load the sensor config YAML via parse_untyped_config (resolves ${load_json:...} automatically)."""
    config = parse_untyped_config(str(_TESTDATA_CONFIG_DIR / "sensor/lidar_model"), hydra_args=[])
    # An absolute config path is parent-rooted (flat package), so the lidar models
    # land at .lidar_models rather than .sensor.lidar_models.
    return config.lidar_models


def _ncore_params_from_config(lidar_config) -> ncore.data.RowOffsetStructuredSpinningLidarModelParameters:
    """Convert an OmegaConf dict (from YAML) to ncore lidar model parameters."""
    plain = OmegaConf.to_container(lidar_config, resolve=True)
    return ncore.data.RowOffsetStructuredSpinningLidarModelParameters.from_json(json.dumps(plain))


def _gsplat_params_from_ncore(
    ncore_params: ncore.data.RowOffsetStructuredSpinningLidarModelParameters,
) -> gsplat.RowOffsetStructuredSpinningLidarModelParameters:
    """Build gsplat lidar model parameters from ncore parameters."""
    spinning_direction = (
        gsplat.SpinningDirection.CLOCKWISE
        if ncore_params.spinning_direction == "cw"
        else gsplat.SpinningDirection.COUNTER_CLOCKWISE
    )
    return gsplat.RowOffsetStructuredSpinningLidarModelParameters(
        row_elevations_rad=torch.tensor(ncore_params.row_elevations_rad, dtype=torch.float32, device=DEVICE),
        column_azimuths_rad=torch.tensor(ncore_params.column_azimuths_rad, dtype=torch.float32, device=DEVICE),
        row_azimuth_offsets_rad=torch.tensor(ncore_params.row_azimuth_offsets_rad, dtype=torch.float32, device=DEVICE),
        spinning_frequency_hz=ncore_params.spinning_frequency_hz,
        spinning_direction=spinning_direction,
    )


@pytest.fixture(scope="module")
def lidar_models_config():
    config = _load_lidar_models_config()
    return config


@pytest.fixture(
    scope="module",
    params=["HESAI_Pandar128", "HESAI_AT128"],
)
def sensor_params(request, lidar_models_config):
    """Yields (ncore_params, gsplat_params) for each sensor."""
    sensor_name = request.param
    lidar_config = lidar_models_config[sensor_name]
    ncore_params = _ncore_params_from_config(lidar_config)
    gsplat_params = _gsplat_params_from_ncore(ncore_params)
    return ncore_params, gsplat_params


class TestAnglestoColumnsMap:
    """Test equivalence of angles_to_columns_map between ncore and gsplat."""

    def test_equivalence(self, sensor_params):
        ncore_params, gsplat_params = sensor_params

        # ncore reference
        ncore_model = RowOffsetStructuredSpinningLidarModel(
            ncore_params,
            angles_to_columns_map_init=True,
            angles_to_columns_map_resolution_factor=RESOLUTION_FACTOR,
            angles_to_columns_map_dtype=torch.int32,
            device=DEVICE,
        )
        ncore_map = ncore_model.angles_to_columns_map.int().to(DEVICE)

        # gsplat
        gsplat_map = gsplat.compute_lidar_angles_to_columns_map(
            gsplat_params, resolution_factor=RESOLUTION_FACTOR, dtype=torch.int32
        )

        assert ncore_map.shape == gsplat_map.shape, (
            f"Shape mismatch: ncore {ncore_map.shape} vs gsplat {gsplat_map.shape}"
        )

        assert torch.equal(ncore_map, gsplat_map), (
            f"angles_to_columns_map mismatch: "
            f"max diff = {(ncore_map - gsplat_map).abs().max().item()}, "
            f"num mismatches = {(ncore_map != gsplat_map).sum().item()} / {ncore_map.numel()}"
        )


class TestLidarTiling:
    """Test equivalence of tiling structures between vren and gsplat."""

    def test_equivalence(self, sensor_params):
        ncore_params, gsplat_params = sensor_params

        # vren/ncore reference
        vren_preprocessed = preprocess_lidar(
            ncore_params,
            n_bins_elevation=TILING_N_BINS_ELEVATION,
            max_pts_per_tile=TILING_MAX_PTS_PER_TILE,
            resolution_elevation=TILING_RESOLUTION_ELEVATION,
            densification_factor_azimuth=TILING_DENSIFICATION_FACTOR_AZIMUTH,
            device=DEVICE,
        )
        vren_tiling = vren_preprocessed._tiling

        # gsplat
        gsplat_tiling = gsplat.compute_lidar_tiling(
            gsplat_params,
            n_bins_elevation=TILING_N_BINS_ELEVATION,
            max_pts_per_tile=TILING_MAX_PTS_PER_TILE,
            resolution_elevation=TILING_RESOLUTION_ELEVATION,
            densification_factor_azimuth=TILING_DENSIFICATION_FACTOR_AZIMUTH,
        )

        # Compare scalar fields
        assert vren_tiling.n_bins_azimuth == gsplat_tiling.n_bins_azimuth, (
            f"n_bins_azimuth mismatch: vren={vren_tiling.n_bins_azimuth} vs gsplat={gsplat_tiling.n_bins_azimuth}"
        )
        assert vren_tiling.n_bins_elevation == gsplat_tiling.n_bins_elevation, (
            f"n_bins_elevation mismatch: vren={vren_tiling.n_bins_elevation} vs gsplat={gsplat_tiling.n_bins_elevation}"
        )

        # Compare tensor fields
        assert torch.equal(vren_tiling.cdf_elevation.cpu(), gsplat_tiling.cdf_elevation.cpu()), (
            f"cdf_elevation mismatch: max diff = {(vren_tiling.cdf_elevation.cpu() - gsplat_tiling.cdf_elevation.cpu()).abs().max().item()}"
        )

        # Adapt vren tensors to gsplat's convention before comparison:
        # - cdf_dense_ray_mask: vren (azimuth, elevation) → gsplat (elevation, azimuth)
        # - tiles_pack_info: vren tile_id = az*n_el+el → gsplat tile_id = el*n_az+az
        # - tiles_to_elements_map: vren (elevation, azimuth) → gsplat (azimuth, elevation)
        n_az = gsplat_tiling.n_bins_azimuth
        n_el = gsplat_tiling.n_bins_elevation
        vren_mask = vren_tiling.cdf_dense_ray_mask.cpu().T
        # Reorder tiles_pack_info from vren's tile ordering (az*n_el+el) to gsplat's (el*n_az+az).
        # Only the counts are meaningful after reordering; recompute offsets from the reordered counts.
        vren_pack_raw = vren_tiling.tiles_pack_info.cpu()
        reordered_counts = vren_pack_raw[:, 1].reshape(n_az, n_el).T.reshape(-1)
        reordered_offsets = torch.zeros_like(reordered_counts)
        reordered_offsets[1:] = reordered_counts[:-1].cumsum(0)
        vren_pack = torch.stack([reordered_offsets, reordered_counts], dim=-1)
        # Reorder tiles_to_elements_map: gather elements per tile in gsplat tile order,
        # and swap columns from vren (elevation, azimuth) to gsplat (azimuth, elevation).
        vren_elems_raw = vren_tiling.tiles_to_elements_map.cpu()
        vren_elems_list = []
        for gsplat_tid in range(n_az * n_el):
            el, az = gsplat_tid // n_az, gsplat_tid % n_az
            vren_tid = az * n_el + el
            off = vren_pack_raw[vren_tid, 0].item()
            cnt = vren_pack_raw[vren_tid, 1].item()
            vren_elems_list.append(vren_elems_raw[off : off + cnt, [1, 0]])  # swap (el,az) → (az,el)
        vren_elems = torch.cat(vren_elems_list, dim=0)

        gsplat_mask = gsplat_tiling.cdf_dense_ray_mask.cpu()
        gsplat_pack = gsplat_tiling.tiles_pack_info.cpu()
        gsplat_elems = gsplat_tiling.tiles_to_elements_map.cpu()

        # For the dense ray mask integral image, the total ray count (bottom-right corner)
        # must match exactly. The interior can differ by up to n_rows at 360-degree FOV
        # wrap-around boundaries due to float32/float64 modulo precision in
        # the angle-to-bin quantization between vren C++ and gsplat Python.
        assert vren_mask.shape == gsplat_mask.shape, (
            f"cdf_dense_ray_mask shape mismatch: vren={vren_mask.shape} vs gsplat={gsplat_mask.shape}"
        )
        # For 360-degree FOV sensors, float32/float64 modulo precision differences
        # between vren C++ and gsplat Python cause up to n_rows rays at the azimuth
        # wrap-around boundary to be assigned to the first vs last azimuth bin.
        # This affects the integral image by up to n_rows * n_azimuth_bins.
        is_full_rotation = abs(gsplat_params.fov_horiz_rad.span - 2 * torch.pi) < 0.01
        if is_full_rotation:
            mask_diff = (vren_mask - gsplat_mask).abs()
            max_mask_diff = mask_diff.max().item()
            max_allowed = ncore_params.n_rows * gsplat_tiling.n_bins_azimuth
            assert max_mask_diff <= max_allowed, (
                f"cdf_dense_ray_mask mismatch exceeds tolerance: max diff = {max_mask_diff} > {max_allowed}"
            )
        else:
            assert torch.equal(vren_mask, gsplat_mask), (
                f"cdf_dense_ray_mask mismatch: max diff = {(vren_mask - gsplat_mask).abs().max().item()}"
            )

        # gsplat!536 stopped folding the closing azimuth edge into the first bin: the last
        # column normalizes to exactly `n_bins_azimuth`, which the modulo wrapped to bin 0.
        # vren still folds. The two therefore disagree by those edge rays alone, moving from
        # the first azimuth column to the last; every interior tile must still agree exactly.
        pack_diff = (vren_pack[:, 1] - gsplat_pack[:, 1]).reshape(n_el, n_az)
        assert not pack_diff[:, 1:-1].any(), (
            f"tiles_pack_info differs away from the closing azimuth edge: {pack_diff.nonzero()}"
        )
        assert torch.equal(pack_diff[:, 0], -pack_diff[:, -1]), (
            "rays left the first azimuth column without landing in the last: "
            f"first={pack_diff[:, 0].tolist()} last={pack_diff[:, -1].tolist()}"
        )
        assert pack_diff[:, 0].min() >= 0, "gsplat put rays into the first azimuth column"
        # Every row of tiles_to_elements_map is a tile coordinate pair, so its content is
        # already pinned by the per-tile counts above; only the total needs checking here.
        assert vren_elems.shape == gsplat_elems.shape, (
            f"tiles_to_elements_map size mismatch: {vren_elems.shape} vs {gsplat_elems.shape}"
        )


class TestLidarSparseSegments:
    """Full-coverage sparse segments on the lidar render path must be a no-op.

    NuRec's segments eligibility is not gated on the sensor, so the lidar renderer can
    take the segments path. v0 segments are a full-coverage draw list, which is the
    dense set in a different wrapper, so the rendered distance and opacity must match
    the dense render exactly.
    """

    _N = 4_096

    @staticmethod
    def _config(sparse_segments: bool, use_rays: bool):
        from nurec.config.model import GSplatRendererConfig

        return GSplatRendererConfig.model_validate(
            {
                "name": "3dgut-gsplat",
                "use_rays": use_rays,
                "sparse_segments": sparse_segments,
                "background": {"color": 0.0},
                "culling": {"near_clip_distance": 0.01, "far_clip_distance_lidar": 1000.0},
                "projection": {"min_projected_ray_radius": 0.5477},
                "rasterize_mode": "classic",
                "packed": False,
                "sparse_grad": False,
                "absgrad": False,
                # Mirrors production: the lidar render is depth-only, which is also
                # the segments branch with no colour assembly.
                "outputs": {
                    "lidar": {
                        "enable_features": False,
                        "enable_normals": False,
                        "enable_distance": True,
                        "enable_extended_features": False,
                        "enable_sensor_features": False,
                    }
                },
            }
        )

    @staticmethod
    def _mock_model():
        import torch.nn as nn

        class MockModel(nn.Module):
            def serialize_to_json_dict(self, with_state_dict=False):
                return {"type": "mock_model", "config": {}}

        return MockModel()

    def _rendering_data(self, ncore_params, n_rows, n_columns, use_rays):
        from nurec.utils.batch import RenderingData

        # Gaussians sit in a shell about the sensor, which spins at the origin.
        rays = torch.zeros(1, n_rows, n_columns, 6, device=DEVICE)
        rays[..., 3] = 1.0  # unit +x direction; only read when use_rays is set
        pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).unsqueeze(1).repeat(1, 2, 1)
        timestamps = torch.tensor([[0, 100000]], device=DEVICE, dtype=torch.int64)
        return RenderingData(
            rays=rays if use_rays else torch.zeros(1, 1, 1, 6, device=DEVICE),
            sensor_model_parameters=[ncore_params],
            poses_tquat_startend=pose.to(DEVICE),
            timestamps_startend_us=timestamps,
            timestamps_startend_us_cpu=timestamps.cpu(),
            rays_timestamps_us=None,
            _rays_footprints=None,
        )

    def _gaussian_parameters(self):
        g = torch.Generator(device=DEVICE).manual_seed(7)
        n = self._N
        directions = torch.nn.functional.normalize(torch.randn(n, 3, generator=g, device=DEVICE), dim=-1)
        radii = 5.0 + 15.0 * torch.rand(n, 1, generator=g, device=DEVICE)
        return {
            "positions": directions * radii,
            "rotations": torch.nn.functional.normalize(torch.randn(n, 4, generator=g, device=DEVICE), dim=-1),
            "scales": torch.rand(n, 3, generator=g, device=DEVICE) * 0.15 + 0.05,
            "densities": torch.rand(n, generator=g, device=DEVICE),
            "features": torch.rand(n, 3, generator=g, device=DEVICE),
            "extra_signal": torch.zeros(n, 0, device=DEVICE),
            "camera_extra_signal": torch.zeros(n, 0, device=DEVICE),
            "lidar_extra_signal": torch.zeros(n, 0, device=DEVICE),
        }

    @staticmethod
    def _lidar_grid(ncore_params):
        """The lidar grid the renderer will derive, so `rays` can be sized to match.

        Mirrors GSplatRenderer._build_lidar_params, including fov_eps_factor and the
        LidarTilingConfig defaults; a mismatch here would show up as a rays shape
        error rather than a parity failure.
        """
        params = _gsplat_params_from_ncore(ncore_params)
        params = gsplat.RowOffsetStructuredSpinningLidarModelParameters(
            row_elevations_rad=params.row_elevations_rad,
            column_azimuths_rad=params.column_azimuths_rad,
            row_azimuth_offsets_rad=params.row_azimuth_offsets_rad,
            spinning_frequency_hz=params.spinning_frequency_hz,
            spinning_direction=params.spinning_direction,
            fov_eps_factor=4,
        )
        context = gsplat.RowOffsetStructuredSpinningLidarFrameContext(
            params,
            angles_to_columns_map=gsplat.compute_lidar_angles_to_columns_map(params),
            tiling=gsplat.compute_lidar_tiling(
                params,
                n_bins_elevation=16,
                max_pts_per_tile=64,
                resolution_elevation=1600,
                densification_factor_azimuth=8,
            ),
        )
        # The renderer sizes the image with `height, width = render_shape`. A
        # (width, height) convention would transpose every lidar render and
        # still satisfy a test that read n_rows / n_columns directly, so pin
        # the order here -- these sensors have n_rows != n_columns.
        parameters = context.parameters
        assert parameters.n_rows != parameters.n_columns, "fixture must not be square"
        assert context.render_shape == (parameters.n_rows, parameters.n_columns)
        return context.render_shape

    @pytest.mark.parametrize("use_rays", [False, True])
    def test_full_coverage_segments_match_the_dense_lidar_render(self, sensor_params, use_rays):
        from nurec.models.gaussians.renderers import GSplatRenderer
        from nurec.models.gaussians.segment_plan import SegmentPlan

        ncore_params, _gsplat_params = sensor_params
        n_rows, n_columns = self._lidar_grid(ncore_params)
        rendering_data = self._rendering_data(ncore_params, n_rows, n_columns, use_rays)
        gaussian_parameters = self._gaussian_parameters()
        model = self._mock_model()

        # Shape [2, 1]: one start row and one row count cover the full scene.
        full_plan = SegmentPlan(
            segments=torch.tensor([[0], [self._N]], dtype=torch.int32, device=DEVICE),
            segment_offsets=torch.tensor([0, self._N], dtype=torch.int32, device=DEVICE),
            active_count=self._N,
            total_rows=self._N,
        )

        # Key False stores the dense result. Key True stores the segment result.
        outputs = {}
        for sparse_segments in (False, True):
            renderer = GSplatRenderer(self._config(sparse_segments, use_rays), model)
            with torch.no_grad():
                outputs[sparse_segments] = renderer.render(
                    rendering_data,
                    gaussian_parameters,
                    n_active_features=0,
                    extra_ray_signal_infos=([], [], []),
                    segment_plan=(full_plan if sparse_segments else None),
                )

        dense, segments = outputs[False], outputs[True]
        # Non-vacuity: parity over an empty render would assert nothing.
        assert segments.opacity.max().item() > 0.0, "lidar scene rendered empty; parity is vacuous"
        assert torch.equal(dense.opacity, segments.opacity), "segments changed the lidar opacity"
        assert torch.equal(dense.distance, segments.distance), "segments changed the lidar distance"
        assert segments.visibility is None, "active-space visibility must not escape as scene metadata"
