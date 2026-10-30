# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard: dataset.lidar_ray_mode must be consistent with the renderer.

The model-free unstructured LiDAR path spans two configs -- dataset.lidar_ray_mode
and model.renderer.lidar_unstructured. They must agree, and only
GSplatRendererConfig supports the unstructured path. The NuRecConfig validator
enforces this and fails loudly on a mismatch. We exercise the validator directly
with lightweight typed stand-ins (building a full NuRecConfig is heavy and
unnecessary -- the check only reads self.dataset.lidar_ray_mode and
self.model.renderer).
"""

from types import SimpleNamespace

import pytest

from nurec.config.model import GSplatRendererConfig
from nurec.config.nurec import NuRecConfig


def _gsplat_renderer(lidar_unstructured: bool, active_tile_mask: dict | None = None) -> GSplatRendererConfig:
    payload: dict = {"name": "3dgut-gsplat", "lidar_unstructured": lidar_unstructured}
    if active_tile_mask is not None:
        payload["active_tile_mask"] = active_tile_mask
    return GSplatRendererConfig.model_validate(payload)


def _run_guard(
    *,
    ray_mode,
    renderer,
    floating_point_filter: bool = False,
    floating_point_filter_neighbor_source: str = "rangeview",
    raydrop_lambda: float = 0.0,
    virtual_rays="off",
    use_real_lidar_rays=False,
    lattice_n_bins_azimuth=None,
    lattice_n_bins_elevation=None,
    fov_vert_deg=None,
    fov_horiz_deg=None,
    lidar_ids: list[str] | None = None,
    train_batch_size: int = 1,
    use_stored_no_returns=False,
) -> None:
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            lidar_ids=lidar_ids if lidar_ids is not None else ["lidar_a"],
            lidar_ray_mode=ray_mode,
            lidar_unstructured_raydrop_virtual_rays=virtual_rays,
            lidar_unstructured_raydrop_use_stored_no_returns=use_stored_no_returns,
            use_real_lidar_rays=use_real_lidar_rays,
            lidar_unstructured_raydrop_rays_per_tile=1,
            lidar_unstructured_raydrop_lattice_time_modes=1,
            lidar_unstructured_raydrop_lattice_max_frames=64,
            lidar_unstructured_raydrop_lattice_n_bins_azimuth=lattice_n_bins_azimuth,
            lidar_unstructured_raydrop_lattice_n_bins_elevation=lattice_n_bins_elevation,
            lidar_unstructured_raydrop_occupancy_n_bins_azimuth=None,
            lidar_unstructured_raydrop_occupancy_n_bins_elevation=None,
            lidar_unstructured_fov_vert_deg=fov_vert_deg,
            lidar_unstructured_fov_horiz_deg=fov_horiz_deg,
        ),
        model=SimpleNamespace(renderer=renderer),
        datamodule=SimpleNamespace(train_batch_size=train_batch_size),
        system=SimpleNamespace(
            test=SimpleNamespace(
                lidar=SimpleNamespace(
                    floating_point_filter=SimpleNamespace(
                        enabled=floating_point_filter,
                        neighbor_source=floating_point_filter_neighbor_source,
                    )
                ),
            ),
        ),
        loss={"raydrop": SimpleNamespace(lambda_=raydrop_lambda)},
    )
    # Pydantic's runtime-callable decorator proxy is marked non-callable in its stubs.
    NuRecConfig._validate_lidar_ray_mode_consistency(cfg)  # type: ignore[operator]


def test_unstructured_requires_lidar_unstructured_renderer():
    with pytest.raises(ValueError, match="requires model.renderer.lidar_unstructured=True"):
        _run_guard(ray_mode="unstructured", renderer=_gsplat_renderer(lidar_unstructured=False))


def test_structured_forbids_lidar_unstructured_renderer():
    with pytest.raises(ValueError, match="requires dataset.lidar_ray_mode='unstructured'"):
        _run_guard(ray_mode="structured", renderer=_gsplat_renderer(lidar_unstructured=True))


def test_unstructured_requires_gsplat_renderer():
    # A non-GSplat renderer (no lidar_unstructured field) cannot serve unstructured.
    nrend_like = SimpleNamespace()  # deliberately not a GSplatRendererConfig
    with pytest.raises(ValueError, match="requires a GSplat renderer"):
        _run_guard(ray_mode="unstructured", renderer=nrend_like)


def test_unstructured_forbids_rangeview_floating_point_filter():
    with pytest.raises(ValueError, match="neighbor_source='rangeview' is not supported"):
        _run_guard(
            ray_mode="unstructured",
            renderer=_gsplat_renderer(lidar_unstructured=True),
            floating_point_filter=True,
            floating_point_filter_neighbor_source="rangeview",
        )


def test_unstructured_allows_angle_floating_point_filter():
    # The sensor-agnostic "angle" source works from ray directions alone.
    _run_guard(
        ray_mode="unstructured",
        renderer=_gsplat_renderer(lidar_unstructured=True),
        floating_point_filter=True,
        floating_point_filter_neighbor_source="angle",
    )


def test_unstructured_forbids_raydrop_loss_without_virtual_rays():
    """Neither stored no-returns nor synthesized virtual rays: the raydrop head would
    train against an all-zero target, which is what this guard exists to prevent."""
    with pytest.raises(ValueError, match=r"requires a source of raydrop negatives"):
        _run_guard(
            ray_mode="unstructured",
            renderer=_gsplat_renderer(lidar_unstructured=True),
            raydrop_lambda=0.1,
            virtual_rays="off",
        )


def test_unstructured_allows_raydrop_loss_with_model_elements():
    _run_guard(
        ray_mode="unstructured",
        renderer=_gsplat_renderer(lidar_unstructured=True),
        raydrop_lambda=0.1,
        virtual_rays="model_elements",
    )


def test_unstructured_allows_raydrop_loss_with_stored_no_returns():
    """Stored no-return emissions satisfy the raydrop-negatives requirement on their
    own, with no synthesized virtual rays."""
    _run_guard(
        ray_mode="unstructured",
        renderer=_gsplat_renderer(lidar_unstructured=True),
        raydrop_lambda=0.1,
        virtual_rays="off",
        use_stored_no_returns=True,
        use_real_lidar_rays=True,
    )


def test_unstructured_allows_stored_no_returns_with_synthesized_rays():
    """The two negative sources are independent and may be combined."""
    _run_guard(
        ray_mode="unstructured",
        renderer=_gsplat_renderer(lidar_unstructured=True),
        raydrop_lambda=0.1,
        virtual_rays="aggregate_lattice",
        use_stored_no_returns=True,
        use_real_lidar_rays=True,
    )


def test_unstructured_allows_raydrop_loss_with_aggregate_lattice():
    _run_guard(
        ray_mode="unstructured",
        renderer=_gsplat_renderer(lidar_unstructured=True),
        raydrop_lambda=0.1,
        virtual_rays="aggregate_lattice",
        use_real_lidar_rays=True,  # aggregate_lattice requires measured world rays
    )


def test_unstructured_aggregate_lattice_requires_measured_rays():
    with pytest.raises(ValueError, match=r"aggregate_lattice.*requires dataset.use_real_lidar_rays=True"):
        _run_guard(
            ray_mode="unstructured",
            renderer=_gsplat_renderer(lidar_unstructured=True),
            raydrop_lambda=0.1,
            virtual_rays="aggregate_lattice",
            use_real_lidar_rays=False,
        )


def test_unstructured_model_elements_forbids_measured_rays():
    with pytest.raises(ValueError, match=r"model_elements.*requires dataset.use_real_lidar_rays=False"):
        _run_guard(
            ray_mode="unstructured",
            renderer=_gsplat_renderer(lidar_unstructured=True),
            raydrop_lambda=0.1,
            virtual_rays="model_elements",
            use_real_lidar_rays=True,
        )


def test_matched_unstructured_ok():
    _run_guard(ray_mode="unstructured", renderer=_gsplat_renderer(lidar_unstructured=True))


def test_matched_structured_ok():
    _run_guard(ray_mode="structured", renderer=_gsplat_renderer(lidar_unstructured=False))


def test_structured_allows_floating_point_filter():
    _run_guard(
        ray_mode="structured",
        renderer=_gsplat_renderer(lidar_unstructured=False),
        floating_point_filter=True,
        floating_point_filter_neighbor_source="rangeview",
    )


def test_structured_allows_raydrop_loss():
    _run_guard(
        ray_mode="structured",
        renderer=_gsplat_renderer(lidar_unstructured=False),
        raydrop_lambda=0.1,
    )


def test_unstructured_lattice_bins_both_or_neither():
    # both set -> OK
    _run_guard(
        ray_mode="unstructured",
        renderer=_gsplat_renderer(lidar_unstructured=True),
        lattice_n_bins_azimuth=3600,
        lattice_n_bins_elevation=256,
    )
    # neither set -> OK (built-in defaults)
    _run_guard(
        ray_mode="unstructured",
        renderer=_gsplat_renderer(lidar_unstructured=True),
    )


def test_unstructured_lattice_bins_partial_errors():
    with pytest.raises(ValueError, match=r"n_bins_azimuth and .*must be set together"):
        _run_guard(
            ray_mode="unstructured",
            renderer=_gsplat_renderer(lidar_unstructured=True),
            lattice_n_bins_azimuth=3600,  # elevation left None
        )


# ---- PER-SENSOR (PerSensor union) cases ----


def test_per_sensor_dict_ray_mode_pairs_with_renderer():
    # A {lidar_id: mode} dict must still pair with a per-sensor lidar_unstructured.
    _run_guard(
        ray_mode={"lidar_a": "unstructured", "lidar_b": "structured"},
        renderer=_gsplat_renderer(lidar_unstructured={"lidar_a": True, "lidar_b": False}),
        lidar_ids=["lidar_a", "lidar_b"],
    )


def test_per_sensor_unstructured_sensor_needs_renderer_true():
    # lidar_a is unstructured but the renderer flag for it is False -> error.
    with pytest.raises(ValueError, match=r"sensor 'lidar_a'.*requires\s+model.renderer.lidar_unstructured=True"):
        _run_guard(
            ray_mode={"lidar_a": "unstructured", "lidar_b": "structured"},
            renderer=_gsplat_renderer(lidar_unstructured={"lidar_a": False, "lidar_b": False}),
            lidar_ids=["lidar_a", "lidar_b"],
        )


def test_per_sensor_dict_covers_all_ids():
    # A per-sensor dict listing every active id resolves per sensor (no fallback key).
    _run_guard(
        ray_mode={"lidar_a": "unstructured", "lidar_b": "structured"},
        renderer=_gsplat_renderer(lidar_unstructured={"lidar_a": True, "lidar_b": False}),
        lidar_ids=["lidar_a", "lidar_b"],
    )


def test_per_sensor_dict_missing_id_errors():
    # lidar_b has no entry -> coverage error (there is no fallback key).
    with pytest.raises(ValueError, match=r"missing entries for \['lidar_b'\]"):
        _run_guard(
            ray_mode={"lidar_a": "unstructured"},
            renderer=_gsplat_renderer(lidar_unstructured=True),
            lidar_ids=["lidar_a", "lidar_b"],
        )


def test_per_sensor_aggregate_lattice_requires_measured_rays():
    # aggregate_lattice for lidar_a with per-sensor use_real_lidar_rays False -> error.
    with pytest.raises(ValueError, match=r"aggregate_lattice.*sensor 'lidar_a'.*use_real_lidar_rays=True"):
        _run_guard(
            ray_mode="unstructured",
            renderer=_gsplat_renderer(lidar_unstructured=True),
            raydrop_lambda=0.1,
            virtual_rays={"lidar_a": "aggregate_lattice"},
            use_real_lidar_rays={"lidar_a": False},
        )


def test_per_sensor_fov_lo_hi_ordering_errors():
    with pytest.raises(ValueError, match=r"fov_vert_deg.*sensor 'lidar_a'.*lo < hi"):
        _run_guard(
            ray_mode="unstructured",
            renderer=_gsplat_renderer(lidar_unstructured=True),
            fov_vert_deg={"lidar_a": [15.0, -25.0]},
        )


def test_per_sensor_valid_mixed_rig_ok():
    # Spinning P128 (K=1) + MEMS (K=2, explicit FOV, model-less) in one rig.
    _run_guard(
        ray_mode={"lidar_p128": "unstructured", "lidar_mems": "unstructured"},
        renderer=_gsplat_renderer(lidar_unstructured={"lidar_p128": True, "lidar_mems": True}),
        raydrop_lambda=0.1,
        virtual_rays={"lidar_p128": "aggregate_lattice", "lidar_mems": "aggregate_lattice"},
        use_real_lidar_rays={"lidar_p128": True, "lidar_mems": True},
        fov_vert_deg={"lidar_mems": [-12.5, 12.5]},
        fov_horiz_deg={"lidar_mems": [-60.0, 60.0]},
        lidar_ids=["lidar_p128", "lidar_mems"],
    )


# --- active tile mask x unstructured -----------------------------------------
#
# The lidar half of VALID_CAMERA_LIDAR_OVERLAP builds its mask from a structured
# spinning-lidar model (build_camera_to_lidar_tile_mapping ->
# compute_lidar_element_to_tile -> reshape(n_rows, n_columns)), which the
# model-free path has no equivalent of. CAMERA_INVALID_ONLY stops before that and
# is therefore safe.


def test_unstructured_forbids_valid_camera_lidar_overlap_mask():
    with pytest.raises(ValueError, match=r"valid_camera_lidar_overlap.*unstructured.*camera_invalid_only"):
        _run_guard(
            ray_mode="unstructured",
            renderer=_gsplat_renderer(
                lidar_unstructured=True,
                active_tile_mask={"enabled": True, "strategy": "valid_camera_lidar_overlap"},
            ),
        )


def test_unstructured_allows_camera_invalid_only_mask():
    # car2sim_lidar selects this strategy, and the unstructured preset inherits it.
    _run_guard(
        ray_mode="unstructured",
        renderer=_gsplat_renderer(
            lidar_unstructured=True,
            active_tile_mask={"enabled": True, "strategy": "camera_invalid_only"},
        ),
    )


def test_unstructured_allows_disabled_mask():
    _run_guard(
        ray_mode="unstructured",
        renderer=_gsplat_renderer(
            lidar_unstructured=True,
            active_tile_mask={"enabled": False, "strategy": "valid_camera_lidar_overlap"},
        ),
    )


def test_structured_allows_valid_camera_lidar_overlap_mask():
    # The guard must not restrict the structured path, which is what the strategy
    # was designed for.
    _run_guard(
        ray_mode="structured",
        renderer=_gsplat_renderer(
            lidar_unstructured=False,
            active_tile_mask={"enabled": True, "strategy": "valid_camera_lidar_overlap"},
        ),
    )


# --- unstructured tiling is single-frame -------------------------------------


def test_unstructured_forbids_train_batch_size_gt_one():
    with pytest.raises(ValueError, match=r"train_batch_size=4.*unstructured.*train_batch_size=1"):
        _run_guard(
            ray_mode="unstructured",
            renderer=_gsplat_renderer(lidar_unstructured=True),
            train_batch_size=4,
        )


def test_unstructured_allows_train_batch_size_one():
    _run_guard(
        ray_mode="unstructured",
        renderer=_gsplat_renderer(lidar_unstructured=True),
        train_batch_size=1,
    )


def test_structured_allows_train_batch_size_gt_one():
    _run_guard(
        ray_mode="structured",
        renderer=_gsplat_renderer(lidar_unstructured=False),
        train_batch_size=8,
    )
