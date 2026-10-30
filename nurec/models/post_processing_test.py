# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import pytest
import torch

from omegaconf import DictConfig

from nurec.config.model import DINOv2PEPostProcessingConfig
from nurec.config.trainer import TrainerConfig
from nurec.models.post_processing import DinoV2PEPostProcessing, PPISPPostProcessing
from nurec.utils.batch import CameraFrameLabels, DataAndRenderingBatch, DataBatch, FrameMeta, RenderingBatch
from nurec.utils.types import ExtraSignal, GaussiansCompositeReturn, GaussiansRenderReturn


class MockTrainerConfig(TrainerConfig):
    def __init__(self):
        super().__init__(
            max_epochs=1,
            check_val_every_n_epoch=1,
            precision="32",
            log_every_n_steps=1,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )


def _make_pp(*, test_no_pe_map: bool, n_dims: int = 8) -> DinoV2PEPostProcessing:
    cfg = DictConfig(
        {
            "name": "dinov2-pe-post-processing",
            "test_no_pe_map": test_no_pe_map,
            "pe_height": 4,
            "pe_width": 4,
            "n_dims": n_dims,
        }
    )
    return DinoV2PEPostProcessing(
        DINOv2PEPostProcessingConfig.model_validate(cfg),
        MockTrainerConfig(),
        n_training_frames_per_camera=[1],
        n_training_frames_per_lidar=[1],
    )


def _make_render_results(extras: ExtraSignal | None) -> GaussiansCompositeReturn:
    h, w = 4, 4
    return GaussiansCompositeReturn(
        rendered_cam=GaussiansRenderReturn(
            rgb=torch.zeros((h, w, 3)),
            opacity=torch.ones((h, w)),
            distance=torch.zeros((h, w)),
            extra_ray_signals=extras,
        ),
    )


def _make_camera_data_batch() -> DataBatch:
    return DataBatch(
        camera=DataBatch.Camera(
            meta=[FrameMeta(unique_sensor_idx=0, unique_frame_idx=0, training_frame_idx=0)],
            labels=CameraFrameLabels(),
        )
    )


def test_dinov2_pe_post_processing_asserts_when_slot_missing_during_training() -> None:
    """Training: missing dinov2_feats is a real misconfiguration — fail loudly."""
    pp = _make_pp(test_no_pe_map=False)
    pp.train()
    with pytest.raises(AssertionError, match="during training"):
        pp.forward(_make_render_results(extras=ExtraSignal(dinov2_feats=None)), rays=torch.zeros((16, 6)))


def test_dinov2_pe_post_processing_does_not_raise_at_inference_when_slot_missing() -> None:
    """Inference: missing dinov2_feats is legitimate (the override defaults
    disabled the gate that produces it). The training assert must NOT fire.
    """
    pp = _make_pp(test_no_pe_map=False)
    pp.eval()
    pp.forward(_make_render_results(extras=ExtraSignal(dinov2_feats=None)), rays=torch.zeros((16, 6)))


def test_dinov2_pe_post_processing_modifies_dinov2_feats_during_training() -> None:
    """Positive path: with a populated dinov2_feats slot during training, forward
    must run the PE addition and actually modify the slot. Catches a regression
    that turns the PE step into a silent no-op.
    """
    pp = _make_pp(test_no_pe_map=False, n_dims=8)
    pp.train()

    n_rays = 16
    dinov2_feats = torch.zeros((n_rays, 8))
    initial = dinov2_feats.clone()
    results = _make_render_results(extras=ExtraSignal(dinov2_feats=dinov2_feats))

    pp.forward(
        results,
        rays=torch.zeros((n_rays, 6)),
        pixel_coords=torch.zeros((n_rays, 2)),
        resolution=(4, 4),
    )

    rendered_cam = results.rendered_cam
    assert rendered_cam is not None
    extras_after = rendered_cam.extra_ray_signals
    assert extras_after is not None
    dinov2_feats_after = extras_after.dinov2_feats
    assert dinov2_feats_after is not None
    assert not torch.equal(dinov2_feats_after, initial)


@pytest.mark.parametrize(
    ("batch", "match"),
    [
        (
            DataAndRenderingBatch(data=_make_camera_data_batch(), rendering=None),
            "requires batch.rendering",
        ),
        (
            DataAndRenderingBatch(data=_make_camera_data_batch(), rendering=RenderingBatch(camera=None)),
            "requires batch.rendering.camera",
        ),
        (
            DataAndRenderingBatch(
                data=DataBatch(camera=None),
                rendering=RenderingBatch(camera=cast(Any, object())),
            ),
            "requires batch.data.camera",
        ),
    ],
)
def test_ppisp_forward_rendered_batch_asserts_when_camera_batch_invariant_is_missing(
    batch: DataAndRenderingBatch, match: str
) -> None:
    ppisp = cast(PPISPPostProcessing, object.__new__(PPISPPostProcessing))

    with pytest.raises(AssertionError, match=match):
        ppisp.forward_rendered_batch(_make_render_results(extras=None), batch)
