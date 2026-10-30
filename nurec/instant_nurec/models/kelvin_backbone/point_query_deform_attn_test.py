# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

from types import SimpleNamespace

import pytest
import torch

from nurec.instant_nurec.models.kelvin_backbone.point_query_deform_attn import (
    PointQueryDeformAttn,
    apply_bounded_rgb_logit_residual,
    bilinear_sample_context_atlas,
    bilinear_sample_image_points,
    build_context_image_atlas,
    masked_softmax,
    project_points_to_context_views,
)


def test_bounded_rgb_logit_residual_has_exact_zero_baseline_and_finite_gradients() -> None:
    source_rgb = torch.tensor([[[0.0, 0.25, 1.0], [0.1, 0.5, 0.9]]])
    zero_residual = torch.zeros_like(source_rgb)
    corrected_zero = apply_bounded_rgb_logit_residual(
        source_rgb,
        zero_residual,
        max_logit_delta=2.0,
    )
    torch.testing.assert_close(corrected_zero, source_rgb, rtol=0.0, atol=0.0)

    raw_residual = torch.tensor([[[1.0, -1.0, 1.0], [-0.5, 0.5, -0.5]]], requires_grad=True)
    corrected = apply_bounded_rgb_logit_residual(source_rgb, raw_residual, max_logit_delta=2.0)
    assert bool(((corrected >= 0.0) & (corrected <= 1.0)).all())
    assert not torch.equal(corrected, source_rgb)
    corrected.sum().backward()
    assert raw_residual.grad is not None and torch.isfinite(raw_residual.grad).all()


def test_projection_inverts_rendering_pose_and_scatter_preserves_query_order() -> None:
    points = torch.tensor([[[12.0, 0.0, 0.0], [99.0, 0.0, 0.0], [13.0, 0.0, 0.0]]])
    query_valid = torch.tensor([[True, False, True]])
    sensor_to_world = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    poses = sensor_to_world.reshape(1, 1, 1, 7).expand(1, 1, 2, 7).clone()
    timestamps = torch.tensor([[[100, 200]]], dtype=torch.int64)

    def projection_fn(_parameters, candidate_points, world_to_sensor, _timestamps, return_all):
        assert return_all
        torch.testing.assert_close(world_to_sensor[:, 0], torch.tensor([-10.0, -10.0]))
        torch.testing.assert_close(candidate_points[:, 0], torch.tensor([12.0, 13.0]))
        return SimpleNamespace(
            image_points=torch.tensor([[0.5, 0.5], [1.5, 0.5]]),
            valid_indices=torch.tensor([1], dtype=torch.int64),
            timestamps_us=torch.tensor([150], dtype=torch.int64),
            T_world_sensors=world_to_sensor[:1],
        )

    projected = project_points_to_context_views(
        points,
        query_valid,
        [[object()]],
        poses,
        timestamps,
        projection_fn=projection_fn,
    )

    assert projected.valid_mask.tolist() == [[[False, False, True]]]
    torch.testing.assert_close(projected.image_points[0, 0, 2], torch.tensor([1.5, 0.5]))
    torch.testing.assert_close(projected.query_metric_distance[0, 0, 2], torch.tensor(3.0))
    assert projected.timestamps_us[0, 0, 2].item() == 150


def test_bilinear_sampling_uses_exact_image_pixel_centers() -> None:
    image = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    points = torch.tensor([[[[0.5, 0.5], [1.5, 1.5], [1.0, 1.0]]]])
    values, valid = bilinear_sample_image_points(image, points, torch.tensor([[2.0, 2.0]]))

    torch.testing.assert_close(values[0, 0, :, 0], torch.tensor([1.0, 4.0, 2.5]))
    assert valid.all()


def test_packed_context_atlas_matches_per_view_sampling_and_backpropagates() -> None:
    dense_image = torch.randn(2, 3, 4, 5, 2, requires_grad=True)
    atlas_image = dense_image.detach().clone().requires_grad_()
    points = torch.tensor(
        [
            [[0.0, 0.0], [2.5, 1.5]],
            [[4.9, 3.9], [5.0, 2.0]],
            [[1.25, 3.0], [float("nan"), 1.0]],
        ]
    )
    batch_indices = torch.tensor([0, 1, 0], dtype=torch.int64)
    view_indices = torch.tensor([2, 0, 1], dtype=torch.int64)
    resolution = torch.full((2, 3, 2), 1.0)
    resolution[..., 0] = 5.0
    resolution[..., 1] = 4.0

    atlas_values, atlas_valid = bilinear_sample_context_atlas(
        build_context_image_atlas(atlas_image),
        4,
        points,
        batch_indices,
        view_indices,
        resolution,
    )
    dense_values, dense_valid = bilinear_sample_image_points(
        dense_image[batch_indices, view_indices],
        points.unsqueeze(1),
        resolution[batch_indices, view_indices],
    )
    dense_values = dense_values.squeeze(1)
    dense_valid = dense_valid.squeeze(1)

    torch.testing.assert_close(atlas_values, dense_values)
    torch.testing.assert_close(atlas_valid, dense_valid)
    atlas_values.nan_to_num().sum().backward()
    dense_values.nan_to_num().sum().backward()
    torch.testing.assert_close(atlas_image.grad, dense_image.grad)


def test_masked_softmax_is_zero_for_empty_rows_and_normalized_otherwise() -> None:
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    valid = torch.tensor([[False, False], [True, False]])
    weights = masked_softmax(logits, valid, dim=-1)

    torch.testing.assert_close(weights[0], torch.zeros(2))
    torch.testing.assert_close(weights[1], torch.tensor([1.0, 0.0]))


def test_deformable_attention_always_keeps_exact_reference_and_backpropagates() -> None:
    module = PointQueryDeformAttn(
        encoder_dim=8,
        fusion_dim=8,
        num_sampling_points=4,
        offset_radius_feature_px=2.0,
        depth_residual_scale_m=1.0,
        occlusion_abs_tolerance_m=0.5,
        occlusion_rel_tolerance=0.05,
        same_time_tolerance_us=5000,
    )
    queries = torch.randn(1, 3, 8, requires_grad=True)
    offsets = module._sampling_offsets(queries)
    torch.testing.assert_close(offsets[:, :, 0], torch.zeros(1, 3, 2))

    context = torch.randn(1, 2, 2, 2, 8, requires_grad=True)
    output = module.fuse_chunk(
        queries,
        torch.tensor([[100, 100, 100]], dtype=torch.int64),
        torch.tensor([[False, False, False]]),
        torch.tensor([[True, True, True]]),
        build_context_image_atlas(context),
        build_context_image_atlas(torch.ones(1, 2, 2, 2, 1)),
        torch.full((1, 2, 3, 2), 0.5),
        torch.ones(1, 2, 3, dtype=torch.bool),
        torch.full((1, 2, 3), 100, dtype=torch.int64),
        torch.ones(1, 2, 3),
        torch.full((1, 2, 2), 2.0),
    )
    assert output.features.shape == (1, 3, 8)
    assert output.valid_observation_count.tolist() == [[2, 2, 2]]
    assert output.view_valid_query_count.tolist() == [[3, 3]]
    torch.testing.assert_close(output.view_attention_mass.sum(dim=1), torch.tensor([3.0]))
    assert torch.isfinite(output.deformed_sample_attention_mass).all()
    assert torch.isfinite(output.learned_offset_magnitude_sum).all()
    output.features.sum().backward()
    assert queries.grad is not None and torch.isfinite(queries.grad).all()
    assert context.grad is not None and torch.isfinite(context.grad).all()


def test_dynamic_queries_reject_cross_time_views_and_depth_residual_weights_static_views() -> None:
    module = PointQueryDeformAttn(
        encoder_dim=4,
        fusion_dim=4,
        num_sampling_points=1,
        offset_radius_feature_px=0.0,
        depth_residual_scale_m=1.0,
        occlusion_abs_tolerance_m=0.5,
        occlusion_rel_tolerance=0.05,
        same_time_tolerance_us=5000,
    )
    final_view_score = module.view_score[-1]
    assert isinstance(final_view_score, torch.nn.Linear)
    torch.nn.init.zeros_(final_view_score.weight)
    torch.nn.init.zeros_(final_view_score.bias)

    common_args = dict(
        queries=torch.randn(1, 1, 4),
        query_timestamps_us=torch.tensor([[100]], dtype=torch.int64),
        query_valid_mask=torch.tensor([[True]]),
        context_feature_atlas=build_context_image_atlas(torch.ones(1, 2, 2, 2, 4)),
        context_distance_atlas=build_context_image_atlas(
            torch.cat(
                [torch.ones(1, 1, 2, 2, 1), torch.full((1, 1, 2, 2, 1), 10.0)],
                dim=1,
            )
        ),
        reference_image_points=torch.full((1, 2, 1, 2), 0.5),
        reference_valid_mask=torch.ones(1, 2, 1, dtype=torch.bool),
        reference_timestamps_us=torch.tensor([[[100], [20_000]]], dtype=torch.int64),
        query_metric_distance=torch.ones(1, 2, 1),
        full_resolution=torch.full((1, 2, 2), 2.0),
    )

    dynamic = module.fuse_chunk(
        query_dynamic_mask=torch.tensor([[True]]),
        **common_args,
    )
    assert dynamic.valid_observation_count.tolist() == [[1]]
    assert dynamic.view_valid_query_count.tolist() == [[1, 0]]
    torch.testing.assert_close(dynamic.view_attention_mass, torch.tensor([[1.0, 0.0]]))

    static = module.fuse_chunk(
        query_dynamic_mask=torch.tensor([[False]]),
        **common_args,
    )
    assert static.valid_observation_count.tolist() == [[2]]
    assert static.view_attention_mass[0, 0] > static.view_attention_mass[0, 1]


def test_depth_aware_interaction_fusion_uses_exact_samples_and_reports_entropy() -> None:
    torch.manual_seed(7)
    module = PointQueryDeformAttn(
        encoder_dim=4,
        fusion_dim=4,
        num_sampling_points=1,
        offset_radius_feature_px=0.0,
        depth_residual_scale_m=1.0,
        occlusion_abs_tolerance_m=0.5,
        occlusion_rel_tolerance=0.05,
        same_time_tolerance_us=5000,
        view_fusion_mode="depth-aware-interaction",
        view_interaction_dim=3,
    )
    queries = torch.randn(1, 1, 4, requires_grad=True)
    torch.testing.assert_close(module._sampling_offsets(queries), torch.zeros(1, 1, 1, 2))
    context = torch.tensor(
        [
            [
                [[[1.0, 0.0, 0.0, 0.0]] * 2] * 2,
                [[[0.0, 1.0, 0.0, 0.0]] * 2] * 2,
            ]
        ],
        requires_grad=True,
    )
    distance = torch.cat(
        [torch.ones(1, 1, 2, 2, 1), torch.full((1, 1, 2, 2, 1), 1.25)],
        dim=1,
    )
    output = module.fuse_chunk(
        queries=queries,
        query_timestamps_us=torch.tensor([[100]], dtype=torch.int64),
        query_dynamic_mask=torch.tensor([[False]]),
        query_valid_mask=torch.tensor([[True]]),
        context_feature_atlas=build_context_image_atlas(context),
        context_distance_atlas=build_context_image_atlas(distance),
        reference_image_points=torch.full((1, 2, 1, 2), 0.5),
        reference_valid_mask=torch.ones(1, 2, 1, dtype=torch.bool),
        reference_timestamps_us=torch.tensor([[[100], [1_000_100]]], dtype=torch.int64),
        query_metric_distance=torch.ones(1, 2, 1),
        full_resolution=torch.full((1, 2, 2), 2.0),
    )

    assert output.valid_observation_count.tolist() == [[2]]
    assert output.attention_entropy.shape == (1, 1)
    assert 0.0 < output.attention_entropy.item() <= math.log(2.0)
    output.features.square().sum().backward()
    assert queries.grad is not None and torch.isfinite(queries.grad).all()
    assert context.grad is not None and torch.isfinite(context.grad).all()
    depth_linear = module.depth_residual_embedding[0]
    assert isinstance(depth_linear, torch.nn.Linear)
    assert depth_linear.weight.grad is not None and torch.isfinite(depth_linear.weight.grad).all()


def test_feature_attention_hard_rejects_out_of_frustum_and_occluded_views() -> None:
    module = PointQueryDeformAttn(
        encoder_dim=4,
        fusion_dim=4,
        num_sampling_points=1,
        offset_radius_feature_px=0.0,
        depth_residual_scale_m=1.0,
        occlusion_abs_tolerance_m=0.1,
        occlusion_rel_tolerance=0.0,
        same_time_tolerance_us=5000,
    )
    context = torch.randn(1, 3, 2, 2, 4, requires_grad=True)
    distances = torch.cat(
        [
            torch.ones(1, 1, 2, 2, 1),
            torch.full((1, 1, 2, 2, 1), 0.1),
            torch.ones(1, 1, 2, 2, 1),
        ],
        dim=1,
    )
    output = module.fuse_chunk(
        queries=torch.randn(1, 1, 4, requires_grad=True),
        query_timestamps_us=torch.tensor([[100]], dtype=torch.int64),
        query_dynamic_mask=torch.tensor([[False]]),
        query_valid_mask=torch.tensor([[True]]),
        context_feature_atlas=build_context_image_atlas(context),
        context_distance_atlas=build_context_image_atlas(distances),
        reference_image_points=torch.tensor([[[[0.5, 0.5]], [[0.5, 0.5]], [[2.5, 0.5]]]]),
        reference_valid_mask=torch.ones(1, 3, 1, dtype=torch.bool),
        reference_timestamps_us=torch.full((1, 3, 1), 100, dtype=torch.int64),
        query_metric_distance=torch.ones(1, 3, 1),
        full_resolution=torch.full((1, 3, 2), 2.0),
    )

    assert output.in_frustum_observation_count.tolist() == [2]
    assert output.depth_tested_observation_count.tolist() == [2]
    assert output.occluded_observation_count.tolist() == [1]
    assert output.valid_observation_count.tolist() == [[1]]
    assert output.view_valid_query_count.tolist() == [[1, 0, 0]]
    torch.testing.assert_close(output.view_attention_mass, torch.tensor([[1.0, 0.0, 0.0]]))
    output.features.sum().backward()
    assert context.grad is not None
    assert bool((context.grad[:, 0].abs().sum() > 0.0))
    torch.testing.assert_close(context.grad[:, 1:], torch.zeros_like(context.grad[:, 1:]))


@pytest.mark.parametrize("num_sampling_points", [1, 4])
@pytest.mark.parametrize("view_fusion_mode", ["scalar", "depth-aware-interaction"])
@pytest.mark.parametrize("in_frustum", [False, True])
def test_attention_without_eligible_observations_has_finite_query_gradients(
    num_sampling_points: int, view_fusion_mode: str, in_frustum: bool
) -> None:
    module = PointQueryDeformAttn(
        encoder_dim=4,
        fusion_dim=4,
        num_sampling_points=num_sampling_points,
        offset_radius_feature_px=2.0,
        depth_residual_scale_m=1.0,
        occlusion_abs_tolerance_m=0.1,
        occlusion_rel_tolerance=0.0,
        same_time_tolerance_us=5000,
        view_fusion_mode=view_fusion_mode,
    )
    queries = torch.randn(1, 2, 4, requires_grad=True)
    context = torch.randn(1, 2, 2, 2, 4, requires_grad=True)
    output = module.fuse_chunk(
        queries=queries,
        query_timestamps_us=torch.full((1, 2), 100, dtype=torch.int64),
        query_dynamic_mask=torch.zeros(1, 2, dtype=torch.bool),
        query_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        context_feature_atlas=build_context_image_atlas(context),
        # Every in-frustum query is behind the predicted surface.
        context_distance_atlas=build_context_image_atlas(torch.ones(1, 2, 2, 2, 1)),
        reference_image_points=torch.full((1, 2, 2, 2), 0.5),
        reference_valid_mask=torch.full((1, 2, 2), in_frustum, dtype=torch.bool),
        reference_timestamps_us=torch.full((1, 2, 2), 100, dtype=torch.int64),
        query_metric_distance=torch.full((1, 2, 2), 2.0),
        full_resolution=torch.full((1, 2, 2), 2.0),
    )

    assert torch.isfinite(output.features).all()
    assert not output.valid_observation_count.any()
    assert not output.view_attention_mass.any()
    assert torch.isfinite(output.attention_entropy).all()
    assert output.in_frustum_observation_count.tolist() == [4 if in_frustum else 0]
    assert output.occluded_observation_count.tolist() == [4 if in_frustum else 0]
    output.features.square().sum().backward()
    assert queries.grad is not None and torch.isfinite(queries.grad).all()
    assert queries.grad.abs().sum() > 0.0
    assert context.grad is None or not context.grad.any()
