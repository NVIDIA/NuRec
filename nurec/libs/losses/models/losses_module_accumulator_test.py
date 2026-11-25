# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts at the NRE preparation / owned gsplat module boundary."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from gsplat.losses.functional import JointSemantics, fused_bg_track_node_semantic_losses
from gsplat.losses.functional.accumulator import SSIMLossAccumulator

from nurec.libs.losses.functional.cuda_losses_accumulator import (
    GRID_NUM_CHANNELS,
    LOSS_SLOT_BY_NAME,
    N_LOSS_SLOTS,
)
from nurec.libs.losses.models.accumulator_module import AccumulatorModuleLosses
from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.models.gaussians.gaussians_model import RigidGaussianModel


def _selected(**weights):
    return {name: (SimpleNamespace(name=name, lambda_=weight), weight) for name, weight in weights.items()}


def _composite(nodes, device="cpu"):
    model = object.__new__(GaussiansComposite)
    torch.nn.Module.__init__(model)
    model.register_buffer("_device_indicator", torch.empty(0, device=device), persistent=False)
    model.gaussians_nodes = nodes
    model.get_gaussians_node_ids = lambda: list(nodes)
    model.post_processings = []
    return model


def test_camera_semantic_only_does_not_require_rgb_or_background():
    losses = AccumulatorModuleLosses()
    logits = torch.arange(24.0).reshape(6, 4)[:, ::2]
    labels = SimpleNamespace(
        flags=torch.zeros((1, 2, 3, 1), dtype=torch.int32),
        semantic=torch.zeros((1, 2, 3, 1), dtype=torch.uint8),
    )
    target = SimpleNamespace(data=SimpleNamespace(camera=SimpleNamespace(labels=labels)))
    results = SimpleNamespace(rendered_cam=SimpleNamespace(extra_ray_signals=SimpleNamespace(semantic_logits=logits)))
    vector = torch.zeros(N_LOSS_SLOTS)
    selected = _selected(rgb_l1_mean=0.0, background_mse_mean=0.0, semantic_cross_entropy_mean=-2.0)
    with mock.patch.object(losses.accumulator.camera_loss_accumulator, "forward", return_value=vector) as dispatch:
        output = losses._accumulate_camera(vector, results, target, selected)
    assert output is vector
    kwargs = dispatch.call_args.kwargs
    assert kwargs["rgb_pred"] is kwargs["rgb_gt"] is kwargs["bg_pred"] is None
    assert kwargs["semantic_logits"] is logits
    assert kwargs["loss_weights"] == (0.0, 0.0, -2.0)
    assert kwargs["flags"].numel() == 6


def test_owned_camera_dispatch_preserves_module_hooks_and_autograd():
    losses = AccumulatorModuleLosses()
    prediction = torch.arange(18.0, requires_grad=True)
    labels = SimpleNamespace(
        flags=torch.zeros((1, 2, 3, 1), dtype=torch.int32),
        rgb=torch.zeros((1, 2, 3, 3)),
    )
    target = SimpleNamespace(data=SimpleNamespace(camera=SimpleNamespace(labels=labels)))
    results = SimpleNamespace(rendered_cam=SimpleNamespace(rgb=prediction))
    owned = losses.accumulator.camera_loss_accumulator
    hook_calls = []

    def forward(**kwargs):
        assert torch.is_grad_enabled()
        output = kwargs["losses_vec"].clone()
        output[LOSS_SLOT_BY_NAME["rgb_l1_mean"]] = kwargs["rgb_pred"].sum() * kwargs["loss_weights"][0]
        return output

    def hook(module, args, output):
        hook_calls.append(module)
        return 3.0 * output

    handle = owned.register_forward_hook(hook)
    try:
        with mock.patch.object(owned, "forward", side_effect=forward):
            output = losses._accumulate_camera(torch.zeros(N_LOSS_SLOTS), results, target, _selected(rgb_l1_mean=2.0))
        output.sum().backward()
    finally:
        handle.remove()

    assert hook_calls == [owned]
    torch.testing.assert_close(prediction.grad, torch.full_like(prediction, 6.0))


def test_ssim_dispatch_uses_current_configuration_and_reuses_previous_modules():
    losses = AccumulatorModuleLosses()
    labels = SimpleNamespace(
        flags=torch.zeros((1, 2, 3, 1), dtype=torch.int32),
        rgb=torch.zeros((1, 2, 3, 3)),
    )
    target = SimpleNamespace(data=SimpleNamespace(camera=SimpleNamespace(labels=labels)))
    results = SimpleNamespace(rendered_cam=SimpleNamespace(rgb=torch.zeros(18)))
    config = SimpleNamespace(window_size=11, mask_mode="constant", mask_value=0.25, channel=3)
    calls = []

    def forward(module, **kwargs):
        calls.append((module, kwargs["loss_weights"]))
        return kwargs["losses_vec"]

    first = losses.accumulator._get_ssim_loss_accumulator(11, "constant", 0.25)
    with mock.patch.object(SSIMLossAccumulator, "forward", autospec=True, side_effect=forward):
        for window_size, mask_mode, mask_value, weight in (
            (11, "constant", 0.25, 2.0),
            (11, "GT", 0.5, -3.0),
            (11, "constant", 0.25, 4.0),
        ):
            config.window_size, config.mask_mode, config.mask_value = window_size, mask_mode, mask_value
            losses._accumulate_ssim(torch.zeros(N_LOSS_SLOTS), results, target, {"ssim_ssim_mean": (config, weight)})

    second = losses.accumulator._get_ssim_loss_accumulator(11, "gt", 0.5)
    assert second is not first
    assert calls == [(first, (2.0,)), (second, (-3.0,)), (first, (4.0,))]
    assert len(losses.accumulator.ssim_loss_accumulators) == 2


@pytest.mark.parametrize(
    "family,name,field",
    [
        ("camera", "rgb_l1_mean", "rgb"),
        ("camera", "background_mse_mean", "opacity"),
        ("camera", "semantic_cross_entropy_mean", "semantic_logits"),
        ("lidar", "lidar_l1_mean", "distance"),
        ("lidar", "intensity_mse_mean", "intensity"),
        ("lidar", "raydrop_mse_mean", "raydrop"),
        ("lidar", "background_lidar_mse_mean", "opacity"),
    ],
)
def test_active_member_missing_prediction_is_rejected_before_dispatch(family, name, field):
    losses = AccumulatorModuleLosses()
    labels = SimpleNamespace(
        flags=torch.zeros((1, 2, 3, 1), dtype=torch.int32),
        rgb=torch.zeros((1, 2, 3, 3)),
        semantic=torch.zeros((1, 2, 3, 1), dtype=torch.uint8),
        distance=torch.zeros((1, 2, 3, 1)),
        intensity=torch.zeros((1, 2, 3, 1)),
        raydrop=torch.zeros((1, 2, 3, 1)),
    )
    rendered = SimpleNamespace(extra_ray_signals=SimpleNamespace())
    signals = rendered.extra_ray_signals if field in ("semantic_logits", "intensity", "raydrop") else rendered
    setattr(signals, field, None)
    target = SimpleNamespace(data=SimpleNamespace(**{family: SimpleNamespace(labels=labels)}))
    results = SimpleNamespace(**{"rendered_cam" if family == "camera" else "rendered_lidar": rendered})
    accumulator = (
        losses.accumulator.camera_loss_accumulator if family == "camera" else losses.accumulator.lidar_loss_accumulator
    )
    group = losses._accumulate_camera if family == "camera" else losses._accumulate_lidar
    with mock.patch.object(accumulator, "forward") as dispatch:
        with pytest.raises(ValueError, match="Can't unpack empty optional"):
            group(torch.zeros(N_LOSS_SLOTS), results, target, _selected(**{name: -0.7}))
    dispatch.assert_not_called()


@pytest.mark.parametrize(
    "name,label,prediction,slot",
    [
        ("lidar_l1_mean", "distance", "distance", 0),
        ("intensity_mse_mean", "intensity", "intensity", 1),
        ("raydrop_mse_mean", "raydrop", "raydrop", 2),
        ("background_lidar_mse_mean", None, "opacity", 3),
    ],
)
def test_lidar_members_only_prepare_their_active_inputs(name, label, prediction, slot):
    losses = AccumulatorModuleLosses()
    labels = SimpleNamespace(flags=torch.zeros((1, 2, 3, 1), dtype=torch.int32))
    if label is not None:
        setattr(labels, label, torch.zeros((1, 2, 3, 1)))
    values = torch.arange(12.0)[::2]
    rendered = SimpleNamespace(extra_ray_signals=SimpleNamespace())
    if prediction in ("intensity", "raydrop"):
        setattr(rendered.extra_ray_signals, prediction, values)
    else:
        setattr(rendered, prediction, values)
    results = SimpleNamespace(rendered_lidar=rendered)
    target = SimpleNamespace(data=SimpleNamespace(lidar=SimpleNamespace(labels=labels)))
    vector = torch.zeros(N_LOSS_SLOTS)
    with mock.patch.object(losses.accumulator.lidar_loss_accumulator, "forward", return_value=vector) as dispatch:
        output = losses._accumulate_lidar(vector, results, target, _selected(**{name: -0.7}))
    assert output is vector
    kwargs = dispatch.call_args.kwargs
    assert kwargs["loss_weights"] == tuple(-0.7 if index == slot else 0.0 for index in range(4))
    for index, prefix in enumerate(("distance", "intensity", "raydrop", "bg")):
        if index == slot:
            assert kwargs[f"{prefix}_pred"].data_ptr() == values.data_ptr()
            assert not kwargs[f"{prefix}_pred"].is_contiguous()
        else:
            assert kwargs[f"{prefix}_pred"] is None
            if prefix != "bg":
                assert kwargs[f"{prefix}_gt"] is None


@pytest.mark.parametrize("camera_cells,frame_cells", [(60, 14), (60, 0), (0, 14)])
def test_grid_drift_uses_combined_cell_count_without_texture(camera_cells, frame_cells):
    losses = AccumulatorModuleLosses()
    camera = (
        [SimpleNamespace(bilateral_grid=SimpleNamespace(grid=torch.zeros((1, GRID_NUM_CHANNELS, 1, 1, camera_cells))))]
        if camera_cells
        else []
    )
    frame = (
        [SimpleNamespace(bilateral_grid=SimpleNamespace(grid=torch.zeros((1, GRID_NUM_CHANNELS, 1, 1, frame_cells))))]
        if frame_cells
        else []
    )
    vector = torch.zeros(N_LOSS_SLOTS)
    with mock.patch.object(losses.accumulator.bg_grid_loss_accumulator, "forward", return_value=vector) as dispatch:
        losses._accumulate_bg_grid(
            vector, _composite({}), camera, frame, _selected(bilateral_grid_drift_identity_distance_mean=-3.0)
        )
    kwargs = dispatch.call_args.kwargs
    assert kwargs["bg_tex"] is None
    assert kwargs["grid_drift_camera_factor"] == kwargs["grid_drift_frame_factor"] == 1 / (camera_cells + frame_cells)
    assert kwargs["loss_weights"] == (0.0, -3.0, 0.0, 0.0)


def _joint_inputs(classes, domain_sizes, device):
    primary = torch.tensor([-21.0, -1.0, 0.0, 0.5, 2.0], device=device, requires_grad=True)
    other = torch.linspace(-2.0, 2.0, sum(domain_sizes), device=device, requires_grad=True)
    positions = torch.tensor([[0.0, 0, 0], [0, 0, 0], [2, 0, 0], [0, 0, 0], [0, 0, 0]], device=device)
    primary_semantics = torch.zeros((5, classes), device=device)
    primary_semantics[:, -1] = 1.0
    primary_semantics[3, 0] = 2.0
    nodes = {
        "background": SimpleNamespace(
            get_positions=lambda: positions,
            get_densities=lambda preactivation: primary,
            get_extra_signal_by_key=lambda key: primary_semantics,
        )
    }
    other_semantics, segments, selected_indices = [], [], []
    start = 0
    for index, size in enumerate(domain_sizes):
        semantic = torch.zeros((size, classes), device=device)
        semantic[::2, -1] = 1.0
        semantic[1::2, 0] = 1.0
        density = other[start : start + size]
        nodes[f"node_{index}"] = SimpleNamespace(
            get_positions=lambda size=size: torch.zeros((size, 3), device=device),
            get_densities=lambda preactivation, density=density: density,
            get_extra_signal_by_key=lambda key, semantic=semantic: semantic,
        )
        other_semantics.append(semantic)
        selected_indices.extend(range(start, start + size, 2))
        start += size
        segments.append((start, (classes - 1,), True))
    tracks = SimpleNamespace(
        tracks_packinfo=torch.tensor([[0, 2]], dtype=torch.int32, device=device),
        tracks_poses=SimpleNamespace(data=torch.tensor([[0.0, 0, 0, 0, 0, 0, 1]] * 2, device=device)),
        tracks_timestamps_us=torch.tensor([0, 10], dtype=torch.int64, device=device),
        cuboids_dims=torch.full((1, 3), 2.0, device=device),
    )
    rigid = mock.Mock(spec=RigidGaussianModel, owns_tracks=True, cuboid_tracks=tracks)
    nodes["track"] = rigid
    timestamps = torch.tensor([[0, 10]], dtype=torch.int64, device=device)
    target = SimpleNamespace(rendering=SimpleNamespace(camera=SimpleNamespace(timestamps_startend_us=timestamps)))
    bg_name = "background_in_track_gaussian_bce_with_logits_mean"
    node_name = "node_semantic_gaussians_bce_with_logits_mean"
    background = SimpleNamespace(
        name=bg_name,
        layer_names=["background"],
        layer_labels_to_use={"background": ["selected"]},
        semantic_classes_map={"selected": classes - 1},
        density_logits_min=-20.0,
    )
    node = SimpleNamespace(
        name=node_name,
        layer_labels_to_use={},
        layer_labels_to_exclude={f"node_{i}": ["selected"] for i in range(len(domain_sizes))},
        semantic_classes_map={"selected": classes - 1},
    )
    reference_kwargs = dict(
        positions=positions,
        density_logits=primary,
        camera_timestamps_startend_us=timestamps,
        tracks_packinfo=tracks.tracks_packinfo,
        tracks_poses=tracks.tracks_poses.data,
        tracks_timestamps_us=tracks.tracks_timestamps_us,
        cuboids_dims=tracks.cuboids_dims,
        n_semantic_points=5,
        background_semantic_logits=(primary_semantics,),
        background_segments=((5, (classes - 1,)),),
        other_density_logits=other,
        other_semantic_logits=tuple(other_semantics),
        other_segments=tuple(segments),
        density_logits_min=-20.0,
    )
    return _composite(nodes, device), target, background, node, primary, other, selected_indices, reference_kwargs


@pytest.mark.parametrize("classes,domain_sizes", [(320, (2, 5)), (2, (1, 3, 2, 4) * 5)])
@pytest.mark.parametrize(
    "background_weight,node_weight", [(2.0, 3.0), (-2.0, 3.0), (2.0, -3.0), (0.0, -3.0), (-2.0, 0.0)]
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_large_joint_domains_preserve_scalar_global_means_and_gradients(
    classes, domain_sizes, background_weight, node_weight
):
    model, target, bg, node, primary, other, selected_indices, reference_kwargs = _joint_inputs(
        classes, domain_sizes, "cuda"
    )
    losses = AccumulatorModuleLosses()
    selected = {bg.name: (bg, background_weight), node.name: (node, node_weight)}
    accumulator = losses.accumulator._get_bg_track_node_semantic_loss_accumulator(-20.0)
    with mock.patch.object(accumulator, "forward", wraps=accumulator.forward) as dispatch:
        output = losses._accumulate_bg_track_node(torch.zeros(N_LOSS_SLOTS, device="cuda"), model, target, selected)
    dispatch.assert_called_once()
    background_slot, node_slot = LOSS_SLOT_BY_NAME[bg.name], LOSS_SLOT_BY_NAME[node.name]
    if background_weight == 0.0:
        reference_kwargs["semantic_logits"] = reference_kwargs["background_semantic_logits"][0]
        reference_kwargs["node_primary_predicate"] = ((), True)
        reference_kwargs["n_semantic_points"] = None
        reference_kwargs["background_semantic_logits"] = ()
        reference_kwargs["background_segments"] = ()
    background_raw, _, node_raw, _ = fused_bg_track_node_semantic_losses(
        **reference_kwargs,
        background_lambda=0.0 if background_weight else -1.0,
        node_lambda=0.0 if node_weight else -1.0,
    )
    scalar_reference = torch.zeros_like(output)
    scalar_reference[background_slot] = background_weight * background_raw
    scalar_reference[node_slot] = node_weight * node_raw
    torch.testing.assert_close(output, scalar_reference, atol=1e-6, rtol=1e-5)
    expected = torch.zeros_like(output)
    expected[background_slot] = background_weight * torch.nn.functional.softplus(primary.detach()[[1, 4]]).mean()
    expected[node_slot] = node_weight * torch.nn.functional.softplus(other.detach()[selected_indices]).mean()
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
    cotangent = torch.linspace(-0.7, 1.3, N_LOSS_SLOTS, device="cuda")
    actual = torch.autograd.grad(output, (primary, other), cotangent, allow_unused=True)
    reference = torch.autograd.grad(scalar_reference, (primary, other), cotangent, allow_unused=True)
    primary_gradient = torch.zeros_like(primary)
    primary_gradient[[1, 4]] = background_weight * cotangent[background_slot] * primary.detach()[[1, 4]].sigmoid() / 2
    other_gradient = torch.zeros_like(other)
    other_gradient[selected_indices] = (
        node_weight * cotangent[node_slot] * other.detach()[selected_indices].sigmoid() / len(selected_indices)
    )
    for source, gradient, expected_gradient, analytical in zip(
        (primary, other), actual, reference, (primary_gradient, other_gradient)
    ):
        gradient = torch.zeros_like(source) if gradient is None else gradient
        expected_gradient = torch.zeros_like(source) if expected_gradient is None else expected_gradient
        torch.testing.assert_close(gradient, expected_gradient, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(gradient, analytical, atol=1e-4, rtol=1e-4)


def test_node_only_does_not_prepare_background_inputs():
    model, target, bg, node, primary, other, _, _ = _joint_inputs(320, (2, 5), "cpu")
    del model.gaussians_nodes["background"]
    del model.gaussians_nodes["track"]
    target = SimpleNamespace()  # Camera timestamps must not be read.
    losses = AccumulatorModuleLosses()
    accumulator = losses.accumulator._get_bg_track_node_semantic_loss_accumulator(-20.0)
    vector = torch.zeros(N_LOSS_SLOTS)
    with mock.patch.object(accumulator, "forward", return_value=vector) as dispatch:
        losses._accumulate_bg_track_node(vector, model, target, {bg.name: (bg, 0.0), node.name: (node, -3.0)})
    kwargs = dispatch.call_args.kwargs
    for name in (
        "positions",
        "camera_timestamps_startend_us",
        "tracks_packinfo",
        "tracks_poses",
        "tracks_timestamps_us",
        "cuboids_dims",
    ):
        assert kwargs[name] is None
    assert isinstance(kwargs["semantics"], JointSemantics)
    assert kwargs["semantics"].semantic_logits.shape[1] == 320
    assert kwargs["loss_weights"] == (0.0, -3.0)


@pytest.mark.parametrize("weight", [float("inf"), float("nan")])
def test_node_nonfinite_weight_reaches_gsplat_validation(weight):
    model, target, bg, node, *_ = _joint_inputs(2, (2, 3), "cpu")
    with pytest.raises(ValueError, match="loss weights must be finite"):
        AccumulatorModuleLosses()._accumulate_bg_track_node(
            torch.zeros(N_LOSS_SLOTS), model, target, {node.name: (node, weight)}
        )


@pytest.mark.parametrize("weight", [1e40, -1e40, 1e-50, -1e-50, 1e-40, -1e-40])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_node_unrepresentable_weight_reaches_native_validation(weight):
    model, target, bg, node, *_ = _joint_inputs(2, (2, 3), "cuda")
    with pytest.raises(RuntimeError, match="float32"):
        AccumulatorModuleLosses()._accumulate_bg_track_node(
            torch.zeros(N_LOSS_SLOTS, device="cuda"), model, target, {node.name: (node, weight)}
        )


@pytest.mark.parametrize(
    "name", ["gaussian_z_scale_abs_mean", "deform_smoothness_abs_mean", "node_semantic_gaussians_bce_with_logits_mean"]
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_missing_data_keeps_existing_omission_at_zero_weight(name):
    losses = AccumulatorModuleLosses()
    losses.losses = [
        SimpleNamespace(
            name=name,
            lambda_=0.0,
            should_run_fn=lambda step: True,
            layer_name="missing",
            layer_labels_to_use={"missing": []},
            layer_labels_to_exclude={},
        )
    ]
    vector, weights = losses.accumulate(
        0, _composite({}, "cuda"), SimpleNamespace(deform_smoothness=None), SimpleNamespace()
    )
    assert name not in weights
    torch.testing.assert_close(vector, torch.zeros(N_LOSS_SLOTS, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_zero_weight_node_semantic_reports_present_empty_domain_without_preparation():
    name = "node_semantic_gaussians_bce_with_logits_mean"
    node = SimpleNamespace(
        get_extra_signal_by_key=mock.Mock(return_value=torch.empty((0, 2), device="cuda")),
        get_positions=mock.Mock(side_effect=AssertionError("zero-weight positions must not be read")),
        get_densities=mock.Mock(side_effect=AssertionError("zero-weight densities must not be read")),
    )
    losses = AccumulatorModuleLosses()
    losses.losses = [
        SimpleNamespace(
            name=name,
            lambda_=0.0,
            should_run_fn=lambda step: True,
            layer_labels_to_use={"empty": []},
            layer_labels_to_exclude={},
        )
    ]
    vector, weights = losses.accumulate(0, _composite({"empty": node}, "cuda"), SimpleNamespace(), SimpleNamespace())
    assert name in weights
    torch.testing.assert_close(vector[LOSS_SLOT_BY_NAME[name]], torch.zeros((), device="cuda"))
    node.get_positions.assert_not_called()
    node.get_densities.assert_not_called()
