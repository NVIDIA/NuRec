# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from nurec.models.rig_spline import CameraExtrinsicResiduals, PackedRigSpline, residual_vectors_to_transforms


def _make_spline(
    *,
    starts: tuple[int, ...] = (1_000_000_000_000_000,),
    ends: tuple[int, ...] = (1_000_000_003_000_000,),
    degree: int = 3,
    spacing: float = 1.0,
    anchor: bool = False,
    max_translation: float = 100.0,
    max_rotation: float = math.pi,
) -> PackedRigSpline:
    return PackedRigSpline(
        torch.tensor(starts, dtype=torch.int64),
        torch.tensor(ends, dtype=torch.int64),
        degree=degree,
        knot_spacing_seconds=spacing,
        skip_first_control=anchor,
        max_translation_m=max_translation,
        max_rotation_rad=max_rotation,
    )


def _dense_reference_basis(u: float, degree: int, spans: int, spacing: float) -> torch.Tensor:
    knots = [0.0] * (degree + 1)
    knots += [i * spacing for i in range(1, spans)]
    knots += [spans * spacing] * (degree + 1)
    n_controls = spans + degree
    u = min(max(u, 0.0), spans * spacing)
    values = [
        float((knots[i] <= u < knots[i + 1]) or (u == knots[-1] and i == n_controls - 1)) for i in range(n_controls)
    ]
    for p in range(1, degree + 1):
        next_values = []
        for i in range(n_controls):
            left_denominator = knots[i + p] - knots[i]
            right_denominator = knots[i + p + 1] - knots[i + 1]
            left = (u - knots[i]) / left_denominator * values[i] if left_denominator else 0.0
            right = (
                (knots[i + p + 1] - u) / right_denominator * values[i + 1]
                if right_denominator and i + 1 < n_controls
                else 0.0
            )
            next_values.append(left + right)
        values = next_values
    return torch.tensor(values)


@pytest.mark.parametrize("degree", [1, 2, 3, 5])
def test_basis_partition_of_unity_and_reference_agreement(degree: int) -> None:
    spline = _make_spline(degree=degree)
    origin = int(spline.trajectory_origins_us[0])
    timestamps = origin + torch.tensor([0, 123_456, 999_999, 1_000_000, 2_700_000, 3_000_000], dtype=torch.int64)
    basis, indices = spline._local_basis_and_control_indices(timestamps, torch.zeros_like(timestamps))
    torch.testing.assert_close(basis.sum(dim=-1), torch.ones(len(timestamps)))
    for query_idx, timestamp in enumerate(timestamps):
        dense = _dense_reference_basis((int(timestamp) - origin) / 1e6, degree, spans=3, spacing=1.0)
        local_dense = torch.zeros_like(dense)
        local_dense[indices[query_idx]] = basis[query_idx]
        torch.testing.assert_close(local_dense, dense, atol=2e-6, rtol=2e-6)


def test_endpoint_clamping_short_clip_and_arbitrary_interpolation() -> None:
    spline = _make_spline(ends=(1_000_000_000_250_000,), degree=1)
    with torch.no_grad():
        spline.controls[:, 0] = torch.tensor([0.0, 2.0])
    origin = int(spline.trajectory_origins_us[0])
    timestamps = torch.tensor([origin - 1, origin, origin + 250_000, origin + 500_000, origin + 2_000_000])
    values = spline.evaluate(timestamps, 0)[:, 0]
    torch.testing.assert_close(values, torch.tensor([0.0, 0.0, 0.5, 0.5, 0.5]))


def test_int64_origin_preserves_large_timestamp_precision() -> None:
    origin = 8_000_000_000_000_000
    spline = _make_spline(starts=(origin,), ends=(origin + 1_000_000,), degree=1)
    with torch.no_grad():
        spline.controls[:, 0] = torch.tensor([0.0, 1.0])
    timestamps = torch.tensor([origin + 1, origin + 10, origin + 100, origin + 500_000], dtype=torch.int64)
    values = spline.evaluate(timestamps, 0)[:, 0]
    assert values[0] > 0.0
    assert values[1] > values[0]
    assert values[2] > values[1]
    torch.testing.assert_close(values[-1], torch.tensor(0.5))


def test_multiple_trajectories_are_packed_and_locally_supported() -> None:
    spline = _make_spline(
        starts=(10_000_000, 90_000_000),
        ends=(13_000_000, 92_000_000),
        degree=3,
    )
    with torch.no_grad():
        spline.controls.copy_(torch.arange(spline.controls.numel()).reshape_as(spline.controls) / 100.0)
    timestamps = torch.tensor([11_250_000, 90_500_000], dtype=torch.int64)
    trajectories = torch.tensor([0, 1], dtype=torch.int64)
    output = spline.evaluate(timestamps, trajectories)
    output.sum().backward()
    control_gradients = spline.controls.grad
    assert control_gradients is not None
    nonzero_control_gradients = control_gradients.abs().sum(dim=-1) > 0
    assert int(nonzero_control_gradients.sum()) <= 2 * (spline.degree + 1)
    assert output.shape == (2, 6)


def test_exact_anchor_radial_bounds_rotation_and_finite_gradients() -> None:
    spline = _make_spline(anchor=True, max_translation=3.0, max_rotation=math.radians(3.0))
    with torch.no_grad():
        spline.controls.fill_(10.0)
    bounded = spline.bounded_controls()
    assert torch.equal(bounded[0], torch.zeros(6))
    assert torch.linalg.vector_norm(bounded[:, :3], dim=-1).max() <= 3.0
    assert torch.linalg.vector_norm(bounded[:, 3:], dim=-1).max() <= math.radians(3.0)

    origin = int(spline.trajectory_origins_us[0])
    transform = spline(torch.tensor([origin, origin + 500_000]), 0)
    torch.testing.assert_close(transform[0], torch.eye(4), atol=0.0, rtol=0.0)
    transform.sum().backward()
    control_gradients = spline.controls.grad
    assert control_gradients is not None
    assert torch.isfinite(control_gradients).all()
    assert torch.equal(control_gradients[0], torch.zeros(6))

    quarter_turn = residual_vectors_to_transforms(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2]]))
    expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    torch.testing.assert_close(quarter_turn[0, :3, :3], expected, atol=1e-6, rtol=1e-6)


def test_curvature_does_not_penalize_constant_or_linear_drift() -> None:
    spline = _make_spline()
    with torch.no_grad():
        ramp = torch.arange(spline.controls.shape[0], dtype=spline.controls.dtype)
        spline.controls[:, 0] = 0.1 + 0.2 * ramp
        spline.controls[:, 5] = -0.03 * ramp
    translation, rotation = spline.second_differences()
    torch.testing.assert_close(translation, torch.zeros_like(translation), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(rotation, torch.zeros_like(rotation), atol=1e-6, rtol=0.0)
    with torch.no_grad():
        spline.controls[2, 0] += 1.0
    assert spline.second_differences()[0].square().sum() > 0.0


def test_group_l1_curvature_is_linear_and_rotation_invariant() -> None:
    spline = _make_spline()
    with torch.no_grad():
        ramp = torch.arange(spline.controls.shape[0], dtype=spline.controls.dtype)
        spline.controls[:, 0] = 0.1 * ramp
        spline.controls[:, 4] = -0.01 * ramp
    torch.testing.assert_close(
        spline.group_l1_curvature_loss(
            translation_normalization_m_s2=2.0,
            rotation_normalization_rad_s2=math.radians(20.0),
        ),
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0.0,
    )

    with torch.no_grad():
        spline.controls.zero_()
        spline.controls[2, 0] = 0.1
    loss = spline.group_l1_curvature_loss(
        translation_normalization_m_s2=2.0,
        rotation_normalization_rad_s2=math.radians(20.0),
    )
    with torch.no_grad():
        spline.controls[2, 0] = 0.2
    doubled_loss = spline.group_l1_curvature_loss(
        translation_normalization_m_s2=2.0,
        rotation_normalization_rad_s2=math.radians(20.0),
    )
    torch.testing.assert_close(doubled_loss, 2.0 * loss)

    with torch.no_grad():
        spline.controls.zero_()
        spline.controls[2, 3] = 0.1
    axis_aligned_loss = spline.group_l1_curvature_loss(
        translation_normalization_m_s2=2.0,
        rotation_normalization_rad_s2=1.0,
    )
    with torch.no_grad():
        spline.controls[2, 3:5] = 0.1 / math.sqrt(2.0)
    rotated_loss = spline.group_l1_curvature_loss(
        translation_normalization_m_s2=2.0,
        rotation_normalization_rad_s2=1.0,
    )
    torch.testing.assert_close(rotated_loss, axis_aligned_loss, atol=1e-7, rtol=1e-6)


def test_group_l1_curvature_uses_knot_spacing_squared() -> None:
    origin = 1_000_000_000_000_000
    unit_spacing = _make_spline(starts=(origin,), ends=(origin + 3_000_000,), spacing=1.0)
    half_spacing = _make_spline(starts=(origin,), ends=(origin + 1_500_000,), spacing=0.5)
    assert unit_spacing.controls.shape == half_spacing.controls.shape
    with torch.no_grad():
        unit_spacing.controls[2, 0] = 0.1
        half_spacing.controls.copy_(unit_spacing.controls)
    unit_loss = unit_spacing.group_l1_curvature_loss(
        translation_normalization_m_s2=1.0,
        rotation_normalization_rad_s2=1.0,
    )
    half_loss = half_spacing.group_l1_curvature_loss(
        translation_normalization_m_s2=1.0,
        rotation_normalization_rad_s2=1.0,
    )
    torch.testing.assert_close(half_loss, 4.0 * unit_loss)


def test_camera_extrinsics_projection_bounds_gradients_and_checkpoint() -> None:
    trajectory_indices = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int64)
    extrinsics = CameraExtrinsicResiduals(
        trajectory_indices,
        num_trajectories=2,
        max_translation_m=0.1,
        max_rotation_rad=math.radians(3.0),
    )
    assert torch.equal(extrinsics.residuals, torch.zeros_like(extrinsics.residuals))
    assert list(extrinsics.state_dict()) == ["residuals"]

    with torch.no_grad():
        extrinsics.residuals.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, -1.0, 0.2, 0.1, -0.3],
                    [-2.0, 1.0, 0.5, -0.1, 0.4, 0.2],
                    [0.3, -1.0, 2.0, 0.2, 0.3, 0.4],
                    [2.0, 0.1, -0.4, -0.5, 0.1, 0.2],
                    [-0.7, 0.4, 1.0, 0.3, -0.2, 0.1],
                ]
            )
        )
    bounded = extrinsics.bounded_residuals()
    for trajectory_idx in range(2):
        trajectory_residuals = bounded[trajectory_indices == trajectory_idx]
        torch.testing.assert_close(
            trajectory_residuals.mean(dim=0),
            torch.zeros(6),
            atol=2e-8,
            rtol=0.0,
        )
    assert torch.linalg.vector_norm(bounded[:, :3], dim=-1).max() <= 0.1
    assert torch.linalg.vector_norm(bounded[:, 3:], dim=-1).max() <= math.radians(3.0)

    extrinsics(torch.arange(len(trajectory_indices))).sum().backward()
    assert extrinsics.residuals.grad is not None
    assert torch.isfinite(extrinsics.residuals.grad).all()

    clone = CameraExtrinsicResiduals(
        trajectory_indices,
        num_trajectories=2,
        max_translation_m=0.1,
        max_rotation_rad=math.radians(3.0),
    )
    clone.load_state_dict(extrinsics.state_dict())
    torch.testing.assert_close(clone.residuals, extrinsics.residuals)
    torch.testing.assert_close(clone.bounded_residuals(), bounded)


def test_camera_extrinsics_require_two_cameras_in_each_gauge_group() -> None:
    with pytest.raises(ValueError, match="at least two cameras"):
        CameraExtrinsicResiduals(
            torch.tensor([0, 0, 1], dtype=torch.int64),
            num_trajectories=2,
            max_translation_m=0.1,
            max_rotation_rad=math.radians(3.0),
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"degree": 0}, "degree"),
        ({"spacing": 0.0}, "spacing"),
        ({"max_translation": 0.0}, "translation"),
        ({"max_rotation": 0.0}, "rotation"),
    ],
)
def test_invalid_configuration(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _make_spline(**kwargs)
