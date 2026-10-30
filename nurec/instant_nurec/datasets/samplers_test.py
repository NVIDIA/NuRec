# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import numpy as np
import pytest

from nurec.instant_nurec.config.dataset import (
    AdaptiveSequentialFrameBatchSamplerConfig,
    SupervisionFrameBatchParamsConfig,
    UniformFrameBatchSamplerConfig,
    VaryingIntervalFrameBatchSamplerConfig,
)
from nurec.instant_nurec.datasets.instant_nurec_base import InstantNuRecDataError
from nurec.instant_nurec.datasets.samplers import (
    AdaptiveSequentialFrameBatchSampler,
    BaseFrameBatchSampler,
    FrameBatchSamplerReturn,
    UniformFrameBatchSampler,
    VaryingIntervalFrameBatchSampler,
    sample_supervision_frame_batch,
)
from nurec.utils.types import HalfClosedInterval


@pytest.mark.parametrize(
    "context_sampler_config, sampler_class",
    [
        (
            VaryingIntervalFrameBatchSamplerConfig(
                n_frames_per_sample=3,
                n_samples_per_sequence=1,
                sequence_gap_timestamp_us_min=0,
                sequence_gap_timestamp_us_max=10,
            ),
            VaryingIntervalFrameBatchSampler,
        ),
        (
            UniformFrameBatchSamplerConfig(
                n_frames_per_sample=3,
                n_samples_per_sequence=1,
                frame_gap_timestamp_us=1,
            ),
            UniformFrameBatchSampler,
        ),
    ],
)
def test_varying_interval_frame_batch_sampler(context_sampler_config, sampler_class) -> None:
    camera_id = "test_camera"
    camera_frame_timestamps = {camera_id: np.arange(100)}

    context_sampler: BaseFrameBatchSampler = sampler_class(context_sampler_config)
    rng = np.random.default_rng(seed=0)

    context_samples = context_sampler.sample_frame_batch(
        rng, 0, camera_frame_timestamps, [HalfClosedInterval(0, 20)], None
    )
    supervision_samples = sample_supervision_frame_batch(
        config=SupervisionFrameBatchParamsConfig(n_frames_per_sample=3, sample_strategy="random"),
        rng=rng,
        context_frame_batch=context_samples,
        sensor_frame_timestamps_us=camera_frame_timestamps,
        supervision_sensor_ids=[camera_id],
        sensor_sample_ratios=[1.0],
    )

    context_range = range(
        min(context_samples.sampled_sensor_frame_idxs[camera_id]),
        max(context_samples.sampled_sensor_frame_idxs[camera_id]) + 1,
    )
    assert all(frame_idx in context_range for frame_idx in supervision_samples.sampled_sensor_frame_idxs[camera_id])


def test_uniform_frame_batch_sampler_reference_camera_prevents_adjacent_cycle_selection() -> None:
    camera_frame_timestamps = {
        "reference": np.array([0, 100, 200], dtype=np.int64),
        "phase_shifted": np.array([10, 110, 210], dtype=np.int64),
    }
    interval = [HalfClosedInterval(55, 56)]

    unsynchronized = UniformFrameBatchSampler(
        UniformFrameBatchSamplerConfig(
            n_frames_per_sample=1,
            n_samples_per_sequence=1,
            frame_gap_timestamp_us=1,
        )
    ).sample_frame_batch(np.random.default_rng(seed=0), 0, camera_frame_timestamps, interval, None)
    synchronized = UniformFrameBatchSampler(
        UniformFrameBatchSamplerConfig(
            n_frames_per_sample=1,
            n_samples_per_sequence=1,
            frame_gap_timestamp_us=1,
            reference_camera_id="reference",
        )
    ).sample_frame_batch(np.random.default_rng(seed=0), 0, camera_frame_timestamps, interval, None)

    assert unsynchronized.sampled_sensor_frame_idxs == {"reference": [1], "phase_shifted": [0]}
    assert synchronized.sampled_sensor_frame_idxs == {"reference": [1], "phase_shifted": [1]}


def test_uniform_frame_batch_sampler_rejects_missing_reference_camera() -> None:
    sampler = UniformFrameBatchSampler(
        UniformFrameBatchSamplerConfig(
            n_frames_per_sample=1,
            n_samples_per_sequence=1,
            frame_gap_timestamp_us=1,
            reference_camera_id="missing",
        )
    )

    with pytest.raises(ValueError, match="Reference camera 'missing' is absent"):
        sampler.sample_frame_batch(
            np.random.default_rng(seed=0),
            0,
            {"present": np.array([0, 100], dtype=np.int64)},
            [HalfClosedInterval(0, 1)],
            None,
        )


def test_uniform_frame_batch_sampler_excludes_incomplete_reference_frames() -> None:
    camera_frame_timestamps = {
        "reference": np.array([0, 100, 200, 300], dtype=np.int64),
        "ends_early": np.array([10, 110, 210], dtype=np.int64),
    }
    sampler = UniformFrameBatchSampler(
        UniformFrameBatchSamplerConfig(
            n_frames_per_sample=1,
            n_samples_per_sequence=1,
            frame_gap_timestamp_us=1,
            reference_camera_id="reference",
            max_camera_timestamp_delta_us=20,
        )
    )

    synchronized = sampler.sample_frame_batch(
        np.random.default_rng(seed=0),
        0,
        camera_frame_timestamps,
        [HalfClosedInterval(0, 301)],
        None,
    )

    assert synchronized.sampled_sensor_frame_idxs == {"reference": [2], "ends_early": [2]}


def test_uniform_frame_batch_sampler_timestamp_tolerance_requires_reference_camera() -> None:
    with pytest.raises(ValueError, match="max_camera_timestamp_delta_us requires reference_camera_id"):
        UniformFrameBatchSampler(
            UniformFrameBatchSamplerConfig(
                n_frames_per_sample=1,
                n_samples_per_sequence=1,
                frame_gap_timestamp_us=1,
                max_camera_timestamp_delta_us=20,
            )
        )


@pytest.mark.parametrize("n_frames,frame_gap", [(3, 100), (2, 200)])
def test_synchronized_sampler_rejects_selected_holes_but_allows_striding_over_them(
    n_frames: int, frame_gap: int
) -> None:
    sampler = UniformFrameBatchSampler(
        UniformFrameBatchSamplerConfig(
            n_frames_per_sample=n_frames,
            n_samples_per_sequence=1,
            frame_gap_timestamp_us=frame_gap,
            reference_camera_id="reference",
            max_camera_timestamp_delta_us=20,
        )
    )
    timestamps = {
        "reference": np.array([0, 100, 200], dtype=np.int64),
        "missing_middle": np.array([10, 210], dtype=np.int64),
    }
    args = (np.random.default_rng(seed=0), 0, timestamps, [HalfClosedInterval(0, 201)], None)
    if n_frames == 3:
        # One attempted sample fails through the normal bounded dataset retry;
        # it must not remap capture 100 to a duplicate of capture 0 or 200.
        with pytest.raises(InstantNuRecDataError, match="without synchronized camera support"):
            sampler.sample_frame_batch(*args)
    else:
        sampled = sampler.sample_frame_batch(*args)
        assert sampled.sampled_sensor_frame_idxs == {"reference": [0, 2], "missing_middle": [0, 1]}


def test_synchronized_sampler_rejects_repeated_frames_in_sparse_reference_stream() -> None:
    sampler = UniformFrameBatchSampler(
        UniformFrameBatchSamplerConfig(
            n_frames_per_sample=3,
            n_samples_per_sequence=1,
            frame_gap_timestamp_us=100,
            reference_camera_id="reference",
            max_camera_timestamp_delta_us=20,
        )
    )
    with pytest.raises(InstantNuRecDataError, match="repeats reference frames"):
        sampler.sample_frame_batch(
            np.random.default_rng(seed=0),
            0,
            {"reference": np.array([0, 200]), "other": np.array([10, 210])},
            [HalfClosedInterval(0, 201)],
            None,
        )


def test_adaptive_sequential_frame_batch_sampler() -> None:
    camera_id = "test_camera"
    camera_frame_timestamps = {camera_id: np.arange(100)}
    context_sampler = AdaptiveSequentialFrameBatchSampler(
        AdaptiveSequentialFrameBatchSamplerConfig(
            n_frames_per_sample=5,
            n_samples_per_sequence=3,
            max_frame_gap_timestamp_us=10,
        )
    )
    rng = np.random.default_rng(seed=0)

    context_samples_0 = context_sampler.sample_frame_batch(
        rng, 0, camera_frame_timestamps, [HalfClosedInterval(0, 100)], None
    )
    context_samples_1 = context_sampler.sample_frame_batch(
        rng, 1, camera_frame_timestamps, [HalfClosedInterval(0, 100)], None
    )
    context_samples_2 = context_sampler.sample_frame_batch(
        rng, 2, camera_frame_timestamps, [HalfClosedInterval(0, 100)], None
    )

    assert context_sampler.num_samples_per_sequence == 3
    sampled_frame_idxs_0 = context_samples_0.sampled_sensor_frame_idxs[camera_id]
    sampled_frame_idxs_1 = context_samples_1.sampled_sensor_frame_idxs[camera_id]
    assert sampled_frame_idxs_0 == [0, 10, 20, 30, 40]
    assert sampled_frame_idxs_1 == [50, 60, 70, 80, 90]
    assert sampled_frame_idxs_1[0] - sampled_frame_idxs_0[-1] == 10
    assert context_samples_2.sampled_sensor_frame_idxs == {}


@pytest.mark.parametrize("sample_strategy", ["random", "stratified"])
@pytest.mark.parametrize(
    "target_timestamps,horizon,expected_indices",
    [
        ([10, 110, 210], 50, [1]),  # The nearest frame already lies after context.
        ([0, 100, 200], 80, []),  # The nearest horizon frame lies beyond the bound.
        ([0, 100, 200], 100, [2]),  # Exclude context, include the exact horizon.
        ([0, 100], 100, []),  # Do not fold an exhausted window back onto context.
        ([0, 100, 200], 0, []),
    ],
)
def test_future_only_supervision_obeys_exact_timestamp_bounds(
    sample_strategy: Literal["random", "stratified"],
    target_timestamps: list[int],
    horizon: int,
    expected_indices: list[int],
) -> None:
    context = FrameBatchSamplerReturn(sampled_sensor_frame_idxs={"context": [0]})
    supervision = sample_supervision_frame_batch(
        config=SupervisionFrameBatchParamsConfig(
            n_frames_per_sample=1,
            sample_strategy=sample_strategy,
            extrapolate_future_only=True,
            extrapolation_horizon_us=horizon,
        ),
        rng=np.random.default_rng(seed=0),
        context_frame_batch=context,
        sensor_frame_timestamps_us={
            "context": np.array([100], dtype=np.int64),
            "target": np.array(target_timestamps, dtype=np.int64),
        },
        supervision_sensor_ids=["target"],
        sensor_sample_ratios=[1.0],
    )

    assert supervision.sampled_sensor_frame_idxs == ({"target": expected_indices} if expected_indices else {})


@pytest.mark.parametrize("horizon,append", [(-1, 0), (None, -1)])
def test_future_only_supervision_rejects_negative_effective_horizon(horizon: int | None, append: int) -> None:
    with pytest.raises(ValueError, match="greater than or equal to 0|must be non-negative"):
        SupervisionFrameBatchParamsConfig(
            extrapolate_future_only=True,
            extrapolation_horizon_us=horizon,
            append_timestamps_us=append,
        )


def test_future_only_horizon_validation_preserves_explicit_override_and_normal_sampling() -> None:
    explicit = SupervisionFrameBatchParamsConfig(
        extrapolate_future_only=True, extrapolation_horizon_us=0, append_timestamps_us=-100
    )
    normal = SupervisionFrameBatchParamsConfig(append_timestamps_us=-100)

    assert explicit.extrapolation_horizon_us == 0
    assert normal.append_timestamps_us == -100


def test_sample_supervision_frame_batch_include_context_frames() -> None:
    camera_id = "cam"
    camera_frame_timestamps = {camera_id: np.arange(100, dtype=np.int64)}
    rng = np.random.default_rng(seed=42)
    context = FrameBatchSamplerReturn(sampled_sensor_frame_idxs={camera_id: [10, 25, 40]})

    supervision = sample_supervision_frame_batch(
        config=SupervisionFrameBatchParamsConfig(
            n_frames_per_sample=2,
            sample_strategy="random",
            include_context_frames=True,
        ),
        rng=rng,
        context_frame_batch=context,
        sensor_frame_timestamps_us=camera_frame_timestamps,
        supervision_sensor_ids=[camera_id],
        sensor_sample_ratios=[1.0],
    )

    sup_idxs = set(supervision.sampled_sensor_frame_idxs[camera_id])
    assert {10, 25, 40}.issubset(sup_idxs)
