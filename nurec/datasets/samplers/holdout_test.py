# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest

from collections import Counter
from typing import Optional
from unittest.mock import Mock

import numpy as np
import numpy.typing as npt
import torch

from omegaconf import DictConfig

from nurec.datasets.samplers.base import BaseBatchSampler
from nurec.datasets.samplers.holdout import (
    CoverageFrameSampler,
    CoveragePlan,
    CoverageSensorSampler,
    HoldOutFrameSampler,
)
from nurec.datasets.samplers.uniform import UniformFrameSampler, UniformSensorSampler


def create_deterministic_rng():
    """Create a mock np.random.Generator.choice() that returns subsequent indices of the non-zero elements of the
    probability array when called in sequence
    """
    current_index: int = 0

    def choice(
        frame_range: npt.ArrayLike,
        size: Optional[int] = None,
        replace: bool = False,
        p: Optional[npt.ArrayLike] = None,
        axis: Optional[int] = None,
        shuffle: bool = False,
    ):
        # This mock is limited to some specific inputs of np.random.Generator.choice().
        assert isinstance(frame_range, range)
        assert size == 1
        assert not replace
        assert not shuffle
        assert axis is None
        assert p is not None

        p_array = np.asarray(p)
        nonzero_indices = np.where(p_array > 0)[0]

        nonlocal current_index
        if current_index >= len(nonzero_indices):
            current_index = 0  # Reset to beginning for cycling

        selected_index = nonzero_indices[current_index]
        selected_frame = list(frame_range)[selected_index]

        current_index += 1
        return np.array([selected_frame])

    # Create a mock that behaves like np.random.Generator
    mock_rng = Mock(spec=np.random.Generator)
    mock_rng.choice = choice
    return mock_rng


class TestHoldOutFrameSampler(unittest.TestCase):
    """Test cases for HoldOutFrameSampler"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_dataset = Mock(spec="nurec.datasets.ncore.NCORETrainDataset")
        self.rng = create_deterministic_rng()

        # Define frame_range as the indices { 20, 30, 40, 50, 60, 70, 80 }.
        self.frame_range = range(20, 81, 10)

    def test_sample_frame_exclude_mode(self):
        """Test whether sample_frame() correctly excludes every Nth frame in exclude mode"""

        # Define config to skip every 3rd frame, starting from index 2.
        # Frames 40, 70 are excluded, only 20, 30, 50, 60, 80 are included (5 indices).
        config = DictConfig(
            {
                # Notice: include_frames_start and include_every_n_frames are missing from the config.
                "exclude_frames_start": 2,
                "exclude_every_n_frames": 3,
            }
        )

        # Create the sampler to be tested.
        sampler = HoldOutFrameSampler(config, self.mock_dataset)

        # Sample a frame multiple times to verify behavior.
        sampled_frames = []
        for _ in range(10):  # Test 2 full cycles over the valid 5 frame indices.
            result = sampler.sample_frame(
                rng=self.rng, batch_idx=0, frame_range=self.frame_range, unique_sensor_id="test_sensor"
            )
            sampled_frames.append(result.sampled_frame_idx)

        expected_sequence = [20, 30, 50, 60, 80, 20, 30, 50, 60, 80]  # 2 full cycles over the valid 5 frame indices.
        self.assertEqual(sampled_frames, expected_sequence)

    def test_sample_frame_include_mode(self):
        """Test whether sample_frame() correctly includes every Nth frame in include mode"""

        # Define config to include every 3rd frame, starting from index 2.
        # Frames 40, 70 are included (2 indices), but 20, 30, 50, 60, 80 are excluded.
        config = DictConfig(
            {
                # Notice: exclude_frames_start and exclude_every_n_frames are missing from the config.
                "include_frames_start": 2,
                "include_every_n_frames": 3,
            }
        )
        # Create the sampler to be tested.
        sampler = HoldOutFrameSampler(config, self.mock_dataset)

        # Sample a frame multiple times to verify behavior.
        sampled_frames = []
        for _ in range(4):  # Test 2 full cycles over the valid 2 frame indices.
            result = sampler.sample_frame(
                rng=self.rng, batch_idx=0, frame_range=self.frame_range, unique_sensor_id="test_sensor"
            )
            sampled_frames.append(result.sampled_frame_idx)

        expected_sequence = [40, 70, 40, 70]  # 2 full cycles over the valid 2 frame indices.
        self.assertEqual(sampled_frames, expected_sequence)

    def test_construction_with_empty_config(self):
        """Test whether HoldOutFrameSampler raises AssertionError with completely missing config"""

        with self.assertRaises(AssertionError):
            HoldOutFrameSampler(config=DictConfig({}), dataset=self.mock_dataset)

    def test_construction_with_contradicting_config(self):
        """Test whether HoldOutFrameSampler raises AssertionError with contradicting include/exclude config"""

        with self.assertRaises(AssertionError):
            HoldOutFrameSampler(
                config=DictConfig(
                    {
                        "include_frames_start": 1,
                        "include_every_n_frames": 2,
                        "exclude_frames_start": 3,
                        "exclude_every_n_frames": 4,
                    }
                ),
                dataset=self.mock_dataset,
            )

    def test_construction_with_unspecific_config(self):
        """Test whether HoldOutFrameSampler raises AssertionError with unspecific config (all None values)"""

        with self.assertRaises(AssertionError):
            HoldOutFrameSampler(
                config=DictConfig(
                    {
                        "include_frames_start": None,
                        "include_every_n_frames": None,
                        "exclude_frames_start": None,
                        "exclude_every_n_frames": None,
                    }
                ),
                dataset=self.mock_dataset,
            )

    def test_construction_with_unspecific_and_incomplete_config(self):
        """Test whether HoldOutFrameSampler raises AssertionError with incomplete config (missing start parameters)"""

        with self.assertRaises(AssertionError):
            HoldOutFrameSampler(
                config=DictConfig(
                    {
                        "include_every_n_frames": None,
                        "exclude_every_n_frames": None,
                    }
                ),
                dataset=self.mock_dataset,
            )


class TestHoldoutTrainingFrameMask(unittest.TestCase):
    """training_frame_mask must mirror sample_frame's reachable frame set."""

    def test_include_mode_mask(self):
        config = DictConfig({"include_frames_start": 1, "include_every_n_frames": 3})
        mask = HoldOutFrameSampler.training_frame_mask(config, n_frames=7)
        np.testing.assert_array_equal(mask, [False, True, False, False, True, False, False])

    def test_include_mode_default_start(self):
        config = DictConfig({"include_every_n_frames": 2})
        mask = HoldOutFrameSampler.training_frame_mask(config, n_frames=5)
        np.testing.assert_array_equal(mask, [True, False, True, False, True])

    def test_exclude_mode_mask(self):
        config = DictConfig({"exclude_frames_start": 0, "exclude_every_n_frames": 2})
        mask = HoldOutFrameSampler.training_frame_mask(config, n_frames=5)
        np.testing.assert_array_equal(mask, [False, True, False, True, False])

    def test_exclude_mode_default_start(self):
        config = DictConfig({"exclude_every_n_frames": 3})
        mask = HoldOutFrameSampler.training_frame_mask(config, n_frames=6)
        np.testing.assert_array_equal(mask, [False, True, True, False, True, True])

    def test_mask_matches_sample_frame_probabilities(self):
        """The mask is exactly the support of sample_frame's probability vector."""
        for config_dict in [
            {"include_frames_start": 1, "include_every_n_frames": 3},
            {"exclude_frames_start": 2, "exclude_every_n_frames": 2},
        ]:
            with self.subTest(config=config_dict):
                config = DictConfig(config_dict)
                n_frames = 9
                mask = HoldOutFrameSampler.training_frame_mask(config, n_frames)
                sampler = HoldOutFrameSampler(config=config, dataset=Mock())
                if sampler.include_every_n_frames is not None:
                    p = np.zeros(n_frames)
                    p[sampler.include_frames_start :: sampler.include_every_n_frames] = 1
                else:
                    p = np.ones(n_frames)
                    p[sampler.exclude_frames_start :: sampler.exclude_every_n_frames] = 0
                np.testing.assert_array_equal(mask, p > 0)

    def test_missing_modes_raise(self):
        with self.assertRaises(ValueError):
            HoldOutFrameSampler.training_frame_mask(DictConfig({}), n_frames=5)

    def test_include_mode_selecting_no_frame_raises(self):
        # include_frames_start past the range selects nothing; sample_frame
        # would then divide by p.sum() == 0, so fail fast in the mask.
        with self.assertRaises(IndexError):
            HoldOutFrameSampler.training_frame_mask(
                DictConfig({"include_frames_start": 9, "include_every_n_frames": 2}), n_frames=5
            )


class TestCoverageFrameSampler(unittest.TestCase):
    """CoverageFrameSampler: coverage draw with an OPTIONAL hold-out."""

    def setUp(self):
        self.mock_dataset = Mock(spec="nurec.datasets.ncore.NCORETrainDataset")
        self.frame_range = range(20, 81, 10)  # {20,30,40,50,60,70,80}
        self.rng = Mock(spec=np.random.Generator)  # unused: coverage seeds its own generator

    def _pass(self, sampler, n, sensor_id="cam0"):
        return [
            sampler.sample_frame(
                rng=self.rng, batch_idx=i, frame_range=self.frame_range, unique_sensor_id=sensor_id
            ).sampled_frame_idx
            for i in range(n)
        ]

    def test_no_holdout_covers_all_frames_each_pass(self):
        s = CoverageFrameSampler(config=DictConfig({}), dataset=self.mock_dataset)
        self.assertEqual(sorted(self._pass(s, len(self.frame_range))), list(self.frame_range))

    def test_coverage_plan_property_requires_injection(self):
        sampler = CoverageFrameSampler(config=DictConfig({}), dataset=self.mock_dataset)
        with self.assertRaisesRegex(RuntimeError, "without an injected CoveragePlan"):
            _ = sampler.coverage_plan

    def test_coverage_plan_property_returns_injected_plan(self):
        sampler = CoverageFrameSampler(config=DictConfig({}), dataset=self.mock_dataset)
        plan = CoveragePlan.single_sensor("cam0", np.asarray([20], dtype=np.int64))
        sampler.set_coverage_plan(plan)
        self.assertIs(sampler.coverage_plan, plan)

    def test_exclude_holdout_covers_only_training_subset(self):
        # exclude every 3rd from index 2 -> held out {40,70}; training {20,30,50,60,80}
        s = CoverageFrameSampler(
            DictConfig({"exclude_frames_start": 2, "exclude_every_n_frames": 3}), self.mock_dataset
        )
        training = [20, 30, 50, 60, 80]
        self.assertEqual(sorted(self._pass(s, len(training))), training)
        self.assertEqual(set(self._pass(s, 2 * len(training))), set(training))  # held-out never sampled

    def test_include_holdout_covers_only_included_subset(self):
        s = CoverageFrameSampler(
            DictConfig({"include_frames_start": 2, "include_every_n_frames": 3}), self.mock_dataset
        )
        self.assertEqual(sorted(self._pass(s, 2)), [40, 70])

    def test_deterministic_per_batch_idx(self):
        a = CoverageFrameSampler(DictConfig({}), self.mock_dataset)
        b = CoverageFrameSampler(DictConfig({}), self.mock_dataset)
        self.assertEqual(self._pass(a, 20), self._pass(b, 20))

    def test_non_positive_stride_raises(self):
        # A stride of 0 (cryptic ValueError in the slice) or negative (silent reverse-slice subset)
        # must be rejected up front, for both include and exclude modes.
        for cfg in [
            {"include_frames_start": 0, "include_every_n_frames": 0},
            {"include_frames_start": 0, "include_every_n_frames": -2},
            {"exclude_frames_start": 0, "exclude_every_n_frames": 0},
            {"exclude_frames_start": 0, "exclude_every_n_frames": -3},
        ]:
            with self.subTest(cfg=cfg):
                with self.assertRaises(AssertionError):
                    CoverageFrameSampler(DictConfig(cfg), self.mock_dataset)
                with self.assertRaises(AssertionError):
                    CoverageFrameSampler.training_frame_mask(DictConfig(cfg), n_frames=7)

    def test_empty_training_set_raises(self):
        # exclude_every_n_frames=1 (start 0) excludes every frame -> 0 eligible training frames. Mirroring
        # include-mode, training_frame_mask must fail fast instead of returning an all-empty mask that
        # mapping setup / sample_frame would silently accept.
        cfg = DictConfig({"exclude_frames_start": 0, "exclude_every_n_frames": 1})
        with self.assertRaises(IndexError):
            CoverageFrameSampler.training_frame_mask(cfg, n_frames=len(self.frame_range))
        s = CoverageFrameSampler(cfg, self.mock_dataset)
        with self.assertRaises(IndexError):
            s.sample_frame(rng=self.rng, batch_idx=0, frame_range=self.frame_range, unique_sensor_id="cam0")

    def test_degenerate_exclude_keeping_one_frame_is_allowed(self):
        # exclude_frames_start=1, exclude_every_n_frames=1 keeps only local frame 0 -> 1 training frame.
        # This is unusual but satisfiable, so it must NOT be rejected (result-based guard, not stride-based).
        s = CoverageFrameSampler(
            DictConfig({"exclude_frames_start": 1, "exclude_every_n_frames": 1}), self.mock_dataset
        )
        self.assertEqual(set(self._pass(s, 3)), {list(self.frame_range)[0]})

    def test_both_include_and_exclude_raises(self):
        # Mutual exclusivity must be enforced on BOTH the constructor and the training_frame_mask
        # classmethod (the latter is reached directly by group_training_frame_mask during mapping setup).
        cfg = DictConfig(
            {
                "include_frames_start": 1,
                "include_every_n_frames": 2,
                "exclude_frames_start": 3,
                "exclude_every_n_frames": 4,
            }
        )
        with self.assertRaises(AssertionError):
            CoverageFrameSampler(cfg, self.mock_dataset)
        with self.assertRaises(AssertionError):
            CoverageFrameSampler.training_frame_mask(cfg, n_frames=7)

    def test_training_frame_mask_no_holdout_is_all_true(self):
        np.testing.assert_array_equal(
            CoverageFrameSampler.training_frame_mask(DictConfig({}), n_frames=5), np.ones(5, dtype=bool)
        )

    def test_training_frame_mask_matches_sample_reachable_set(self):
        # mask must mirror the frames sample_frame can actually return
        for cfg in [
            {},
            {"exclude_frames_start": 2, "exclude_every_n_frames": 3},
            {"include_frames_start": 2, "include_every_n_frames": 3},
        ]:
            with self.subTest(cfg=cfg):
                mask = CoverageFrameSampler.training_frame_mask(DictConfig(cfg), n_frames=len(self.frame_range))
                reachable = sorted(set(self._pass(CoverageFrameSampler(DictConfig(cfg), self.mock_dataset), 200)))
                reachable_local = [list(self.frame_range).index(f) for f in reachable]
                expect = np.zeros(len(self.frame_range), dtype=bool)
                expect[reachable_local] = True
                np.testing.assert_array_equal(mask, expect)


# --- Shared helpers for the joint (sensor, frame) coverage tests -----------------------------------


def _make_plan(sensors: list[tuple[str, str, list[int]]]) -> CoveragePlan:
    """Build a CoveragePlan directly from ``[(sensor_id, unique_id, [frame values]), ...]``."""
    per_sensor = tuple(np.asarray(frames, dtype=np.int64) for _, _, frames in sensors)
    prefix = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum([v.shape[0] for v in per_sensor], dtype=np.int64)])
    return CoveragePlan(
        sensor_ids=tuple(s for s, _, _ in sensors),
        unique_sensor_ids=tuple(u for _, u, _ in sensors),
        per_sensor_frame_idxs=per_sensor,
        prefix=prefix,
    )


class _CoveragePlanDataset(torch.utils.data.Dataset):
    """Minimal map-style dataset that decodes coverage work inside a dataloader worker."""

    def __init__(self, plan: CoveragePlan, length: int) -> None:
        self.plan = plan
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, batch_idx: int) -> dict[str, int]:
        return {"pid": os.getpid(), "frame_idx": self.plan.decode(batch_idx).frame_idx}


class TestCoveragePlanRestriction(unittest.TestCase):
    def test_iter_pairs_uses_stable_sensor_then_frame_order(self):
        plan = _make_plan([("camA", "uidA", [2, 4]), ("camB", "uidB", [1])])
        self.assertEqual(list(plan.iter_pairs()), [("camA", 2), ("camA", 4), ("camB", 1)])

    def test_restriction_is_uniform_over_eligible_pairs_and_allows_empty_sensor(self):
        plan = _make_plan([("camA", "uidA", [0, 1, 2]), ("camB", "uidB", [10, 11])])
        restriction = plan.restrict_to_pairs({("camB", 10), ("camB", 11)})

        self.assertEqual(restriction.original_total, 5)
        self.assertEqual(restriction.eligible_total, 2)
        self.assertEqual(restriction.eligible_fraction, 2 / 5)
        self.assertEqual(restriction.plan.per_sensor_frame_idxs[0].tolist(), [])
        decoded = [restriction.plan.decode(i) for i in range(2)]
        self.assertEqual({(item.sensor_id, item.frame_idx) for item in decoded}, {("camB", 10), ("camB", 11)})

    def test_restriction_preserves_pair_weight_across_unequal_cameras(self):
        plan = _make_plan([("camA", "uidA", list(range(600))), ("camB", "uidB", [0])])
        restriction = plan.restrict_to_pairs(plan.iter_pairs())
        counts = Counter(restriction.plan.decode(i).sensor_id for i in range(restriction.plan.total))

        self.assertIs(restriction.plan, plan)
        self.assertEqual(counts, {"camA": 600, "camB": 1})
        self.assertEqual(restriction.eligible_fraction, 1.0)

    def test_unknown_eligible_pair_raises(self):
        plan = _make_plan([("camA", "uidA", [0])])
        with self.assertRaisesRegex(ValueError, "not present"):
            plan.restrict_to_pairs({("camA", 1)})

    def test_globally_empty_restriction_raises(self):
        plan = _make_plan([("camA", "uidA", [0])])
        with self.assertRaisesRegex(ValueError, "at least one"):
            plan.restrict_to_pairs([])

    def test_restricting_empty_source_plan_raises(self):
        plan = _make_plan([("camA", "uidA", [])])
        with self.assertRaisesRegex(ValueError, "empty CoveragePlan"):
            plan.restrict_to_pairs([])

    def test_epoch_local_indices_continue_the_global_coverage_schedule(self):
        baseline = _make_plan([("camA", "uidA", [0, 1, 2, 3, 4])])
        plan = _make_plan([("camA", "uidA", [0, 1, 2, 3, 4])])
        samples_per_epoch = 3

        decoded = []
        for epoch in range(2):
            plan.set_batch_index_offset(epoch * samples_per_epoch)
            decoded.extend(plan.decode(batch_idx) for batch_idx in range(samples_per_epoch))

        expected = [baseline.decode(global_batch_idx) for global_batch_idx in range(2 * samples_per_epoch)]
        self.assertEqual(
            [(item.sensor_id, item.frame_idx) for item in decoded],
            [(item.sensor_id, item.frame_idx) for item in expected],
        )
        self.assertTrue(plan._batch_index_offset.is_shared())

    def test_negative_batch_index_offset_raises(self):
        plan = _make_plan([("camA", "uidA", [0])])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            plan.set_batch_index_offset(-1)

    def test_persistent_worker_observes_epoch_offset_updates(self):
        baseline = _make_plan([("camA", "uidA", [0, 1, 2, 3, 4])])
        plan = _make_plan([("camA", "uidA", [0, 1, 2, 3, 4])])
        samples_per_epoch = 3
        plan.set_batch_index_offset(0)
        loader = torch.utils.data.DataLoader(
            _CoveragePlanDataset(plan, samples_per_epoch),
            batch_size=None,
            num_workers=1,
            persistent_workers=True,
            prefetch_factor=1,
        )
        try:
            first_epoch = list(loader)
            plan.set_batch_index_offset(samples_per_epoch)
            second_epoch = list(loader)
        finally:
            if loader._iterator is not None:
                loader._iterator._shutdown_workers()

        worker_pids = {item["pid"] for item in first_epoch + second_epoch}
        self.assertEqual(len(worker_pids), 1)
        self.assertNotIn(os.getpid(), worker_pids)
        expected = [baseline.decode(global_batch_idx).frame_idx for global_batch_idx in range(6)]
        self.assertEqual([item["frame_idx"] for item in first_epoch + second_epoch], expected)


def _simulate_batch_loop(
    sensor_sampler,
    frame_sampler,
    active_ids: list[str],
    unique_of: dict[str, str],
    frange_of: dict[str, range],
    n_steps: int,
) -> list[tuple[str, int]]:
    """Mimic ``NCORETrainDataset.get_train_batch``: pick a sensor, then a frame of that sensor, per step."""
    rng = np.random.default_rng(0)
    pairs: list[tuple[str, int]] = []
    for batch_idx in range(n_steps):
        sensor_id = sensor_sampler.sample_sensor(rng, batch_idx, list(active_ids)).sampled_sensor_ids[0]
        frame_idx = frame_sampler.sample_frame(
            rng, batch_idx, frame_range=frange_of[sensor_id], unique_sensor_id=unique_of[sensor_id]
        ).sampled_frame_idx
        pairs.append((sensor_id, int(frame_idx)))
    return pairs


class TestCoverageMultiSensor(unittest.TestCase):
    """The joint schedule must cover every (sensor, training-frame) pair once per pass across ALL sensors,
    even though the pipeline picks the sensor first and the frame second."""

    def setUp(self):
        self.mock_dataset = Mock(spec="nurec.datasets.ncore.NCORETrainDataset")
        # Three cameras with DIFFERENT frame counts, to exercise the ragged prefix-sum decode.
        self.sensors = [("camA", "uidA", [10, 20, 30]), ("camB", "uidB", [0, 1]), ("camC", "uidC", [5, 6, 7, 8])]
        self.active_ids = [s for s, _, _ in self.sensors]
        self.unique_of = {s: u for s, u, _ in self.sensors}
        self.frange_of = {"camA": range(10, 31, 10), "camB": range(0, 2), "camC": range(5, 9)}
        self.plan = _make_plan(self.sensors)
        self.total = 9  # 3 + 2 + 4
        self.expected_pairs = {(s, f) for s, _, frames in self.sensors for f in frames}

    def _paired_samplers(self):
        sensor = CoverageSensorSampler(DictConfig({"sample_all_sensors": False}), self.mock_dataset)
        frame = CoverageFrameSampler(DictConfig({}), self.mock_dataset)
        sensor.set_coverage_plan(self.plan)
        frame.set_coverage_plan(self.plan)
        return sensor, frame

    def test_one_pass_covers_every_sensor_frame_pair_exactly_once(self):
        sensor, frame = self._paired_samplers()
        pairs = _simulate_batch_loop(sensor, frame, self.active_ids, self.unique_of, self.frange_of, self.total)
        self.assertEqual(sorted(pairs), sorted(self.expected_pairs))  # each pair exactly once (no dup, no miss)

    def test_two_passes_cover_each_pair_exactly_twice(self):
        sensor, frame = self._paired_samplers()
        pairs = _simulate_batch_loop(sensor, frame, self.active_ids, self.unique_of, self.frange_of, 2 * self.total)
        counts = Counter(pairs)
        self.assertEqual(set(counts), self.expected_pairs)
        self.assertTrue(all(c == 2 for c in counts.values()))

    def test_deterministic_across_instances(self):
        s1, f1 = self._paired_samplers()
        s2, f2 = self._paired_samplers()
        a = _simulate_batch_loop(s1, f1, self.active_ids, self.unique_of, self.frange_of, 3 * self.total)
        b = _simulate_batch_loop(s2, f2, self.active_ids, self.unique_of, self.frange_of, 3 * self.total)
        self.assertEqual(a, b)

    def test_frame_sampler_rejects_sensor_the_plan_did_not_schedule(self):
        # If the sensor chosen upstream is not the one the shared plan scheduled (e.g. a non-coverage
        # sensor sampler was wired in by mistake), fail loudly instead of silently mis-covering.
        _, frame = self._paired_samplers()
        with self.assertRaises(AssertionError):
            frame.sample_frame(rng=Mock(), batch_idx=0, frame_range=range(10, 31, 10), unique_sensor_id="uidZ")


class TestCoverageSingleSensorInvariance(unittest.TestCase):
    """With a single active sensor, frame coverage must hold regardless of the sensor sampler, since the
    lone sensor is drawn every step."""

    def setUp(self):
        self.mock_dataset = Mock(spec="nurec.datasets.ncore.NCORETrainDataset")
        self.frame_range = range(20, 81, 10)  # {20,30,40,50,60,70,80}
        self.active_ids = ["cam0"]
        self.unique_of = {"cam0": "uid0"}
        self.frange_of = {"cam0": self.frame_range}

    def _sensor_samplers(self):
        return {
            "coverage": CoverageSensorSampler(DictConfig({"sample_all_sensors": False}), self.mock_dataset),
            "uniform_one": UniformSensorSampler(DictConfig({"sample_all_sensors": False}), self.mock_dataset),
            "uniform_all": UniformSensorSampler(DictConfig({"sample_all_sensors": True}), self.mock_dataset),
        }

    def test_coverage_holds_with_injected_plan_for_any_sensor_sampler(self):
        plan = _make_plan([("cam0", "uid0", list(self.frame_range))])
        for name, sensor in self._sensor_samplers().items():
            with self.subTest(sensor=name):
                sensor.set_coverage_plan(plan)  # no-op for the uniform samplers
                frame = CoverageFrameSampler(DictConfig({}), self.mock_dataset)
                frame.set_coverage_plan(plan)
                pairs = _simulate_batch_loop(
                    sensor, frame, self.active_ids, self.unique_of, self.frange_of, len(self.frame_range)
                )
                self.assertEqual(sorted(f for _, f in pairs), list(self.frame_range))

    def test_fallback_single_sensor_coverage_without_plan(self):
        # No plan injected: the frame sampler's single-sensor fall-back must still cover every frame,
        # because the single sensor is drawn on every step.
        for name, sensor in self._sensor_samplers().items():
            if name == "coverage":
                continue  # the coverage sensor sampler requires a plan
            with self.subTest(sensor=name):
                frame = CoverageFrameSampler(DictConfig({}), self.mock_dataset)  # no plan
                pairs = _simulate_batch_loop(
                    sensor, frame, self.active_ids, self.unique_of, self.frange_of, len(self.frame_range)
                )
                self.assertEqual(sorted(f for _, f in pairs), list(self.frame_range))


class TestCoveragePairingGuard(unittest.TestCase):
    """BaseBatchSampler must reject sensor/frame sampler combinations that silently break coverage."""

    def test_lone_coverage_sensor_sampler_rejected(self):
        with self.assertRaises(AssertionError):
            BaseBatchSampler._validate_coverage_pairing("coverage", "uniform", 1, "camera")

    def test_multi_sensor_coverage_frame_requires_coverage_sensor(self):
        with self.assertRaises(AssertionError):
            BaseBatchSampler._validate_coverage_pairing("uniform", "coverage", 3, "camera")

    def test_single_sensor_coverage_frame_allows_any_sensor_sampler(self):
        BaseBatchSampler._validate_coverage_pairing("uniform", "coverage", 1, "camera")  # must not raise

    def test_paired_coverage_ok_for_any_sensor_count(self):
        BaseBatchSampler._validate_coverage_pairing("coverage", "coverage", 5, "camera")  # must not raise

    def test_non_coverage_combo_ok(self):
        BaseBatchSampler._validate_coverage_pairing("uniform", "uniform", 5, "camera")  # must not raise


class TestCoveragePlanBuild(unittest.TestCase):
    """CoveragePlan.build must enumerate the ACTIVE sensors' training frames using the same reachable-set
    definition as training_frame_mask."""

    @staticmethod
    def _fake_dataset(sensor_ids: list[str], frame_ranges: dict[str, range], sensor_kind: str = "camera"):
        """Minimal stand-in exposing only the attributes CoveragePlan.build reads, for either kind."""
        dataset = Mock(spec="nurec.datasets.ncore.NCORETrainDataset")
        datasource = Mock()
        unique_ids = {sid: Mock(id=f"uid_{sid}") for sid in frame_ranges}
        if sensor_kind == "camera":
            dataset.camera_ids = sensor_ids
            datasource.camera_frame_ranges = frame_ranges
            datasource.camera_unique_ids = unique_ids
        else:
            dataset.lidar_ids = sensor_ids
            datasource.lidar_frame_ranges = frame_ranges
            datasource.lidar_unique_ids = unique_ids
        dataset.datasource = datasource
        return dataset

    def test_build_no_holdout_covers_all_active_frames(self):
        dataset = self._fake_dataset(["camA", "camB"], {"camA": range(0, 3), "camB": range(0, 2)})
        plan = CoveragePlan.build(dataset, "camera", DictConfig({}))
        self.assertEqual(plan.total, 5)
        pairs = sorted((plan.decode(b).sensor_id, plan.decode(b).frame_idx) for b in range(plan.total))
        self.assertEqual(pairs, sorted([("camA", 0), ("camA", 1), ("camA", 2), ("camB", 0), ("camB", 1)]))

    def test_build_lidar_branch_is_parallel_to_camera(self):
        # The lidar side must build identically from the lidar_* attributes / active lidar set.
        dataset = self._fake_dataset(["lidarA"], {"lidarA": range(0, 4)}, sensor_kind="lidar")
        plan = CoveragePlan.build(
            dataset, "lidar", DictConfig({"exclude_frames_start": 0, "exclude_every_n_frames": 2})
        )
        # exclude indices 0, 2 -> training frames {1, 3}
        self.assertEqual(sorted(plan.per_sensor_frame_idxs[0].tolist()), [1, 3])
        self.assertEqual(plan.decode(0).unique_sensor_id, "uid_lidarA")

    def test_build_carries_unique_ids_for_coupling_assert(self):
        dataset = self._fake_dataset(["camA"], {"camA": range(0, 2)})
        plan = CoveragePlan.build(dataset, "camera", DictConfig({}))
        self.assertEqual(plan.decode(0).unique_sensor_id, "uid_camA")

    def test_build_respects_exclude_holdout(self):
        dataset = self._fake_dataset(["camA"], {"camA": range(0, 6)})
        plan = CoveragePlan.build(
            dataset, "camera", DictConfig({"exclude_frames_start": 0, "exclude_every_n_frames": 3})
        )
        # exclude indices 0, 3 -> training frames {1, 2, 4, 5}
        self.assertEqual(sorted(plan.per_sensor_frame_idxs[0].tolist()), [1, 2, 4, 5])

    def test_build_empty_training_set_raises(self):
        # An exclude config that leaves no training frame must fail fast (via training_frame_mask) at build.
        dataset = self._fake_dataset(["camA"], {"camA": range(0, 4)})
        with self.assertRaises(IndexError):
            CoveragePlan.build(dataset, "camera", DictConfig({"exclude_frames_start": 0, "exclude_every_n_frames": 1}))

    def test_build_and_inject_wires_both_samplers(self):
        # build_and_inject_coverage_plan (invoked by the batch sampler, keeping CoveragePlan out of `base`)
        # must inject the SAME plan into the frame sampler and the paired sensor sampler.
        dataset = self._fake_dataset(["camA", "camB"], {"camA": range(0, 3), "camB": range(0, 2)})
        frame = CoverageFrameSampler(DictConfig({}), dataset)
        sensor = CoverageSensorSampler(DictConfig({"sample_all_sensors": False}), dataset)
        frame.build_and_inject_coverage_plan(dataset, "camera", sensor)
        self.assertIs(frame._coverage_plan, sensor._coverage_plan)
        self.assertIsNotNone(frame._coverage_plan)
        self.assertEqual(frame._coverage_plan.total, 5)

    def test_base_frame_sampler_build_and_inject_is_noop(self):
        # A non-coverage frame sampler must accept the call and do nothing (no plan, no error), so the
        # batch sampler can invoke it uniformly without importing/knowing about the holdout module.
        dataset = self._fake_dataset(["camA"], {"camA": range(0, 3)})
        uniform_frame = UniformFrameSampler(DictConfig({}), dataset)
        sensor = UniformSensorSampler(DictConfig({"sample_all_sensors": False}), dataset)
        uniform_frame.build_and_inject_coverage_plan(dataset, "camera", sensor)  # must not raise
