# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from typing import Optional, Sequence

import numpy as np
import torch

from nurec.models.gaussians.collect import TrackTimeData
from nurec.models.gaussians.segment_plan import (
    LayerSegmentTable,
    assert_sparse_segments_supported,
    collect_ranges,
    rows_grouped_by_track,
    upload_segments,
    whole_layer_ranges,
)


CPU = torch.device("cpu")


def _time_data(ranges_us: Sequence[Optional[tuple[int, int]]], gap_us: int = 100) -> TrackTimeData:
    """Two poses per track at the given (first, last) timestamps; None means no poses."""
    timestamps: list[int] = []
    packinfo: list[tuple[int, int]] = []
    for r in ranges_us:
        if r is None:
            packinfo.append((len(timestamps), 0))
            continue
        packinfo.append((len(timestamps), 2))
        timestamps.extend(r)
    return TrackTimeData(
        tracks_timestamps_us=torch.tensor(timestamps, dtype=torch.int64),
        tracks_packinfo=torch.tensor(packinfo, dtype=torch.int32).reshape(-1, 2),
        max_positive_timestamp_gap_us=torch.full((len(ranges_us),), gap_us, dtype=torch.int64),
    )


def _selection(starts: Sequence[int], counts: Sequence[int]) -> np.ndarray:
    """A partial-layer selection built by hand, as a future selector would."""
    return np.array(list(zip(starts, counts)), dtype=np.int64).reshape(-1, 2)


def _kernel_accepts(first: int, last: int, tol: int, query_us: int) -> bool:
    """Acceptance rule of fetch_tracks in collector.slang for one query timestamp."""
    if last < first:
        return False
    if first <= query_us <= last:
        return True
    return tol > 0 and min(abs(query_us - first), abs(query_us - last)) <= tol


class TestLayerSegmentTable(unittest.TestCase):
    def test_live_is_a_superset_of_the_kernel_rule(self) -> None:
        rng = np.random.default_rng(0)
        ranges = [(int(a), int(a + b)) for a, b in zip(rng.integers(0, 1000, 40), rng.integers(0, 300, 40))]
        table = LayerSegmentTable.build(torch.arange(40), _time_data(ranges, gap_us=25))
        for _ in range(200):
            start = int(rng.integers(-200, 1400))
            exposure = int(rng.integers(0, 60))
            end = start + exposure
            selected = table.select(start, end, exposure)
            live = np.zeros(40, dtype=bool)
            for selected_start, selected_count in selected:
                live[selected_start : selected_start + selected_count] = True
            queries = np.linspace(start, end, 7).astype(np.int64)
            for track, (first, last) in enumerate(ranges):
                accepted = any(_kernel_accepts(first, last, 25 + exposure, int(q)) for q in queries)
                if accepted:
                    self.assertTrue(live[track], f"track {track} accepted by the kernel but not live")
                # The host rule is exact for query timestamps inside the frame window.
                if not live[track]:
                    self.assertFalse(accepted)

    def test_grouped_rows_give_one_range_per_track(self) -> None:
        table = LayerSegmentTable.build(
            torch.tensor([2, 2, 2, 0, 0, 5], dtype=torch.int32),
            _time_data([(100, 110), (100, 110), (0, 10), None, None, (100, 110)], gap_us=0),
        )
        np.testing.assert_array_equal(table.select(0, 10, 0), [[0, 3]])
        self.assertEqual(table.row_count, 6)

    def test_track_without_poses_is_never_selected(self) -> None:
        table = LayerSegmentTable.build(torch.tensor([0, 1, 2]), _time_data([(10, 20), None, (5, 5)]))
        np.testing.assert_array_equal(table.select(0, 1_000, 0), [[0, 1], [2, 1]])

    def test_empty_layer(self) -> None:
        table = LayerSegmentTable.build(torch.empty(0, dtype=torch.int32), None)
        self.assertEqual(table.select(0, 0, 0).shape, (0, 2))

    def test_selection_follows_the_tracks_time_ranges(self) -> None:
        track_ids = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int32)
        time_data = _time_data([(0, 10), (100, 110), (200, 210)], gap_us=0)
        table = LayerSegmentTable.build(track_ids, time_data)
        self.assertIs(LayerSegmentTable.for_layer(table, track_ids, time_data), table)
        self.assertIsNot(LayerSegmentTable.for_layer(table, track_ids.clone(), time_data), table)

        selection = table.select(frame_start_us=100, frame_end_us=105, frame_exposure_us=5)
        np.testing.assert_array_equal(selection, [[2, 3]])

        everything = table.select(frame_start_us=0, frame_end_us=210, frame_exposure_us=0)
        np.testing.assert_array_equal(everything, whole_layer_ranges(6))

        nothing = table.select(frame_start_us=500, frame_end_us=510, frame_exposure_us=0)
        self.assertEqual(nothing.shape, (0, 2))

    def test_new_pose_timestamps_replace_the_live_ranges(self) -> None:
        # An edit that moves a track in time while every row keeps its track id: the
        # windows still hold, last frame's live ranges do not.
        track_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        table = LayerSegmentTable.build(track_ids, _time_data([(0, 10), (100, 110)], gap_us=0))
        np.testing.assert_array_equal(table.select(0, 10, 0), _selection([0], [2]))

        moved = LayerSegmentTable.for_layer(table, track_ids, _time_data([(500, 510), (100, 110)], gap_us=0))
        self.assertIs(moved, table, "new times do not rebuild the row runs")
        self.assertEqual(moved.select(0, 10, 0).shape, (0, 2), "track 0 no longer lives at time 0")
        np.testing.assert_array_equal(moved.select(500, 510, 0), _selection([0], [2]))

    def test_timestamp_only_replacement_refreshes_time_bounds(self) -> None:
        track_ids = torch.tensor([0, 0], dtype=torch.int32)
        original = _time_data([(0, 10)], gap_us=0)
        table = LayerSegmentTable.build(track_ids, original)
        moved = TrackTimeData(
            tracks_timestamps_us=torch.tensor([500, 510], dtype=torch.int64),
            tracks_packinfo=original.tracks_packinfo,
            max_positive_timestamp_gap_us=original.max_positive_timestamp_gap_us,
        )

        self.assertIs(LayerSegmentTable.for_layer(table, track_ids, moved), table)
        self.assertEqual(table.select(0, 10, 0).shape, (0, 2))
        np.testing.assert_array_equal(table.select(500, 510, 0), whole_layer_ranges(2))

    def test_selection_covers_the_same_rows_while_the_live_tracks_are_unchanged(self) -> None:
        track_ids = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int32)
        table = LayerSegmentTable.build(track_ids, _time_data([(0, 10), (100, 110), (200, 210)], gap_us=0))
        first = table.select(100, 105, 0)
        np.testing.assert_array_equal(table.select(101, 106, 0), first)
        self.assertFalse(np.array_equal(table.select(200, 205, 0), first))
        np.testing.assert_array_equal(table.select(0, 210, 0), whole_layer_ranges(5))

    def test_same_size_replacement_rebuilds_the_table(self) -> None:
        original = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        replacement = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
        time_data = _time_data([(0, 10), (100, 110)], gap_us=0)
        table = LayerSegmentTable.build(original, time_data)

        rebuilt = LayerSegmentTable.for_layer(table, replacement, time_data)
        self.assertIsNot(rebuilt, table)
        selection = rebuilt.select(frame_start_us=0, frame_end_us=10, frame_exposure_us=0)
        np.testing.assert_array_equal(selection, [[0, 1], [2, 1]])

    def test_interleaved_live_tracks_preserve_each_run(self) -> None:
        track_ids = torch.tensor([0, 1, 0, 2, 1, 2], dtype=torch.int32)
        table = LayerSegmentTable.build(track_ids, _time_data([(0, 10), (100, 110), (0, 10)], gap_us=0))

        selection = table.select(frame_start_us=0, frame_end_us=10, frame_exposure_us=0)
        np.testing.assert_array_equal(selection, [[0, 1], [2, 1], [3, 1], [5, 1]])

    def test_static_layer_is_always_whole(self) -> None:
        track_ids = torch.tensor([0, 0, 1], dtype=torch.int32)
        table = LayerSegmentTable.build(track_ids, time_data=None)
        np.testing.assert_array_equal(table.select(0, 0, 0), whole_layer_ranges(3))
        self.assertIs(LayerSegmentTable.for_layer(table, track_ids, None), table)

    def test_fragmented_layer_is_rejected(self) -> None:
        # 200 runs for 2 tracks: not grouped by track. The composite re-sorts such a
        # layer at load, so a table built from one is a bug rather than a slow path.
        track_ids = torch.tensor([0, 1] * 100, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "grouped by track"):
            LayerSegmentTable.build(track_ids, _time_data([(0, 10), (1_000, 1_010)]), layer_name="rigid")


class TestUploadSegments(unittest.TestCase):
    def test_cacheable_layer_is_whole_when_cold_and_nothing_when_warm_while_draw_is_unchanged(self) -> None:
        # static layer (cacheable, layer 0) + rigid layer (variant, layer 1) with one live track.
        # static [0, 4) + track 1 (layer rows [2, 4)) -> draw segments [0, 4) and [6, 8).
        rigid = LayerSegmentTable.build(
            torch.tensor([0, 0, 1, 1, 2], dtype=torch.int32), _time_data([(0, 10), (100, 110), (200, 210)], gap_us=0)
        )
        selections = [whole_layer_ranges(4), rigid.select(100, 105, 0)]

        cold_collect, cold_draw = upload_segments(
            CPU,
            selections,
            layer_sizes=[4, 5],
            layer_offsets=[0, 4],
            variant_ids=[1],
            cold=True,
        )
        self.assertEqual(cold_collect[0].n_segments, 0, "the cacheable layer is collected whole on a cold frame")
        self.assertEqual(cold_collect[0].active_count, 4)
        rigid_segments = cold_collect[1]
        self.assertEqual(rigid_segments.n_segments, 1, "the variant layer is windowed")
        assert rigid_segments.starts is not None and rigid_segments.window_size_prefix is not None
        self.assertEqual(rigid_segments.starts.tolist(), [2])
        self.assertEqual(rigid_segments.window_size_prefix.tolist(), [0, 2])
        self.assertEqual(rigid_segments.active_count, 2)
        assert cold_draw is not None
        self.assertEqual(cold_draw.segments.tolist(), [[0, 6], [4, 2]])
        self.assertEqual(cold_draw.segment_offsets.tolist(), [0, 4, 6])
        self.assertEqual(cold_draw.active_count, 6)
        self.assertEqual(cold_draw.total_rows, 9)

        warm_collect, warm_draw = upload_segments(
            CPU,
            selections,
            layer_sizes=[4, 5],
            layer_offsets=[0, 4],
            variant_ids=[1],
            cold=False,
        )
        self.assertEqual(warm_collect[0].n_segments, 0, "a warm cacheable layer is still nothing(), not windowed")
        self.assertEqual(warm_collect[0].active_count, 0, "a warm cacheable layer collects no rows")
        assert warm_collect[1].starts is not None
        self.assertEqual(warm_collect[1].starts.tolist(), [2], "the variant layer is windowed on a warm frame too")
        self.assertEqual(warm_collect[1].active_count, 2)
        assert warm_draw is not None
        self.assertEqual(warm_draw.segments.tolist(), cold_draw.segments.tolist(), "draw ignores cold and variants")
        self.assertEqual(warm_draw.segment_offsets.tolist(), cold_draw.segment_offsets.tolist())
        self.assertEqual(warm_draw.active_count, cold_draw.active_count)

    def test_no_track_is_live(self) -> None:
        rigid = LayerSegmentTable.build(torch.tensor([0, 0], dtype=torch.int32), _time_data([(0, 1)], gap_us=0))
        collect, draw = upload_segments(
            CPU,
            [whole_layer_ranges(3), rigid.select(100, 100, 0)],
            layer_sizes=[3, 2],
            layer_offsets=[0, 3],
            variant_ids=[0, 1],
            cold=True,
        )
        rigid_segments = collect[1]
        self.assertEqual(rigid_segments.active_count, 0)
        self.assertIsNone(rigid_segments.window_size_prefix, "an empty collect binds no segment tensor")
        assert draw is not None
        self.assertEqual(draw.segments.tolist(), [[0], [3]])
        self.assertEqual(draw.active_count, 3)

    def test_touching_windows_merge_across_layers(self) -> None:
        # static layer [0, 4) + rigid layer with live windows [4, 6) and [7, 9) of 5 rows.
        rigid = LayerSegmentTable.build(
            torch.tensor([0, 0, 1, 2, 2], dtype=torch.int32), _time_data([(0, 10), (500, 510), (0, 10)], gap_us=0)
        ).select(frame_start_us=0, frame_end_us=10, frame_exposure_us=0)
        _, draw = upload_segments(
            CPU,
            [whole_layer_ranges(4), rigid],
            layer_sizes=[4, 5],
            layer_offsets=[0, 4],
            variant_ids=[1],
            cold=True,
        )
        assert draw is not None
        self.assertEqual(draw.segments.tolist(), [[0, 7], [6, 2]])
        self.assertEqual(draw.segment_offsets.tolist(), [0, 6, 8])
        self.assertEqual(draw.active_count, 8)
        self.assertEqual(draw.total_rows, 9)
        self.assertEqual(draw.segments.dtype, torch.int32)

    def test_fully_live_scene_gives_no_draw_plan(self) -> None:
        # Everything live: draw merges to the whole scene, so there is nothing sparse to draw.
        _, draw = upload_segments(
            CPU,
            [whole_layer_ranges(3), whole_layer_ranges(5)],
            layer_sizes=[3, 5],
            layer_offsets=[0, 3],
            variant_ids=[],
            cold=True,
        )
        self.assertIsNone(draw)

    def test_empty_layers_and_empty_selection(self) -> None:
        rigid = LayerSegmentTable.build(torch.tensor([0, 0], dtype=torch.int32), _time_data([(0, 1)], gap_us=0)).select(
            frame_start_us=100, frame_end_us=100, frame_exposure_us=0
        )
        _, draw = upload_segments(
            CPU,
            [whole_layer_ranges(0), rigid],
            layer_sizes=[0, 2],
            layer_offsets=[0, 0],
            variant_ids=[1],
            cold=True,
        )
        assert draw is not None
        self.assertEqual(draw.segments.shape, (2, 0))
        self.assertEqual(draw.segment_offsets.tolist(), [0])
        self.assertEqual(draw.active_count, 0)

    def test_total_rows_must_fit_int32(self) -> None:
        with self.assertRaisesRegex(ValueError, "fit in int32"):
            upload_segments(
                CPU,
                [whole_layer_ranges(torch.iinfo(torch.int32).max + 1)],
                layer_sizes=[torch.iinfo(torch.int32).max + 1],
                layer_offsets=[0],
                variant_ids=[],
                cold=True,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_tensors_land_on_the_device(self) -> None:
        device = torch.device("cuda:0")
        track_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        data = _time_data([(0, 10), (100, 110)], gap_us=0)
        data = TrackTimeData(
            tracks_timestamps_us=data.tracks_timestamps_us.to(device),
            tracks_packinfo=data.tracks_packinfo.to(device),
            max_positive_timestamp_gap_us=data.max_positive_timestamp_gap_us.to(device),
        )
        table = LayerSegmentTable.build(track_ids, data)
        selection = table.select(frame_start_us=0, frame_end_us=10, frame_exposure_us=0)
        collect, draw = upload_segments(
            device,
            [selection],
            layer_sizes=[4],
            layer_offsets=[0],
            variant_ids=[0],
            cold=True,
        )
        assert draw is not None
        torch.cuda.synchronize()
        assert collect[0].starts is not None
        self.assertEqual(collect[0].starts.device, device)
        self.assertEqual(draw.segments.device, device)
        self.assertEqual(draw.segments.cpu().tolist(), [[0], [2]])


class TestRowsGroupedByTrack(unittest.TestCase):
    def test_grouped_and_fragmented_layers(self) -> None:
        self.assertTrue(rows_grouped_by_track(torch.tensor([0, 0, 1, 1, 2], dtype=torch.int32), track_count=3))
        self.assertTrue(rows_grouped_by_track(torch.empty(0, dtype=torch.int32), track_count=0))
        # 200 runs for 2 tracks exceeds the window limit (max(64, 8 * 2)).
        self.assertFalse(rows_grouped_by_track(torch.tensor([0, 1] * 100, dtype=torch.int32), track_count=2))
        # Below the absolute floor a fragmented small layer is still fine.
        self.assertTrue(rows_grouped_by_track(torch.tensor([0, 1] * 30, dtype=torch.int32), track_count=2))


class TestCollectRanges(unittest.TestCase):
    """The collect list in host rows: what upload_segments marshals and post_process reads."""

    SELECTIONS = [
        np.array([[0, 4]], dtype=np.int64),  # layer 0, whole
        np.array([[2, 3]], dtype=np.int64),  # layer 1, one window
    ]
    SIZES = [4, 8]

    def test_a_warm_cacheable_layer_writes_nothing(self) -> None:
        ranges = collect_ranges(self.SELECTIONS, self.SIZES, variant_ids=[1], cold=False)
        self.assertEqual(ranges[0].shape, (0, 2))
        self.assertTrue(np.array_equal(ranges[1], self.SELECTIONS[1]))

    def test_a_cold_cacheable_layer_is_written_whole(self) -> None:
        ranges = collect_ranges(self.SELECTIONS, self.SIZES, variant_ids=[1], cold=True)
        self.assertTrue(np.array_equal(ranges[0], np.array([[0, 4]])))
        self.assertTrue(np.array_equal(ranges[1], self.SELECTIONS[1]))

    def test_without_selections_every_written_layer_is_whole(self) -> None:
        ranges = collect_ranges(None, self.SIZES, variant_ids=[1], cold=False)
        self.assertEqual(ranges[0].shape, (0, 2))
        self.assertTrue(np.array_equal(ranges[1], np.array([[0, 8]])))

    def test_an_empty_layer_writes_nothing(self) -> None:
        ranges = collect_ranges(None, [0], variant_ids=[0], cold=True)
        self.assertEqual(ranges[0].shape, (0, 2))


class TestAssertSparseSegmentsSupported(unittest.TestCase):
    SUPPORTED = dict(
        has_extra_signals=False,
        has_active_tile_mask=False,
        grad_enabled=False,
    )

    def test_an_inference_render_without_extras_is_supported(self) -> None:
        assert_sparse_segments_supported(**self.SUPPORTED)

    def test_every_unsupported_feature_names_itself(self) -> None:
        expected = {
            "has_extra_signals": "extra signals",
            "has_active_tile_mask": "active tile masks",
            "grad_enabled": "without gradients only",
        }
        for feature, wording in expected.items():
            with self.subTest(feature=feature), self.assertRaisesRegex(ValueError, wording):
                assert_sparse_segments_supported(**{**self.SUPPORTED, feature: True})


if __name__ == "__main__":
    unittest.main()
