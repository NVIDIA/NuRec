# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Archive assertions shared by lightweight and full-model parity tests."""

from pathlib import Path

import numpy as np

from nurec.utils.ncore_utils import EGO_MASK_BASE_GROUP, AuxShardDataLoader, CamAuxDataLoader


def assert_segmentation_archives_match(
    serial_output: Path, parallel_output: Path, camera_id: str, seg_logits: bool, *, mask_tolerance: float = 0.0
) -> tuple[CamAuxDataLoader, AuxShardDataLoader]:
    """Compare all persisted outputs and metadata; return shared readers for further checks."""
    readers = []
    for directory in (serial_output, parallel_output):
        semantic_files = list(directory.glob("*.aux.sseg.zarr.itar"))
        assert len(semantic_files) == 1
        assert len(list(directory.glob("*.aux.sseg-logits.zarr.itar"))) == int(seg_logits)
        assert len(list(directory.glob("*.aux.egomask.zarr.itar"))) == 1
        path = semantic_files[0]
        readers.append(
            (
                CamAuxDataLoader(path),
                AuxShardDataLoader(sequence_id=path.name.partition(".aux.")[0], dataset_paths=[path]),
            )
        )
    (serial, serial_aux), (shared, shared_aux) = readers
    semantic_meta = serial.get_semantic_segmentation_meta(camera_id)
    assert semantic_meta == shared.get_semantic_segmentation_meta(camera_id)
    timestamps = set(serial._camera_group(camera_id).keys())
    assert timestamps == set(shared._camera_group(camera_id).keys())
    assert timestamps
    if seg_logits:
        assert serial_aux.get_semantic_logits_meta(camera_id) == shared_aux.get_semantic_logits_meta(camera_id)
    for timestamp in timestamps:
        labels = [
            np.asarray(reader.get_semantic_segmentation(camera_id, int(timestamp))) for reader in (serial, shared)
        ]
        np.testing.assert_array_equal(labels[0], labels[1])
        if seg_logits:
            logits = [reader.get_semantic_logits(camera_id, int(timestamp)) for reader in (serial_aux, shared_aux)]
            for value in logits:
                assert value.dtype == np.float16
                assert value.shape == (*labels[0].shape, len(semantic_meta["stuff_classes"]))
            np.testing.assert_array_equal(logits[0], logits[1])

    assert serial_aux.get_egomask_meta(camera_id) == shared_aux.get_egomask_meta(camera_id)
    mask_timestamps = [
        set(reader.base_groups[EGO_MASK_BASE_GROUP][0][camera_id].keys()) for reader in (serial_aux, shared_aux)
    ]
    assert mask_timestamps[0] == mask_timestamps[1]
    assert "0" in mask_timestamps[0]
    for timestamp in mask_timestamps[0]:
        masks = [reader.get_egomask(camera_id, int(timestamp)) for reader in (serial_aux, shared_aux)]
        assert masks[0].shape == masks[1].shape
        assert masks[0].dtype == masks[1].dtype
        assert np.count_nonzero(masks[0] != masks[1]) / masks[0].size <= mask_tolerance
    return shared, shared_aux
