# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from nurec.utils.ncore_utils import INSTANCE_SEG_BASE_GROUP, AuxDataWriter, AuxShardDataLoader


@pytest.mark.parametrize("store_type", ["directory", "itar"])
@pytest.mark.parametrize("reader_type", ["writer", "loader"])
@pytest.mark.parametrize("resolution", [(129, 97), (128, 96), (7, 1), (1, 1)])
@pytest.mark.parametrize("instance_count", [0, 1, 2, 8])
def test_instance_masks_round_trip(tmp_path, store_type, reader_type, resolution, instance_count):
    """Read the producer's packed input, trimming only the final byte's padding."""
    width, height = resolution
    masks = np.random.default_rng(0).integers(0, 2, (instance_count, height, width), dtype=np.uint8).astype(bool)
    scores = np.linspace(0.5, 1.0, instance_count, dtype=np.float32)
    classes = np.zeros(instance_count, dtype=np.int32)
    writer = AuxDataWriter(tmp_path, "scene", "scene", 0, 1, store_type)
    # Same whole-tensor packing and metadata order as ObjectInstanceGeneration.
    packed = np.packbits(masks)
    writer.store_instance_segmentation("cam", 20, packed, scores, classes)
    writer.store_instance_meta("cam", list(resolution), "test", "test", "", ["object"], [[255, 128, 0]])
    frame = writer.get_base_group(INSTANCE_SEG_BASE_GROUP)["cam"]["20"]
    np.testing.assert_array_equal(np.asarray(frame["instance_masks"]), packed)
    reader: AuxDataWriter | AuxShardDataLoader
    if reader_type == "loader":
        writer.finalize()
        reader = AuxShardDataLoader("scene", [tmp_path / "scene.zarr"])
    else:
        reader = writer
    try:
        result = reader.get_instance_segmentation("cam", 20)
        np.testing.assert_array_equal(result["instance_masks"], masks)
        assert result["instance_masks"].dtype == np.uint8
        np.testing.assert_array_equal(result["scores"], scores)
        np.testing.assert_array_equal(result["classes"], classes)
    finally:
        if reader_type == "writer":
            writer.finalize()


@pytest.mark.parametrize("store_type", ["directory", "itar"])
@pytest.mark.parametrize("reader_type", ["writer", "loader"])
def test_instance_masks_pack_across_instance_boundary(tmp_path, store_type, reader_type):
    """Two 1x7 masks share a byte; no padding may be inserted between instances."""
    masks = np.array([[[1, 0, 1, 0, 1, 0, 1]], [[1, 1, 0, 0, 1, 0, 0]]], dtype=bool)
    # The producer uses packbits without an axis: C-order and MSB-first.
    packed = np.packbits(masks)
    np.testing.assert_array_equal(packed, np.array([0xAB, 0x90], dtype=np.uint8))
    separately_packed = np.packbits(masks.reshape(2, -1), axis=1)
    assert separately_packed.size == packed.size
    assert not np.array_equal(separately_packed.ravel(), packed)
    writer = AuxDataWriter(tmp_path, "scene", "scene", 0, 1, store_type)
    writer.store_instance_segmentation("cam", 20, packed, np.ones(2), np.zeros(2, dtype=np.int32))
    writer.store_instance_meta("cam", [7, 1], "test", "test", "", ["object"], [[255, 128, 0]])
    reader: AuxDataWriter | AuxShardDataLoader
    if reader_type == "loader":
        writer.finalize()
        reader = AuxShardDataLoader("scene", [tmp_path / "scene.zarr"])
    else:
        reader = writer
    try:
        np.testing.assert_array_equal(reader.get_instance_segmentation("cam", 20)["instance_masks"], masks)
    finally:
        if reader_type == "writer":
            writer.finalize()


@pytest.mark.parametrize("store_type", ["directory", "itar"])
@pytest.mark.parametrize("invalid", ["per_instance", "raw", "dtype"])
def test_instance_mask_writer_rejects_invalid_arrays(tmp_path, store_type, invalid):
    """Reject arrays that do not satisfy the packed-input interface."""
    masks = np.ones((2, 1, 7), dtype=np.uint8)
    if invalid == "per_instance":
        packed = np.packbits(masks.reshape(2, -1), axis=1)
    elif invalid == "raw":
        packed = masks
    else:
        packed = np.packbits(masks).astype(np.int32)
    writer = AuxDataWriter(tmp_path, "scene", "scene", 0, 1, store_type)
    with pytest.raises(ValueError, match="flat uint8 bitstream"):
        writer.store_instance_segmentation("cam", 20, packed, np.ones(2), np.zeros(2, dtype=np.int32))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("store_type", ["directory", "itar"])
@pytest.mark.parametrize("reader_type", ["writer", "loader"])
@pytest.mark.parametrize("invalid", ["per_instance", "per_row", "truncated", "extra_byte", "dtype"])
def test_instance_mask_readers_reject_invalid_arrays(tmp_path, store_type, reader_type, invalid):
    """Reject malformed stored arrays before unpackbits can flatten or zero-fill them."""
    masks = np.ones((2, 1, 7), dtype=np.uint8)
    packed = np.packbits(masks)
    if invalid == "per_instance":
        packed = np.packbits(masks.reshape(2, -1), axis=1)
    elif invalid == "per_row":
        packed = np.packbits(masks, axis=-1)
    elif invalid == "truncated":
        packed = packed[:-1]
    elif invalid == "extra_byte":
        packed = np.append(packed, np.uint8(0))
    elif invalid == "dtype":
        packed = packed.astype(np.int32)
    writer = AuxDataWriter(tmp_path, "scene", "scene", 0, 1, store_type)
    writer.store_instance_meta("cam", [7, 1], "test", "test", "", ["object"], [[255, 128, 0]])
    # Construct a stored frame directly to exercise validation of existing data.
    frame = writer.get_base_group(INSTANCE_SEG_BASE_GROUP)["cam"].create_group("20")
    frame.create_dataset("instance_masks", data=packed)
    frame.create_dataset("scores", data=np.ones(2))
    frame.create_dataset("classes", data=np.zeros(2, dtype=np.int32))
    reader: AuxDataWriter | AuxShardDataLoader
    if reader_type == "loader":
        writer.finalize()
        reader = AuxShardDataLoader("scene", [tmp_path / "scene.zarr"])
    else:
        reader = writer
    try:
        message = "bytes; expected" if invalid in ["truncated", "extra_byte"] else "flat uint8 bitstream"
        with pytest.raises(ValueError, match=message):
            reader.get_instance_segmentation("cam", 20)
    finally:
        if reader_type == "writer":
            writer.finalize()
