# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import imageio as iio
import numpy as np
import pytest
import torch

from nurec.grpc.ego_hood import EgocarHood, EgocarRig, EgocarRigBank
from nurec.grpc.protos.sensorsim_pb2 import RGBRenderRequest


@pytest.fixture
def test_image_rgba():
    # Create a simple RGBA test image (100x100 with a red square in the middle)
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    # Set the middle 50x50 area to red with alpha=255
    img[25:75, 25:75, 0] = 255  # Red
    img[25:75, 25:75, 3] = 255  # Alpha
    return img


@pytest.fixture
def test_rig_bank_dir(test_image_rgba, tmp_path):
    rig_ids = ["rig1", "rig2"]
    camera_ids = ["front", "left", "right"]

    # Create multiple rig subdirectories in the pytest-provided temporary directory
    for rig_id in rig_ids:
        rig_dir = tmp_path / rig_id
        rig_dir.mkdir()

        # Create multiple hood images for each rig
        for camera_id in camera_ids:
            image_path = rig_dir / f"{camera_id}.png"
            iio.imwrite(image_path, test_image_rgba)

    # Return the base directory, rig IDs, camera IDs,
    # a specific hood image path, and a specific rig directory path
    hood_image_path = tmp_path / rig_ids[0] / f"{camera_ids[0]}.png"
    rig_dir_path = tmp_path / rig_ids[0]

    return {
        "base_dir": tmp_path,
        "rig_ids": rig_ids,
        "camera_ids": camera_ids,
        "hood_image_path": hood_image_path,
        "rig_dir_path": rig_dir_path,
    }


class TestEgocarHood:
    def test_load_from_file(self, test_rig_bank_dir):
        # Test creating an EgocarHood from a file
        hood_path = test_rig_bank_dir["hood_image_path"]
        hood = EgocarHood.load_from_file("front", "rig1", hood_path)

        assert hood.camera_logical_id == "front"
        assert hood.rig_id == "rig1"
        assert hood._hood_rgba.shape[0] == 4  # RGBA
        assert hood._hood_rgba.dtype == torch.float32
        assert torch.max(hood._hood_rgba) <= 1.0  # Normalized to [0, 1]

    def test_draw_egocar(self, test_rig_bank_dir):
        # Test overlaying the hood on a render
        hood_path = test_rig_bank_dir["hood_image_path"]
        hood = EgocarHood.load_from_file("front", "rig1", hood_path)

        # Create a test render (blue background)
        render = torch.zeros(100, 100, 3, dtype=torch.float32)
        render[:, :, 2] = 1.0  # Blue background

        # Apply the hood to the render
        result = hood.overlay_on_image(render)

        # The result should have red pixels where the hood was and blue pixels elsewhere
        assert result.shape == (100, 100, 3)

        # Check middle pixel (should be hood/red)
        assert result[50, 50, 0] > 0.9  # Red channel high
        assert result[50, 50, 1] < 0.1  # Green channel low
        assert result[50, 50, 2] < 0.1  # Blue channel low

        # Check corner pixel (should be render/blue)
        assert result[0, 0, 0] < 0.1  # Red channel low
        assert result[0, 0, 1] < 0.1  # Green channel low
        assert result[0, 0, 2] > 0.9  # Blue channel high

    def test_resize_cache(self, test_rig_bank_dir, monkeypatch):
        # Test that the resize cache works correctly
        hood_path = test_rig_bank_dir["hood_image_path"]
        hood = EgocarHood.load_from_file("front", "rig1", hood_path)

        # Create a render with a different size
        render1 = torch.zeros(50, 50, 3, dtype=torch.float32)

        # Setup to monitor logger calls
        logger_calls = []

        def mock_info(message):
            logger_calls.append(message)

        # Patch the logger.info method
        monkeypatch.setattr("nurec.grpc.ego_hood.logger.info", mock_info)

        # First draw should update the cache
        hood.overlay_on_image(render1)
        assert len(logger_calls) == 1

        # Cache should now be at the new size
        assert hood._resize_cache.shape[1:3] == (50, 50)

        # Clear the call log
        logger_calls.clear()

        # Second draw should use the cache
        hood.overlay_on_image(render1)
        assert len(logger_calls) == 0

    def test_metadata(self, test_rig_bank_dir):
        # Test that the metadata is correctly generated
        hood_path = test_rig_bank_dir["hood_image_path"]
        hood = EgocarHood.load_from_file("front", "rig1", hood_path)
        metadata = hood.metadata

        assert metadata.ego_mask_id.camera_logical_id == "front"
        assert metadata.ego_mask_id.rig_config_id == "rig1"


class TestEgocarRig:
    def test_load_from_dir(self, test_rig_bank_dir):
        # Test creating an EgocarRig from a directory
        rig_dir = test_rig_bank_dir["rig_dir_path"]
        camera_ids = test_rig_bank_dir["camera_ids"]
        rig = EgocarRig.load_from_dir("test_rig", rig_dir)

        assert rig.rig_id == "test_rig"
        assert len(rig._hoods) == len(camera_ids)
        for camera_id in camera_ids:
            assert camera_id in rig._hoods

    def test_load_from_dir_ignores_non_png_files_and_directories(self, test_rig_bank_dir):
        rig_dir = test_rig_bank_dir["rig_dir_path"]
        camera_ids = test_rig_bank_dir["camera_ids"]
        (rig_dir / "notes.txt").write_text("ignore me")
        (rig_dir / "nested.png").mkdir()

        rig = EgocarRig.load_from_dir("test_rig", rig_dir)

        assert len(rig._hoods) == len(camera_ids)
        assert "nested" not in rig._hoods

    def test_get(self, test_rig_bank_dir):
        # Test the get method
        rig_dir = test_rig_bank_dir["rig_dir_path"]
        camera_ids = test_rig_bank_dir["camera_ids"]
        rig = EgocarRig.load_from_dir("test_rig", rig_dir)

        # Existing camera_id
        hood = rig.get(camera_ids[0])
        assert hood is not None
        assert hood.camera_logical_id == camera_ids[0]

        # Non-existent camera_id
        hood = rig.get("nonexistent")
        assert hood is None

    def test_available_metadata(self, test_rig_bank_dir):
        # Test that available_metadata returns the correct list
        rig_dir = test_rig_bank_dir["rig_dir_path"]
        camera_ids = test_rig_bank_dir["camera_ids"]
        rig = EgocarRig.load_from_dir("test_rig", rig_dir)

        metadata_list = rig.available_metadata()
        assert len(metadata_list) == len(camera_ids)

        camera_ids_in_metadata = [m.ego_mask_id.camera_logical_id for m in metadata_list]
        for camera_id in camera_ids:
            assert camera_id in camera_ids_in_metadata


class TestEgocarRigBank:
    def test_load_from_dir(self, test_rig_bank_dir):
        # Test creating an EgocarRigBank from a directory
        bank_dir = test_rig_bank_dir["base_dir"]
        rig_ids = test_rig_bank_dir["rig_ids"]
        bank = EgocarRigBank.load_from_dir(bank_dir)

        assert len(bank._rigs) == len(rig_ids)
        for rig_id in rig_ids:
            assert rig_id in bank._rigs

    def test_load_from_dir_ignores_root_files(self, test_rig_bank_dir):
        bank_dir = test_rig_bank_dir["base_dir"]
        rig_ids = test_rig_bank_dir["rig_ids"]
        (bank_dir / ".nvcf_manifest.json").write_text("{}")
        (bank_dir / "README.txt").write_text("ignore me")

        bank = EgocarRigBank.load_from_dir(bank_dir)

        assert len(bank._rigs) == len(rig_ids)
        for rig_id in rig_ids:
            assert rig_id in bank._rigs
        assert ".nvcf_manifest.json" not in bank._rigs
        assert "README.txt" not in bank._rigs

    def test_empty(self):
        # Test creating an empty EgocarRigBank
        bank = EgocarRigBank.empty()

        assert len(bank._rigs) == 0

    def test_available_metadata(self, test_rig_bank_dir):
        # Test that available_metadata returns the correct list
        bank_dir = test_rig_bank_dir["base_dir"]
        rig_ids = test_rig_bank_dir["rig_ids"]
        camera_ids = test_rig_bank_dir["camera_ids"]
        bank = EgocarRigBank.load_from_dir(bank_dir)

        metadata_list = bank.available_metadata()
        assert len(metadata_list) == len(rig_ids) * len(camera_ids)

    def test_select_from_request_specific(self, test_rig_bank_dir):
        # Test selecting a specific hood by ID
        bank_dir = test_rig_bank_dir["base_dir"]
        rig_ids = test_rig_bank_dir["rig_ids"]
        camera_ids = test_rig_bank_dir["camera_ids"]
        bank = EgocarRigBank.load_from_dir(bank_dir)

        # Create a request with specific camera_logical_id and rig_config_id
        request = RGBRenderRequest()
        request.insert_ego_mask = True
        request.ego_mask_id.camera_logical_id = camera_ids[1]
        request.ego_mask_id.rig_config_id = rig_ids[1]

        hood = bank.select_from_request(request)
        assert hood is not None
        assert hood.camera_logical_id == camera_ids[1]
        assert hood.rig_id == rig_ids[1]

    def test_select_from_request_nonexistent_config(self, test_rig_bank_dir):
        # Test selecting a hood with nonexistent config_id
        bank_dir = test_rig_bank_dir["base_dir"]
        camera_ids = test_rig_bank_dir["camera_ids"]
        bank = EgocarRigBank.load_from_dir(bank_dir)

        # Create a request with nonexistent config_id
        request = RGBRenderRequest()
        request.insert_ego_mask = True
        request.ego_mask_id.camera_logical_id = camera_ids[0]
        request.ego_mask_id.rig_config_id = "nonexistent"

        with pytest.raises(KeyError):
            bank.select_from_request(request)

    def test_select_from_request_nonexistent_camera(self, test_rig_bank_dir):
        # Test selecting a hood with nonexistent camera_id
        bank_dir = test_rig_bank_dir["base_dir"]
        rig_ids = test_rig_bank_dir["rig_ids"]
        bank = EgocarRigBank.load_from_dir(bank_dir)

        # Create a request with nonexistent camera_id
        request = RGBRenderRequest()
        request.ego_mask_id.camera_logical_id = "nonexistent"
        request.ego_mask_id.rig_config_id = rig_ids[0]

        hood = bank.select_from_request(request)
        assert hood is None

    def test_select_from_request_no_use_ego_mask(self, test_rig_bank_dir):
        bank_dir = test_rig_bank_dir["base_dir"]
        bank = EgocarRigBank.load_from_dir(bank_dir)

        # Create a request with empty camera_id
        request = RGBRenderRequest()
        request.camera_intrinsics.logical_id = "front"  # This should be ignored

        hood = bank.select_from_request(request)
        assert hood is None

    def test_per_camera_mask_selection_returns_correct_mask_per_camera(self, test_rig_bank_dir):
        """Verify that different cameras within the same rig return distinct hood objects."""
        bank_dir = test_rig_bank_dir["base_dir"]
        rig_ids = test_rig_bank_dir["rig_ids"]
        camera_ids = test_rig_bank_dir["camera_ids"]
        bank = EgocarRigBank.load_from_dir(bank_dir)

        for camera_id in camera_ids:
            request = RGBRenderRequest()
            request.insert_ego_mask = True
            request.ego_mask_id.camera_logical_id = camera_id
            request.ego_mask_id.rig_config_id = rig_ids[0]

            hood = bank.select_from_request(request)
            assert hood is not None
            assert hood.camera_logical_id == camera_id
            assert hood.rig_id == rig_ids[0]

    def test_per_camera_mask_selection_across_rigs(self, test_rig_bank_dir):
        """Verify correct mask is returned when same camera exists in different rigs."""
        bank_dir = test_rig_bank_dir["base_dir"]
        rig_ids = test_rig_bank_dir["rig_ids"]
        camera_ids = test_rig_bank_dir["camera_ids"]
        bank = EgocarRigBank.load_from_dir(bank_dir)

        for rig_id in rig_ids:
            request = RGBRenderRequest()
            request.insert_ego_mask = True
            request.ego_mask_id.camera_logical_id = camera_ids[0]
            request.ego_mask_id.rig_config_id = rig_id

            hood = bank.select_from_request(request)
            assert hood is not None
            assert hood.rig_id == rig_id
            assert hood.camera_logical_id == camera_ids[0]

    def test_select_from_request_logs_mask_used_once_for_new_logical_camera_id(self, test_rig_bank_dir, monkeypatch):
        bank_dir = test_rig_bank_dir["base_dir"]
        rig_ids = test_rig_bank_dir["rig_ids"]
        camera_ids = test_rig_bank_dir["camera_ids"]
        bank = EgocarRigBank.load_from_dir(bank_dir)
        logger_calls = []

        monkeypatch.setattr("nurec.grpc.ego_hood.logger.info", lambda message: logger_calls.append(message))

        request = RGBRenderRequest()
        request.camera_intrinsics.logical_id = "camera_new_front"
        request.insert_ego_mask = True
        request.ego_mask_id.camera_logical_id = camera_ids[0]
        request.ego_mask_id.rig_config_id = rig_ids[0]

        hood = bank.select_from_request(request)
        assert hood is not None
        assert logger_calls == [
            "New logical camera id 'camera_new_front' using ego mask "
            f"rig_config_id='{rig_ids[0]}', camera_logical_id='{camera_ids[0]}' "
            f"(requested rig_config_id='{rig_ids[0]}', camera_logical_id='{camera_ids[0]}')."
        ]

        bank.select_from_request(request)
        assert len(logger_calls) == 1

    def test_select_from_request_logs_missing_mask_for_new_logical_camera_id(self, test_rig_bank_dir, monkeypatch):
        bank_dir = test_rig_bank_dir["base_dir"]
        rig_ids = test_rig_bank_dir["rig_ids"]
        bank = EgocarRigBank.load_from_dir(bank_dir)
        logger_calls = []

        monkeypatch.setattr("nurec.grpc.ego_hood.logger.info", lambda message: logger_calls.append(message))

        request = RGBRenderRequest()
        request.camera_intrinsics.logical_id = "camera_new_rear"
        request.insert_ego_mask = True
        request.ego_mask_id.camera_logical_id = "camera_missing_mask"
        request.ego_mask_id.rig_config_id = rig_ids[0]

        with pytest.raises(KeyError, match="No egocar hood found"):
            bank.select_from_request(request)
        assert logger_calls == [
            "New logical camera id 'camera_new_rear' requested ego mask "
            f"rig_config_id='{rig_ids[0]}', camera_logical_id='camera_missing_mask'; "
            "no matching mask was found."
        ]


class TestEgoMaskOverlayAccuracy:
    """Tests verifying that mask overlay produces pixel-accurate results via gRPC-style requests."""

    @pytest.fixture
    def distinct_masks_dir(self, tmp_path):
        """Create masks with distinct pixel patterns per camera so we can verify the right mask was applied."""
        rig_id = "test_rig"
        rig_dir = tmp_path / rig_id
        rig_dir.mkdir()

        cameras = {
            "camera_front_wide": (255, 0, 0),  # Red
            "camera_left": (0, 255, 0),  # Green
            "camera_right": (0, 0, 255),  # Blue
        }

        for camera_id, (r, g, b) in cameras.items():
            img = np.zeros((100, 100, 4), dtype=np.uint8)
            # Bottom 20 rows = hood region (fully opaque with distinct color)
            img[80:100, :, 0] = r
            img[80:100, :, 1] = g
            img[80:100, :, 2] = b
            img[80:100, :, 3] = 255
            iio.imwrite(rig_dir / f"{camera_id}.png", img)

        return {"base_dir": tmp_path, "rig_id": rig_id, "cameras": cameras}

    def test_overlay_applies_correct_mask_per_camera(self, distinct_masks_dir):
        """Verify that after overlay, the mask region contains the correct camera-specific color."""
        bank = EgocarRigBank.load_from_dir(distinct_masks_dir["base_dir"])
        rig_id = distinct_masks_dir["rig_id"]

        for camera_id, (r, g, b) in distinct_masks_dir["cameras"].items():
            # Simulate a white rendered image
            rendered = torch.ones(100, 100, 3, dtype=torch.float32)

            # Select mask via request (same path as gRPC server)
            request = RGBRenderRequest()
            request.insert_ego_mask = True
            request.ego_mask_id.camera_logical_id = camera_id
            request.ego_mask_id.rig_config_id = rig_id

            hood = bank.select_from_request(request)
            assert hood is not None, f"No mask found for {camera_id}"

            result = hood.overlay_on_image(rendered)

            # Check the hood region (row 90, middle column) has the camera-specific color
            pixel = result[90, 50]
            assert pixel[0] == pytest.approx(r / 255.0, abs=0.02), (
                f"Camera {camera_id}: expected R={r / 255:.2f}, got {pixel[0]:.2f}"
            )
            assert pixel[1] == pytest.approx(g / 255.0, abs=0.02), (
                f"Camera {camera_id}: expected G={g / 255:.2f}, got {pixel[1]:.2f}"
            )
            assert pixel[2] == pytest.approx(b / 255.0, abs=0.02), (
                f"Camera {camera_id}: expected B={b / 255:.2f}, got {pixel[2]:.2f}"
            )

            # Check the non-hood region (row 50, middle column) is still white (render untouched)
            pixel_clear = result[50, 50]
            assert pixel_clear[0] == pytest.approx(1.0, abs=0.01), "Non-hood region should be untouched"
            assert pixel_clear[1] == pytest.approx(1.0, abs=0.01), "Non-hood region should be untouched"
            assert pixel_clear[2] == pytest.approx(1.0, abs=0.01), "Non-hood region should be untouched"

    def test_overlay_with_partial_alpha(self, tmp_path):
        """Verify alpha blending accuracy with semi-transparent masks."""
        rig_id = "alpha_rig"
        rig_dir = tmp_path / rig_id
        rig_dir.mkdir()

        # Create a mask with 50% alpha in the bottom region
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        img[80:100, :, 0] = 200  # Red
        img[80:100, :, 3] = 128  # ~50% alpha
        iio.imwrite(rig_dir / "cam_semi.png", img)

        bank = EgocarRigBank.load_from_dir(tmp_path)

        # White background
        rendered = torch.ones(100, 100, 3, dtype=torch.float32)

        request = RGBRenderRequest()
        request.insert_ego_mask = True
        request.ego_mask_id.camera_logical_id = "cam_semi"
        request.ego_mask_id.rig_config_id = rig_id

        hood = bank.select_from_request(request)
        result = hood.overlay_on_image(rendered)

        # At 50% alpha, the pixel should be a blend of mask (200/255, 0, 0) and white (1, 1, 1)
        alpha = 128 / 255.0
        expected_r = (1.0 - alpha) * 1.0 + alpha * (200 / 255.0)
        pixel = result[90, 50]
        assert pixel[0] == pytest.approx(expected_r, abs=0.02), f"Expected R={expected_r:.3f}, got {pixel[0]:.3f}"

    def test_missing_camera_mask_raises_for_explicit_request(self, distinct_masks_dir):
        """Verify that an explicit mask request fails when the selected rig has no matching camera."""
        bank = EgocarRigBank.load_from_dir(distinct_masks_dir["base_dir"])

        request = RGBRenderRequest()
        request.insert_ego_mask = True
        request.ego_mask_id.camera_logical_id = "camera_nonexistent"
        request.ego_mask_id.rig_config_id = distinct_masks_dir["rig_id"]

        with pytest.raises(KeyError, match="No egocar hood found"):
            bank.select_from_request(request)
