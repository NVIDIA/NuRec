# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os

import click
import numpy as np

from PIL import Image
from upath import UPath

import ncore.data
import nurec.utils.ncore_utils as ncore_utils

from nurec.ncore_backends import (
    DatasetRequest,
    require_explicit_sequence_input_provider,
    resolve_dataset,
)


logger = logging.getLogger(__name__)


# TODO: remove this script in favor of nurec/benchmark/gt_from_ncore.py or nurec/utils/io/export/ncore_diagnostic.py
@click.command("export-ego-mask")
@click.option("--dataset-path", type=str, required=False, help="Path to an NCore sequence metadata file")
@click.option(
    "--output-dir",
    "output_dir",
    type=str,
    required=True,
    help="Path to the output images directory. It should be the same directory of the checkpoint for grpc usage",
)
@click.option(
    "--camera-ids", "camera_ids", type=str, multiple=True, help="List of camera ids. [default: all available cameras]"
)
@click.option(
    "--invert-mask/--no-invert-mask",
    default=False,
    help="Whether to show the masked region and blank the unmasked region (--no-invert-mask) or show the unmasked region and blank the masked region (--invert-mask)",
)
@click.option(
    "--camera-frame-idx",
    "camera_frame_idx",
    type=int,
    default=50,
    help="Index of the camera frame to export the ego mask for",
)
def export_ego_mask(
    dataset_path: str | None,
    output_dir: str,
    camera_ids: list[str],
    invert_mask: bool,
    camera_frame_idx: int,
    explicit_path_pattern: str | None = None,
) -> None:
    if bool(dataset_path) == bool(explicit_path_pattern):
        raise click.ClickException("Exactly one sequence input must be provided")

    if explicit_path_pattern:
        backend, loader, _ = require_explicit_sequence_input_provider().open_group(
            explicit_path_pattern,
            open_consolidated=True,
            loader_max_workers=None,
        )
    else:
        assert dataset_path is not None
        resolved = resolve_dataset(
            DatasetRequest(
                metadata_path=UPath(dataset_path),
                masks_component_group="default",
                cuboids_component_group=None,
            )
        )
        backend = resolved.backend
        loader = resolved.create_sequence_loader()

    output_dir = os.path.join(output_dir, "ego-hoods")
    os.makedirs(output_dir, exist_ok=True)
    resized_h = 540
    resized_w = 960

    selected_camera_ids = camera_ids or loader.camera_ids
    aux_loader: ncore_utils.AuxShardDataLoader | None = None

    for camera_id in selected_camera_ids:
        camera_sensor = loader.get_camera_sensor(camera_id)
        camera_mask_image = ncore_utils.get_camera_sensor_mask(camera_sensor, backend=backend)

        if camera_mask_image is None:
            if aux_loader is None:
                aux_loader = ncore_utils.AuxShardDataLoader(
                    sequence_id=loader.sequence_id,
                    dataset_paths=loader.sequence_paths,
                    open_consolidated=True,
                )
            try:
                camera_mask_image = aux_loader.get_aggregated_egomask(camera_id)
                logger.info("Using auxiliary ego mask for %s", camera_id)
            except KeyError:
                camera_mask_image = None

        _export_camera_ego_mask(
            camera_sensor,
            camera_mask_image,
            camera_id,
            output_dir,
            invert_mask,
            camera_frame_idx,
            resized_w,
            resized_h,
        )


def _export_camera_ego_mask(
    camera_sensor: ncore.data.CameraSensorProtocol,
    camera_mask_image: np.ndarray | None,
    camera_id: str,
    output_dir: str,
    invert_mask: bool,
    camera_frame_idx: int,
    resized_w: int,
    resized_h: int,
) -> None:
    if camera_mask_image is None:
        logger.warning(f"Camera {camera_id} does not have a mask image")
        return

    camera_mask_array = (
        np.asarray(
            Image.fromarray(camera_mask_image)
            .convert("L")
            .resize((resized_w, resized_h), resample=Image.Resampling.BICUBIC)
        )
        != 0
    )
    if not invert_mask:
        camera_mask_array = ~camera_mask_array

    alpha_array = np.expand_dims(np.ones_like(camera_mask_array), axis=-1).astype(np.uint8) * 255
    assert camera_frame_idx < camera_sensor.frames_count, (
        f"Camera {camera_id} only has {camera_sensor.frames_count} frames"
    )
    camera_image_array = np.asarray(
        camera_sensor.get_frame_image(camera_frame_idx).resize((resized_w, resized_h))
    ).copy()
    camera_image_array[camera_mask_array] = 0
    alpha_array[camera_mask_array] = 0
    camera_image_array = np.concatenate((camera_image_array, alpha_array), axis=-1)
    ego_mask = Image.fromarray(camera_image_array, mode="RGBA")
    ego_mask.save(os.path.join(output_dir, f"{camera_id}.png"))
    logging.info(f"Wrote: {os.path.abspath(os.path.join(output_dir, f'{camera_id}.png'))}")
