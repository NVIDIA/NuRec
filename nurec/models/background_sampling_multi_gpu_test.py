# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import tempfile

from datetime import timedelta

import pytest
import torch

from nurec.config.model import SkyEnvMapBackgroundConfig
from nurec.config.trainer import TrainerConfig
from nurec.models.background import SkyEnvMapBackground
from nurec.models.background_sampling import (
    _cubemap_padding_topology,
    cubemap_coverage_seamless,
    sample_cubemap_seamless,
)


_CUDA_DEVICE_INDICES = (0, 1)
_CUBEMAP_SIZE = 4
_CUBEMAP_FACE_COUNT = 6
_SAMPLER_CHANNEL_COUNT = 3

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")


def _cubemap(device: torch.device) -> torch.Tensor:
    return torch.arange(
        _CUBEMAP_FACE_COUNT * _CUBEMAP_SIZE * _CUBEMAP_SIZE * _SAMPLER_CHANNEL_COUNT,
        device=device,
        dtype=torch.float32,
    ).reshape(_CUBEMAP_FACE_COUNT, _CUBEMAP_SIZE, _CUBEMAP_SIZE, _SAMPLER_CHANNEL_COUNT)


def _rays(device: torch.device) -> torch.Tensor:
    # One strictly interior ray per distinct face keeps the sampler-backward
    # footprints disjoint on both devices, making their results bit-exact.
    return torch.tensor(
        [
            [1.0, 0.25, -0.5],
            [-0.5, 1.0, 0.25],
            [0.25, -0.5, 1.0],
        ],
        device=device,
    )


def test_sample_cubemap_rejects_exact_cross_device_mismatch() -> None:
    cubemap = _cubemap(torch.device("cuda", _CUDA_DEVICE_INDICES[0]))
    rays = _rays(torch.device("cuda", _CUDA_DEVICE_INDICES[1]))

    with pytest.raises(RuntimeError) as error:
        sample_cubemap_seamless(cubemap, rays)

    assert str(error.value) == (
        "Cubemap and rays must be on the same CUDA device; got cubemap on cuda:0 and rays on cuda:1"
    )


def test_topology_cache_and_sampling_are_isolated_by_concrete_cuda_device() -> None:
    original_device_index = torch.cuda.current_device()
    compute_capabilities = {torch.cuda.get_device_capability(device_index) for device_index in _CUDA_DEVICE_INDICES}
    topologies = []
    sampled_by_device = []
    cubemap_gradients_by_device = []

    try:
        for requested_device_index, ambient_device_index in (
            (_CUDA_DEVICE_INDICES[0], _CUDA_DEVICE_INDICES[1]),
            (_CUDA_DEVICE_INDICES[1], _CUDA_DEVICE_INDICES[0]),
        ):
            requested_device = torch.device("cuda", requested_device_index)
            torch.cuda.set_device(ambient_device_index)

            topology = _cubemap_padding_topology(_CUBEMAP_SIZE, requested_device)
            cached_topology = _cubemap_padding_topology(_CUBEMAP_SIZE, requested_device)
            topologies.append(topology)

            assert torch.cuda.current_device() == ambient_device_index
            assert topology is cached_topology
            topology_tensors = tuple(field for field in topology if isinstance(field, torch.Tensor))
            assert all(index.device == requested_device for index in topology_tensors)

            cubemap = _cubemap(requested_device).requires_grad_(True)
            sampled = sample_cubemap_seamless(cubemap, _rays(requested_device))
            weights = torch.tensor(
                [
                    [0.5, -0.25, 0.75],
                    [-1.0, 0.125, 0.25],
                    [0.375, 0.625, -0.5],
                ],
                device=requested_device,
            )
            sampled.backward(weights)
            torch.cuda.synchronize(requested_device)
            assert torch.cuda.current_device() == ambient_device_index
            assert sampled.device == requested_device
            assert sampled.shape == (3, _SAMPLER_CHANNEL_COUNT)
            assert cubemap.grad is not None
            sampled_by_device.append(sampled.detach().cpu())
            cubemap_gradients_by_device.append(cubemap.grad.detach().cpu())
    finally:
        torch.cuda.set_device(original_device_index)

    assert topologies[0] is not topologies[1]
    # Device isolation is architecture-independent; exact arithmetic parity is not.
    if len(compute_capabilities) == 1:
        torch.testing.assert_close(sampled_by_device[0], sampled_by_device[1], atol=0, rtol=0)
        torch.testing.assert_close(cubemap_gradients_by_device[0], cubemap_gradients_by_device[1], atol=0, rtol=0)


def _make_cubemap_background(device: torch.device) -> SkyEnvMapBackground:
    config = SkyEnvMapBackgroundConfig.model_validate(
        {
            "name": "sky-env-map",
            "envmap_type": "cubemap",
            "width": _CUBEMAP_SIZE,
            "height": _CUBEMAP_SIZE,
            "min_grad_updates": 0,
            "should_inpaint": False,
            "inpaint_threshold": 5e-2,
            "inpaint_kernel_size": 10,
        }
    )
    trainer_config = TrainerConfig(
        max_epochs=1,
        check_val_every_n_epoch=1,
        precision="32",
        log_every_n_steps=1,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    return SkyEnvMapBackground(config, trainer_config).to(device)


def _coverage_ddp_worker(rank: int, init_file: str) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=len(_CUDA_DEVICE_INDICES),
        timeout=timedelta(seconds=45),
    )
    try:
        rays = _rays(device).roll(rank, dims=0)
        weights = torch.tensor([0.2, 0.5, 0.9], device=device) + rank * 0.05
        local_coverage = cubemap_coverage_seamless(_CUBEMAP_SIZE, rays, weights)
        gathered = [torch.empty_like(local_coverage) for _ in _CUDA_DEVICE_INDICES]
        torch.distributed.all_gather(gathered, local_coverage)
        expected = torch.stack(gathered).mean(dim=0)
        background = _make_cubemap_background(device)
        raw_rgb = background._sample_raw(rays)

        background._track_texture_grads(raw_rgb, rays, 1 - weights, texture_update_mask=None, is_training=True)

        assert background.n_grad_updates == 2
        torch.testing.assert_close(background.texture_grads, expected, atol=2e-5, rtol=2e-5)
    finally:
        torch.distributed.destroy_process_group()


def test_cubemap_coverage_tracking_averages_across_two_ranks() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = f"{tmpdir}/process-group"
        workers = [
            subprocess.Popen([sys.executable, __file__, "--coverage-worker", str(rank), init_file])
            for rank in range(len(_CUDA_DEVICE_INDICES))
        ]
        try:
            return_codes = [worker.wait(timeout=60) for worker in workers]
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.kill()
        assert return_codes == [0, 0]


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--coverage-worker":
    _coverage_ddp_worker(int(sys.argv[2]), sys.argv[3])
