# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import queue
import threading

from typing import Any, Iterator, Optional, TypeVar

import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler

from nurec.config.datamodule import SODatamoduleConfig
from nurec.config.nurec import NuRecConfig
from nurec.datamodules.registry import register as register_datamodule
from nurec.datasets import make as make_dataset
from nurec.datasets.base import BaseDataSource
from nurec.utils.batch import DataAndRenderingBatch, DataBatch, batch_collate_fn
from nurec.utils.misc import compute_process_local_rng_seed, distributed_all_gather_nested
from nurec.utils.trainer import adjust_num_workers_for_world_size


logger = logging.getLogger(__name__)

_T_co = TypeVar("_T_co", covariant=True)


class MockDataLoader(DataLoader):
    """
    MockDataLoader is a mock dataloader that runs data fetching and processing only ones.
    The fetched batch is then repeated in a loop.
    This allows to observe the performance of training without the impact of actual data loading.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.data = None

    def __iter__(self):
        while True:
            if self.data is None:
                self.data = next(super().__iter__())
            yield self.data


class BaseDataModule(LightningDataModule):
    def update_epoch(self, epoch: int, system, **kwargs) -> None: ...


class PrefetchDataLoader(DataLoader):
    """DataLoader with GPU prefetching capabilities that runs in a separate thread.

    This prefetcher continuously loads batches in a background thread and transfers them to the GPU
    using a dedicated CUDA stream, allowing computation and data loading to overlap.
    """

    def __init__(
        self,
        dataset: Dataset[_T_co],
        device: str = "cuda",
        queue_size: int = 2,
        pin_memory: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the PrefetchDataLoader.

        Args:
            dataset: The dataset to load data from
            device: The device to prefetch data to (default: "cuda")
            queue_size: Size of the prefetch queue (default: 2)
            pin_memory: Whether to use pinned memory (default: True, required for prefetching)
            **kwargs: Additional arguments to pass to DataLoader
        """
        assert pin_memory, "PrefetchDataLoader requires pin_memory=True for optimal performance"

        # Initialize parent DataLoader class
        super().__init__(dataset, pin_memory=pin_memory, **kwargs)

        self.device = device
        self.queue_size = queue_size
        self.prefetcher: Optional[ThreadedPrefetcher] = None

    def reset(self) -> None:
        """Reset the prefetcher to prepare for a new iteration."""
        self.shutdown()

    def __iter__(self) -> Iterator[DataAndRenderingBatch]:  # type: ignore[override]
        """Create and start a prefetcher and yield batches from it."""
        # Use parent class __iter__ to get the correct iterator (including DDP sampler logic)
        base_iterator = super().__iter__()

        # Create the prefetcher only if needed
        if self.prefetcher is None:
            self.prefetcher = ThreadedPrefetcher(base_iterator, self.device, self.queue_size)
            self.prefetcher.start()
        # If prefetcher exists but is not active, restart it
        elif not self.prefetcher.prefetch_thread.is_alive():
            self.reset()
            self.prefetcher = ThreadedPrefetcher(base_iterator, self.device, self.queue_size)
            self.prefetcher.start()

        # Make sure prefetcher is initialized at this point
        if self.prefetcher is None:
            return iter(())  # Return empty iterator instead of None

        try:
            # Yield batches until exhausted
            while True:
                try:
                    batch = self.prefetcher.next()
                    yield batch
                except StopIteration:
                    break
        finally:
            pass

    def shutdown(self) -> None:
        """Shutdown the prefetcher to clean up resources."""
        if self.prefetcher is not None:
            self.prefetcher.shutdown()
            self.prefetcher = None


class ThreadedPrefetcher:
    """Threaded data prefetcher for overlapping data loading and GPU computation.

    Architecture:
    This background thread runs in the main process to enable parallel operations:
    DataLoader workers → Prefetcher thread → Main training thread

    Workflow:
    1. DataLoader worker processes load and return data to main process (CPU)
    2. Prefetcher thread transfers CPU data to GPU using dedicated CUDA stream
    3. GPU data is queued with CUDA event marking transfer completion
    4. Main thread retrieves data and waits on CUDA event before use

    Benefits:
    Enables three concurrent operations for maximum throughput:
    • Data loading (worker processes)
    • CPU→GPU transfer (prefetcher thread)
    • Model computation (main thread)

    Note: CUDA events ensure proper synchronization between streams.
    """

    def __init__(self, data_iterator: Iterator, device: str = "cuda", queue_size: int = 2) -> None:
        """Initialize the ThreadedPrefetcher.

        Args:
            data_iterator: The data iterator to prefetch from
            device: The device to prefetch data to (default: "cuda")
            queue_size: Size of the prefetch queue (default: 2)
        """
        self.data_iterator = data_iterator
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        # Queue stores tuples of (batch data, CUDA event)
        # The CUDA event is used to ensure transfer to GPU is complete
        self.queue: queue.Queue[tuple[DataAndRenderingBatch | None, torch.cuda.Event | None]] = queue.Queue(
            maxsize=queue_size
        )
        self.stop_event = threading.Event()
        self.not_full_event = threading.Event()
        self.not_full_event.set()  # Initially not full
        self.prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True, name="PrefetchThread")

    def start(self) -> None:
        """Start the prefetcher thread."""
        self.prefetch_thread.start()

    def shutdown(self) -> None:
        """Shutdown the prefetcher thread gracefully and clean up resources."""
        self.stop_event.set()
        self.not_full_event.set()  # Wake up the thread if it's waiting
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=0.1)

    def _prefetch_loop(self) -> None:
        """Main prefetch loop that runs in a background thread."""
        logger = logging.getLogger(__name__)
        try:
            while not self.stop_event.is_set():
                # If queue is full, wait for not_full_event instead of polling
                if self.queue.full():
                    self.not_full_event.wait(timeout=0.1)
                    continue

                # Try to get next batch
                batch: DataAndRenderingBatch
                try:
                    batch = next(self.data_iterator)
                except StopIteration:
                    break

                # Transfer batch to GPU using dedicated CUDA stream
                with torch.cuda.stream(self.stream):
                    batch = batch.to(self.device, non_blocking=True)
                    # Record a CUDA event to mark when the data transfer is complete
                    # This event will be used by the main thread to wait for the transfer to finish
                    cuda_event: torch.cuda.Event = self.stream.record_event()

                # Put batch and associated event into queue
                self.queue.put((batch, cuda_event))

                # If queue is now full, clear the not_full_event
                if self.queue.full():
                    self.not_full_event.clear()
        except Exception as e:
            logger.error(f"Error in prefetch thread: {e}")
        finally:
            # Signal that no more data is coming
            self.queue.put((None, None))

    def next(self) -> DataAndRenderingBatch:
        """
        Get the next batch, with proper CUDA stream synchronization.

        Returns:
            The next batch.

        Raises:
            StopIteration: If no more batches are available.
        """
        # Get batch and event from queue
        batch, cuda_event = self.queue.get()

        if batch is None:
            raise StopIteration()

        # Wait for the CUDA event to ensure data transfer is complete
        if cuda_event is not None:
            torch.cuda.current_stream().wait_event(cuda_event)

        # IMPORTANT: Record this tensor in the current stream before notifying the prefetch thread
        # When a tensor is created in one CUDA stream and then used in another stream,
        # PyTorch needs to know this to avoid premature memory deallocation.
        # Without record_stream(), PyTorch might think the tensor is no longer needed in the
        # original stream after the operation completes, and could reclaim the memory too early.
        # This prevents potential race conditions where the prefetch thread starts loading a new
        # batch (after we set not_full_event) and deallocates our current batch before we've
        # registered it with the current stream.
        self._record_stream(batch, torch.cuda.current_stream())

        # Now that we've registered the tensors with the current stream, it's safe to
        # notify the prefetch thread that the queue has space for a new batch
        if not self.not_full_event.is_set():
            self.not_full_event.set()

        return batch

    def _record_stream(self, obj: Any, stream: torch.cuda.Stream) -> None:
        """Record stream for all tensors to prevent memory issues."""
        if isinstance(obj, torch.Tensor):
            obj.record_stream(stream)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                if isinstance(item, torch.Tensor):
                    item.record_stream(stream)
                elif isinstance(item, (list, tuple, dict)) or hasattr(item, "__dict__"):
                    self._record_stream(item, stream)
        elif isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, torch.Tensor):
                    value.record_stream(stream)
                elif isinstance(value, (list, tuple, dict)) or hasattr(value, "__dict__"):
                    self._record_stream(value, stream)
        elif hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                if isinstance(value, torch.Tensor):
                    value.record_stream(stream)
                elif isinstance(value, (list, tuple, dict)) or hasattr(value, "__dict__"):
                    self._record_stream(value, stream)


@register_datamodule("so")
class SODataModule(BaseDataModule):
    config: SODatamoduleConfig

    _TRAIN_LOADER_GENERATOR_STATES_KEY = "train_loader_generator_states"
    _TRAIN_LOADER_GENERATOR_RANK_COUNT_KEY = "train_loader_generator_rank_count"

    @staticmethod
    def __batch_collate_fn(*args, **kwargs) -> DataBatch | DataAndRenderingBatch:
        """Intermediate function wrapping the collate_fn passed to the dataloader."""
        return batch_collate_fn(*args, **kwargs)

    def __init__(self, nurec_config: NuRecConfig) -> None:
        super().__init__()
        self.nurec_config = nurec_config
        assert isinstance(nurec_config.datamodule, SODatamoduleConfig)
        self.config = nurec_config.datamodule.model_copy()
        self._train_loader_generator: torch.Generator | None = None
        self.config.train_num_workers = adjust_num_workers_for_world_size(
            nurec_config.trainer,
            self.config.train_num_workers,
        )
        self.config.val_num_workers = adjust_num_workers_for_world_size(
            nurec_config.trainer,
            self.config.val_num_workers,
        )
        self.config.test_num_workers = adjust_num_workers_for_world_size(
            nurec_config.trainer,
            self.config.test_num_workers,
        )

        logger.info(
            "SODataModule: train_num_workers=%d val_num_workers=%d test_num_workers=%d",
            self.config.train_num_workers,
            self.config.val_num_workers,
            self.config.test_num_workers,
        )

        self.max_train_num_rays: int | None = None

        if "train" in self.nurec_config.mode:
            self.train_dataset = make_dataset(self.nurec_config.dataset.name, self.nurec_config, split="train")

        if "val" in self.nurec_config.mode or "test" in self.nurec_config.mode:
            self.val_dataset = make_dataset(self.nurec_config.dataset.name, self.nurec_config, split="val")

    def get_datasource(self) -> BaseDataSource:
        if "train" in self.nurec_config.mode:
            return self.train_dataset.get_datasource()
        return self.val_dataset.get_datasource()

    def state_dict(self) -> dict[str, Any]:
        """Return rank-indexed training-loader generator states for checkpointing."""
        local_state = self._train_loader_generator.get_state() if self._train_loader_generator is not None else None
        # Lightning persists only rank 0's checkpoint, so gather every rank's independent DDP stream.
        states = distributed_all_gather_nested([local_state], only_on_rank_zero=False)
        return {
            self._TRAIN_LOADER_GENERATOR_STATES_KEY: states,
            self._TRAIN_LOADER_GENERATOR_RANK_COUNT_KEY: len(states),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore this rank's training-loader generator state when available."""
        states = state_dict.get(self._TRAIN_LOADER_GENERATOR_STATES_KEY)
        if states is None:
            return

        assert self.trainer is not None
        rank = self.trainer.global_rank
        if rank >= len(states):
            if rank < state_dict.get(self._TRAIN_LOADER_GENERATOR_RANK_COUNT_KEY, len(states)):
                raise ValueError(f"Checkpoint has no training-loader generator state for rank {rank}")
            return

        rank_state = states[rank]
        if rank_state is not None:
            self._train_loader_generator = torch.Generator(device="cpu")
            self._train_loader_generator.set_state(rank_state.cpu())

    def train_dataloader(self) -> DataLoader:
        """Returns a training dataloader with optional prefetching"""
        if self._train_loader_generator is None:
            # A private generator prevents worker seeding from consuming the process-global CPU RNG.
            self._train_loader_generator = torch.Generator(device="cpu")
            self._train_loader_generator.manual_seed(compute_process_local_rng_seed())

        if self.config.prefetch.enabled:
            return PrefetchDataLoader(
                self.train_dataset,
                device="cuda",
                queue_size=self.config.prefetch.queue_size,
                num_workers=self.config.train_num_workers,
                persistent_workers=True if self.config.train_num_workers > 0 else False,
                batch_size=self.config.train_batch_size,
                pin_memory=True,
                collate_fn=self.__batch_collate_fn,
                generator=self._train_loader_generator,
            )
        else:
            DataLoaderClass = MockDataLoader if self.config.mock_dataloader else DataLoader
            return DataLoaderClass(
                self.train_dataset,
                num_workers=self.config.train_num_workers,
                persistent_workers=True if self.config.train_num_workers > 0 else False,
                batch_size=self.config.train_batch_size,
                pin_memory=True,
                collate_fn=self.__batch_collate_fn,
                generator=self._train_loader_generator,
            )

    def val_dataloader(self, dataloader_sampler: Sampler | None = None) -> DataLoader | None:
        """
        Returns a validation dataloader.
        Args:
            dataloader_sampler: Sampler to use for the dataloader. Note that this is not the same as the
        nurec.datasets.samplers!
        """
        if "val" in self.nurec_config.mode or "test" in self.nurec_config.mode:
            return DataLoader(
                self.val_dataset,
                num_workers=self.config.val_num_workers,
                persistent_workers=True if self.config.val_num_workers > 0 else False,
                batch_size=1,
                collate_fn=self.__batch_collate_fn,
                pin_memory=True,
                sampler=dataloader_sampler,
            )
        else:
            return None

    def test_dataloader(self, dataloader_sampler: Sampler | None = None) -> DataLoader | None:
        """
        Returns a test dataloader.
        Args:
            dataloader_sampler: Sampler to use for the dataloader. Note that this is not the same as the
        nurec.datasets.samplers!
        """
        if "test" in self.nurec_config.mode:
            return DataLoader(
                self.val_dataset,
                num_workers=self.config.test_num_workers,
                persistent_workers=True if self.config.test_num_workers > 0 else False,
                batch_size=1,
                collate_fn=self.__batch_collate_fn,
                pin_memory=True,
                sampler=dataloader_sampler,
            )
        else:
            return None

    def train_dataloader_sequential(self) -> DataLoader:
        """
        Returns a dataloader that sequentially goes over the potentially subsampled training set.
        Used for modules such as the uncertainty estimator that need to be fitted against training rays
        at the end of training.
        """

        return DataLoader(
            make_dataset(self.nurec_config.dataset.name, self.nurec_config, split="train-sequential"),
            num_workers=self.config.train_num_workers,
            persistent_workers=True if self.config.train_num_workers > 0 else False,
            batch_size=1,
            collate_fn=self.__batch_collate_fn,
            pin_memory=True,
        )

    def update_epoch(self, epoch: int, system, **kwargs) -> None:
        if "train" in self.nurec_config.mode:
            update_n_epochs = self.config.update_n_epochs
            assert update_n_epochs != 0 and (epoch + 1) % update_n_epochs == 0
            self.train_dataset.update_epoch(epoch, system, **kwargs)

    def get_max_train_num_rays(self) -> int:
        """Returns the number of rays per training batch"""
        assert "train" in self.nurec_config.mode, "can only be called if train mode is enabled"

        if self.max_train_num_rays is not None:
            return self.max_train_num_rays

        self.max_train_num_rays = self.config.train_batch_size * self.train_dataset.get_max_num_rays_per_train_sample()

        return self.max_train_num_rays
