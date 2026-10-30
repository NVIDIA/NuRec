# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture-safe CUDA Generator singleton for training-hot-path RNG.

PyTorch officially supports ``torch.randn`` / ``torch.rand`` / ``torch.randint``
against the default CUDA generator inside ``torch.cuda.graph`` captured
regions via intra-graph Philox offset tracking — see the "CUDA Graph-safe
RNG states" Note in ``aten/src/ATen/cuda/CUDAGeneratorImpl.h`` and the
prose at `docs.pytorch.org/docs/main/notes/cuda.html
<https://docs.pytorch.org/docs/main/notes/cuda.html>`_. However, in
practice training runs hit ``RuntimeError: Offset increment outside graph
capture encountered unexpectedly`` — a corrupted-generator-state symptom
from a prior failed capture (see open issue `pytorch#171263
<https://github.com/pytorch/pytorch/issues/171263>`_, where
``capture_end()`` throws before ``capture_epilogue()`` leaving
``capturing_ = true`` and tripping all subsequent default-generator RNG).
This module side-steps that failure mode by routing training-hot-path
draws through a user-allocated ``torch.Generator(device='cuda')`` whose
state is independent of the default generator's; if pytorch#171263 lands
a fix, the singleton can be retired with confidence.

A user-allocated ``torch.Generator(device='cuda')`` stores its
``(seed, offset)`` in device memory. Once the generator is registered
with the target ``CUDAGraph`` via ``g.register_generator_state(gen)``,
kernels launched with ``generator=gen`` inside
``with torch.cuda.graph(g):`` advance the offset device-side per launch,
so replay produces fresh randomness every iteration. The kernel-launch
arguments stay pointer-stable -- exactly what graph capture requires.

Hot-path call sites pass ``generator=get_cuda_generator(device)`` to
``torch.rand`` / ``torch.randn`` / ``torch.randint``. Code paths that
wrap these inside ``torch.cuda.graph(...)`` must call
``register_with_graph(graph, device)`` once on the ``CUDAGraph`` before
entering the capture context, otherwise the kernel raises
``RuntimeError: Attempt to increase offset for a CUDA generator not in
capture mode``.

API surface:

    get_cuda_generator(device)
        Lazy accessor. First call allocates a `torch.Generator(device=cuda)`
        seeded with ``0``; subsequent calls return that singleton.

    seed_cuda_generator(device, seed)
        Explicit re-seed. Idempotent across allocation state -- creates
        the singleton if missing, re-seeds it if already allocated.
        Trainer init uses this to wire per-rank seeding so DDP ranks
        decorrelate (see ``nurec.utils.misc.compute_process_local_rng_seed``).

    register_with_graph(graph, device)
        Bind the singleton to a specific ``CUDAGraph`` so RNG ops capture
        cleanly. Required once per graph before ``with torch.cuda.graph(g):``.

    reset_singleton()
        Forget the cached generator. Test/init use only.

    read_checkpoint_state(checkpoint)
        Return the persisted per-rank generator states stored under
        ``CHECKPOINT_STATE_KEY`` (or the pre-rename
        ``LEGACY_CHECKPOINT_STATE_KEY``) in a Lightning checkpoint dict, or
        ``None`` when absent.

Limitations:

    Not ``torch.compile(fullgraph=True)`` clean. Dynamo lacks an
    ``as_proxy()`` for ``torch.Generator`` and graph-breaks on
    ``torch.rand(generator=...)``; under ``fullgraph=True`` that break is
    escalated to a hard error. Only torch's global default seed is
    handled inside compiled regions today (``replace_random.py``).
    See https://github.com/pytorch/pytorch/issues/88576 for the
    documented limitation.
"""

from __future__ import annotations

from typing import Optional

import torch

from nurec.utils.renamed_keys import normalize_renamed_key


_GEN: Optional[torch.Generator] = None

# Lightning checkpoint key that carries the singleton's per-rank states.
CHECKPOINT_STATE_KEY = "nurec.cuda_rng.singleton_state"
# Spelling written by checkpoints that predate the nre -> nurec rename.
LEGACY_CHECKPOINT_STATE_KEY = "nre.cuda_rng.singleton_state"


CheckpointState = torch.Tensor | list[Optional[torch.Tensor]]


def _states_equal(a: Optional[CheckpointState], b: Optional[CheckpointState]) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_states_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.equal(a, b)
    return a is None and b is None


def read_checkpoint_state(checkpoint: dict) -> Optional[CheckpointState]:
    """Return the persisted generator state(s) under the current or the legacy key, or None.

    Raises:
        ValueError: If both keys are present with different states.
    """
    return normalize_renamed_key(
        checkpoint, CHECKPOINT_STATE_KEY, LEGACY_CHECKPOINT_STATE_KEY, equal=_states_equal
    ).get(CHECKPOINT_STATE_KEY)


def get_cuda_generator(device: torch.device) -> torch.Generator:
    """Return the process-shared CUDA Generator bound to ``device``.

    First call lazily allocates with ``seed=0`` and returns the new
    singleton; subsequent calls return the same object so all hot-path
    RNG draws share state and naturally decorrelate across call sites.

    For explicit (re-)seeding, use :func:`seed_cuda_generator`. The
    `seed` parameter the previous API exposed only honored on the lazy
    path was a footgun -- subsequent calls silently dropped it -- so
    seeding is now a separate entry point that is idempotent across
    allocation state.

    ``device`` MUST be a cuda device. A non-cuda device passed in after
    :func:`register_with_graph` has wired the singleton into a live
    ``CUDAGraph`` would silently overwrite ``_GEN`` with a fresh
    non-cuda generator, stranding the live graph's pre-registered
    handle; the next captured ``torch.rand(generator=...)`` would then
    raise ``RuntimeError: Attempt to increase offset for a CUDA
    generator not in capture mode``. The assert below makes that misuse
    fail-fast at the API surface instead of by-construction-only.

    Match policy: any cuda-device request hits an existing cuda-bound
    generator regardless of index. Lightning, torch.cuda.current_stream,
    and tensors created by user code all return ``torch.device`` objects
    with inconsistent ``index=None`` vs ``index=0`` representations even
    on the same physical GPU, and a strict-equality check would
    re-create the generator on every representation drift -- which would
    leave the new generator unregistered with the active CUDAGraph and
    abort capture with ``Offset increment outside graph capture
    encountered unexpectedly``.

    Single-device-per-process is the supported mode: the first
    cuda-bound caller wins. Pipeline-parallel / model-sharded execution
    that wants distinct per-device generators would need a per-device
    dict instead of a single scalar slot; left as a future change.
    """
    _assert_cuda(device)
    global _GEN
    if _GEN is not None:
        return _GEN
    return _allocate(device, seed=0)


def seed_cuda_generator(device: torch.device, seed: int) -> torch.Generator:
    """Re-seed the shared CUDA Generator (creating it if missing).

    Idempotent across allocation state: if the singleton is unallocated,
    allocates and seeds; if already allocated, calls ``manual_seed(seed)``
    on the existing generator. Either way the post-condition is the same:
    the singleton exists, is bound to a cuda device, and starts the next
    draw at ``(seed, offset=0)``.

    Trainer init wires this per-rank to keep DDP ranks decorrelated::

        from nurec.utils.cuda_rng import seed_cuda_generator
        from nurec.utils.misc import compute_process_local_rng_seed
        seed_cuda_generator(device, compute_process_local_rng_seed())

    Lightning's ``pl.seed_everything(config.seed, workers=True)`` only
    seeds torch's default generators; the user-allocated cuda_rng
    singleton has to be re-seeded explicitly here. ``config.seed`` flows
    through Lightning's ``PL_GLOBAL_SEED`` env var, so calling
    ``compute_process_local_rng_seed()`` with no arg picks it up
    automatically.
    """
    _assert_cuda(device)
    global _GEN
    if _GEN is None:
        return _allocate(device, seed=seed)
    _GEN.manual_seed(seed)
    return _GEN


def reset_singleton() -> None:
    """Forget the cached generator. Test/init use only.

    The next :func:`get_cuda_generator` (or :func:`seed_cuda_generator`)
    call lazily allocates a fresh ``torch.Generator(device='cuda')``.
    Tests use this between cases so each test sees a clean singleton
    that can be registered with its own ``CUDAGraph`` without colliding
    with a prior test's ``g.reset()``-ed graph state.
    """
    global _GEN
    _GEN = None


def register_with_graph(graph: "torch.cuda.CUDAGraph", device: torch.device) -> None:
    """Register the singleton CUDA Generator with ``graph``.

    Must be called once per ``CUDAGraph``, before entering the
    ``with torch.cuda.graph(...)`` context. Without this call,
    ``torch.rand`` / ``torch.randn`` / ``torch.randint`` against the
    singleton inside the captured region trips ``RuntimeError: Attempt
    to increase offset for a CUDA generator not in capture mode``.
    """
    graph.register_generator_state(get_cuda_generator(device))


def _assert_cuda(device: torch.device) -> None:
    assert device.type == "cuda", (
        f"cuda_rng requires a cuda device, got device={device!r}; "
        "the singleton stays cuda-bound for the life of the process."
    )


def _allocate(device: torch.device, seed: int) -> torch.Generator:
    global _GEN
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    _GEN = gen
    return _GEN
