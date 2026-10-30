# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for ``nurec.utils.cuda_rng``.

The module documents that, when a ``torch.Generator(device='cuda')``
returned by :func:`get_cuda_generator` is registered with a
``torch.cuda.CUDAGraph`` via :func:`register_with_graph`, RNG ops invoked
with ``generator=`` advance the generator's offset device-side and
therefore work inside a ``torch.cuda.graph(...)`` capture region.

These tests exercise that contract for the three ``torch`` RNG
primitives the module is documented to support (``torch.rand``,
``torch.randn``, ``torch.randint``), plus the per-element
``randn * std + mean`` composition pattern used by
``GSplatStrategy._split_gaussians`` (in lieu of the direct
``torch.normal(mean, std, generator=...)``, which is not
stream-capture-safe in this PyTorch version). Each test:

  1. Allocates the singleton generator and registers it with a fresh
     ``CUDAGraph``.
  2. Captures one invocation of the RNG op against that generator.
  3. Replays the captured graph twice and verifies the output buffer
     receives fresh, finite values whose seed-advance contract holds
     (consecutive replays produce different draws).

The tests are pure ``cuda_rng`` unit tests; they do not depend on any
NuRec training-step machinery or on the gsplat rendering backends.
"""

from __future__ import annotations

import pytest
import torch

from nurec.utils.cuda_rng import (
    get_cuda_generator,
    register_with_graph,
    reset_singleton,
    seed_cuda_generator,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda_rng tests require a CUDA device")


@pytest.fixture(autouse=True)
def _reset_cuda_rng_singleton():
    """Reset the cuda_rng module-level singleton between tests.

    Production will have one ``CUDAGraph`` (the training-step wrapper's);
    currently zero — this module is enablement, capture-mode call sites
    land in a follow-up. In a test suite, several tests each create their
    own ``CUDAGraph`` and call ``register_generator_state`` on the same
    singleton; once one test's graph is reset, the next test's first
    eager use of the same generator trips ``"Offset increment outside
    graph capture encountered unexpectedly"``. Resetting the singleton
    between tests gives each test a fresh generator instance and avoids
    the multi-graph interaction.
    """
    reset_singleton()
    yield
    reset_singleton()


def _capture_and_replay_twice(
    body,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture ``body()`` under ``torch.cuda.graph(...)`` and replay twice.

    Returns a (snapshot1, snapshot2) tuple of cloned outputs from the
    first and second replay. ``body`` should be a closure that writes
    its result into a tensor reachable from outside (we clone to detach
    from the captured graph's memory pool).

    The generator must already have been used at least once eagerly so
    its first-call lazy init has settled before capture begins.
    """
    g = torch.cuda.CUDAGraph()
    register_with_graph(g, gen.device)
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        out = body()
    g.replay()
    torch.cuda.synchronize()
    snap1 = out.detach().clone()
    g.replay()
    torch.cuda.synchronize()
    snap2 = out.detach().clone()
    g.reset()
    return snap1, snap2


def test_get_cuda_generator_returns_singleton() -> None:
    """Repeated calls with cuda devices return the same generator object."""
    g1 = get_cuda_generator(torch.device("cuda"))
    g2 = get_cuda_generator(torch.device("cuda", 0))
    g3 = get_cuda_generator(torch.device("cuda:0"))
    assert g1 is g2 is g3, "expected singleton policy across cuda-device aliases"
    assert g1.device.type == "cuda"


def test_get_cuda_generator_rejects_non_cuda_device() -> None:
    """A non-cuda device is rejected up front so a stale singleton can't be stranded.

    See ``get_cuda_generator``'s docstring: silently re-binding the
    singleton to a cpu device after :func:`register_with_graph` would
    leave a live ``CUDAGraph``'s pre-registered handle pointing at a
    free'd cuda generator and trip ``RuntimeError: Attempt to increase
    offset for a CUDA generator not in capture mode`` on the next
    captured ``torch.rand(generator=...)``. The assert moves the
    detection from by-construction-only (every call site has to remember
    the invariant) to fail-fast at the API surface.
    """
    with pytest.raises(AssertionError, match="cuda device"):
        get_cuda_generator(torch.device("cpu"))


def test_seed_cuda_generator_re_seeds_existing_singleton() -> None:
    """``seed_cuda_generator`` re-seeds even if the singleton already exists.

    Pins the new split-API contract (vs the old footgun where
    ``get_cuda_generator(device, seed=N)`` silently dropped ``seed`` on
    the second call). After re-seeding to the same seed twice, two
    ``torch.rand(generator=gen)`` draws must agree bit-for-bit.
    """
    device = torch.device("cuda")
    seed_cuda_generator(device, 42)
    gen1 = get_cuda_generator(device)
    s1 = torch.rand(64, generator=gen1, device=device)

    # Re-seed via the explicit entry point; the singleton object is
    # preserved (so any pre-registered CUDAGraph stays valid), only its
    # internal `(seed, offset)` resets.
    gen2 = seed_cuda_generator(device, 42)
    assert gen2 is gen1, "seed_cuda_generator must preserve the singleton object"
    s2 = torch.rand(64, generator=gen2, device=device)

    assert torch.equal(s1, s2), "same seed twice must produce identical draws"


def test_reset_singleton_clears_cached_generator() -> None:
    """``reset_singleton`` makes the next ``get_cuda_generator`` allocate fresh."""
    device = torch.device("cuda")
    g1 = get_cuda_generator(device)
    reset_singleton()
    g2 = get_cuda_generator(device)
    assert g1 is not g2, "reset_singleton should force re-allocation"


def test_unregistered_generator_under_capture_raises() -> None:
    """Forgetting :func:`register_with_graph` is documented to trip a specific error.

    Pins the contract spelled out in the module docstring -- if PyTorch
    evolves the message or the failure mode (e.g., changes which op
    surfaces it), this test fails and the docstring needs to follow.
    """
    device = torch.device("cuda")
    gen = get_cuda_generator(device)
    # Eager warmup -- exercise the kernel path once so the caching
    # allocator has stable allocations before capture begins.
    torch.rand(64, generator=gen, device=device)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    # Deliberately NO register_with_graph(g, gen.device) here.
    with pytest.raises(RuntimeError, match="not in capture mode"):
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            torch.rand(64, generator=gen, device=device)


def test_torch_rand_with_cuda_rng_under_torch_cuda_graph() -> None:
    """``torch.rand(generator=cuda_rng_gen)`` works inside ``torch.cuda.graph(...)``."""
    device = torch.device("cuda")
    gen = get_cuda_generator(device)
    out = torch.empty(1024, device=device)

    def body() -> torch.Tensor:
        out.copy_(torch.rand(1024, generator=gen, device=device))
        return out

    body()  # eager warmup
    torch.cuda.synchronize()
    s1, s2 = _capture_and_replay_twice(body, gen)
    assert torch.isfinite(s1).all() and torch.isfinite(s2).all()
    # Per-replay seed advance should yield distinct draws.
    assert not torch.equal(s1, s2)
    # Uniform[0, 1) range.
    assert s1.min() >= 0.0 and s1.max() < 1.0


def test_torch_randn_with_cuda_rng_under_torch_cuda_graph() -> None:
    """``torch.randn(generator=cuda_rng_gen)`` works inside ``torch.cuda.graph(...)``."""
    device = torch.device("cuda")
    gen = get_cuda_generator(device)
    out = torch.empty(512, 3, device=device)

    def body() -> torch.Tensor:
        out.copy_(torch.randn(512, 3, generator=gen, device=device))
        return out

    body()
    torch.cuda.synchronize()
    s1, s2 = _capture_and_replay_twice(body, gen)
    assert torch.isfinite(s1).all() and torch.isfinite(s2).all()
    assert not torch.equal(s1, s2)


def test_torch_randint_with_cuda_rng_under_torch_cuda_graph() -> None:
    """``torch.randint(generator=cuda_rng_gen)`` works inside ``torch.cuda.graph(...)``."""
    device = torch.device("cuda")
    gen = get_cuda_generator(device)
    out = torch.empty(256, dtype=torch.int64, device=device)

    def body() -> torch.Tensor:
        out.copy_(torch.randint(0, 1000, (256,), generator=gen, device=device))
        return out

    body()
    torch.cuda.synchronize()
    s1, s2 = _capture_and_replay_twice(body, gen)
    assert s1.min() >= 0 and s1.max() < 1000
    assert s2.min() >= 0 and s2.max() < 1000
    assert not torch.equal(s1, s2)


def test_per_element_normal_via_randn_under_torch_cuda_graph() -> None:
    """``randn(generator=cuda_rng_gen) * std + mean`` works inside ``torch.cuda.graph(...)``.

    Per-element mean/std Normal sampling is the pattern used by gsplat's
    densify-split path (and any other caller that needs heterogeneous
    Normal noise). ``torch.normal(mean, std, generator=...)`` itself is
    *not* stream-capture-safe in this PyTorch version (raises
    ``"CUDA error: operation not permitted when stream is capturing"``);
    the canonical capture-clean replacement is to compose
    ``torch.randn(generator=) * std + mean``, which is what production
    callers must use. This test pins that contract.
    """
    device = torch.device("cuda")
    gen = get_cuda_generator(device)
    means = torch.zeros(1024, 3, device=device)
    stds = torch.full((1024, 3), 0.1, device=device)
    out = torch.empty(1024, 3, device=device)

    def body() -> torch.Tensor:
        out.copy_(torch.randn(stds.shape, generator=gen, device=device, dtype=stds.dtype) * stds + means)
        return out

    body()
    torch.cuda.synchronize()
    s1, s2 = _capture_and_replay_twice(body, gen)
    assert torch.isfinite(s1).all() and torch.isfinite(s2).all()
    assert not torch.equal(s1, s2)
