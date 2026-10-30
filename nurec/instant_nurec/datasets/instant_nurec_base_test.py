# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib

from typing import NoReturn
from unittest.mock import MagicMock

import numpy as np
import pytest

import nurec.instant_nurec.datasets.registry as registry_module

from nurec.instant_nurec.config.dataset import InstantNuRecMixedDatasetConfig
from nurec.instant_nurec.config.dataset import TestIndexInstantNuRecDatasetConfig as IndexDatasetConfig
from nurec.instant_nurec.datasets.instant_nurec_base import InstantNuRecDataError, MixedInstantNuRecDataset
from nurec.instant_nurec.datasets.instant_nurec_base import TestIndexInstantNuRecDataset as IndexDataset
from nurec.instant_nurec.datasets.registry import make


def _sample_rng(dataset: IndexDataset, batch_idx: int) -> int:
    return int(dataset._get_rng(batch_idx).integers(0, 2**31))


def _expected_sample(*, epoch: int, batch_idx: int, seed: int) -> int:
    digest = hashlib.sha256(f"{epoch}_{batch_idx}_{seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return int(rng.integers(0, 2**31))


def test_rng_uses_lightning_environment_when_seed_is_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PL_GLOBAL_SEED", "17")
    config = IndexDatasetConfig(name="instant_nurec-test-index", size=1)
    dataset = IndexDataset(config)
    dataset.set_rng_epoch(3)

    assert _sample_rng(dataset, 5) == _expected_sample(epoch=3, batch_idx=5, seed=17)


def test_omitted_seed_preserves_dynamic_environment_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    config = IndexDatasetConfig(name="instant_nurec-test-index", size=1)
    dataset = IndexDataset(config)

    monkeypatch.setenv("PL_GLOBAL_SEED", "1")
    first = _sample_rng(dataset, 0)
    monkeypatch.setenv("PL_GLOBAL_SEED", "2")

    assert _sample_rng(dataset, 0) != first


def test_omitted_seed_requires_lightning_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PL_GLOBAL_SEED", raising=False)
    config = IndexDatasetConfig(name="instant_nurec-test-index", size=1)
    dataset = IndexDataset(config)

    with pytest.raises(AssertionError, match="Please call pl.seed_everything"):
        _sample_rng(dataset, 0)


def test_explicit_seed_does_not_depend_on_lightning_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    config = IndexDatasetConfig(name="instant_nurec-test-index", size=1)
    dataset = IndexDataset(config, seed=0)

    monkeypatch.delenv("PL_GLOBAL_SEED", raising=False)
    first = _sample_rng(dataset, 0)
    monkeypatch.setenv("PL_GLOBAL_SEED", "1")
    assert _sample_rng(dataset, 0) == first


def test_registry_preserves_environment_fallback_when_seed_is_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PL_GLOBAL_SEED", "7")
    config = IndexDatasetConfig(name="instant_nurec-test-index", size=1)

    dataset = make(config.name, config, "train")

    assert isinstance(dataset, IndexDataset)
    assert _sample_rng(dataset, 0) == _expected_sample(epoch=-1, batch_idx=0, seed=7)


def test_registry_omits_keyword_for_legacy_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    config = IndexDatasetConfig(name="instant_nurec-test-index", size=1)
    calls: list[tuple[IndexDatasetConfig, str]] = []

    def legacy_constructor(dataset_config: IndexDatasetConfig, split: str) -> IndexDataset:
        calls.append((dataset_config, split))
        return IndexDataset(dataset_config, split)

    monkeypatch.setitem(registry_module.datasets, "legacy", legacy_constructor)

    omitted = registry_module.make("legacy", config, "train")
    explicit_none = registry_module.make("legacy", config, "train", seed=None)
    with pytest.raises(TypeError, match="unexpected keyword argument 'seed'"):
        registry_module.make("legacy", config, "train", seed=1)

    assert isinstance(omitted, IndexDataset)
    assert isinstance(explicit_none, IndexDataset)
    assert calls == [(config, "train"), (config, "train")]


def test_registry_forwards_explicit_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PL_GLOBAL_SEED", raising=False)
    config = IndexDatasetConfig(name="instant_nurec-test-index", size=1)

    zero = make(config.name, config, "train", seed=0)
    other = make(config.name, config, "train", seed=12)

    assert isinstance(zero, IndexDataset)
    assert isinstance(other, IndexDataset)
    assert _sample_rng(zero, 0) == _expected_sample(epoch=-1, batch_idx=0, seed=0)
    assert _sample_rng(other, 0) == _expected_sample(epoch=-1, batch_idx=0, seed=12)
    assert _sample_rng(zero, 0) != _sample_rng(other, 0)


def test_registry_rejects_constructor_that_drops_explicit_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PL_GLOBAL_SEED", "7")
    config = IndexDatasetConfig(name="instant_nurec-test-index", size=1)

    def drops_seed(
        dataset_config: IndexDatasetConfig,
        split: str,
        *,
        seed: int | None = None,
    ) -> IndexDataset:
        return IndexDataset(dataset_config, split)

    monkeypatch.setitem(registry_module.datasets, "drops-seed", drops_seed)

    with pytest.raises(RuntimeError, match="accepted seed=12 but did not forward"):
        registry_module.make("drops-seed", config, "train", seed=12)


def test_mixed_dataset_forwards_explicit_seed_to_components() -> None:
    component_config = IndexDatasetConfig(name="instant_nurec-test-index", size=2)
    config = InstantNuRecMixedDatasetConfig(
        name="instant_nurec-mixed",
        mixture={
            "component": InstantNuRecMixedDatasetConfig.MixtureComponentConfig(
                sample_ratio=1.0,
                config=component_config,
            )
        },
    )

    dataset = MixedInstantNuRecDataset(config, seed=23)
    component = dataset.datasets[0].dataset

    assert isinstance(component, IndexDataset)
    assert _sample_rng(component, 0) == _sample_rng(IndexDataset(component_config, seed=23), 0)


@pytest.mark.parametrize("dataset_size", [1, 3, 10, 12])
def test_failed_dataset_retries_are_bounded_even_when_rng_repeats(
    dataset_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = IndexDataset(IndexDatasetConfig(name="instant_nurec-test-index", size=dataset_size), seed=0)
    attempted: list[int] = []
    draw_count = 0
    rng = MagicMock()

    def draw_zero(low: int, high: int) -> int:
        nonlocal draw_count
        draw_count += 1
        # Fail quickly, rather than hanging the test, if a rejection loop returns.
        assert draw_count <= dataset_size, "Retry selection did not terminate"
        assert low == 0 and high > 0
        return 0

    def fail(batch_idx: int, rng: np.random.Generator) -> NoReturn:
        attempted.append(batch_idx)
        raise InstantNuRecDataError("No synchronized context available")

    rng.integers.side_effect = draw_zero
    monkeypatch.setattr(dataset, "_get_rng", lambda batch_idx: rng)
    monkeypatch.setattr(dataset, "getitem_allow_exceptions", fail)
    expected_attempts = min(10, dataset_size)

    with pytest.raises(InstantNuRecDataError, match=f"tried out {expected_attempts} attempts") as error:
        dataset[0]

    assert isinstance(error.value.__cause__, InstantNuRecDataError)
    assert error.value.__cause__.message == "No synchronized context available"
    assert attempted == list(range(expected_attempts))
    assert draw_count == expected_attempts - 1
