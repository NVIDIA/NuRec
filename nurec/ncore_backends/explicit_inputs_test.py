# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

import nurec.ncore_backends.explicit_inputs as explicit_inputs

from nurec.ncore_backends.interface import UnsupportedBackendOperationError


def test_registered_explicit_sequence_input_provider_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(explicit_inputs, "_provider", None)
    provider = MagicMock()

    explicit_inputs.register_explicit_sequence_input_provider(provider)

    assert explicit_inputs.require_explicit_sequence_input_provider() is provider


def test_missing_explicit_sequence_input_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(explicit_inputs, "_provider", None)

    with pytest.raises(
        UnsupportedBackendOperationError,
        match="does not support explicit sequence path patterns",
    ):
        explicit_inputs.require_explicit_sequence_input_provider()


def test_explicit_sequence_input_provider_cannot_be_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(explicit_inputs, "_provider", MagicMock())

    with pytest.raises(ValueError, match="already registered"):
        explicit_inputs.register_explicit_sequence_input_provider(MagicMock())
