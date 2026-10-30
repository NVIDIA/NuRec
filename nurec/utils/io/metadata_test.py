# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import pytest

from nurec.utils import media_log
from nurec.utils.io.metadata import get_logger_metadata


class _RunBackend:
    """Backend with an active run; only active_run is exercised here."""

    def active_run(self) -> Optional[tuple[str, str]]:
        return ("https://wandb.example/run/abc", "abc")


@pytest.fixture
def registered_backend():
    previous = media_log.media_backend()
    media_log.register_media_backend(_RunBackend())
    yield
    media_log._backend = previous


@pytest.mark.parametrize("logger_name", ["dummy", "tensorboard"])
def test_non_wandb_logger_ignores_active_run(registered_backend, logger_name):
    # A registered backend with an open run must not attach that run's URL and
    # id to metadata written under a different logger.
    logger = get_logger_metadata(logger_name)
    assert logger.name == logger_name
    assert logger.run_url is None
    assert logger.run_id is None


def test_wandb_logger_records_active_run(registered_backend):
    logger = get_logger_metadata("wandb")
    assert (logger.run_url, logger.run_id) == ("https://wandb.example/run/abc", "abc")


def test_wandb_logger_without_backend_leaves_run_unset():
    previous = media_log.media_backend()
    media_log._backend = None
    try:
        logger = get_logger_metadata("wandb")
    finally:
        media_log._backend = previous
    assert logger.run_url is None
    assert logger.run_id is None
