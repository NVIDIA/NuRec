# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Class-label normalisation for the Gaussian statistics export."""

from omegaconf import OmegaConf

from nurec.utils.io.export.gaussian_statistics import classLabelsAsList


def test_class_labels_from_a_config_node() -> None:
    node = OmegaConf.create(["car", "truck"])

    assert classLabelsAsList(node) == ["car", "truck"]


def test_class_labels_from_a_plain_list() -> None:
    """A config parsed from a written parsed.yaml yields a plain list, which must not raise."""
    assert classLabelsAsList(["car", "truck"]) == ["car", "truck"]


def test_class_labels_when_absent() -> None:
    assert classLabelsAsList(None) == []
    assert classLabelsAsList([]) == []
    assert classLabelsAsList(OmegaConf.create([])) == []
