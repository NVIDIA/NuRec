# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Keep this file, otherwise find_packages() in setup.py will not find this package and its sub-packages,
# and they do not get installed with 'pip install .' (as copies) or 'pip install -e .' (as references
# to the source tree). This will lead to import errors outside of the repo root.
# In turn, find_namespace_packages() can be used in setup.py but it is very sensitive and would
# need too many exclusions to achieve the desired result.

import logging
import os


# pytorch-lightning 2.6.5's checkpoint loader (Trainer fit/validate resume ->
# lightning_fabric cloud_io._load) defaults torch.load(weights_only=True),
# unlike 2.4.0. NuRec checkpoints carry OmegaConf containers and internal pickled
# types from trusted training runs, which weights_only=True rejects. Force the
# legacy full-unpickle load process-wide rather than threading weights_only=False
# through every PL resume call site.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from nurec.config.version import get_version
from nurec.repo_root import __reporoot__  # noqa: F401  # re-exported as nurec.__reporoot__ for downstream consumers
from nurec.utils.colored_exceptions import enable_colored_exceptions
from nurec.utils.debug.remote_debug import breakpoint_env
from nurec.utils.legacy_pickle import install_torch_load_module_remap
from nurec.utils.misc import is_env_true


# Checkpoints written before the package was renamed from `nre` to `nurec` pickle
# `nre.config.*` enums in `hyper_parameters`; remap them on the default load path.
install_torch_load_module_remap()


# Conditionally activate remote debugging based on environment variables
breakpoint_env()

# Note: version information is not available *intentionally* in some environments
# (like build / test sandboxes) to prevent build / test cache invalidations
__version__ = get_version()

log_level = os.environ.get("LOGLEVEL", "INFO")
level = getattr(logging, log_level)

log_format = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
logging.basicConfig(format=log_format, level=level)

# Enable colored exception handling if not explicitly disabled
if is_env_true("NUREC_ENABLE_COLORED_EXCEPTIONS", True):
    enable_colored_exceptions()

logger = logging.getLogger(__name__)
