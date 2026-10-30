# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import TracebackType
from typing import Protocol


class LockLike(Protocol):
    """Protocol for a lock object. Used to swap out threading.Lock with MockLock below."""

    def __enter__(self) -> bool: ...
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None: ...
    def acquire(self, blocking: bool = ..., timeout: float = ...) -> bool: ...
    def release(self) -> None: ...


class MockLock:
    """USDZ viewer does not need to lock anything"""

    def __enter__(self) -> bool:
        return True

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        pass

    def acquire(self, blocking: bool = False, timeout: float = 0.0) -> bool:
        return True

    def release(self) -> None:
        pass
