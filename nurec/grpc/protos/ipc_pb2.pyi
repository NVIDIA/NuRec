from typing import ClassVar as _ClassVar
from typing import Iterable as _Iterable
from typing import Optional as _Optional
from typing import Union as _Union

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor

class IpcCompletionMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COMPLETION_ON_RETURN: _ClassVar[IpcCompletionMode]
    EVENT_ORDERED: _ClassVar[IpcCompletionMode]

COMPLETION_ON_RETURN: IpcCompletionMode
EVENT_ORDERED: IpcCompletionMode

class OpenIpcSessionRequest(_message.Message):
    __slots__ = ("requested_width", "requested_height", "requested_buffer_count", "completion_mode")
    REQUESTED_WIDTH_FIELD_NUMBER: _ClassVar[int]
    REQUESTED_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    REQUESTED_BUFFER_COUNT_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_MODE_FIELD_NUMBER: _ClassVar[int]
    requested_width: int
    requested_height: int
    requested_buffer_count: int
    completion_mode: IpcCompletionMode
    def __init__(
        self,
        requested_width: _Optional[int] = ...,
        requested_height: _Optional[int] = ...,
        requested_buffer_count: _Optional[int] = ...,
        completion_mode: _Optional[_Union[IpcCompletionMode, str]] = ...,
    ) -> None: ...

class OpenIpcSessionResponse(_message.Message):
    __slots__ = (
        "session_id",
        "cuda_ipc_mem_handles",
        "buffer_size",
        "device_uuid",
        "actual_width",
        "actual_height",
        "completion_mode",
        "cuda_ipc_ready_event_handles",
        "linux_boot_id",
        "cuda_runtime_version",
    )
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    CUDA_IPC_MEM_HANDLES_FIELD_NUMBER: _ClassVar[int]
    BUFFER_SIZE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_UUID_FIELD_NUMBER: _ClassVar[int]
    ACTUAL_WIDTH_FIELD_NUMBER: _ClassVar[int]
    ACTUAL_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_MODE_FIELD_NUMBER: _ClassVar[int]
    CUDA_IPC_READY_EVENT_HANDLES_FIELD_NUMBER: _ClassVar[int]
    LINUX_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    CUDA_RUNTIME_VERSION_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    cuda_ipc_mem_handles: _containers.RepeatedScalarFieldContainer[bytes]
    buffer_size: int
    device_uuid: bytes
    actual_width: int
    actual_height: int
    completion_mode: IpcCompletionMode
    cuda_ipc_ready_event_handles: _containers.RepeatedScalarFieldContainer[bytes]
    linux_boot_id: str
    cuda_runtime_version: int
    def __init__(
        self,
        session_id: _Optional[str] = ...,
        cuda_ipc_mem_handles: _Optional[_Iterable[bytes]] = ...,
        buffer_size: _Optional[int] = ...,
        device_uuid: _Optional[bytes] = ...,
        actual_width: _Optional[int] = ...,
        actual_height: _Optional[int] = ...,
        completion_mode: _Optional[_Union[IpcCompletionMode, str]] = ...,
        cuda_ipc_ready_event_handles: _Optional[_Iterable[bytes]] = ...,
        linux_boot_id: _Optional[str] = ...,
        cuda_runtime_version: _Optional[int] = ...,
    ) -> None: ...

class CloseIpcSessionRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class CloseIpcSessionResponse(_message.Message):
    __slots__ = ("success", "error_message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error_message: str
    def __init__(self, success: bool = ..., error_message: _Optional[str] = ...) -> None: ...
