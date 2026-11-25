from typing import ClassVar as _ClassVar
from typing import Optional as _Optional

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message

DESCRIPTOR: _descriptor.FileDescriptor

class OpenRdmaSessionRequest(_message.Message):
    __slots__ = ("gid", "gid_index", "qp_num", "psn", "max_buf_size")
    GID_FIELD_NUMBER: _ClassVar[int]
    GID_INDEX_FIELD_NUMBER: _ClassVar[int]
    QP_NUM_FIELD_NUMBER: _ClassVar[int]
    PSN_FIELD_NUMBER: _ClassVar[int]
    MAX_BUF_SIZE_FIELD_NUMBER: _ClassVar[int]
    gid: bytes
    gid_index: int
    qp_num: int
    psn: int
    max_buf_size: int
    def __init__(
        self,
        gid: _Optional[bytes] = ...,
        gid_index: _Optional[int] = ...,
        qp_num: _Optional[int] = ...,
        psn: _Optional[int] = ...,
        max_buf_size: _Optional[int] = ...,
    ) -> None: ...

class OpenRdmaSessionResponse(_message.Message):
    __slots__ = ("session_id", "gid", "gid_index", "qp_num", "psn")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    GID_FIELD_NUMBER: _ClassVar[int]
    GID_INDEX_FIELD_NUMBER: _ClassVar[int]
    QP_NUM_FIELD_NUMBER: _ClassVar[int]
    PSN_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    gid: bytes
    gid_index: int
    qp_num: int
    psn: int
    def __init__(
        self,
        session_id: _Optional[str] = ...,
        gid: _Optional[bytes] = ...,
        gid_index: _Optional[int] = ...,
        qp_num: _Optional[int] = ...,
        psn: _Optional[int] = ...,
    ) -> None: ...

class RegisterReceiveBufferRequest(_message.Message):
    __slots__ = ("session_id", "remote_addr", "rkey", "buffer_size", "buffer_id")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    REMOTE_ADDR_FIELD_NUMBER: _ClassVar[int]
    RKEY_FIELD_NUMBER: _ClassVar[int]
    BUFFER_SIZE_FIELD_NUMBER: _ClassVar[int]
    BUFFER_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    remote_addr: int
    rkey: int
    buffer_size: int
    buffer_id: int
    def __init__(
        self,
        session_id: _Optional[str] = ...,
        remote_addr: _Optional[int] = ...,
        rkey: _Optional[int] = ...,
        buffer_size: _Optional[int] = ...,
        buffer_id: _Optional[int] = ...,
    ) -> None: ...

class RegisterReceiveBufferResponse(_message.Message):
    __slots__ = ("success", "error_message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error_message: str
    def __init__(self, success: bool = ..., error_message: _Optional[str] = ...) -> None: ...

class CloseRdmaSessionRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class CloseRdmaSessionResponse(_message.Message):
    __slots__ = ("success", "error_message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error_message: str
    def __init__(self, success: bool = ..., error_message: _Optional[str] = ...) -> None: ...
