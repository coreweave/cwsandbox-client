# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client
import datetime

from google.api import field_behavior_pb2 as _field_behavior_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ExecStreamRequest(_message.Message):
    __slots__ = ("init", "stdin", "resize", "close")
    INIT_FIELD_NUMBER: _ClassVar[int]
    STDIN_FIELD_NUMBER: _ClassVar[int]
    RESIZE_FIELD_NUMBER: _ClassVar[int]
    CLOSE_FIELD_NUMBER: _ClassVar[int]
    init: ExecStreamInit
    stdin: ExecStreamData
    resize: ExecStreamResize
    close: ExecStreamClose
    def __init__(self, init: _Optional[_Union[ExecStreamInit, _Mapping]] = ..., stdin: _Optional[_Union[ExecStreamData, _Mapping]] = ..., resize: _Optional[_Union[ExecStreamResize, _Mapping]] = ..., close: _Optional[_Union[ExecStreamClose, _Mapping]] = ...) -> None: ...

class ExecStreamInit(_message.Message):
    __slots__ = ("sandbox_id", "command", "tty", "tty_width", "tty_height", "env")
    class EnvEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    TTY_FIELD_NUMBER: _ClassVar[int]
    TTY_WIDTH_FIELD_NUMBER: _ClassVar[int]
    TTY_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    ENV_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    command: _containers.RepeatedScalarFieldContainer[str]
    tty: bool
    tty_width: int
    tty_height: int
    env: _containers.ScalarMap[str, str]
    def __init__(self, sandbox_id: _Optional[str] = ..., command: _Optional[_Iterable[str]] = ..., tty: _Optional[bool] = ..., tty_width: _Optional[int] = ..., tty_height: _Optional[int] = ..., env: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ExecStreamData(_message.Message):
    __slots__ = ("data",)
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    def __init__(self, data: _Optional[bytes] = ...) -> None: ...

class ExecStreamResize(_message.Message):
    __slots__ = ("width", "height")
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    width: int
    height: int
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ...) -> None: ...

class ExecStreamClose(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StreamingExecReady(_message.Message):
    __slots__ = ("ready_at",)
    READY_AT_FIELD_NUMBER: _ClassVar[int]
    ready_at: _timestamp_pb2.Timestamp
    def __init__(self, ready_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ExecStreamResponse(_message.Message):
    __slots__ = ("output", "exit", "error", "ready")
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    EXIT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    READY_FIELD_NUMBER: _ClassVar[int]
    output: ExecStreamOutput
    exit: ExecStreamExit
    error: ExecStreamError
    ready: StreamingExecReady
    def __init__(self, output: _Optional[_Union[ExecStreamOutput, _Mapping]] = ..., exit: _Optional[_Union[ExecStreamExit, _Mapping]] = ..., error: _Optional[_Union[ExecStreamError, _Mapping]] = ..., ready: _Optional[_Union[StreamingExecReady, _Mapping]] = ...) -> None: ...

class ExecStreamOutput(_message.Message):
    __slots__ = ("stream_type", "data", "timestamp")
    class StreamType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STREAM_TYPE_UNSPECIFIED: _ClassVar[ExecStreamOutput.StreamType]
        STREAM_TYPE_STDOUT: _ClassVar[ExecStreamOutput.StreamType]
        STREAM_TYPE_STDERR: _ClassVar[ExecStreamOutput.StreamType]
    STREAM_TYPE_UNSPECIFIED: ExecStreamOutput.StreamType
    STREAM_TYPE_STDOUT: ExecStreamOutput.StreamType
    STREAM_TYPE_STDERR: ExecStreamOutput.StreamType
    STREAM_TYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    stream_type: ExecStreamOutput.StreamType
    data: bytes
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, stream_type: _Optional[_Union[ExecStreamOutput.StreamType, str]] = ..., data: _Optional[bytes] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ExecStreamExit(_message.Message):
    __slots__ = ("exit_code", "completed_at")
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_AT_FIELD_NUMBER: _ClassVar[int]
    exit_code: int
    completed_at: _timestamp_pb2.Timestamp
    def __init__(self, exit_code: _Optional[int] = ..., completed_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ExecStreamError(_message.Message):
    __slots__ = ("message", "code")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    message: str
    code: str
    def __init__(self, message: _Optional[str] = ..., code: _Optional[str] = ...) -> None: ...

class LogStreamRequest(_message.Message):
    __slots__ = ("init", "close")
    INIT_FIELD_NUMBER: _ClassVar[int]
    CLOSE_FIELD_NUMBER: _ClassVar[int]
    init: LogStreamInit
    close: LogStreamClose
    def __init__(self, init: _Optional[_Union[LogStreamInit, _Mapping]] = ..., close: _Optional[_Union[LogStreamClose, _Mapping]] = ...) -> None: ...

class LogStreamInit(_message.Message):
    __slots__ = ("sandbox_id", "follow", "tail_lines", "since_time", "timestamps")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    FOLLOW_FIELD_NUMBER: _ClassVar[int]
    TAIL_LINES_FIELD_NUMBER: _ClassVar[int]
    SINCE_TIME_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    follow: bool
    tail_lines: int
    since_time: _timestamp_pb2.Timestamp
    timestamps: bool
    def __init__(self, sandbox_id: _Optional[str] = ..., follow: _Optional[bool] = ..., tail_lines: _Optional[int] = ..., since_time: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., timestamps: _Optional[bool] = ...) -> None: ...

class LogStreamClose(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class LogStreamResponse(_message.Message):
    __slots__ = ("data", "error", "complete")
    DATA_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    COMPLETE_FIELD_NUMBER: _ClassVar[int]
    data: LogStreamData
    error: LogStreamError
    complete: LogStreamComplete
    def __init__(self, data: _Optional[_Union[LogStreamData, _Mapping]] = ..., error: _Optional[_Union[LogStreamError, _Mapping]] = ..., complete: _Optional[_Union[LogStreamComplete, _Mapping]] = ...) -> None: ...

class LogStreamData(_message.Message):
    __slots__ = ("data", "timestamp")
    DATA_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, data: _Optional[bytes] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class LogStreamComplete(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class LogStreamError(_message.Message):
    __slots__ = ("message", "code")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    message: str
    code: str
    def __init__(self, message: _Optional[str] = ..., code: _Optional[str] = ...) -> None: ...

class StreamingExecCommand(_message.Message):
    __slots__ = ("session_id", "container_id", "command", "tty", "tty_width", "tty_height", "env", "issued_at")
    class EnvEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_ID_FIELD_NUMBER: _ClassVar[int]
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    TTY_FIELD_NUMBER: _ClassVar[int]
    TTY_WIDTH_FIELD_NUMBER: _ClassVar[int]
    TTY_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    ENV_FIELD_NUMBER: _ClassVar[int]
    ISSUED_AT_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    container_id: str
    command: _containers.RepeatedScalarFieldContainer[str]
    tty: bool
    tty_width: int
    tty_height: int
    env: _containers.ScalarMap[str, str]
    issued_at: _timestamp_pb2.Timestamp
    def __init__(self, session_id: _Optional[str] = ..., container_id: _Optional[str] = ..., command: _Optional[_Iterable[str]] = ..., tty: _Optional[bool] = ..., tty_width: _Optional[int] = ..., tty_height: _Optional[int] = ..., env: _Optional[_Mapping[str, str]] = ..., issued_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class StreamingLogCommand(_message.Message):
    __slots__ = ("session_id", "container_id", "follow", "tail_lines", "since_time", "timestamps", "issued_at")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_ID_FIELD_NUMBER: _ClassVar[int]
    FOLLOW_FIELD_NUMBER: _ClassVar[int]
    TAIL_LINES_FIELD_NUMBER: _ClassVar[int]
    SINCE_TIME_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    ISSUED_AT_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    container_id: str
    follow: bool
    tail_lines: int
    since_time: _timestamp_pb2.Timestamp
    timestamps: bool
    issued_at: _timestamp_pb2.Timestamp
    def __init__(self, session_id: _Optional[str] = ..., container_id: _Optional[str] = ..., follow: _Optional[bool] = ..., tail_lines: _Optional[int] = ..., since_time: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., timestamps: _Optional[bool] = ..., issued_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class TerminalResize(_message.Message):
    __slots__ = ("width", "height")
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    width: int
    height: int
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ...) -> None: ...

class StreamingDataCommand(_message.Message):
    __slots__ = ("session_id", "stdin_data", "resize", "close_stdin", "cancel_stream")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    STDIN_DATA_FIELD_NUMBER: _ClassVar[int]
    RESIZE_FIELD_NUMBER: _ClassVar[int]
    CLOSE_STDIN_FIELD_NUMBER: _ClassVar[int]
    CANCEL_STREAM_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    stdin_data: bytes
    resize: TerminalResize
    close_stdin: bool
    cancel_stream: bool
    def __init__(self, session_id: _Optional[str] = ..., stdin_data: _Optional[bytes] = ..., resize: _Optional[_Union[TerminalResize, _Mapping]] = ..., close_stdin: _Optional[bool] = ..., cancel_stream: _Optional[bool] = ...) -> None: ...

class StreamOutput(_message.Message):
    __slots__ = ("type", "data", "timestamp")
    class OutputType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        OUTPUT_TYPE_UNSPECIFIED: _ClassVar[StreamOutput.OutputType]
        OUTPUT_TYPE_STDOUT: _ClassVar[StreamOutput.OutputType]
        OUTPUT_TYPE_STDERR: _ClassVar[StreamOutput.OutputType]
    OUTPUT_TYPE_UNSPECIFIED: StreamOutput.OutputType
    OUTPUT_TYPE_STDOUT: StreamOutput.OutputType
    OUTPUT_TYPE_STDERR: StreamOutput.OutputType
    TYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    type: StreamOutput.OutputType
    data: bytes
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, type: _Optional[_Union[StreamOutput.OutputType, str]] = ..., data: _Optional[bytes] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class StreamExit(_message.Message):
    __slots__ = ("exit_code", "completed_at")
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_AT_FIELD_NUMBER: _ClassVar[int]
    exit_code: int
    completed_at: _timestamp_pb2.Timestamp
    def __init__(self, exit_code: _Optional[int] = ..., completed_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class StreamError(_message.Message):
    __slots__ = ("message", "code")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    message: str
    code: str
    def __init__(self, message: _Optional[str] = ..., code: _Optional[str] = ...) -> None: ...

class StreamingExecData(_message.Message):
    __slots__ = ("session_id", "output", "exit", "error", "ready")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    EXIT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    READY_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    output: StreamOutput
    exit: StreamExit
    error: StreamError
    ready: StreamingExecReady
    def __init__(self, session_id: _Optional[str] = ..., output: _Optional[_Union[StreamOutput, _Mapping]] = ..., exit: _Optional[_Union[StreamExit, _Mapping]] = ..., error: _Optional[_Union[StreamError, _Mapping]] = ..., ready: _Optional[_Union[StreamingExecReady, _Mapping]] = ...) -> None: ...

class StreamingLogData(_message.Message):
    __slots__ = ("session_id", "log_data", "error", "complete", "timestamp")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    LOG_DATA_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    COMPLETE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    log_data: bytes
    error: StreamError
    complete: bool
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, session_id: _Optional[str] = ..., log_data: _Optional[bytes] = ..., error: _Optional[_Union[StreamError, _Mapping]] = ..., complete: _Optional[bool] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...
