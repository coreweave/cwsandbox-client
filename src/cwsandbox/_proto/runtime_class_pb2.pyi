# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client
from google.api import field_behavior_pb2 as _field_behavior_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RuntimeClass(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RUNTIME_CLASS_UNSPECIFIED: _ClassVar[RuntimeClass]
    RUNTIME_CLASS_CPU_DEFAULT: _ClassVar[RuntimeClass]
    RUNTIME_CLASS_CPU_NESTED_VIRT: _ClassVar[RuntimeClass]
    RUNTIME_CLASS_CPU_SNAPSHOT_RESTORE: _ClassVar[RuntimeClass]
    RUNTIME_CLASS_GPU_DEFAULT: _ClassVar[RuntimeClass]
RUNTIME_CLASS_UNSPECIFIED: RuntimeClass
RUNTIME_CLASS_CPU_DEFAULT: RuntimeClass
RUNTIME_CLASS_CPU_NESTED_VIRT: RuntimeClass
RUNTIME_CLASS_CPU_SNAPSHOT_RESTORE: RuntimeClass
RUNTIME_CLASS_GPU_DEFAULT: RuntimeClass

class RuntimeClassMapping(_message.Message):
    __slots__ = ("runtime_class", "kubernetes_runtime_class_name", "node_default")
    RUNTIME_CLASS_FIELD_NUMBER: _ClassVar[int]
    KUBERNETES_RUNTIME_CLASS_NAME_FIELD_NUMBER: _ClassVar[int]
    NODE_DEFAULT_FIELD_NUMBER: _ClassVar[int]
    runtime_class: RuntimeClass
    kubernetes_runtime_class_name: str
    node_default: bool
    def __init__(self, runtime_class: _Optional[_Union[RuntimeClass, str]] = ..., kubernetes_runtime_class_name: _Optional[str] = ..., node_default: bool = ...) -> None: ...
