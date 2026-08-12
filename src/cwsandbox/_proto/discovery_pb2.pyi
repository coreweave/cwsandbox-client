# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client
from cwsandbox._proto import sandbox_pb2 as _sandbox_pb2
from google.api import annotations_pb2 as _annotations_pb2
from google.api import field_behavior_pb2 as _field_behavior_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RunnerView(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RUNNER_VIEW_UNSPECIFIED: _ClassVar[RunnerView]
    RUNNER_VIEW_BASIC: _ClassVar[RunnerView]
    RUNNER_VIEW_FULL: _ClassVar[RunnerView]
RUNNER_VIEW_UNSPECIFIED: RunnerView
RUNNER_VIEW_BASIC: RunnerView
RUNNER_VIEW_FULL: RunnerView

class AvailableRunner(_message.Message):
    __slots__ = ("runner_id", "runner_group_id", "tags", "healthy", "capabilities", "resources", "connected_at", "is_shared", "organization_id")
    RUNNER_ID_FIELD_NUMBER: _ClassVar[int]
    RUNNER_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    CONNECTED_AT_FIELD_NUMBER: _ClassVar[int]
    IS_SHARED_FIELD_NUMBER: _ClassVar[int]
    ORGANIZATION_ID_FIELD_NUMBER: _ClassVar[int]
    runner_id: str
    runner_group_id: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    healthy: bool
    capabilities: RunnerCapabilitySummary
    resources: RunnerResourceSummary
    connected_at: _timestamp_pb2.Timestamp
    is_shared: bool
    organization_id: str
    def __init__(self, runner_id: _Optional[str] = ..., runner_group_id: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., healthy: bool = ..., capabilities: _Optional[_Union[RunnerCapabilitySummary, _Mapping]] = ..., resources: _Optional[_Union[RunnerResourceSummary, _Mapping]] = ..., connected_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., is_shared: bool = ..., organization_id: _Optional[str] = ...) -> None: ...

class RunnerCapabilitySummary(_message.Message):
    __slots__ = ("max_cpu_millicores", "max_memory_bytes", "max_gpu_count", "supported_gpu_types", "supported_architectures", "supports_privileged", "available_storage_classes", "supported_service_visibilities")
    MAX_CPU_MILLICORES_FIELD_NUMBER: _ClassVar[int]
    MAX_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    MAX_GPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_GPU_TYPES_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_ARCHITECTURES_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_PRIVILEGED_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_STORAGE_CLASSES_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_SERVICE_VISIBILITIES_FIELD_NUMBER: _ClassVar[int]
    max_cpu_millicores: int
    max_memory_bytes: int
    max_gpu_count: int
    supported_gpu_types: _containers.RepeatedScalarFieldContainer[str]
    supported_architectures: _containers.RepeatedScalarFieldContainer[str]
    supports_privileged: bool
    available_storage_classes: _containers.RepeatedScalarFieldContainer[str]
    supported_service_visibilities: _containers.RepeatedScalarFieldContainer[_sandbox_pb2.Visibility]
    def __init__(self, max_cpu_millicores: _Optional[int] = ..., max_memory_bytes: _Optional[int] = ..., max_gpu_count: _Optional[int] = ..., supported_gpu_types: _Optional[_Iterable[str]] = ..., supported_architectures: _Optional[_Iterable[str]] = ..., supports_privileged: bool = ..., available_storage_classes: _Optional[_Iterable[str]] = ..., supported_service_visibilities: _Optional[_Iterable[_Union[_sandbox_pb2.Visibility, str]]] = ...) -> None: ...

class RunnerResourceSummary(_message.Message):
    __slots__ = ("available_cpu_millicores", "available_memory_bytes", "available_gpu_count", "running_sandboxes")
    AVAILABLE_CPU_MILLICORES_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_GPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    RUNNING_SANDBOXES_FIELD_NUMBER: _ClassVar[int]
    available_cpu_millicores: int
    available_memory_bytes: int
    available_gpu_count: int
    running_sandboxes: int
    def __init__(self, available_cpu_millicores: _Optional[int] = ..., available_memory_bytes: _Optional[int] = ..., available_gpu_count: _Optional[int] = ..., running_sandboxes: _Optional[int] = ...) -> None: ...

class ListAvailableRunnersRequest(_message.Message):
    __slots__ = ("runner_group_id", "gpu_type", "architecture", "healthy_only", "page_size", "page_token", "view", "service_visibility")
    RUNNER_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    GPU_TYPE_FIELD_NUMBER: _ClassVar[int]
    ARCHITECTURE_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_ONLY_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    VIEW_FIELD_NUMBER: _ClassVar[int]
    SERVICE_VISIBILITY_FIELD_NUMBER: _ClassVar[int]
    runner_group_id: str
    gpu_type: str
    architecture: str
    healthy_only: bool
    page_size: int
    page_token: str
    view: RunnerView
    service_visibility: _sandbox_pb2.Visibility
    def __init__(self, runner_group_id: _Optional[str] = ..., gpu_type: _Optional[str] = ..., architecture: _Optional[str] = ..., healthy_only: bool = ..., page_size: _Optional[int] = ..., page_token: _Optional[str] = ..., view: _Optional[_Union[RunnerView, str]] = ..., service_visibility: _Optional[_Union[_sandbox_pb2.Visibility, str]] = ...) -> None: ...

class ListAvailableRunnersResponse(_message.Message):
    __slots__ = ("runners", "next_page_token")
    RUNNERS_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    runners: _containers.RepeatedCompositeFieldContainer[AvailableRunner]
    next_page_token: str
    def __init__(self, runners: _Optional[_Iterable[_Union[AvailableRunner, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class GetAvailableRunnerRequest(_message.Message):
    __slots__ = ("runner_id", "view", "organization_id")
    RUNNER_ID_FIELD_NUMBER: _ClassVar[int]
    VIEW_FIELD_NUMBER: _ClassVar[int]
    ORGANIZATION_ID_FIELD_NUMBER: _ClassVar[int]
    runner_id: str
    view: RunnerView
    organization_id: str
    def __init__(self, runner_id: _Optional[str] = ..., view: _Optional[_Union[RunnerView, str]] = ..., organization_id: _Optional[str] = ...) -> None: ...
