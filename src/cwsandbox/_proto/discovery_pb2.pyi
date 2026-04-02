# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client
from google.api import annotations_pb2 as _annotations_pb2
from google.api import field_behavior_pb2 as _field_behavior_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TowerView(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TOWER_VIEW_UNSPECIFIED: _ClassVar[TowerView]
    TOWER_VIEW_BASIC: _ClassVar[TowerView]
    TOWER_VIEW_FULL: _ClassVar[TowerView]
TOWER_VIEW_UNSPECIFIED: TowerView
TOWER_VIEW_BASIC: TowerView
TOWER_VIEW_FULL: TowerView

class AvailableTower(_message.Message):
    __slots__ = ("tower_id", "tower_group_id", "tags", "healthy", "capabilities", "resources", "runway_names", "connected_at", "is_shared")
    TOWER_ID_FIELD_NUMBER: _ClassVar[int]
    TOWER_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    RUNWAY_NAMES_FIELD_NUMBER: _ClassVar[int]
    CONNECTED_AT_FIELD_NUMBER: _ClassVar[int]
    IS_SHARED_FIELD_NUMBER: _ClassVar[int]
    tower_id: str
    tower_group_id: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    healthy: bool
    capabilities: TowerCapabilitySummary
    resources: TowerResourceSummary
    runway_names: _containers.RepeatedScalarFieldContainer[str]
    connected_at: _timestamp_pb2.Timestamp
    is_shared: bool
    def __init__(self, tower_id: _Optional[str] = ..., tower_group_id: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., healthy: bool = ..., capabilities: _Optional[_Union[TowerCapabilitySummary, _Mapping]] = ..., resources: _Optional[_Union[TowerResourceSummary, _Mapping]] = ..., runway_names: _Optional[_Iterable[str]] = ..., connected_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., is_shared: bool = ...) -> None: ...

class TowerCapabilitySummary(_message.Message):
    __slots__ = ("max_cpu_millicores", "max_memory_bytes", "max_gpu_count", "supported_gpu_types", "supported_architectures", "supports_privileged", "available_storage_classes")
    MAX_CPU_MILLICORES_FIELD_NUMBER: _ClassVar[int]
    MAX_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    MAX_GPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_GPU_TYPES_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_ARCHITECTURES_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_PRIVILEGED_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_STORAGE_CLASSES_FIELD_NUMBER: _ClassVar[int]
    max_cpu_millicores: int
    max_memory_bytes: int
    max_gpu_count: int
    supported_gpu_types: _containers.RepeatedScalarFieldContainer[str]
    supported_architectures: _containers.RepeatedScalarFieldContainer[str]
    supports_privileged: bool
    available_storage_classes: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, max_cpu_millicores: _Optional[int] = ..., max_memory_bytes: _Optional[int] = ..., max_gpu_count: _Optional[int] = ..., supported_gpu_types: _Optional[_Iterable[str]] = ..., supported_architectures: _Optional[_Iterable[str]] = ..., supports_privileged: bool = ..., available_storage_classes: _Optional[_Iterable[str]] = ...) -> None: ...

class TowerResourceSummary(_message.Message):
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

class ServiceExposureMode(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class EgressMode(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class RunwaySummary(_message.Message):
    __slots__ = ("runway_name", "tower_id", "supported_gpu_types", "supported_architectures", "service_exposure_modes", "egress_modes")
    RUNWAY_NAME_FIELD_NUMBER: _ClassVar[int]
    TOWER_ID_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_GPU_TYPES_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_ARCHITECTURES_FIELD_NUMBER: _ClassVar[int]
    SERVICE_EXPOSURE_MODES_FIELD_NUMBER: _ClassVar[int]
    EGRESS_MODES_FIELD_NUMBER: _ClassVar[int]
    runway_name: str
    tower_id: str
    supported_gpu_types: _containers.RepeatedScalarFieldContainer[str]
    supported_architectures: _containers.RepeatedScalarFieldContainer[str]
    service_exposure_modes: _containers.RepeatedCompositeFieldContainer[ServiceExposureMode]
    egress_modes: _containers.RepeatedCompositeFieldContainer[EgressMode]
    def __init__(self, runway_name: _Optional[str] = ..., tower_id: _Optional[str] = ..., supported_gpu_types: _Optional[_Iterable[str]] = ..., supported_architectures: _Optional[_Iterable[str]] = ..., service_exposure_modes: _Optional[_Iterable[_Union[ServiceExposureMode, _Mapping]]] = ..., egress_modes: _Optional[_Iterable[_Union[EgressMode, _Mapping]]] = ...) -> None: ...

class ListAvailableTowersRequest(_message.Message):
    __slots__ = ("tower_group_id", "runway_name", "gpu_type", "architecture", "exclude_shared", "healthy_only", "page_size", "page_token", "view")
    TOWER_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    RUNWAY_NAME_FIELD_NUMBER: _ClassVar[int]
    GPU_TYPE_FIELD_NUMBER: _ClassVar[int]
    ARCHITECTURE_FIELD_NUMBER: _ClassVar[int]
    EXCLUDE_SHARED_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_ONLY_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    VIEW_FIELD_NUMBER: _ClassVar[int]
    tower_group_id: str
    runway_name: str
    gpu_type: str
    architecture: str
    exclude_shared: bool
    healthy_only: bool
    page_size: int
    page_token: str
    view: TowerView
    def __init__(self, tower_group_id: _Optional[str] = ..., runway_name: _Optional[str] = ..., gpu_type: _Optional[str] = ..., architecture: _Optional[str] = ..., exclude_shared: bool = ..., healthy_only: bool = ..., page_size: _Optional[int] = ..., page_token: _Optional[str] = ..., view: _Optional[_Union[TowerView, str]] = ...) -> None: ...

class ListAvailableTowersResponse(_message.Message):
    __slots__ = ("towers", "next_page_token")
    TOWERS_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    towers: _containers.RepeatedCompositeFieldContainer[AvailableTower]
    next_page_token: str
    def __init__(self, towers: _Optional[_Iterable[_Union[AvailableTower, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class GetAvailableTowerRequest(_message.Message):
    __slots__ = ("tower_id", "view")
    TOWER_ID_FIELD_NUMBER: _ClassVar[int]
    VIEW_FIELD_NUMBER: _ClassVar[int]
    tower_id: str
    view: TowerView
    def __init__(self, tower_id: _Optional[str] = ..., view: _Optional[_Union[TowerView, str]] = ...) -> None: ...

class GetRunwayRequest(_message.Message):
    __slots__ = ("runway_name", "tower_id")
    RUNWAY_NAME_FIELD_NUMBER: _ClassVar[int]
    TOWER_ID_FIELD_NUMBER: _ClassVar[int]
    runway_name: str
    tower_id: str
    def __init__(self, runway_name: _Optional[str] = ..., tower_id: _Optional[str] = ...) -> None: ...

class ListRunwaysRequest(_message.Message):
    __slots__ = ("gpu_type", "architecture", "exclude_shared", "page_size", "page_token", "tower_id")
    GPU_TYPE_FIELD_NUMBER: _ClassVar[int]
    ARCHITECTURE_FIELD_NUMBER: _ClassVar[int]
    EXCLUDE_SHARED_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    TOWER_ID_FIELD_NUMBER: _ClassVar[int]
    gpu_type: str
    architecture: str
    exclude_shared: bool
    page_size: int
    page_token: str
    tower_id: str
    def __init__(self, gpu_type: _Optional[str] = ..., architecture: _Optional[str] = ..., exclude_shared: bool = ..., page_size: _Optional[int] = ..., page_token: _Optional[str] = ..., tower_id: _Optional[str] = ...) -> None: ...

class ListRunwaysResponse(_message.Message):
    __slots__ = ("runways", "next_page_token")
    RUNWAYS_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    runways: _containers.RepeatedCompositeFieldContainer[RunwaySummary]
    next_page_token: str
    def __init__(self, runways: _Optional[_Iterable[_Union[RunwaySummary, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
