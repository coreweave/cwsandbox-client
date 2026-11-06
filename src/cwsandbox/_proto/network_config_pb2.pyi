# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client
from cwsandbox._proto import sandbox_pb2 as _sandbox_pb2
from google.api import annotations_pb2 as _annotations_pb2
from google.api import field_behavior_pb2 as _field_behavior_pb2
from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import field_mask_pb2 as _field_mask_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class NetworkConfig(_message.Message):
    __slots__ = ("network_config_id", "spec", "status", "etag")
    NETWORK_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    SPEC_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ETAG_FIELD_NUMBER: _ClassVar[int]
    network_config_id: str
    spec: _sandbox_pb2.NetworkOptions
    status: NetworkConfigStatus
    etag: str
    def __init__(self, network_config_id: _Optional[str] = ..., spec: _Optional[_Union[_sandbox_pb2.NetworkOptions, _Mapping]] = ..., status: _Optional[_Union[NetworkConfigStatus, _Mapping]] = ..., etag: _Optional[str] = ...) -> None: ...

class NetworkConfigStatus(_message.Message):
    __slots__ = ("sandbox_id", "platform_managed", "create_time", "update_time")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    PLATFORM_MANAGED_FIELD_NUMBER: _ClassVar[int]
    CREATE_TIME_FIELD_NUMBER: _ClassVar[int]
    UPDATE_TIME_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    platform_managed: bool
    create_time: _timestamp_pb2.Timestamp
    update_time: _timestamp_pb2.Timestamp
    def __init__(self, sandbox_id: _Optional[str] = ..., platform_managed: bool = ..., create_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., update_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class CreateNetworkConfigRequest(_message.Message):
    __slots__ = ("network_config", "request_id")
    NETWORK_CONFIG_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    network_config: NetworkConfig
    request_id: str
    def __init__(self, network_config: _Optional[_Union[NetworkConfig, _Mapping]] = ..., request_id: _Optional[str] = ...) -> None: ...

class GetNetworkConfigRequest(_message.Message):
    __slots__ = ("network_config_id",)
    NETWORK_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    network_config_id: str
    def __init__(self, network_config_id: _Optional[str] = ...) -> None: ...

class ListNetworkConfigsRequest(_message.Message):
    __slots__ = ("page_size", "page_token", "platform_managed")
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    PLATFORM_MANAGED_FIELD_NUMBER: _ClassVar[int]
    page_size: int
    page_token: str
    platform_managed: bool
    def __init__(self, page_size: _Optional[int] = ..., page_token: _Optional[str] = ..., platform_managed: bool = ...) -> None: ...

class ListNetworkConfigsResponse(_message.Message):
    __slots__ = ("network_configs", "next_page_token")
    NETWORK_CONFIGS_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    network_configs: _containers.RepeatedCompositeFieldContainer[NetworkConfig]
    next_page_token: str
    def __init__(self, network_configs: _Optional[_Iterable[_Union[NetworkConfig, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class UpdateNetworkConfigRequest(_message.Message):
    __slots__ = ("network_config_id", "network_config", "update_mask")
    NETWORK_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    NETWORK_CONFIG_FIELD_NUMBER: _ClassVar[int]
    UPDATE_MASK_FIELD_NUMBER: _ClassVar[int]
    network_config_id: str
    network_config: NetworkConfig
    update_mask: _field_mask_pb2.FieldMask
    def __init__(self, network_config_id: _Optional[str] = ..., network_config: _Optional[_Union[NetworkConfig, _Mapping]] = ..., update_mask: _Optional[_Union[_field_mask_pb2.FieldMask, _Mapping]] = ...) -> None: ...

class DeleteNetworkConfigRequest(_message.Message):
    __slots__ = ("network_config_id", "allow_missing")
    NETWORK_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    ALLOW_MISSING_FIELD_NUMBER: _ClassVar[int]
    network_config_id: str
    allow_missing: bool
    def __init__(self, network_config_id: _Optional[str] = ..., allow_missing: bool = ...) -> None: ...
