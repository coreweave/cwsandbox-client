# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client
from google.api import annotations_pb2 as _annotations_pb2
from google.api import field_behavior_pb2 as _field_behavior_pb2
from google.protobuf import field_mask_pb2 as _field_mask_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class VolumeState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    VOLUME_STATE_UNSPECIFIED: _ClassVar[VolumeState]
    VOLUME_STATE_VALIDATING: _ClassVar[VolumeState]
    VOLUME_STATE_READY: _ClassVar[VolumeState]
    VOLUME_STATE_ERROR: _ClassVar[VolumeState]
    VOLUME_STATE_DELETING: _ClassVar[VolumeState]

class VolumeLocality(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    VOLUME_LOCALITY_UNSPECIFIED: _ClassVar[VolumeLocality]
    VOLUME_LOCALITY_CLUSTER_LOCAL: _ClassVar[VolumeLocality]
    VOLUME_LOCALITY_GLOBAL: _ClassVar[VolumeLocality]
VOLUME_STATE_UNSPECIFIED: VolumeState
VOLUME_STATE_VALIDATING: VolumeState
VOLUME_STATE_READY: VolumeState
VOLUME_STATE_ERROR: VolumeState
VOLUME_STATE_DELETING: VolumeState
VOLUME_LOCALITY_UNSPECIFIED: VolumeLocality
VOLUME_LOCALITY_CLUSTER_LOCAL: VolumeLocality
VOLUME_LOCALITY_GLOBAL: VolumeLocality

class Volume(_message.Message):
    __slots__ = ("volume_id", "spec", "status")
    VOLUME_ID_FIELD_NUMBER: _ClassVar[int]
    SPEC_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    volume_id: str
    spec: VolumeSpec
    status: VolumeStatus
    def __init__(self, volume_id: _Optional[str] = ..., spec: _Optional[_Union[VolumeSpec, _Mapping]] = ..., status: _Optional[_Union[VolumeStatus, _Mapping]] = ...) -> None: ...

class VolumeSpec(_message.Message):
    __slots__ = ("pvc", "read_only", "description")
    PVC_FIELD_NUMBER: _ClassVar[int]
    READ_ONLY_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    pvc: PvcVolumeSource
    read_only: bool
    description: str
    def __init__(self, pvc: _Optional[_Union[PvcVolumeSource, _Mapping]] = ..., read_only: bool = ..., description: _Optional[str] = ...) -> None: ...

class PvcVolumeSource(_message.Message):
    __slots__ = ("runner_id", "namespace", "claim_name", "sub_path")
    RUNNER_ID_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    CLAIM_NAME_FIELD_NUMBER: _ClassVar[int]
    SUB_PATH_FIELD_NUMBER: _ClassVar[int]
    runner_id: str
    namespace: str
    claim_name: str
    sub_path: str
    def __init__(self, runner_id: _Optional[str] = ..., namespace: _Optional[str] = ..., claim_name: _Optional[str] = ..., sub_path: _Optional[str] = ...) -> None: ...

class VolumeStatus(_message.Message):
    __slots__ = ("state", "state_reason", "locality", "create_time", "update_time", "last_validated_time", "access_modes", "capacity", "attached_sandbox_count")
    STATE_FIELD_NUMBER: _ClassVar[int]
    STATE_REASON_FIELD_NUMBER: _ClassVar[int]
    LOCALITY_FIELD_NUMBER: _ClassVar[int]
    CREATE_TIME_FIELD_NUMBER: _ClassVar[int]
    UPDATE_TIME_FIELD_NUMBER: _ClassVar[int]
    LAST_VALIDATED_TIME_FIELD_NUMBER: _ClassVar[int]
    ACCESS_MODES_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_FIELD_NUMBER: _ClassVar[int]
    ATTACHED_SANDBOX_COUNT_FIELD_NUMBER: _ClassVar[int]
    state: VolumeState
    state_reason: str
    locality: VolumeLocality
    create_time: _timestamp_pb2.Timestamp
    update_time: _timestamp_pb2.Timestamp
    last_validated_time: _timestamp_pb2.Timestamp
    access_modes: _containers.RepeatedScalarFieldContainer[str]
    capacity: str
    attached_sandbox_count: int
    def __init__(self, state: _Optional[_Union[VolumeState, str]] = ..., state_reason: _Optional[str] = ..., locality: _Optional[_Union[VolumeLocality, str]] = ..., create_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., update_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., last_validated_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., access_modes: _Optional[_Iterable[str]] = ..., capacity: _Optional[str] = ..., attached_sandbox_count: _Optional[int] = ...) -> None: ...

class CreateVolumeRequest(_message.Message):
    __slots__ = ("volume", "request_id")
    VOLUME_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    volume: Volume
    request_id: str
    def __init__(self, volume: _Optional[_Union[Volume, _Mapping]] = ..., request_id: _Optional[str] = ...) -> None: ...

class GetVolumeRequest(_message.Message):
    __slots__ = ("volume_id",)
    VOLUME_ID_FIELD_NUMBER: _ClassVar[int]
    volume_id: str
    def __init__(self, volume_id: _Optional[str] = ...) -> None: ...

class ListVolumesRequest(_message.Message):
    __slots__ = ("page_size", "page_token", "states", "runner_ids")
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    STATES_FIELD_NUMBER: _ClassVar[int]
    RUNNER_IDS_FIELD_NUMBER: _ClassVar[int]
    page_size: int
    page_token: str
    states: _containers.RepeatedScalarFieldContainer[VolumeState]
    runner_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, page_size: _Optional[int] = ..., page_token: _Optional[str] = ..., states: _Optional[_Iterable[_Union[VolumeState, str]]] = ..., runner_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class ListVolumesResponse(_message.Message):
    __slots__ = ("volumes", "next_page_token")
    VOLUMES_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    volumes: _containers.RepeatedCompositeFieldContainer[Volume]
    next_page_token: str
    def __init__(self, volumes: _Optional[_Iterable[_Union[Volume, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class UpdateVolumeRequest(_message.Message):
    __slots__ = ("volume_id", "volume", "update_mask")
    VOLUME_ID_FIELD_NUMBER: _ClassVar[int]
    VOLUME_FIELD_NUMBER: _ClassVar[int]
    UPDATE_MASK_FIELD_NUMBER: _ClassVar[int]
    volume_id: str
    volume: Volume
    update_mask: _field_mask_pb2.FieldMask
    def __init__(self, volume_id: _Optional[str] = ..., volume: _Optional[_Union[Volume, _Mapping]] = ..., update_mask: _Optional[_Union[_field_mask_pb2.FieldMask, _Mapping]] = ...) -> None: ...

class DeleteVolumeRequest(_message.Message):
    __slots__ = ("volume_id", "allow_missing", "force")
    VOLUME_ID_FIELD_NUMBER: _ClassVar[int]
    ALLOW_MISSING_FIELD_NUMBER: _ClassVar[int]
    FORCE_FIELD_NUMBER: _ClassVar[int]
    volume_id: str
    allow_missing: bool
    force: bool
    def __init__(self, volume_id: _Optional[str] = ..., allow_missing: bool = ..., force: bool = ...) -> None: ...

class ValidateVolumeRequest(_message.Message):
    __slots__ = ("volume_id",)
    VOLUME_ID_FIELD_NUMBER: _ClassVar[int]
    volume_id: str
    def __init__(self, volume_id: _Optional[str] = ...) -> None: ...
