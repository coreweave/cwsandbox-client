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
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FileSystemSnapshotBucketMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FILE_SYSTEM_SNAPSHOT_BUCKET_MODE_UNSPECIFIED: _ClassVar[FileSystemSnapshotBucketMode]
    FILE_SYSTEM_SNAPSHOT_BUCKET_MODE_CW_MANAGED: _ClassVar[FileSystemSnapshotBucketMode]
    FILE_SYSTEM_SNAPSHOT_BUCKET_MODE_BRING_YOUR_OWN: _ClassVar[FileSystemSnapshotBucketMode]

class SecretStoreProviderType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SECRET_STORE_PROVIDER_TYPE_UNSPECIFIED: _ClassVar[SecretStoreProviderType]
    SECRET_STORE_PROVIDER_TYPE_WANDB: _ClassVar[SecretStoreProviderType]
FILE_SYSTEM_SNAPSHOT_BUCKET_MODE_UNSPECIFIED: FileSystemSnapshotBucketMode
FILE_SYSTEM_SNAPSHOT_BUCKET_MODE_CW_MANAGED: FileSystemSnapshotBucketMode
FILE_SYSTEM_SNAPSHOT_BUCKET_MODE_BRING_YOUR_OWN: FileSystemSnapshotBucketMode
SECRET_STORE_PROVIDER_TYPE_UNSPECIFIED: SecretStoreProviderType
SECRET_STORE_PROVIDER_TYPE_WANDB: SecretStoreProviderType

class ObjectStorageWifConfig(_message.Message):
    __slots__ = ("wif_config_id", "enabled", "allowed_buckets", "max_permission", "create_time", "update_time")
    WIF_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    ALLOWED_BUCKETS_FIELD_NUMBER: _ClassVar[int]
    MAX_PERMISSION_FIELD_NUMBER: _ClassVar[int]
    CREATE_TIME_FIELD_NUMBER: _ClassVar[int]
    UPDATE_TIME_FIELD_NUMBER: _ClassVar[int]
    wif_config_id: str
    enabled: bool
    allowed_buckets: _containers.RepeatedScalarFieldContainer[str]
    max_permission: _sandbox_pb2.ObjectStoragePermission
    create_time: _timestamp_pb2.Timestamp
    update_time: _timestamp_pb2.Timestamp
    def __init__(self, wif_config_id: _Optional[str] = ..., enabled: bool = ..., allowed_buckets: _Optional[_Iterable[str]] = ..., max_permission: _Optional[_Union[_sandbox_pb2.ObjectStoragePermission, str]] = ..., create_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., update_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class GetObjectStorageWifConfigRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class UpdateObjectStorageWifConfigRequest(_message.Message):
    __slots__ = ("object_storage_wif_config", "update_mask")
    OBJECT_STORAGE_WIF_CONFIG_FIELD_NUMBER: _ClassVar[int]
    UPDATE_MASK_FIELD_NUMBER: _ClassVar[int]
    object_storage_wif_config: ObjectStorageWifConfig
    update_mask: _field_mask_pb2.FieldMask
    def __init__(self, object_storage_wif_config: _Optional[_Union[ObjectStorageWifConfig, _Mapping]] = ..., update_mask: _Optional[_Union[_field_mask_pb2.FieldMask, _Mapping]] = ...) -> None: ...

class DeleteObjectStorageWifConfigRequest(_message.Message):
    __slots__ = ("allow_missing",)
    ALLOW_MISSING_FIELD_NUMBER: _ClassVar[int]
    allow_missing: bool
    def __init__(self, allow_missing: bool = ...) -> None: ...

class FileSystemSnapshotBucketConfig(_message.Message):
    __slots__ = ("bucket_name", "region", "mode", "effective_bucket_name")
    BUCKET_NAME_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    EFFECTIVE_BUCKET_NAME_FIELD_NUMBER: _ClassVar[int]
    bucket_name: str
    region: str
    mode: FileSystemSnapshotBucketMode
    effective_bucket_name: str
    def __init__(self, bucket_name: _Optional[str] = ..., region: _Optional[str] = ..., mode: _Optional[_Union[FileSystemSnapshotBucketMode, str]] = ..., effective_bucket_name: _Optional[str] = ...) -> None: ...

class GetFileSystemSnapshotBucketConfigRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class UpdateFileSystemSnapshotBucketConfigRequest(_message.Message):
    __slots__ = ("file_system_snapshot_bucket_config", "update_mask")
    FILE_SYSTEM_SNAPSHOT_BUCKET_CONFIG_FIELD_NUMBER: _ClassVar[int]
    UPDATE_MASK_FIELD_NUMBER: _ClassVar[int]
    file_system_snapshot_bucket_config: FileSystemSnapshotBucketConfig
    update_mask: _field_mask_pb2.FieldMask
    def __init__(self, file_system_snapshot_bucket_config: _Optional[_Union[FileSystemSnapshotBucketConfig, _Mapping]] = ..., update_mask: _Optional[_Union[_field_mask_pb2.FieldMask, _Mapping]] = ...) -> None: ...

class WandBStoreConfig(_message.Message):
    __slots__ = ("api_url", "team")
    API_URL_FIELD_NUMBER: _ClassVar[int]
    TEAM_FIELD_NUMBER: _ClassVar[int]
    api_url: str
    team: str
    def __init__(self, api_url: _Optional[str] = ..., team: _Optional[str] = ...) -> None: ...

class SecretStore(_message.Message):
    __slots__ = ("name", "provider_type", "wandb", "create_time", "update_time")
    NAME_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_TYPE_FIELD_NUMBER: _ClassVar[int]
    WANDB_FIELD_NUMBER: _ClassVar[int]
    CREATE_TIME_FIELD_NUMBER: _ClassVar[int]
    UPDATE_TIME_FIELD_NUMBER: _ClassVar[int]
    name: str
    provider_type: SecretStoreProviderType
    wandb: WandBStoreConfig
    create_time: _timestamp_pb2.Timestamp
    update_time: _timestamp_pb2.Timestamp
    def __init__(self, name: _Optional[str] = ..., provider_type: _Optional[_Union[SecretStoreProviderType, str]] = ..., wandb: _Optional[_Union[WandBStoreConfig, _Mapping]] = ..., create_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., update_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class CreateSecretStoreRequest(_message.Message):
    __slots__ = ("secret_store", "request_id")
    SECRET_STORE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    secret_store: SecretStore
    request_id: str
    def __init__(self, secret_store: _Optional[_Union[SecretStore, _Mapping]] = ..., request_id: _Optional[str] = ...) -> None: ...

class GetSecretStoreRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class ListSecretStoresRequest(_message.Message):
    __slots__ = ("page_size", "page_token")
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    page_size: int
    page_token: str
    def __init__(self, page_size: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class ListSecretStoresResponse(_message.Message):
    __slots__ = ("secret_stores", "next_page_token")
    SECRET_STORES_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    secret_stores: _containers.RepeatedCompositeFieldContainer[SecretStore]
    next_page_token: str
    def __init__(self, secret_stores: _Optional[_Iterable[_Union[SecretStore, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class UpdateSecretStoreRequest(_message.Message):
    __slots__ = ("secret_store", "update_mask")
    SECRET_STORE_FIELD_NUMBER: _ClassVar[int]
    UPDATE_MASK_FIELD_NUMBER: _ClassVar[int]
    secret_store: SecretStore
    update_mask: _field_mask_pb2.FieldMask
    def __init__(self, secret_store: _Optional[_Union[SecretStore, _Mapping]] = ..., update_mask: _Optional[_Union[_field_mask_pb2.FieldMask, _Mapping]] = ...) -> None: ...

class DeleteSecretStoreRequest(_message.Message):
    __slots__ = ("name", "allow_missing")
    NAME_FIELD_NUMBER: _ClassVar[int]
    ALLOW_MISSING_FIELD_NUMBER: _ClassVar[int]
    name: str
    allow_missing: bool
    def __init__(self, name: _Optional[str] = ..., allow_missing: bool = ...) -> None: ...
