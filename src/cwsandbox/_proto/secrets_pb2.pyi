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

class SecretStoreProviderType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SECRET_STORE_PROVIDER_TYPE_UNSPECIFIED: _ClassVar[SecretStoreProviderType]
    SECRET_STORE_PROVIDER_TYPE_WANDB: _ClassVar[SecretStoreProviderType]
SECRET_STORE_PROVIDER_TYPE_UNSPECIFIED: SecretStoreProviderType
SECRET_STORE_PROVIDER_TYPE_WANDB: SecretStoreProviderType

class SecretStoreReference(_message.Message):
    __slots__ = ("store_name", "secrets")
    STORE_NAME_FIELD_NUMBER: _ClassVar[int]
    SECRETS_FIELD_NUMBER: _ClassVar[int]
    store_name: str
    secrets: _containers.RepeatedCompositeFieldContainer[SecretMapping]
    def __init__(self, store_name: _Optional[str] = ..., secrets: _Optional[_Iterable[_Union[SecretMapping, _Mapping]]] = ...) -> None: ...

class SecretMapping(_message.Message):
    __slots__ = ("path", "field", "env_var")
    PATH_FIELD_NUMBER: _ClassVar[int]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    ENV_VAR_FIELD_NUMBER: _ClassVar[int]
    path: str
    field: str
    env_var: str
    def __init__(self, path: _Optional[str] = ..., field: _Optional[str] = ..., env_var: _Optional[str] = ...) -> None: ...

class ResolvedSecret(_message.Message):
    __slots__ = ("env_var", "value")
    ENV_VAR_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    env_var: str
    value: bytes
    def __init__(self, env_var: _Optional[str] = ..., value: _Optional[bytes] = ...) -> None: ...

class WandBStoreConfig(_message.Message):
    __slots__ = ("api_url", "team")
    API_URL_FIELD_NUMBER: _ClassVar[int]
    TEAM_FIELD_NUMBER: _ClassVar[int]
    api_url: str
    team: str
    def __init__(self, api_url: _Optional[str] = ..., team: _Optional[str] = ...) -> None: ...

class SecretStore(_message.Message):
    __slots__ = ("id", "organization_id", "name", "provider_type", "wandb", "created_at", "updated_at")
    ID_FIELD_NUMBER: _ClassVar[int]
    ORGANIZATION_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_TYPE_FIELD_NUMBER: _ClassVar[int]
    WANDB_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    id: str
    organization_id: str
    name: str
    provider_type: SecretStoreProviderType
    wandb: WandBStoreConfig
    created_at: _timestamp_pb2.Timestamp
    updated_at: _timestamp_pb2.Timestamp
    def __init__(self, id: _Optional[str] = ..., organization_id: _Optional[str] = ..., name: _Optional[str] = ..., provider_type: _Optional[_Union[SecretStoreProviderType, str]] = ..., wandb: _Optional[_Union[WandBStoreConfig, _Mapping]] = ..., created_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class CreateSecretStoreRequest(_message.Message):
    __slots__ = ("name", "provider_type", "wandb")
    NAME_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_TYPE_FIELD_NUMBER: _ClassVar[int]
    WANDB_FIELD_NUMBER: _ClassVar[int]
    name: str
    provider_type: SecretStoreProviderType
    wandb: WandBStoreConfig
    def __init__(self, name: _Optional[str] = ..., provider_type: _Optional[_Union[SecretStoreProviderType, str]] = ..., wandb: _Optional[_Union[WandBStoreConfig, _Mapping]] = ...) -> None: ...

class CreateSecretStoreResponse(_message.Message):
    __slots__ = ("secret_store",)
    SECRET_STORE_FIELD_NUMBER: _ClassVar[int]
    secret_store: SecretStore
    def __init__(self, secret_store: _Optional[_Union[SecretStore, _Mapping]] = ...) -> None: ...

class GetSecretStoreRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class GetSecretStoreResponse(_message.Message):
    __slots__ = ("secret_store",)
    SECRET_STORE_FIELD_NUMBER: _ClassVar[int]
    secret_store: SecretStore
    def __init__(self, secret_store: _Optional[_Union[SecretStore, _Mapping]] = ...) -> None: ...

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

class DeleteSecretStoreRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class DeleteSecretStoreResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
