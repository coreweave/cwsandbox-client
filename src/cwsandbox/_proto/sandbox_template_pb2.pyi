# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client
from cwsandbox._proto import sandbox_pb2 as _sandbox_pb2
from google.api import annotations_pb2 as _annotations_pb2
from google.api import field_behavior_pb2 as _field_behavior_pb2
from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import field_mask_pb2 as _field_mask_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SandboxTemplate(_message.Message):
    __slots__ = ("sandbox_template_id", "display_name", "spec", "revision", "create_time", "update_time", "delete_time", "create_user_id", "deleted", "attachments", "requires_cks")
    SANDBOX_TEMPLATE_ID_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_NAME_FIELD_NUMBER: _ClassVar[int]
    SPEC_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    CREATE_TIME_FIELD_NUMBER: _ClassVar[int]
    UPDATE_TIME_FIELD_NUMBER: _ClassVar[int]
    DELETE_TIME_FIELD_NUMBER: _ClassVar[int]
    CREATE_USER_ID_FIELD_NUMBER: _ClassVar[int]
    DELETED_FIELD_NUMBER: _ClassVar[int]
    ATTACHMENTS_FIELD_NUMBER: _ClassVar[int]
    REQUIRES_CKS_FIELD_NUMBER: _ClassVar[int]
    sandbox_template_id: str
    display_name: str
    spec: _sandbox_pb2.PartialSandboxSpec
    revision: int
    create_time: _timestamp_pb2.Timestamp
    update_time: _timestamp_pb2.Timestamp
    delete_time: _timestamp_pb2.Timestamp
    create_user_id: str
    deleted: bool
    attachments: _struct_pb2.Struct
    requires_cks: bool
    def __init__(self, sandbox_template_id: _Optional[str] = ..., display_name: _Optional[str] = ..., spec: _Optional[_Union[_sandbox_pb2.PartialSandboxSpec, _Mapping]] = ..., revision: _Optional[int] = ..., create_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., update_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., delete_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., create_user_id: _Optional[str] = ..., deleted: bool = ..., attachments: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., requires_cks: bool = ...) -> None: ...

class CreateSandboxTemplateRequest(_message.Message):
    __slots__ = ("sandbox_template", "request_id")
    SANDBOX_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    sandbox_template: SandboxTemplate
    request_id: str
    def __init__(self, sandbox_template: _Optional[_Union[SandboxTemplate, _Mapping]] = ..., request_id: _Optional[str] = ...) -> None: ...

class GetSandboxTemplateRequest(_message.Message):
    __slots__ = ("sandbox_template_id", "show_deleted")
    SANDBOX_TEMPLATE_ID_FIELD_NUMBER: _ClassVar[int]
    SHOW_DELETED_FIELD_NUMBER: _ClassVar[int]
    sandbox_template_id: str
    show_deleted: bool
    def __init__(self, sandbox_template_id: _Optional[str] = ..., show_deleted: bool = ...) -> None: ...

class ListSandboxTemplatesRequest(_message.Message):
    __slots__ = ("page_size", "page_token", "show_deleted")
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    SHOW_DELETED_FIELD_NUMBER: _ClassVar[int]
    page_size: int
    page_token: str
    show_deleted: bool
    def __init__(self, page_size: _Optional[int] = ..., page_token: _Optional[str] = ..., show_deleted: bool = ...) -> None: ...

class ListSandboxTemplatesResponse(_message.Message):
    __slots__ = ("sandbox_templates", "next_page_token")
    SANDBOX_TEMPLATES_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    sandbox_templates: _containers.RepeatedCompositeFieldContainer[SandboxTemplate]
    next_page_token: str
    def __init__(self, sandbox_templates: _Optional[_Iterable[_Union[SandboxTemplate, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class UpdateSandboxTemplateRequest(_message.Message):
    __slots__ = ("sandbox_template_id", "sandbox_template", "update_mask")
    SANDBOX_TEMPLATE_ID_FIELD_NUMBER: _ClassVar[int]
    SANDBOX_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    UPDATE_MASK_FIELD_NUMBER: _ClassVar[int]
    sandbox_template_id: str
    sandbox_template: SandboxTemplate
    update_mask: _field_mask_pb2.FieldMask
    def __init__(self, sandbox_template_id: _Optional[str] = ..., sandbox_template: _Optional[_Union[SandboxTemplate, _Mapping]] = ..., update_mask: _Optional[_Union[_field_mask_pb2.FieldMask, _Mapping]] = ...) -> None: ...

class DeleteSandboxTemplateRequest(_message.Message):
    __slots__ = ("sandbox_template_id",)
    SANDBOX_TEMPLATE_ID_FIELD_NUMBER: _ClassVar[int]
    sandbox_template_id: str
    def __init__(self, sandbox_template_id: _Optional[str] = ...) -> None: ...
